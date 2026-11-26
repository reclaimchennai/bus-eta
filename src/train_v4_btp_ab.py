#!/usr/bin/env python3
"""A/B: v4 cached ensemble with vs without BTP congestion-profile features (BMTC).

  --mode screen : BMTC-only LightGBM Huber, arms base / +2 / +4 (fast signal check)
  --mode full   : the full v4 recipe (all 5 cities, LGB+XGB+CAT blend, q10/q90 + AgACI)
                  trained twice with identical seeds: base 74 features vs 74 + BTP arm.
Both arms are evaluated on the same held-out BMTC day (2026-11-23).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))

import btp_features as btp  # type: ignore
import eval_eta  # type: ignore
import extract_dataset as ed  # type: ignore
import train_v2_deepreta_cqr as v2  # type: ignore
import train_v3_ensemble_agaci as v3  # type: ignore
from train_v4_cached_ensemble import KEEP_IDX, V4_FEATURES  # type: ignore

DATA = ROOT / "data"
CACHE = DATA / "v3_cache"
PROFILE = DATA / "btp" / "stop_congestion_how.json"
OUT = ROOT / "models" / "v4_btp_ab"

DAY_PLAN = {
    "mtc":    (["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.16, 0.35),
    "bmtc":   (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.35, 0.35),
    "mumbai": (["2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"], "2026-11-16", "2026-11-17", 0.50, 0.50),
    "ncr":    (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"], "2026-11-16", "2026-11-17", 1.00, 0.50),
    "delhi":  (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.04, 0.10),
}

_routes = None
_prof = None


def load(city, day, sr, stride, max_q):
    cp = CACHE / f"v3_{city}_{day}_{int(sr*1000)}_{stride}_{max_q}.npz"
    if not cp.exists():
        return None
    z = np.load(cp, allow_pickle=False)
    X = z["X"][:, KEEP_IDX]
    y, b = z["y"], z["b"]
    if city == "bmtc":
        fc = DATA / "btp" / f"btpfeat_{cp.stem}.npy"
        if fc.exists():
            F = np.load(fc)
        else:
            global _routes, _prof
            if _routes is None:
                _routes = ed.Routes(DATA / "meta" / "bmtc" / "route_stops.json", DATA / "meta" / "bmtc" / "stop_dist.json")
                _prof = btp.load_profile(PROFILE)
            F = btp.compute(X, z["rids"], b, V4_FEATURES, _routes.ids, _routes.ds, _prof)
            np.save(fc, F)
    else:
        F = np.full((len(X), 4), np.nan, dtype=np.float32)
        F[:, 3] = 0.0
    return X, F, y, b


def gather(cities):
    tr, va, te = [], [], {}
    for city in cities:
        tr_days, va_day, te_day, tr_sr, te_sr = DAY_PLAN[city]
        for d in tr_days:
            g = load(city, d, tr_sr, 4, 220_000)
            if g:
                tr.append(g)
        g = load(city, va_day, te_sr, 3, 260_000)
        if g:
            va.append(g)
        g = load(city, te_day, te_sr, 3, 1_500_000)
        if g:
            te[city] = (te_day, *g)
    cat = lambda L, i: np.vstack([x[i] for x in L]) if i < 2 else np.concatenate([x[i] for x in L])
    return [cat(tr, i) for i in range(4)], [cat(va, i) for i in range(4)], te


def arm_matrix(X, F, n_btp):
    return X if n_btp == 0 else np.hstack([X, F[:, :n_btp]])


def resid(y, b):
    return np.clip(np.log(y / b), -1.5, 1.5).astype(np.float32)


def screen():
    (Xtr, Ftr, ytr, btr), (Xva, Fva, yva, bva), te = gather(["bmtc"])
    te_day, Xte, Fte, yte, bte = te["bmtc"]
    rtr, rva = resid(ytr, btr), resid(yva, bva)
    cov = Fte[:, 3]
    print(f"BMTC train {len(Xtr):,} val {len(Xva):,} test {len(Xte):,}; test rows with any BTP stop on path: {np.mean(cov > 0)*100:.1f}%, next stop covered: {np.mean(~np.isnan(Fte[:,0]))*100:.1f}%")
    res = {}
    for n_btp in (0, 2, 4):
        names = V4_FEATURES + btp.BTP_FEATURES[:n_btp]
        dtr = lgb.Dataset(arm_matrix(Xtr, Ftr, n_btp), label=rtr, feature_name=names, categorical_feature=["city_idx", "bus_class_idx"])
        dva = lgb.Dataset(arm_matrix(Xva, Fva, n_btp), label=rva, reference=dtr)
        m = lgb.train({"objective": "huber", "alpha": 0.25, "metric": "l1", "learning_rate": 0.04, "num_leaves": 191,
                       "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                       "n_jobs": -1, "verbose": -1, "seed": 42},
                      dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)])
        p = bte * np.exp(m.predict(arm_matrix(Xte, Fte, n_btp), num_iteration=m.best_iteration))
        err = np.abs(p - yte)
        on = cov > 0
        res[n_btp] = {"mae": float(err.mean()), "p80": float(np.percentile(err, 80)),
                      "mae_btp_path": float(err[on].mean()), "mae_no_btp": float(err[~on].mean()), "iters": m.best_iteration}
        if n_btp:
            g = dict(zip(names, m.feature_importance("gain")))
            tot = sum(g.values())
            res[n_btp]["gain_pct"] = {k: round(100 * g[k] / tot, 2) for k in btp.BTP_FEATURES[:n_btp]}
        print(f"arm +{n_btp}: {res[n_btp]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.json").write_text(json.dumps(res, indent=2))


def seeds():
    """Noise check: base vs +2 real vs +2 shuffled (row-permuted) BTP features, 3 seeds, BMTC-only LGB."""
    (Xtr, Ftr, ytr, btr), (Xva, Fva, yva, bva), te = gather(["bmtc"])
    _, Xte, Fte, yte, bte = te["bmtc"]
    rtr, rva = resid(ytr, btr), resid(yva, bva)
    rng = np.random.default_rng(7)
    shuf = lambda F: F[rng.permutation(len(F))]
    arms = {"base": (Xtr, Xva, Xte),
            "btp2": (arm_matrix(Xtr, Ftr, 2), arm_matrix(Xva, Fva, 2), arm_matrix(Xte, Fte, 2)),
            "shuffled2": (arm_matrix(Xtr, shuf(Ftr), 2), arm_matrix(Xva, shuf(Fva), 2), arm_matrix(Xte, shuf(Fte), 2))}
    res = {}
    for seed in (1, 2, 3):
        for tag, (a, b_, c) in arms.items():
            dtr = lgb.Dataset(a, label=rtr, categorical_feature=[0, 1])
            dva = lgb.Dataset(b_, label=rva, reference=dtr)
            m = lgb.train({"objective": "huber", "alpha": 0.25, "metric": "l1", "learning_rate": 0.04, "num_leaves": 191,
                           "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                           "n_jobs": -1, "verbose": -1, "seed": seed},
                          dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)])
            mae = float(np.mean(np.abs(bte * np.exp(m.predict(c, num_iteration=m.best_iteration)) - yte)))
            res.setdefault(tag, []).append(mae)
            print(f"seed {seed} {tag}: {mae:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "seeds.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, f"mean {np.mean(v):.2f} sd {np.std(v):.2f}")


def train_full(Xtr, rtr, Xva, rva, yva, bva, names, tag):
    t0 = time.time()
    cats = ["city_idx", "bus_class_idx"]
    dtr = lgb.Dataset(Xtr, label=rtr, feature_name=names, categorical_feature=cats)
    dva = lgb.Dataset(Xva, label=rva, reference=dtr, feature_name=names, categorical_feature=cats)
    m_lgb = lgb.train({"objective": "huber", "alpha": 0.25, "metric": "l1", "learning_rate": 0.04, "num_leaves": 191,
                       "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                       "n_jobs": -1, "verbose": -1, "seed": 42},
                      dtr, num_boost_round=1000, valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)])
    m_xgb = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=0.25, eval_metric="mae",
                             n_estimators=800, learning_rate=0.05, max_depth=9, min_child_weight=80,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                             random_state=42, early_stopping_rounds=40)
    m_xgb.fit(Xtr, rtr, eval_set=[(Xva, rva)], verbose=False)
    m_cat = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=0.07, depth=8,
                              random_seed=42, thread_count=-1, verbose=0)
    m_cat.fit(Xtr, rtr, eval_set=(Xva, rva), early_stopping_rounds=50, use_best_model=True)
    pl_, px, pc = m_lgb.predict(Xva, num_iteration=m_lgb.best_iteration), m_xgb.predict(Xva), m_cat.predict(Xva)
    best_w, best = (0.5, 0.3, 0.2), 1e9
    for w1 in np.linspace(0.2, 0.7, 11):
        for w2 in np.linspace(0.1, 0.6, 11):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            mae = float(np.mean(np.abs(bva * np.exp(w1 * pl_ + w2 * px + w3 * pc) - yva)))
            if mae < best:
                best, best_w = mae, (round(float(w1), 3), round(float(w2), 3), round(float(w3), 3))
    q = {}
    for a in (0.10, 0.90):
        q[a] = lgb.train({"objective": "quantile", "alpha": a, "metric": "quantile", "learning_rate": 0.05, "num_leaves": 127,
                          "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "verbose": -1, "seed": 42},
                         dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(35, verbose=False)])
    hb = v2.horizon_bucket_idx(bva)
    qlo, qhi = q[0.10].predict(Xva, num_iteration=q[0.10].best_iteration), q[0.90].predict(Xva, num_iteration=q[0.90].best_iteration)
    cqr = {i: float(np.quantile(np.maximum(qlo[hb == i] - rva[hb == i], rva[hb == i] - qhi[hb == i]), 0.80)) if np.sum(hb == i) >= 50 else 0.05 for i in range(5)}
    print(f"[{tag}] trained in {time.time()-t0:.0f}s, weights {best_w}, val MAE {best:.2f}s", flush=True)
    return m_lgb, m_xgb, m_cat, best_w, q, cqr


def eval_city(models, Xte, yte, bte):
    m_lgb, m_xgb, m_cat, (w1, w2, w3), q, cqr = models
    rt = np.clip(np.log(yte / bte), -1.5, 1.5)
    rh = w1 * m_lgb.predict(Xte, num_iteration=m_lgb.best_iteration) + w2 * m_xgb.predict(Xte) + w3 * m_cat.predict(Xte)
    lo, hi = q[0.10].predict(Xte, num_iteration=q[0.10].best_iteration), q[0.90].predict(Xte, num_iteration=q[0.90].best_iteration)
    hb = v2.horizon_bucket_idx(bte)
    alo, ahi = np.empty_like(lo), np.empty_like(hi)
    for i in range(5):
        msk = hb == i
        if np.any(msk):
            alo[msk], ahi[msk] = v3.agaci_calibrate_online(rt[msk], lo[msk], hi[msk], cqr[i])
    pred = bte * np.exp(rh)
    s = eval_eta.summarize_errors(list(zip(yte.tolist(), pred.tolist(), (bte * np.exp(alo)).tolist(), (bte * np.exp(ahi)).tolist())))
    return s, pred


def full(n_btp):
    cities = list(DAY_PLAN)
    (Xtr, Ftr, ytr, btr), (Xva, Fva, yva, bva), te = gather(cities)
    rtr, rva = resid(ytr, btr), resid(yva, bva)
    print(f"all cities: train {len(Xtr):,} val {len(Xva):,}", flush=True)
    report = {"n_btp": n_btp, "arms": {}}
    preds = {}
    for tag, nb in (("base", 0), (f"btp{n_btp}", n_btp)):
        names = V4_FEATURES + btp.BTP_FEATURES[:nb]
        models = train_full(arm_matrix(Xtr, Ftr, nb), rtr, arm_matrix(Xva, Fva, nb), rva, yva, bva, names, tag)
        report["arms"][tag] = {"weights": models[3], "cities": {}}
        for city, (day, Xte, Fte, yte, bte) in te.items():
            s, p = eval_city(models, arm_matrix(Xte, Fte, nb), yte, bte)
            report["arms"][tag]["cities"][city] = s
            if city == "bmtc":
                preds[tag] = p
                on = Fte[:, 3] > 0
                err = np.abs(p - yte)
                report["arms"][tag]["bmtc_subsets"] = {"btp_on_path_n": int(on.sum()), "btp_on_path_mae": float(err[on].mean()),
                                                       "no_btp_mae": float(err[~on].mean())}
        if nb:
            g = dict(zip(names, models[0].feature_importance("gain")))
            tot = sum(g.values())
            report["arms"][tag]["btp_gain_pct"] = {k: round(100 * g[k] / tot, 3) for k in btp.BTP_FEATURES[:nb]}
        print(json.dumps({c: v["ALL(1-30m)"] for c, v in report["arms"][tag]["cities"].items()}), flush=True)
    # paired bootstrap on BMTC by query row (MAE diff)
    _, _, Fte, yte, bte = te["bmtc"]
    d = np.abs(preds[f"btp{n_btp}"] - yte) - np.abs(preds["base"] - yte)
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
    report["bmtc_mae_diff_s"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"full_btp{n_btp}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["bmtc_mae_diff_s"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "full", "seeds"], default="screen")
    ap.add_argument("--n-btp", type=int, default=2)
    a = ap.parse_args()
    {"screen": screen, "seeds": seeds}.get(a.mode, lambda: full(a.n_btp))()
