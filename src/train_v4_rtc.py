#!/usr/bin/env python3
"""v4 recipe retrained with the RTC systems (v4r): the five city caches of train_v4_cached_ensemble plus the RTC days
exported by serve/rtc_audit.py --dump (ETA/data/rtc_cache/<system>_<day>.npz: the live engine's own rows, model.json
causal for each day, y = the interpolated stop crossing).

Each RTC system gets its own city_idx (tricity 5, tgsrtc 6, apsrtc 7) instead of borrowing Gurugram's 3.
Same models, hyperparameters, blend search, P10/P90 and per-bucket CQR margins as v4. Days:
  RTC train 18-22 Nov, validation 23 Nov, held-out 24 Nov (never seen in training or early stopping).
The report scores every held-out day with the current v4 (v4_clean_ensemble) and with v4r on identical rows, so
the five cities' regression check and the RTC gains come from one run.

    ETA/.venv/bin/python src/train_v4_rtc.py [--out ETA/models/v4r_rtc]
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

BASE_DIR = Path("~/projects")
sys.path.insert(0, str(BASE_DIR / "ETA-wt-baseline" / "src"))
sys.path.insert(0, str(BASE_DIR / "ETA-wt-models" / "src"))
sys.path.insert(0, str(BASE_DIR / "ETA-wt-models" / "serve"))

import train_v2_deepreta_cqr as v2  # type: ignore  # noqa: E402
import train_v3_ensemble_agaci as v3  # type: ignore  # noqa: E402
import train_v4_cached_ensemble as v4  # type: ignore  # noqa: E402

DATA = BASE_DIR / "ETA" / "data"
CI = v4.V4_FEATURES.index("city_idx")
RTC_IDX = {"tricity": 5, "tgsrtc": 6, "apsrtc2": 7}
RTC_TRAIN = ["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22"]
RTC_VAL, RTC_TEST = "2026-11-23", "2026-11-24"
CITY_PLAN = {   # as train_v4_cached_ensemble (train days, validation day, test day, train sample, test sample)
    "mtc":    (["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.16, 0.35),
    "bmtc":   (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.35, 0.35),
    "mumbai": (["2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"], "2026-11-16", "2026-11-17", 0.50, 0.50),
    "ncr":    (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"], "2026-11-16", "2026-11-17", 1.00, 0.50),
    "delhi":  (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.04, 0.10),
}


def rtc(system, day):
    p = DATA / "rtc_cache" / f"{system}_{day}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    X = z["X"].astype(np.float32)
    if X.shape[1] != len(v4.V4_FEATURES):
        X = X[:, v4.KEEP_IDX]
    X[:, CI] = RTC_IDX[system]
    return X, z["y"].astype(np.float64), z["b"].astype(np.float64)


def ratio(y, b):
    return np.clip(np.log(y / b), -1.5, 1.5).astype(np.float32)


def summarize(y, p, lo, hi, b):
    err = np.abs(p - y)
    out = {"n": int(len(y)), "mae_s": round(float(err.mean()), 1), "nowcast_mae_s": round(float(np.abs(b - y).mean()), 1),
           "cov80_pct": round(100.0 * float(np.mean((y >= lo) & (y <= hi))), 1), "bias_s": round(float(np.mean(p - y)), 1)}
    for name, (a, c) in {"1-5m": (60, 300), "5-10m": (300, 600), "10-20m": (600, 1200), "20-30m": (1200, 1800)}.items():
        m = (y >= a) & (y < c)
        if m.any():
            out[name] = round(float(err[m].mean()), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE_DIR / "ETA" / "models" / "v4r_rtc"))
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = DATA / "v3_cache"
    tr, va, te = [], [], {}
    for city, (tr_days, va_day, te_day, tr_sr, te_sr) in CITY_PLAN.items():
        for d in tr_days:
            g = v4.load_cached(cache, city, d, tr_sr, 4, 220_000)
            if g:
                tr.append(g)
        g = v4.load_cached(cache, city, va_day, te_sr, 3, 260_000)
        if g:
            va.append(g)
        g = v4.load_cached(cache, city, te_day, te_sr, 3, 1_500_000)
        if g:
            te[city] = (te_day, *g)
    for s in RTC_IDX:
        for d in RTC_TRAIN:
            g = rtc(s, d)
            if g:
                tr.append(g)
        g = rtc(s, RTC_VAL)
        if g:
            va.append(g)
        g = rtc(s, RTC_TEST)
        if g:
            te[s] = (RTC_TEST, *g)
    X_tr = np.vstack([g[0] for g in tr]); y_tr = np.concatenate([g[1] for g in tr]); b_tr = np.concatenate([g[2] for g in tr])
    X_va = np.vstack([g[0] for g in va]); y_va = np.concatenate([g[1] for g in va]); b_va = np.concatenate([g[2] for g in va])
    r_tr, r_va = ratio(y_tr, b_tr), ratio(y_va, b_va)
    print(f"[v4r] train {len(X_tr):,} val {len(X_va):,} | RTC train rows "
          + ", ".join(f"{s} {int(np.sum(X_tr[:, CI] == i)):,}" for s, i in RTC_IDX.items()), flush=True)

    F = v4.V4_FEATURES
    dtr = lgb.Dataset(X_tr, label=r_tr, feature_name=F, categorical_feature=["city_idx", "bus_class_idx"])
    dva = lgb.Dataset(X_va, label=r_va, reference=dtr, feature_name=F, categorical_feature=["city_idx", "bus_class_idx"])
    m_lgb = lgb.train(
        {"objective": "huber", "alpha": 0.25, "metric": "l1", "learning_rate": 0.04, "num_leaves": 191,
         "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "n_jobs": -1, "verbose": -1, "seed": 42},
        dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(45, verbose=False), lgb.log_evaluation(250)])
    m_lgb.save_model(str(out / "v4_lgb_huber.txt"))
    p_lgb = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)
    m_xgb = xgb.XGBRegressor(
        objective="reg:pseudohubererror", huber_slope=0.25, eval_metric="mae", n_estimators=800, learning_rate=0.05,
        max_depth=9, min_child_weight=80, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
        random_state=42, early_stopping_rounds=40)
    m_xgb.fit(X_tr, r_tr, eval_set=[(X_va, r_va)], verbose=250)
    m_xgb.save_model(str(out / "v4_xgb_huber.json"))
    p_xgb = m_xgb.predict(X_va)
    m_cat = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=0.07, depth=8, random_seed=42,
                              thread_count=-1, verbose=300)
    m_cat.fit(X_tr, r_tr, eval_set=(X_va, r_va), early_stopping_rounds=50, use_best_model=True)
    m_cat.save_model(str(out / "v4_cat_mae.cbm"))
    p_cat = m_cat.predict(X_va)

    best_w, best = (0.5, 0.3, 0.2), 1e9
    for w1 in np.linspace(0.2, 0.7, 11):
        for w2 in np.linspace(0.1, 0.6, 11):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            mae = float(np.mean(np.abs(b_va * np.exp(w1 * p_lgb + w2 * p_xgb + w3 * p_cat) - y_va)))
            if mae < best:
                best, best_w = mae, (round(float(w1), 3), round(float(w2), 3), round(float(w3), 3))
    print(f"[v4r] blend {best_w} val MAE {best:.1f}s", flush=True)

    q = {}
    for alpha in (0.10, 0.90):
        qm = lgb.train(
            {"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.05, "num_leaves": 127,
             "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "verbose": -1, "seed": 42},
            dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(35, verbose=False)])
        qm.save_model(str(out / f"v4_lgb_q{int(alpha * 100)}.txt"))
        q[alpha] = qm.predict(X_va, num_iteration=qm.best_iteration)
    hb = v2.horizon_bucket_idx(b_va)
    cqr = {k: float(np.quantile(np.maximum(q[0.10][hb == k] - r_va[hb == k], r_va[hb == k] - q[0.90][hb == k]), 0.80))
           if np.sum(hb == k) >= 50 else 0.05 for k in range(5)}
    rep = {"weights": best_w, "cqr_by_bucket": cqr, "city_idx": {**{c: i for i, c in enumerate(CITY_PLAN)}, **RTC_IDX},
           "rtc_days": {"train": RTC_TRAIN, "val": RTC_VAL, "test": RTC_TEST}, "cities": {}}
    (out / "v4_report.json").write_text(json.dumps(rep, indent=2))   # Ensemble reads weights + cqr from here

    # held-out days: current v4 vs v4r on identical rows (P50 and the 80% range with each model's own margins)
    import eta_engine as E
    old = E.Ensemble(str(BASE_DIR / "ETA" / "models" / "v4_clean_ensemble"), threads=8)
    new = E.Ensemble(str(out), threads=8)
    print(f"\n{'system':<9} {'day':<11} {'n':>9} {'nowcast':>8} {'v4':>7} {'v4r':>7}  cov80 v4/v4r   20-30m v4/v4r")
    for city, (day, X, y, b) in te.items():
        Xo = X.copy()
        if city in RTC_IDX:
            Xo[:, CI] = 3                                  # what the live service feeds v4 today
        po, lo_o, hi_o = old.predict(Xo, b)[:3]
        pn, lo_n, hi_n = new.predict(X, b)[:3]
        so, sn = summarize(y, po, lo_o, hi_o, b), summarize(y, pn, lo_n, hi_n, b)
        rep["cities"][city] = {"day": day, "v4": so, "v4r": sn}
        print(f"{city:<9} {day:<11} {len(y):>9,} {so['nowcast_mae_s']:>8} {so['mae_s']:>7} {sn['mae_s']:>7}  "
              f"{so['cov80_pct']:>5}/{sn['cov80_pct']:<5}  {so.get('20-30m')}/{sn.get('20-30m')}", flush=True)
    (out / "v4_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nsaved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
