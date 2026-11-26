#!/usr/bin/env python3
"""v4 Leak-Free Stacked Ensemble (LightGBM Huber + XGBoost Pseudo-Huber + CatBoost MAE + Online AgACI).

Loads directly from ETA/data/v3_cache/*.npz in ~15 seconds (ZERO raw-file replay!),
drops the 3 in-sample target-encoded route prior columns that caused v3 overfitting on
Delhi/BMTC/NCR, keeping ALL 74 leak-free features:
  - Overpass OSM infrastructure (new Chennai flyovers, signals, rail crossings, roundabouts)
  - Verified 2025-2026 Indian National & State Holidays
  - Baidu ConSTGAT path geometry (stops/km, max sinuosity, cumulative turning angle)
  - Shared-corridor multi-route live probes + speed CV/stop-go traffic discipline proxies
  - Bus bunching headway + driver pace + pol.reclaimchennai.city weather & 3h/6h/24h rain lag
  - Online Adaptive Conformal Inference (AgACI, Zaffran et al. 2022) on UncertBAT Mondrian CQR!
"""
from __future__ import annotations

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

import eval_eta  # type: ignore
import train_v2_deepreta_cqr as v2  # type: ignore
import train_v3_ensemble_agaci as v3  # type: ignore

DROP_COLS = {"route_bias_prior", "route_wknd_bias_prior", "city_hour_wknd_bias_prior"}
KEEP_IDX = [i for i, c in enumerate(v3.V3_FEATURES) if c not in DROP_COLS]
V4_FEATURES = [v3.V3_FEATURES[i] for i in KEEP_IDX]


def load_cached(cache_dir: Path, city: str, day: str, sr: float, stride: int, max_q: int):
    cp = cache_dir / f"v3_{city}_{day}_{int(sr*1000)}_{stride}_{max_q}.npz"
    if not cp.exists():
        return None
    z = np.load(cp, allow_pickle=False)
    return z["X"][:, KEEP_IDX], z["y"], z["b"]


def main():
    t0 = time.time()
    data_root = (ROOT / "data")
    cache_dir = data_root / "v3_cache"
    model_dir = (ROOT / "models/v4_clean_ensemble")
    model_dir.mkdir(parents=True, exist_ok=True)

    day_plan = {
        "mtc":    (["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.16, 0.35),
        "bmtc":   (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.35, 0.35),
        "mumbai": (["2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"], "2026-11-16", "2026-11-17", 0.50, 0.50),
        "ncr":    (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"], "2026-11-16", "2026-11-17", 1.00, 0.50),
        "delhi":  (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.04, 0.10),
    }

    X_tr_l, y_tr_l, b_tr_l = [], [], []
    X_va_l, y_va_l, b_va_l = [], [], []
    test_data = {}

    for city, (tr_days, va_day, te_day, tr_sr, te_sr) in day_plan.items():
        for d in tr_days:
            got = load_cached(cache_dir, city, d, tr_sr, 4, 220_000)
            if got:
                X_tr_l.append(got[0]); y_tr_l.append(got[1]); b_tr_l.append(got[2])
        got_v = load_cached(cache_dir, city, va_day, te_sr, 3, 260_000)
        if got_v:
            X_va_l.append(got_v[0]); y_va_l.append(got_v[1]); b_va_l.append(got_v[2])
        got_t = load_cached(cache_dir, city, te_day, te_sr, 3, 1_500_000)
        if got_t:
            test_data[city] = (te_day, *got_t)

    X_tr = np.vstack(X_tr_l); y_tr = np.concatenate(y_tr_l); b_tr = np.concatenate(b_tr_l)
    X_va = np.vstack(X_va_l); y_va = np.concatenate(y_va_l); b_va = np.concatenate(b_va_l)
    r_tr = np.clip(np.log(y_tr / b_tr), -1.5, 1.5).astype(np.float32)
    r_va = np.clip(np.log(y_va / b_va), -1.5, 1.5).astype(np.float32)
    print(f"[v4 Clean] Loaded from cache in {time.time() - t0:.1f}s | Train: {len(X_tr):,} | Val: {len(X_va):,} | Features: {len(V4_FEATURES)}")

    # 1. LightGBM Huber
    dtr = lgb.Dataset(X_tr, label=r_tr, feature_name=V4_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    dva = lgb.Dataset(X_va, label=r_va, reference=dtr, feature_name=V4_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    print("[v4] 1/3 LightGBM Huber ...")
    m_lgb = lgb.train(
        {"objective": "huber", "alpha": 0.25, "metric": "l1", "learning_rate": 0.04, "num_leaves": 191,
         "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "n_jobs": -1, "verbose": -1, "seed": 42},
        dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(45, verbose=False), lgb.log_evaluation(250)],
    )
    m_lgb.save_model(str(model_dir / "v4_lgb_huber.txt"))
    p_va_lgb = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

    # 2. XGBoost Pseudo-Huber
    print("[v4] 2/3 XGBoost Pseudo-Huber ...")
    m_xgb = xgb.XGBRegressor(
        objective="reg:pseudohubererror", huber_slope=0.25, eval_metric="mae",
        n_estimators=800, learning_rate=0.05, max_depth=9, min_child_weight=80,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
        random_state=42, early_stopping_rounds=40,
    )
    m_xgb.fit(X_tr, r_tr, eval_set=[(X_va, r_va)], verbose=250)
    m_xgb.save_model(str(model_dir / "v4_xgb_huber.json"))
    p_va_xgb = m_xgb.predict(X_va)

    # 3. CatBoost MAE
    print("[v4] 3/3 CatBoost MAE ...")
    m_cat = CatBoostRegressor(
        loss_function="MAE", iterations=1000, learning_rate=0.07, depth=8,
        random_seed=42, thread_count=-1, verbose=300,
    )
    m_cat.fit(X_tr, r_tr, eval_set=(X_va, r_va), early_stopping_rounds=50, use_best_model=True)
    m_cat.save_model(str(model_dir / "v4_cat_mae.cbm"))
    p_va_cat = m_cat.predict(X_va)

    best_w, best_mae = (0.5, 0.3, 0.2), 1e9
    for w1 in np.linspace(0.2, 0.7, 11):
        for w2 in np.linspace(0.1, 0.6, 11):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            mae = float(np.mean(np.abs(b_va * np.exp(w1 * p_va_lgb + w2 * p_va_xgb + w3 * p_va_cat) - y_va)))
            if mae < best_mae:
                best_mae = mae
                best_w = (round(float(w1), 3), round(float(w2), 3), round(float(w3), 3))
    print(f"\n[v4 Blend] Weights (LGB, XGB, CAT) = {best_w} | Val MAE = {best_mae:.2f}s (LGB={np.mean(np.abs(b_va*np.exp(p_va_lgb)-y_va)):.2f}s, XGB={np.mean(np.abs(b_va*np.exp(p_va_xgb)-y_va)):.2f}s, CAT={np.mean(np.abs(b_va*np.exp(p_va_cat)-y_va)):.2f}s)")

    # Quantiles P10/P90 + Online AgACI
    q_models, q_va = {}, {}
    for alpha in (0.10, 0.90):
        qm = lgb.train(
            {"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.05, "num_leaves": 127,
             "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "verbose": -1, "seed": 42},
            dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        qm.save_model(str(model_dir / f"v4_lgb_q{int(alpha * 100)}.txt"))
        q_models[alpha] = qm
        q_va[alpha] = qm.predict(X_va, num_iteration=qm.best_iteration)

    hb_va = v2.horizon_bucket_idx(b_va)
    cqr_by_bucket = {
        b_id: float(np.quantile(np.maximum(q_va[0.10][hb_va == b_id] - r_va[hb_va == b_id], r_va[hb_va == b_id] - q_va[0.90][hb_va == b_id]), 0.80))
        if np.sum(hb_va == b_id) >= 50 else 0.05
        for b_id in range(5)
    }

    report = {"weights": best_w, "cqr_by_bucket": cqr_by_bucket, "cities": {}}
    w1, w2, w3 = best_w
    for city, (te_day, X_te, y_te, b_te) in test_data.items():
        r_te_true = np.clip(np.log(y_te / b_te), -1.5, 1.5)
        r_hat = w1 * m_lgb.predict(X_te, num_iteration=m_lgb.best_iteration) + w2 * m_xgb.predict(X_te) + w3 * m_cat.predict(X_te)
        r_lo = q_models[0.10].predict(X_te, num_iteration=q_models[0.10].best_iteration)
        r_hi = q_models[0.90].predict(X_te, num_iteration=q_models[0.90].best_iteration)

        hb_te = v2.horizon_bucket_idx(b_te)
        adj_lo, adj_hi = np.empty_like(r_lo), np.empty_like(r_hi)
        for b_id in range(5):
            msk = hb_te == b_id
            if np.any(msk):
                adj_lo[msk], adj_hi[msk] = v3.agaci_calibrate_online(r_te_true[msk], r_lo[msk], r_hi[msk], cqr_by_bucket[b_id])

        s_base = eval_eta.summarize_errors([(float(yt), float(yb), None, None) for yt, yb in zip(y_te, b_te)])
        s_v4 = eval_eta.summarize_errors([(float(yt), float(yp), float(pl_), float(ph_)) for yt, yp, pl_, ph_ in zip(y_te, b_te * np.exp(r_hat), b_te * np.exp(adj_lo), b_te * np.exp(adj_hi))])
        eval_eta.print_table(f"{city.upper()} ({te_day}) — v4 Clean Stacked (LGB+XGB+CAT) + OSM + Holidays + AgACI", s_v4)
        b_mae = s_base["ALL(1-30m)"]["mae_s"]
        v4_mae = s_v4["ALL(1-30m)"]["mae_s"]
        lift = (1.0 - v4_mae / b_mae) * 100.0
        print(f">>> {city.upper()} v4 LIFT vs nowcast: {b_mae:.1f}s -> {v4_mae:.1f}s ({lift:+.1f}% MAE reduction!) | Cov80={s_v4['ALL(1-30m)']['cov80_pct']:.1f}%")
        report["cities"][city] = {"day": te_day, "nowcast": s_base, "v4": s_v4, "mae_lift_pct": round(lift, 2)}

    imp = m_lgb.feature_importance(importance_type="gain")
    top_imp = sorted(zip(V4_FEATURES, imp.tolist()), key=lambda x: x[1], reverse=True)[:20]
    print("\n=== v4 TOP 20 FEATURES BY GAIN ===")
    tot_g = sum(x[1] for x in top_imp) or 1.0
    for n, g in top_imp:
        print(f"  {n:<26} {g:>14,.0f} ({100.0*g/tot_g:>5.1f}%)")
    report["top_features"] = [{"name": n, "gain": round(g, 1)} for n, g in top_imp]
    (model_dir / "v4_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved v4 Clean Ensemble to {model_dir} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
