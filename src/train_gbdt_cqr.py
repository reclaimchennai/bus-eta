#!/usr/bin/env python3
"""Unified Multi-City Indian Bus ETA Model: Hierarchical Empirical-Bayes Priors +
LightGBM Quantile Residual Suite (P10 / P50 / P90) + Online Conformal Calibration (CQR).

Predicts log-speed residual:
    y = log(eff_speed_ms / prior_v)
so a single unified model transfers cleanly across Chennai MTC, Delhi DTC, Bengaluru BMTC,
Mumbai BEST, and cold-start Gurugram NCR.

Strict temporal split per city (zero leakage):
  - train : days < val_day
  - val   : second-to-last complete day (early stopping + CQR calibration)
  - test  : last complete day (final held-out evaluation)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[1]  # repo root

FEATURE_COLS = [
    # Spatial & road geometry
    "city_idx",
    "bus_class_idx",
    "stop_seq",
    "route_num_stops",
    "stop_progress",
    "full_seg_len_m",
    "seg_dist_m",
    "sinuosity",
    "mid_lat",
    "mid_lon",
    # Temporal & calendar
    "hour_ist",
    "min_of_day",
    "dow",
    "is_weekend",
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
    # Empirical-Bayes hierarchical priors & road-condition proxy
    "prior_seg_hw",
    "prior_corr_hw",
    "prior_city_hw",
    "prior_v",
    "offpeak_pothole_v",
    "veh_hist_pace",
    # Driver / vehicle online behavior
    "veh_pace_ewma",
    "trip_elapsed_s",
    "trip_dist_m",
    "veh_recent_v",
    "raw_k",
    # Live fleet & shared-corridor multi-route traffic (causal!)
    "seg_live_v_15m",
    "seg_live_n_15m",
    "corr_live_v_5m",
    "corr_live_v_15m",
    "corr_live_v_30m",
    "corr_live_n_15m",
    "corr_speed_cv_30m",     # traffic discipline / friction proxy
    "corr_stopgo_30m",       # congestion stop-go turbulence
    "up_live_v_15m",
    "dn_live_v_15m",
    "city_live_v_15m",
    "corr_live_ratio",       # corr_live_v_15m / prior_v
    "seg_live_ratio",        # seg_live_v_15m / prior_v
    # Bus bunching / headway
    "lead_headway_s",
    "corr_lead_headway_s",
    # Weather, monsoon waterlogging lag & air quality (pol.reclaimchennai.city + Open-Meteo)
    "temp_c",
    "humidity",
    "precip_mm",
    "precip_3h_mm",
    "precip_6h_mm",
    "precip_24h_mm",
    "is_raining",
    "is_heavy_rain",
    "wind_kph",
    "gust_kph",
    "cloud_cover",
    "vis_km",
    "pm25",
    "aqi",
    "local_precip_mm",
    "local_rain_accum_mm",
]


def load_and_split(processed_dir: Path, cities: list[str], max_rows_per_city: int = 1_500_000):
    """Load processed city Parquet files and assign strict temporal train/val/test splits."""
    frames = []
    split_meta = {}
    for city in cities:
        p = processed_dir / f"{city}_segments.parquet"
        if not p.exists():
            print(f"[warn] Missing {p}, skipping {city}")
            continue
        df = pl.read_parquet(p)
        if len(df) == 0:
            continue
        # Drop partial "today" if it has <30% of median daily rows
        day_counts = df.group_by("day").len().sort("day")
        days = day_counts["day"].to_list()
        counts = day_counts["len"].to_list()
        if len(days) >= 3 and counts[-1] < 0.35 * float(np.median(counts[:-1])):
            days = days[:-1]
            df = df.filter(pl.col("day").is_in(days))

        if len(days) < 3:
            print(f"[warn] {city} has only {len(days)} days, putting all in test/cold-start")
            val_day = days[-1]
            test_day = days[-1]
            train_days = days[:-1]
        else:
            test_day = days[-1]
            val_day = days[-2]
            train_days = days[:-2]

        split_meta[city] = {"train_days": len(train_days), "val_day": val_day, "test_day": test_day}
        df = df.with_columns(
            pl.when(pl.col("day") == test_day).then(pl.lit("test"))
            .when(pl.col("day") == val_day).then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
        # Keep valid speeds [0.4, 22.0] m/s
        df = df.filter((pl.col("eff_speed_ms") >= 0.4) & (pl.col("eff_speed_ms") <= 22.0))
        if len(df) > max_rows_per_city:
            # Downsample train only; keep val and test 100% intact!
            tr = df.filter(pl.col("split") == "train").sample(n=max_rows_per_city, seed=42)
            vt = df.filter(pl.col("split") != "train")
            df = pl.concat([tr, vt], how="vertical_relaxed")

        # Ensure optional weather columns exist
        for col in ["vis_km", "pm25", "aqi", "local_precip_mm", "local_rain_accum_mm",
                    "temp_c", "humidity", "precip_mm", "precip_3h_mm", "precip_6h_mm",
                    "precip_24h_mm", "is_raining", "is_heavy_rain", "wind_kph", "gust_kph", "cloud_cover"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        frames.append(df)

    full = pl.concat(frames, how="diagonal_relaxed")
    return full, split_meta


def fit_hierarchical_priors(train_df: pl.DataFrame) -> dict[str, pl.DataFrame | float]:
    """Compute leak-free Empirical-Bayes shrinkage priors strictly on train_df."""
    global_v = float(train_df["eff_speed_ms"].mean())

    # 1. City x hour x weekend
    p_city = (
        train_df.group_by(["city", "hour_ist", "is_weekend"])
        .agg([pl.col("eff_speed_ms").sum().alias("s"), pl.len().alias("n")])
        .with_columns(((pl.col("s") + 20.0 * global_v) / (pl.col("n") + 20.0)).alias("prior_city_hw"))
        .select(["city", "hour_ist", "is_weekend", "prior_city_hw"])
    )

    # 2. Shared physical corridor x hour x weekend (shrunk toward city prior)
    p_corr = (
        train_df.group_by(["city", "phys_link_id", "hour_ist", "is_weekend"])
        .agg([pl.col("eff_speed_ms").sum().alias("s"), pl.len().alias("n")])
        .join(p_city, on=["city", "hour_ist", "is_weekend"], how="left")
        .with_columns(
            ((pl.col("s") + 10.0 * pl.col("prior_city_hw").fill_null(global_v)) / (pl.col("n") + 10.0))
            .alias("prior_corr_hw")
        )
        .select(["city", "phys_link_id", "hour_ist", "is_weekend", "prior_corr_hw"])
    )

    # 3. Exact stop-to-stop segment x hour x weekend (shrunk toward corridor prior)
    p_seg = (
        train_df.group_by(["city", "seg_key", "phys_link_id", "hour_ist", "is_weekend"])
        .agg([pl.col("eff_speed_ms").sum().alias("s"), pl.len().alias("n")])
        .join(p_corr, on=["city", "phys_link_id", "hour_ist", "is_weekend"], how="left")
        .with_columns(
            ((pl.col("s") + 5.0 * pl.col("prior_corr_hw").fill_null(global_v)) / (pl.col("n") + 5.0))
            .alias("prior_seg_hw")
        )
        .select(["city", "seg_key", "hour_ist", "is_weekend", "prior_seg_hw"])
    )

    # 4. Off-peak (23:00 - 05:00) chronic slow-zone speed (data-driven pothole/bottleneck proxy)
    p_offpeak = (
        train_df.filter((pl.col("hour_ist") >= 23) | (pl.col("hour_ist") <= 5))
        .group_by(["city", "phys_link_id"])
        .agg([pl.col("eff_speed_ms").median().alias("offpeak_pothole_v"), pl.len().alias("n_op")])
        .filter(pl.col("n_op") >= 3)
        .select(["city", "phys_link_id", "offpeak_pothole_v"])
    )

    # 5. Per-vehicle historical speed ratio (driver/bus class baseline)
    tr_with_c = train_df.join(p_corr, on=["city", "phys_link_id", "hour_ist", "is_weekend"], how="left")
    p_veh = (
        tr_with_c.with_columns(
            (pl.col("eff_speed_ms") / pl.col("prior_corr_hw").fill_null(global_v)).clip(0.5, 2.0).alias("vr")
        )
        .group_by(["city", "reg"])
        .agg([pl.col("vr").sum().alias("s"), pl.len().alias("n")])
        .with_columns(((pl.col("s") + 15.0 * 1.0) / (pl.col("n") + 15.0)).alias("veh_hist_pace"))
        .select(["city", "reg", "veh_hist_pace"])
    )

    return {
        "global_v": global_v,
        "p_city": p_city,
        "p_corr": p_corr,
        "p_seg": p_seg,
        "p_offpeak": p_offpeak,
        "p_veh": p_veh,
    }


def apply_priors_and_features(df: pl.DataFrame, priors: dict, city_map: dict, class_map: dict) -> pl.DataFrame:
    gv = priors["global_v"]
    df = (
        df.join(priors["p_city"], on=["city", "hour_ist", "is_weekend"], how="left")
        .join(priors["p_corr"], on=["city", "phys_link_id", "hour_ist", "is_weekend"], how="left")
        .join(priors["p_seg"], on=["city", "seg_key", "hour_ist", "is_weekend"], how="left")
        .join(priors["p_offpeak"], on=["city", "phys_link_id"], how="left")
        .join(priors["p_veh"], on=["city", "reg"], how="left")
    )
    df = df.with_columns(
        pl.col("prior_city_hw").fill_null(gv),
        pl.col("prior_corr_hw").fill_null(pl.col("prior_city_hw").fill_null(gv)),
        pl.col("prior_seg_hw").fill_null(pl.col("prior_corr_hw").fill_null(pl.col("prior_city_hw").fill_null(gv))),
        pl.col("offpeak_pothole_v").fill_null(-1.0),
        pl.col("veh_hist_pace").fill_null(1.0),
    )
    df = df.with_columns(
        pl.col("prior_seg_hw").alias("prior_v"),
        (pl.col("stop_seq") / pl.col("route_num_stops").clip(lower_bound=1)).alias("stop_progress"),
        pl.col("city").replace_strict(city_map, default=-1).cast(pl.Int32).alias("city_idx"),
        pl.col("bus_class").replace_strict(class_map, default=-1).cast(pl.Int32).alias("bus_class_idx"),
    )
    df = df.with_columns(
        pl.when(pl.col("corr_live_v_15m") > 0)
        .then(pl.col("corr_live_v_15m") / pl.col("prior_v"))
        .otherwise(-1.0)
        .alias("corr_live_ratio"),
        pl.when(pl.col("seg_live_v_15m") > 0)
        .then(pl.col("seg_live_v_15m") / pl.col("prior_v"))
        .otherwise(-1.0)
        .alias("seg_live_ratio"),
        (pl.col("eff_speed_ms") / pl.col("prior_v")).log().clip(-1.5, 1.5).alias("target_log_ratio"),
    )
    return df


def train_and_evaluate(processed_dir: Path, model_dir: Path, cities: list[str]):
    t0 = time.time()
    full, split_meta = load_and_split(processed_dir, cities)
    print(f"Loaded {len(full):,} total segment events across {list(split_meta.keys())}")
    print("Splits:", json.dumps(split_meta, indent=2))

    train_raw = full.filter(pl.col("split") == "train")
    priors = fit_hierarchical_priors(train_raw)

    city_map = {c: i for i, c in enumerate(sorted(split_meta.keys()))}
    top_classes = (
        train_raw.group_by("bus_class").len().sort("len", descending=True).head(64)["bus_class"].to_list()
    )
    class_map = {c: i for i, c in enumerate(top_classes)}

    full_feat = apply_priors_and_features(full, priors, city_map, class_map)

    tr_df = full_feat.filter(pl.col("split") == "train")
    va_df = full_feat.filter(pl.col("split") == "val")
    te_df = full_feat.filter(pl.col("split") == "test")

    X_tr = tr_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_tr = tr_df["target_log_ratio"].to_numpy().astype(np.float32)
    X_va = va_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_va = va_df["target_log_ratio"].to_numpy().astype(np.float32)
    X_te = te_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y_te = te_df["target_log_ratio"].to_numpy().astype(np.float32)

    print(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,} | Features: {len(FEATURE_COLS)}")

    model_dir.mkdir(parents=True, exist_ok=True)
    quantiles = [0.10, 0.50, 0.90]
    models = {}
    val_preds = {}
    test_preds = {}

    for alpha in quantiles:
        print(f"\n[LightGBM] Training quantile alpha={alpha:.2f} ...")
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 80,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbose": -1,
            "seed": 42,
        }
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS, categorical_feature=["city_idx", "bus_class_idx"])
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=FEATURE_COLS, categorical_feature=["city_idx", "bus_class_idx"])
        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)],
        )
        models[alpha] = bst
        bst.save_model(str(model_dir / f"lgb_q{int(alpha * 100)}.txt"))
        val_preds[alpha] = bst.predict(X_va, num_iteration=bst.best_iteration)
        test_preds[alpha] = bst.predict(X_te, num_iteration=bst.best_iteration)

    # Conformalized Quantile Regression (CQR) calibration on validation set (target 80% coverage for [P10, P90])
    cqr_scores_va = np.maximum(val_preds[0.10] - y_va, y_va - val_preds[0.90])
    q_cqr = float(np.quantile(cqr_scores_va, 0.80))
    print(f"\n[CQR] Calibrated conformal margin on val (target 80% coverage): q_cqr = {q_cqr:.4f} (log-speed)")

    # Convert log-speed predictions -> segment duration seconds on TEST set
    prior_te = te_df["prior_v"].to_numpy()
    dist_te = te_df["seg_dist_m"].to_numpy()
    true_dur = te_df["dur_s"].to_numpy()

    pred_v_p50 = np.clip(prior_te * np.exp(test_preds[0.50]), 0.4, 22.0)
    pred_v_lo = np.clip(prior_te * np.exp(test_preds[0.10] - q_cqr), 0.4, 22.0)
    pred_v_hi = np.clip(prior_te * np.exp(test_preds[0.90] + q_cqr), 0.4, 22.0)

    pred_dur_p50 = dist_te / pred_v_p50
    pred_dur_hi = dist_te / pred_v_lo   # slower speed -> longer duration (P90 ETA)
    pred_dur_lo = dist_te / pred_v_hi   # faster speed -> shorter duration (P10 ETA)
    prior_dur = dist_te / prior_te

    # Overall & per-city segment-level report
    te_cities = te_df["city"].to_numpy()
    results = {"split_meta": split_meta, "q_cqr": q_cqr, "cities": {}}
    print("\n=== HELD-OUT TEST DAY: SEGMENT DURATION PREDICTION (Unified Multi-City Model) ===")
    print(f"{'City':<10} {'N':>8} {'PriorMAE':>10} {'LGB MAE':>10} {'Lift%':>8} {'LGB Med':>9} {'CQR Cov80%':>11}")
    print("-" * 72)
    for c in sorted(split_meta.keys()) + ["ALL"]:
        mask = np.ones(len(te_cities), dtype=bool) if c == "ALL" else (te_cities == c)
        if not np.any(mask):
            continue
        p_mae = float(np.mean(np.abs(prior_dur[mask] - true_dur[mask])))
        m_mae = float(np.mean(np.abs(pred_dur_p50[mask] - true_dur[mask])))
        m_med = float(np.median(np.abs(pred_dur_p50[mask] - true_dur[mask])))
        cov = float(np.mean((true_dur[mask] >= pred_dur_lo[mask]) & (true_dur[mask] <= pred_dur_hi[mask])) * 100.0)
        lift = (1.0 - m_mae / max(p_mae, 1e-6)) * 100.0
        print(f"{c:<10} {int(mask.sum()):>8,} {p_mae:>9.2f}s {m_mae:>9.2f}s {lift:>7.1f}% {m_med:>8.2f}s {cov:>10.1f}%")
        results["cities"][c] = {
            "n": int(mask.sum()),
            "prior_mae_s": round(p_mae, 2),
            "lgb_mae_s": round(m_mae, 2),
            "lift_pct": round(lift, 2),
            "lgb_med_s": round(m_med, 2),
            "cqr_cov80_pct": round(cov, 2),
        }

    # Feature importance (gain) for P50 model
    imp = models[0.50].feature_importance(importance_type="gain")
    top_imp = sorted(zip(FEATURE_COLS, imp.tolist()), key=lambda x: x[1], reverse=True)[:20]
    print("\n=== TOP 20 FEATURES BY GAIN (P50 Model) ===")
    total_gain = sum(x[1] for x in top_imp) or 1.0
    for name, g in top_imp:
        print(f"  {name:<24} {g:>14,.0f}  ({100.0 * g / total_gain:>5.1f}%)")
    results["top_features"] = [{"name": n, "gain": round(g, 1)} for n, g in top_imp]

    # Save priors & metadata for inference / rollout
    priors["p_city"].write_parquet(model_dir / "prior_city.parquet")
    priors["p_corr"].write_parquet(model_dir / "prior_corr.parquet")
    priors["p_seg"].write_parquet(model_dir / "prior_seg.parquet")
    priors["p_offpeak"].write_parquet(model_dir / "prior_offpeak.parquet")
    priors["p_veh"].write_parquet(model_dir / "prior_veh.parquet")
    (model_dir / "metadata.json").write_text(json.dumps({
        "global_v": priors["global_v"],
        "city_map": city_map,
        "class_map": class_map,
        "feature_cols": FEATURE_COLS,
        "q_cqr": q_cqr,
        "results": results,
    }, indent=2))
    print(f"\nSaved model artifacts & report to {model_dir} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=str(ROOT / "data/processed"))
    ap.add_argument("--model-dir", default=str(ROOT / "models/gbdt_cqr_v1"))
    ap.add_argument("--cities", default="delhi,mtc,bmtc,mumbai,ncr")
    a = ap.parse_args()
    train_and_evaluate(Path(a.processed_dir), Path(a.model_dir), [c.strip() for c in a.cities.split(",")])


if __name__ == "__main__":
    main()
