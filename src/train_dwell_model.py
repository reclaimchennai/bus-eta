#!/usr/bin/env python3
"""Explicit Indian Bus Dwell-Time & Bus-Bunching Model (Two-Stage Hurdle LightGBM).

In Indian bus transit, dwell time at stops is zero-inflated and heavily governed by
bus bunching (leading-bus headway) and rain/peak-hour passenger surges:
  Stage 1 (Classifier): P(bus stops at stop_id | headway, stop_prior, time, bus_class, weather)
  Stage 2 (Regressor) : E[log(dwell_s) | stopped=1, headway, stop_prior, time, bus_class, weather]
  Combined expected dwell: P(stop) * exp(E[log(dwell_s)])

Evaluated strictly on held-out test day across all cities!
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

DWELL_FEATURES = [
    "city_idx",
    "bus_class_idx",
    "stop_seq",
    "route_num_stops",
    "stop_progress",
    "lat",
    "lon",
    "hour_ist",
    "dow",
    "is_weekend",
    "sin_tod",
    "cos_tod",
    "lead_headway_s",
    "any_headway_s",
    "headway_ratio",
    "stop_prior_p",
    "stop_prior_dwell",
    "temp_c",
    "humidity",
    "precip_mm",
    "precip_3h_mm",
    "is_raining",
    "is_heavy_rain",
]


def train_dwell(processed_dir: Path, model_dir: Path, cities: list[str]):
    t0 = time.time()
    frames = []
    for city in cities:
        p = processed_dir / f"{city}_stops.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if len(df) < 1000:
            continue
        day_counts = df.group_by("day").len().sort("day")
        days = day_counts["day"].to_list()
        counts = day_counts["len"].to_list()
        if len(days) >= 3 and counts[-1] < 0.35 * float(np.median(counts[:-1])):
            days = days[:-1]
            df = df.filter(pl.col("day").is_in(days))
        if len(days) < 3:
            continue
        test_d, val_d = days[-1], days[-2]
        df = df.with_columns(
            pl.when(pl.col("day") == test_d).then(pl.lit("test"))
            .when(pl.col("day") == val_d).then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
        # Extract IST hour/dow from t_arr
        df = df.with_columns(
            (((pl.col("t_arr").cast(pl.Int64) + 19800) % 86400) // 3600).alias("hour_ist"),
            ((((pl.col("t_arr").cast(pl.Int64) + 19800) // 86400) + 3) % 7).alias("dow"),
            (((pl.col("t_arr").cast(pl.Int64) + 19800) % 86400) / 86400.0 * 2.0 * math.pi).sin().alias("sin_tod"),
            (((pl.col("t_arr").cast(pl.Int64) + 19800) % 86400) / 86400.0 * 2.0 * math.pi).cos().alias("cos_tod"),
            (pl.col("stop_seq") / pl.col("route_num_stops").clip(lower_bound=1)).alias("stop_progress"),
        ).with_columns(
            (pl.col("dow") >= 5).cast(pl.Int8).alias("is_weekend"),
            pl.when((pl.col("lead_headway_s") > 0) & (pl.col("any_headway_s") > 0))
            .then(pl.col("lead_headway_s") / pl.col("any_headway_s").clip(lower_bound=10.0))
            .otherwise(-1.0)
            .alias("headway_ratio"),
        )
        for col in ["temp_c", "humidity", "precip_mm", "precip_3h_mm", "is_raining", "is_heavy_rain"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        # Cap train rows per city for fast training
        tr = df.filter(pl.col("split") == "train")
        if len(tr) > 800_000:
            tr = tr.sample(n=800_000, seed=42)
        vt = df.filter(pl.col("split") != "train")
        frames.append(pl.concat([tr, vt], how="vertical_relaxed"))

    full = pl.concat(frames, how="diagonal_relaxed")
    tr_raw = full.filter(pl.col("split") == "train")

    # Stop-level Empirical-Bayes priors (fit strictly on train!)
    global_p = float(tr_raw["stopped"].mean())
    global_dw = float(tr_raw.filter(pl.col("stopped") == 1)["dwell_s"].median() or 25.0)
    stop_priors = (
        tr_raw.group_by(["city", "stop_id"])
        .agg([
            pl.col("stopped").sum().alias("n_stop"),
            pl.len().alias("n_tot"),
            pl.col("dwell_s").filter(pl.col("stopped") == 1).median().alias("med_dw"),
        ])
        .with_columns(
            ((pl.col("n_stop") + 10.0 * global_p) / (pl.col("n_tot") + 10.0)).alias("stop_prior_p"),
            pl.col("med_dw").fill_null(global_dw).alias("stop_prior_dwell"),
        )
        .select(["city", "stop_id", "stop_prior_p", "stop_prior_dwell"])
    )

    city_map = {c: i for i, c in enumerate(sorted(full["city"].unique().to_list()))}
    top_cls = tr_raw.group_by("bus_class").len().sort("len", descending=True).head(64)["bus_class"].to_list()
    class_map = {c: i for i, c in enumerate(top_cls)}

    full = (
        full.join(stop_priors, on=["city", "stop_id"], how="left")
        .with_columns(
            pl.col("stop_prior_p").fill_null(global_p),
            pl.col("stop_prior_dwell").fill_null(global_dw),
            pl.col("city").replace_strict(city_map, default=-1).cast(pl.Int32).alias("city_idx"),
            pl.col("bus_class").replace_strict(class_map, default=-1).cast(pl.Int32).alias("bus_class_idx"),
        )
    )

    tr = full.filter(pl.col("split") == "train")
    va = full.filter(pl.col("split") == "val")
    te = full.filter(pl.col("split") == "test")

    X_tr = tr.select(DWELL_FEATURES).to_numpy().astype(np.float32)
    y_tr_cls = tr["stopped"].to_numpy().astype(np.float32)
    X_va = va.select(DWELL_FEATURES).to_numpy().astype(np.float32)
    y_va_cls = va["stopped"].to_numpy().astype(np.float32)
    X_te = te.select(DWELL_FEATURES).to_numpy().astype(np.float32)
    y_te_cls = te["stopped"].to_numpy().astype(np.float32)
    y_te_dw = te["dwell_s"].to_numpy().astype(np.float64)

    print(f"[Dwell] Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,} | Stop rate: {y_tr_cls.mean()*100:.1f}%")

    # Stage 1: Stopping probability classifier
    dtr_c = lgb.Dataset(X_tr, label=y_tr_cls, feature_name=DWELL_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    dva_c = lgb.Dataset(X_va, label=y_va_cls, reference=dtr_c, feature_name=DWELL_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    clf = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63, "n_jobs": -1, "verbose": -1, "seed": 42},
        dtr_c,
        num_boost_round=500,
        valid_sets=[dva_c],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    p_stop_te = clf.predict(X_te, num_iteration=clf.best_iteration)

    # Stage 2: Conditional dwell duration given stopped == 1
    tr_s = tr.filter((pl.col("stopped") == 1) & (pl.col("dwell_s") >= 2.0))
    va_s = va.filter((pl.col("stopped") == 1) & (pl.col("dwell_s") >= 2.0))
    dtr_r = lgb.Dataset(tr_s.select(DWELL_FEATURES).to_numpy().astype(np.float32), label=np.log(tr_s["dwell_s"].to_numpy()), feature_name=DWELL_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    dva_r = lgb.Dataset(va_s.select(DWELL_FEATURES).to_numpy().astype(np.float32), label=np.log(va_s["dwell_s"].to_numpy()), reference=dtr_r, feature_name=DWELL_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    reg = lgb.train(
        {"objective": "regression_l1", "metric": "l1", "learning_rate": 0.05, "num_leaves": 63, "n_jobs": -1, "verbose": -1, "seed": 42},
        dtr_r,
        num_boost_round=500,
        valid_sets=[dva_r],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    dw_cond_te = np.exp(reg.predict(X_te, num_iteration=reg.best_iteration))
    pred_expected_dw = p_stop_te * dw_cond_te
    prior_expected_dw = te["stop_prior_p"].to_numpy() * te["stop_prior_dwell"].to_numpy()

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_te_cls, p_stop_te)
    mae_model = float(np.mean(np.abs(pred_expected_dw - y_te_dw)))
    mae_prior = float(np.mean(np.abs(prior_expected_dw - y_te_dw)))
    print(f"[Dwell TEST] Stop-Classifier AUC = {auc:.4f} | Expected Dwell MAE: Prior={mae_prior:.2f}s -> Model={mae_model:.2f}s ({(1-mae_model/mae_prior)*100:.1f}% lift)")

    model_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_dir / "dwell_clf.txt"))
    reg.save_model(str(model_dir / "dwell_reg.txt"))
    stop_priors.write_parquet(model_dir / "stop_priors.parquet")
    (model_dir / "dwell_meta.json").write_text(json.dumps({
        "auc": round(float(auc), 4),
        "mae_prior_s": round(mae_prior, 2),
        "mae_model_s": round(mae_model, 2),
        "lift_pct": round((1.0 - mae_model / mae_prior) * 100.0, 2),
        "top_clf_features": sorted(zip(DWELL_FEATURES, clf.feature_importance("gain").tolist()), key=lambda x: x[1], reverse=True)[:10],
        "top_reg_features": sorted(zip(DWELL_FEATURES, reg.feature_importance("gain").tolist()), key=lambda x: x[1], reverse=True)[:10],
    }, indent=2))
    print(f"[Dwell] Saved to {model_dir} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="~/projects/ETA/data/processed")
    ap.add_argument("--model-dir", default="~/projects/ETA/models/dwell_v1")
    ap.add_argument("--cities", default="delhi,mtc,bmtc,mumbai,ncr")
    a = ap.parse_args()
    train_dwell(Path(a.processed_dir), Path(a.model_dir), [c.strip() for c in a.cities.split(",")])


if __name__ == "__main__":
    main()
