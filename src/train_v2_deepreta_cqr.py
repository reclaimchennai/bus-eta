#!/usr/bin/env python3
"""v2 SOTA Indian Multi-City Bus ETA: DeeprETA-Style Multi-Horizon Residual LightGBM +
Shared-Corridor Live Probes + Headway/Driver/Traffic-Discipline + pol.reclaimchennai.city Weather +
UncertBAT Horizon-Grouped Mondrian CQR.

Fixes the 3 mathematical failure modes discovered in v1 segment-summation rollout:
  1. Standing-time exclusion bias  -> directly supervises TRUE observed stop-arrival time (running + dwell + signals)
  2. Jensen's inequality drift     -> predicts log(true_eta / base_eta) directly at each stop horizon k (1..30 min)
  3. Non-decaying short-term pace  -> explicit horizon features (base_eta_s, n_stops_ahead, dist_ahead_m) let GBDT
                                      decay driver pace and live speed optimally with horizon!

Citations implemented:
  - Hu et al. (2022) DeeprETA: Residual post-processing over physics/nowcast baseline + Huber loss
  - Vanajakshi et al. (2009) & Achar et al. (2020): Shared-corridor probe pooling & spatial pace
  - Newell & Potts (1964) / Daganzo (2009): Leading-bus headway (bus bunching)
  - Mallikarjuna & Rao (2006): Heterogeneous traffic speed CV & stop-go friction proxy
  - Banik & Vanajakshi (2024): Non-linear monsoon rainfall + 3h/6h/24h waterlogging lag
  - Song et al. (2026) UncertBAT & Romano et al. (2019): Horizon-grouped Mondrian CQR
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

BASE_DIR = Path("~/projects")
sys.path.insert(0, str(BASE_DIR / "ETA-wt-baseline" / "src"))
sys.path.insert(0, str(BASE_DIR / "ETA-wt-features" / "src"))
sys.path.insert(0, str(BASE_DIR / "ETA" / "reference" / "mumbai"))

import eval_eta  # type: ignore
import extract_dataset as ed  # type: ignore
import nowcast  # type: ignore
import travel_model as tm  # type: ignore

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

V2_FEATURES = [
    # Identity & Horizon (enables automatic horizon decay of live signals!)
    "city_idx",
    "bus_class_idx",
    "dist_ahead_m",
    "n_stops_ahead",
    "base_eta_nowcast_s",
    "base_eta_hist_s",
    "base_eta_corr_s",
    "live_vs_hist_ratio",
    "stop_seq_now",
    "stop_seq_target",
    "route_num_stops",
    "progress_now",
    "progress_target",
    "path_mean_sinuosity",
    "lat_now",
    "lon_now",
    # Temporal & Calendar (missing from nowcast.py!)
    "hour_ist",
    "min_of_day",
    "dow",
    "is_weekend",
    "sin_tod",
    "cos_tod",
    "sin_dow",
    "cos_dow",
    # Driver / Vehicle Behavior (Cats 2019; Yin & Zhang 2023)
    "nc_pace",
    "veh_pace_ewma",
    "veh_recent_v",
    "trip_elapsed_s",
    "trip_dist_m",
    # Live Fleet & Shared-Corridor Traffic Discipline (LWR / Mallikarjuna & Rao 2006 / MAGNN)
    "cur_seg_live_v_15m",
    "cur_corr_live_v_5m",
    "cur_corr_live_v_15m",
    "cur_corr_live_v_30m",
    "cur_corr_n_15m",
    "cur_corr_cv_30m",
    "cur_corr_stopgo_30m",
    "up_live_v_15m",
    "dn_live_v_15m",
    "path_live_corr_v",
    "path_live_cov_frac",
    "city_live_v_15m",
    # Bus Bunching / Headway (Newell & Potts 1964; Daganzo 2009)
    "lead_headway_s",
    "corr_lead_headway_s",
    # Weather, Monsoon Waterlogging & AQI (pol.reclaimchennai.city + Open-Meteo; Banik & Vanajakshi 2024)
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
    "pol_precip_mean",
    "pol_rain_accum_mean",
]


def load_weather_map(data_root: Path, city: str) -> dict[int, dict]:
    wp = data_root / "weather" / "weather_city_hourly.parquet"
    if not wp.exists():
        return {}
    df = pl.read_parquet(wp).filter(pl.col("city") == city)
    return {int(r["hour_ts"]): r for r in df.iter_rows(named=True)}


def build_day_queries(
    city: str,
    city_idx: int,
    day: str,
    data_root: Path,
    routes_tm: tm.Routes,
    routes_ed: ed.Routes,
    model_json_path: Path | None,
    wx_map: dict[int, dict],
    class_map: dict[str, int],
    sample_rate: float,
    max_gap: int,
    fix_stride: int = 3,
    max_queries: int = 450_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay a single day strictly causally and extract (X, true_eta_s, base_eta_nowcast_s, horizon_bucket)."""
    raw_dir = data_root / "raw" / city
    M = nowcast.Model(str(model_json_path)) if (model_json_path and model_json_path.exists()) else nowcast.Model(None)
    pace_nc = nowcast.Pace()
    tracker = ed.LiveCorridorTracker()

    tracks = eval_eta.load_day_tracks(routes_tm, raw_dir, day)
    trips_by_reg = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(max_gap, 300))

    rng = np.random.default_rng(42)
    picked = {reg for reg in trips_by_reg if rng.random() <= sample_rate}

    events = []
    for reg, trips in trips_by_reg.items():
        for t_idx, trip in enumerate(trips):
            for f_idx, _ in enumerate(trip["fixes"]):
                events.append((trip["fixes"][f_idx][0], reg, t_idx, f_idx))
    events.sort(key=lambda x: x[0])

    bus_state: dict[tuple[str, int], list[float]] = {}
    X_rows: list[list[float]] = []
    y_true: list[float] = []
    y_base: list[float] = []

    for t, reg, t_idx, f_idx in events:
        trip = trips_by_reg[reg][t_idx]
        code = trip["code"]
        if code not in routes_ed.ids:
            continue
        s, i, bus_class = trip["fixes"][f_idx][1], trip["fixes"][f_idx][2], trip["fixes"][f_idx][3]
        ids, ds, phys, sinu = routes_ed.ids[code], routes_ed.ds[code], routes_ed.phys[code], routes_ed.sinuosity[code]
        hour = time.gmtime(t + 19800).tm_hour

        kt = (reg, t_idx)
        if kt not in bus_state:
            bus_state[kt] = [1.0, -1.0, t, s]
        st = bus_state[kt]

        # Strictly causal updates to both nowcast probes AND shared-corridor tracker
        if f_idx > 0:
            pt, ps, pi, _pf = trip["fixes"][f_idx - 1]
            dt_ = t - pt
            d = s - ps
            if 0 < dt_ <= max_gap and d >= 12.0 and d <= 25.0 * dt_:
                v_pair = d / dt_
                for j in range(pi, i + 1):
                    if j + 1 < len(ids):
                        share = max(0.0, min(s, ds[j + 1]) - max(ps, ds[j])) / d
                        if share > 0:
                            sk = f"{ids[j]}|{ids[j + 1]}"
                            ph = phys[j]
                            M.add_probe(sk, t, v_pair, dt_ * share)
                            tracker.record(sk, ph, t, d * share, dt_ * share)
                            tracker.seg_last_pass[sk] = t
                            tracker.corr_last_pass[ph] = t
                pace_nc.update(reg, M.expected_time(ids, ds, ps, s, hour, t), dt_)
                ref_v = tracker.query_corr(phys[min(i, len(phys) - 1)], t)[1]
                if ref_v <= 0:
                    ref_v = 4.2
                st[0] = st[0] + 0.25 * (max(0.5, min(2.0, v_pair / ref_v)) - st[0])
                st[1] = v_pair
            elif 0 < dt_ <= max_gap and abs(d) < 12.0 and i < len(phys):
                M.add_probe(f"{ids[i]}|{ids[i + 1]}", t, 0.0, dt_)
                tracker.record(f"{ids[i]}|{ids[i + 1]}", phys[i], t, 0.0, dt_)
                st[1] = 0.0

        if reg not in picked or (f_idx % fix_stride != 0):
            continue
        if len(X_rows) >= max_queries:
            continue

        next_k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
        stop_times = trip["stop_times"]
        valid_targets = [
            (k, t_true - t)
            for k, t_true in stop_times.items()
            if k >= next_k and 60.0 <= (t_true - t) < 1800.0 and k < next_k + 30
        ]
        if not valid_targets:
            continue

        # Compute 3 cumulative baselines along the upcoming stops up to max_k:
        #   1) base_eta_nowcast_s : exact production nowcast.py (live probes + pace)
        #   2) base_eta_hist_s    : historical segment speed only (no live probes, pace=1)
        #   3) base_eta_corr_s    : shared-corridor multi-route live speed where available!
        max_k = max(k for k, _ in valid_targets)
        nc_p = pace_nc.get(reg)
        etas_nc = dict(M.eta(ids, ds, s, next_k, t, pace=nc_p, n=30))

        # Precompute cumulative historical & shared-corridor ETAs + sinuosity + live coverage
        cum_hist = {}
        cum_corr = {}
        cum_sinu = {}
        cum_lcov = {}
        th = tc = 0.0
        sinu_sum = 0.0
        lcov_n = 0
        tot_n = 0
        here = s
        for seg_i in range(next_k - 1, max_k):
            if seg_i < 0 or seg_i >= len(phys):
                continue
            seg_d = max(0.0, ds[seg_i + 1] - here)
            here = ds[seg_i + 1]
            sk = f"{ids[seg_i]}|{ids[seg_i + 1]}"
            ph = phys[seg_i]
            # Historical speed from model.json
            row_m = (M.m.get("seg") or {}).get(sk)
            vh = -1.0
            if row_m:
                vh = row_m[2][hour] if row_m[2][hour] > 0 else row_m[1]
            if not vh or vh <= 0:
                cv = ((M.m.get("city") or {}).get("v") or [-1] * 24)[hour]
                vh = cv if cv and cv > 0 else 4.2
            vh = min(22.0, max(1.0, vh))
            th += seg_d / vh

            # Shared-corridor live speed fallback to historical
            cv15 = tracker.query_corr(ph, t)[1]
            if cv15 > 0.5:
                vc = min(22.0, max(1.0, cv15))
                lcov_n += 1
            else:
                vc = vh
            tc += seg_d / vc
            sinu_sum += sinu[seg_i]
            tot_n += 1
            k_stop = seg_i + 1
            cum_hist[k_stop] = th
            cum_corr[k_stop] = tc
            cum_sinu[k_stop] = sinu_sum / tot_n
            cum_lcov[k_stop] = lcov_n / tot_n

        # Current-location context features
        cur_i = min(max(0, next_k - 1), len(phys) - 1)
        sk_cur = f"{ids[cur_i]}|{ids[cur_i + 1]}"
        ph_cur = phys[cur_i]
        cur_seg_v15, _ = tracker.query_seg(sk_cur, t)
        c_v5, c_v15, c_v30, c_n15, c_cv, c_sg = tracker.query_corr(ph_cur, t)
        up_v15 = tracker.query_seg(f"{ids[cur_i - 1]}|{ids[cur_i]}", t)[0] if cur_i >= 1 else -1.0
        dn_v15 = tracker.query_seg(f"{ids[cur_i + 1]}|{ids[cur_i + 2]}", t)[0] if cur_i + 2 < len(ids) else -1.0
        city_v = tracker.query_city(t)
        lead_hw = t - tracker.seg_last_pass.get(sk_cur, -1e9)
        lead_hw = lead_hw if 0 < lead_hw <= 7200 else -1.0
        corr_hw = t - tracker.corr_last_pass.get(ph_cur, -1e9)
        corr_hw = corr_hw if 0 < corr_hw <= 7200 else -1.0

        dt_obj = dt.datetime.fromtimestamp(t, IST)
        hour_ist = dt_obj.hour
        min_of_day = hour_ist * 60 + dt_obj.minute
        dow = dt_obj.weekday()
        is_wknd = int(dow >= 5)
        sin_tod = math.sin(min_of_day * (2.0 * math.pi / 1440.0))
        cos_tod = math.cos(min_of_day * (2.0 * math.pi / 1440.0))
        sin_dow = math.sin(dow * (2.0 * math.pi / 7.0))
        cos_dow = math.cos(dow * (2.0 * math.pi / 7.0))
        bc_idx = class_map.get(bus_class, -1)
        lat_now, lon_now = routes_ed.ll[code][cur_i]
        n_stops = len(ds)

        wx = wx_map.get((int(t) // 3600) * 3600, {})
        wx_vals = [
            float(wx["temp_c"]) if wx.get("temp_c") is not None else np.nan,
            float(wx["humidity"]) if wx.get("humidity") is not None else np.nan,
            float(wx.get("precip_mm") or 0.0),
            float(wx.get("precip_3h_mm") or 0.0),
            float(wx.get("precip_6h_mm") or 0.0),
            float(wx.get("precip_24h_mm") or 0.0),
            float(wx.get("is_raining") or 0.0),
            float(wx.get("is_heavy_rain") or 0.0),
            float(wx["wind_kph"]) if wx.get("wind_kph") is not None else np.nan,
            float(wx["gust_kph"]) if wx.get("gust_kph") is not None else np.nan,
            float(wx["cloud_cover"]) if wx.get("cloud_cover") is not None else np.nan,
            float(wx["vis_km"]) if wx.get("vis_km") is not None else np.nan,
            float(wx["pm25"]) if wx.get("pm25") is not None else np.nan,
            float(wx["aqi"]) if wx.get("aqi") is not None else np.nan,
            float(wx["pol_precip_mean"]) if wx.get("pol_precip_mean") is not None else np.nan,
            float(wx["pol_rain_accum_mean"]) if wx.get("pol_rain_accum_mean") is not None else np.nan,
        ]

        for k, true_eta in valid_targets:
            if k not in etas_nc or k not in cum_hist:
                continue
            b_nc = max(5.0, etas_nc[k] - t)
            b_h = max(5.0, cum_hist[k])
            b_c = max(5.0, cum_corr[k])
            dist_ahead = max(1.0, ds[k] - s)
            n_ahead = k - next_k + 1
            path_v = dist_ahead / b_c

            feat = [
                city_idx, bc_idx, dist_ahead, n_ahead,
                b_nc, b_h, b_c, b_c / b_h,
                cur_i, k, n_stops, cur_i / max(n_stops, 1), k / max(n_stops, 1),
                cum_sinu[k], lat_now, lon_now,
                hour_ist, min_of_day, dow, is_wknd, sin_tod, cos_tod, sin_dow, cos_dow,
                nc_p, st[0], st[1], t - st[2], max(0.0, s - st[3]),
                cur_seg_v15, c_v5, c_v15, c_v30, c_n15, c_cv, c_sg,
                up_v15, dn_v15, path_v, cum_lcov[k], city_v,
                lead_hw, corr_hw,
            ] + wx_vals
            X_rows.append(feat)
            y_true.append(true_eta)
            y_base.append(b_nc)

    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_base, dtype=np.float64),
    )


def horizon_bucket_idx(base_eta_s: np.ndarray) -> np.ndarray:
    """Assign each query to one of 5 horizon buckets for UncertBAT Mondrian CQR."""
    return np.digitize(base_eta_s, [180.0, 300.0, 600.0, 900.0])  # 0..4


def run_v2(data_root: Path, model_dir: Path, cities: list[str]):
    t0 = time.time()
    city_map = {c: i for i, c in enumerate(cities)}
    # Stable bus_class vocabulary across cities
    class_vocab = ["", "u", "a", "n", "k", "dn", "ln", "dc", "lc", "ev", "ac", "e", "o", "s", "d"]
    class_map = {c: i for i, c in enumerate(class_vocab)}

    # Explicit train / val / test day plan per city (aligned to exact baseline test days!)
    day_plan = {
        "mtc":    (["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.20, 0.35),
        "bmtc":   (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.35, 0.35),
        "mumbai": (["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"], "2026-11-16", "2026-11-17", 0.50, 0.50),
        "ncr":    (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"], "2026-11-16", "2026-11-17", 1.00, 0.50),
        "delhi":  (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.04, 0.10),
    }

    X_tr_l, y_tr_l, b_tr_l = [], [], []
    X_va_l, y_va_l, b_va_l = [], [], []
    test_data = {}

    for city in cities:
        if city not in day_plan:
            continue
        tr_days, va_day, te_day, tr_sr, te_sr = day_plan[city]
        meta_dir = data_root / "meta" / city
        routes_tm = tm.Routes(str(meta_dir / "route_stops.json"), str(meta_dir / "stop_dist.json"))
        routes_ed = ed.Routes(meta_dir / "route_stops.json", meta_dir / "stop_dist.json")
        m_json = meta_dir / "model.json"
        wx_map = load_weather_map(data_root, city)
        max_gap = 600 if city in ("mtc", "mumbai") else 180

        for d in tr_days:
            if not list(tm.day_files(str(data_root / "raw" / city), "obs", 1, d)):
                continue
            X, yt, yb = build_day_queries(city, city_map[city], d, data_root, routes_tm, routes_ed, m_json, wx_map, class_map, tr_sr, max_gap, fix_stride=4, max_queries=250_000)
            if len(X):
                X_tr_l.append(X); y_tr_l.append(yt); b_tr_l.append(yb)
                print(f"  [train:{city}] {d}: {len(X):,} queries", flush=True)

        if list(tm.day_files(str(data_root / "raw" / city), "obs", 1, va_day)):
            X, yt, yb = build_day_queries(city, city_map[city], va_day, data_root, routes_tm, routes_ed, m_json, wx_map, class_map, te_sr, max_gap, fix_stride=3, max_queries=300_000)
            if len(X):
                X_va_l.append(X); y_va_l.append(yt); b_va_l.append(yb)
                print(f"  [val:{city}]   {va_day}: {len(X):,} queries", flush=True)

        if list(tm.day_files(str(data_root / "raw" / city), "obs", 1, te_day)):
            X, yt, yb = build_day_queries(city, city_map[city], te_day, data_root, routes_tm, routes_ed, m_json, wx_map, class_map, te_sr, max_gap, fix_stride=3, max_queries=1_500_000)
            if len(X):
                test_data[city] = (te_day, X, yt, yb)
                print(f"  [test:{city}]  {te_day}: {len(X):,} queries", flush=True)

    X_tr = np.vstack(X_tr_l); y_tr = np.concatenate(y_tr_l); b_tr = np.concatenate(b_tr_l)
    X_va = np.vstack(X_va_l); y_va = np.concatenate(y_va_l); b_va = np.concatenate(b_va_l)

    r_tr = np.clip(np.log(y_tr / b_tr), -1.5, 1.5).astype(np.float32)
    r_va = np.clip(np.log(y_va / b_va), -1.5, 1.5).astype(np.float32)

    print(f"\n[v2 DeeprETA] Train: {len(X_tr):,} | Val: {len(X_va):,} | Features: {len(V2_FEATURES)}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Point ETA model (Huber loss on log-residual -> robust to outliers & unbiased!)
    dtr = lgb.Dataset(X_tr, label=r_tr, feature_name=V2_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    dva = lgb.Dataset(X_va, label=r_va, reference=dtr, feature_name=V2_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])

    print("[v2] Training Huber Point Residual Model ...")
    m_hub = lgb.train(
        {
            "objective": "huber",
            "alpha": 0.25,
            "metric": "l1",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 100,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbose": -1,
            "seed": 42,
        },
        dtr,
        num_boost_round=800,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)],
    )
    m_hub.save_model(str(model_dir / "lgb_v2_huber.txt"))

    # 2. P10 & P90 Quantile models for Horizon-Grouped Mondrian CQR (UncertBAT 2026)
    q_models = {}
    q_va = {}
    for alpha in (0.10, 0.90):
        print(f"[v2] Training Quantile alpha={alpha:.2f} ...")
        qm = lgb.train(
            {
                "objective": "quantile",
                "alpha": alpha,
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 95,
                "min_child_samples": 100,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "n_jobs": -1,
                "verbose": -1,
                "seed": 42,
            },
            dtr,
            num_boost_round=600,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(200)],
        )
        qm.save_model(str(model_dir / f"lgb_v2_q{int(alpha * 100)}.txt"))
        q_models[alpha] = qm
        q_va[alpha] = qm.predict(X_va, num_iteration=qm.best_iteration)

    # Calibrate Horizon-Grouped Mondrian CQR (UncertBAT, Song et al. 2026) on Val set!
    hb_va = horizon_bucket_idx(b_va)
    cqr_by_bucket = {}
    for b_id in range(5):
        msk = hb_va == b_id
        if np.sum(msk) >= 50:
            scores = np.maximum(q_va[0.10][msk] - r_va[msk], r_va[msk] - q_va[0.90][msk])
            cqr_by_bucket[b_id] = float(np.quantile(scores, 0.80))
        else:
            cqr_by_bucket[b_id] = 0.05
    print("[v2 UncertBAT Mondrian CQR] Margins by horizon bucket:", {k: round(v, 4) for k, v in cqr_by_bucket.items()})

    # Evaluate head-to-head against production nowcast.py on every city's exact held-out test day!
    report = {"cqr_by_bucket": cqr_by_bucket, "cities": {}}
    for city, (te_day, X_te, y_te, b_te) in test_data.items():
        t_inf = time.time()
        r_hat = m_hub.predict(X_te, num_iteration=m_hub.best_iteration)
        r_lo = q_models[0.10].predict(X_te, num_iteration=q_models[0.10].best_iteration)
        r_hi = q_models[0.90].predict(X_te, num_iteration=q_models[0.90].best_iteration)
        inf_ms = (time.time() - t_inf) * 1000.0

        hb_te = horizon_bucket_idx(b_te)
        margins = np.array([cqr_by_bucket[int(b)] for b in hb_te], dtype=np.float64)

        pred_s = b_te * np.exp(r_hat)
        pred_lo = b_te * np.exp(r_lo - margins)
        pred_hi = b_te * np.exp(r_hi + margins)

        recs_base = [(float(yt), float(yb), None, None) for yt, yb in zip(y_te, b_te)]
        recs_v2 = [(float(yt), float(yp), float(pl_), float(ph_)) for yt, yp, pl_, ph_ in zip(y_te, pred_s, pred_lo, pred_hi)]

        s_base = eval_eta.summarize_errors(recs_base)
        s_v2 = eval_eta.summarize_errors(recs_v2)
        eval_eta.print_table(f"{city.upper()} ({te_day}) — Baseline Production nowcast.py", s_base)
        eval_eta.print_table(f"{city.upper()} ({te_day}) — v2 DeeprETA LightGBM + UncertBAT CQR + Weather ({inf_ms/len(X_te)*1000:.2f} us/query)", s_v2)

        b_mae = s_base["ALL(1-30m)"]["mae_s"]
        v2_mae = s_v2["ALL(1-30m)"]["mae_s"]
        lift = (1.0 - v2_mae / b_mae) * 100.0
        print(f">>> {city.upper()} HEAD-TO-HEAD LIFT: nowcast {b_mae:.1f}s -> v2 {v2_mae:.1f}s ({lift:+.1f}% MAE reduction!) | Cov80={s_v2['ALL(1-30m)']['cov80_pct']:.1f}%")
        report["cities"][city] = {"day": te_day, "nowcast": s_base, "v2": s_v2, "mae_lift_pct": round(lift, 2)}

    imp = m_hub.feature_importance(importance_type="gain")
    top_imp = sorted(zip(V2_FEATURES, imp.tolist()), key=lambda x: x[1], reverse=True)[:20]
    print("\n=== v2 TOP 20 FEATURES BY GAIN ===")
    tot_g = sum(x[1] for x in top_imp) or 1.0
    for n, g in top_imp:
        print(f"  {n:<24} {g:>14,.0f} ({100.0*g/tot_g:>5.1f}%)")
    report["top_features"] = [{"name": n, "gain": round(g, 1)} for n, g in top_imp]

    (model_dir / "v2_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved v2 DeeprETA+CQR models & full benchmark to {model_dir} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/projects/ETA/data")
    ap.add_argument("--model-dir", default="~/projects/ETA/models/v2_deepreta_cqr")
    ap.add_argument("--cities", default="mtc,bmtc,mumbai,ncr,delhi")
    a = ap.parse_args()
    run_v2(Path(a.data_root), Path(a.model_dir), [c.strip() for c in a.cities.split(",")])


if __name__ == "__main__":
    main()
