#!/usr/bin/env python3
"""v3 Accuracy Push: Stacked Multi-Model Ensemble (LightGBM Huber + CatBoost + XGBoost Pseudo-Huber)
+ DiDi WDR/HetETA Route & Weekend Residual Priors
+ Baidu ConSTGAT Path Geometry (Stops/km, Max Sinuosity, Cumulative Turning Angle)
+ Online Adaptive Conformal Inference (AgACI, Zaffran et al. 2022) on UncertBAT Mondrian CQR!

Also saves compact v3 train/val/test feature matrices to Google Drive so our Colab A100
Deep Residual Neural Net can train on the exact same queries and ensemble with the 3-GBDT stack!
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference" / "mumbai"))

import eval_eta  # type: ignore
import extract_dataset as ed  # type: ignore
import nowcast  # type: ignore
import osm_infra_fetch as osm  # type: ignore
import train_v2_deepreta_cqr as v2  # type: ignore
import travel_model as tm  # type: ignore
from concurrent.futures import ProcessPoolExecutor

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

V3_EXTRA_FEATURES = [
    # Baidu ConSTGAT (2020) path geometry & intersection/turning complexity
    "path_stops_per_km",
    "path_max_sinuosity",
    "path_turn_rad_sum",
    "path_turn_rad_per_km",
    # Overpass OSM road infrastructure (signals, NEW flyovers, rail crossings, roundabouts)
    "path_osm_signals",
    "path_osm_flyovers",
    "path_osm_rail_crossings",
    "path_osm_roundabouts",
    "path_osm_signals_per_km",
    "path_osm_flyovers_per_km",
    # Verified Indian National & State Holiday / Rain-Holiday Calendar (2025-2026)
    "is_major_holiday",
    "is_festival_or_procession",
    # DiDi WDR / HetETA (2018, 2020) Empirical-Bayes route & weekday/weekend residual priors
    "route_bias_prior",
    "route_wknd_bias_prior",
    "city_hour_wknd_bias_prior",
    # Interaction features: rain x peak, live ratio x horizon, pace x short-horizon decay
    "rain_x_peak",
    "decayed_pace_effect",
    "live_ratio_x_horizon",
]

V3_FEATURES = v2.V2_FEATURES + V3_EXTRA_FEATURES


def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlo = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return math.atan2(y, x)


def precompute_route_turns(routes_ed: ed.Routes) -> dict[str, list[float]]:
    """Precompute cumulative absolute turning angle (radians) along each route's stop chords."""
    out = {}
    for code, ll in routes_ed.ll.items():
        n = len(ll)
        turns = [0.0] * n
        prev_b = None
        acc = 0.0
        for i in range(n - 1):
            b = bearing_rad(ll[i][0], ll[i][1], ll[i + 1][0], ll[i + 1][1])
            if prev_b is not None:
                diff = abs(b - prev_b)
                if diff > math.pi:
                    diff = 2.0 * math.pi - diff
                acc += diff
            turns[i + 1] = acc
            prev_b = b
        out[code] = turns
    return out


def precompute_route_osm(routes_ed: ed.Routes, osm_path: Path) -> dict[str, tuple[list[int], list[int], list[int], list[int]]]:
    """Precompute cumulative (signals, flyovers, rail_crossings, roundabouts) along each route using Overpass KD-Tree."""
    idx = osm.CityOSMIndex(osm_path) if osm_path.exists() else None
    out = {}
    for code, ll in routes_ed.ll.items():
        n = len(ll)
        cs, cf, cr, crb = [0] * n, [0] * n, [0] * n, [0] * n
        if idx is not None:
            as_ = af = ar = arb = 0
            for i in range(n - 1):
                s_, f_, r_, rb_ = idx.count_near_segment(ll[i][0], ll[i][1], ll[i + 1][0], ll[i + 1][1], radius_m=65.0)
                as_ += s_; af += f_; ar += r_; arb += rb_
                cs[i + 1] = as_; cf[i + 1] = af; cr[i + 1] = ar; crb[i + 1] = arb
        out[code] = (cs, cf, cr, crb)
    return out


def holiday_flags(day_str: str, city: str, holidays: dict) -> tuple[float, float]:
    info = holidays.get(day_str)
    if not info:
        return 0.0, 0.0
    scopes = info.get("scope", [])
    if "all" not in scopes and city not in scopes:
        return 0.0, 0.0
    imp = str(info.get("impact", ""))
    return (1.0 if imp == "major_holiday" else 0.0, 1.0 if imp in ("festival_eve", "procession_peak", "restricted_holiday") else 0.0)


def build_v3_day(
    city: str,
    city_idx: int,
    day: str,
    data_root: Path,
    routes_tm: tm.Routes,
    routes_ed: ed.Routes,
    route_turns: dict[str, list[float]],
    route_osm: dict[str, tuple[list[int], list[int], list[int], list[int]]],
    holidays: dict,
    model_json_path: Path | None,
    wx_map: dict[int, dict],
    class_map: dict[str, int],
    sample_rate: float,
    max_gap: int,
    fix_stride: int = 3,
    max_queries: int = 450_000,
    cache_dir: Path | None = None,
):
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"v3_{city}_{day}_{int(sample_rate*1000)}_{fix_stride}_{max_queries}.npz"
        if cp.exists():
            z = np.load(cp, allow_pickle=False)
            return z["X"], z["rids"].tolist(), z["y"], z["b"]
    # Reuse v2's exact causal replay by augmenting with route_id & turning geometry
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
    route_ids: list[str] = []
    y_true: list[float] = []
    y_base: list[float] = []

    for t, reg, t_idx, f_idx in events:
        trip = trips_by_reg[reg][t_idx]
        code = trip["code"]
        if code not in routes_ed.ids:
            continue
        s, i, bus_class = trip["fixes"][f_idx][1], trip["fixes"][f_idx][2], trip["fixes"][f_idx][3]
        ids, ds, phys, sinu = routes_ed.ids[code], routes_ed.ds[code], routes_ed.phys[code], routes_ed.sinuosity[code]
        turns = route_turns[code]
        hour = time.gmtime(t + 19800).tm_hour

        kt = (reg, t_idx)
        if kt not in bus_state:
            bus_state[kt] = [1.0, -1.0, t, s]
        st = bus_state[kt]

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

        if reg not in picked or (f_idx % fix_stride != 0) or len(X_rows) >= max_queries:
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

        max_k = max(k for k, _ in valid_targets)
        nc_p = pace_nc.get(reg)
        etas_nc = dict(M.eta(ids, ds, s, next_k, t, pace=nc_p, n=30))

        cum_hist, cum_corr, cum_sinu, cum_max_sinu, cum_lcov = {}, {}, {}, {}, {}
        th = tc = sinu_sum = 0.0
        max_sinu = 1.0
        lcov_n = tot_n = 0
        here = s
        for seg_i in range(next_k - 1, max_k):
            if seg_i < 0 or seg_i >= len(phys):
                continue
            seg_d = max(0.0, ds[seg_i + 1] - here)
            here = ds[seg_i + 1]
            sk = f"{ids[seg_i]}|{ids[seg_i + 1]}"
            ph = phys[seg_i]
            row_m = (M.m.get("seg") or {}).get(sk)
            vh = -1.0
            if row_m:
                vh = row_m[2][hour] if row_m[2][hour] > 0 else row_m[1]
            if not vh or vh <= 0:
                cv = ((M.m.get("city") or {}).get("v") or [-1] * 24)[hour]
                vh = cv if cv and cv > 0 else 4.2
            vh = min(22.0, max(1.0, vh))
            th += seg_d / vh

            cv15 = tracker.query_corr(ph, t)[1]
            if cv15 > 0.5:
                vc = min(22.0, max(1.0, cv15))
                lcov_n += 1
            else:
                vc = vh
            tc += seg_d / vc
            sinu_sum += sinu[seg_i]
            if sinu[seg_i] > max_sinu:
                max_sinu = sinu[seg_i]
            tot_n += 1
            k_stop = seg_i + 1
            cum_hist[k_stop] = th
            cum_corr[k_stop] = tc
            cum_sinu[k_stop] = sinu_sum / tot_n
            cum_max_sinu[k_stop] = max_sinu
            cum_lcov[k_stop] = lcov_n / tot_n

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
        is_peak = int((8 <= hour_ist <= 11) or (17 <= hour_ist <= 20))
        sin_tod = math.sin(min_of_day * (2.0 * math.pi / 1440.0))
        cos_tod = math.cos(min_of_day * (2.0 * math.pi / 1440.0))
        sin_dow = math.sin(dow * (2.0 * math.pi / 7.0))
        cos_dow = math.cos(dow * (2.0 * math.pi / 7.0))
        bc_idx = class_map.get(bus_class, -1)
        lat_now, lon_now = routes_ed.ll[code][cur_i]
        n_stops = len(ds)

        wx = wx_map.get((int(t) // 3600) * 3600, {})
        precip_3h = float(wx.get("precip_3h_mm") or 0.0)
        wx_vals = [
            float(wx["temp_c"]) if wx.get("temp_c") is not None else np.nan,
            float(wx["humidity"]) if wx.get("humidity") is not None else np.nan,
            float(wx.get("precip_mm") or 0.0),
            precip_3h,
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

        turn_now = turns[cur_i]
        osm_s, osm_f, osm_r, osm_rb = route_osm[code]
        os_now, of_now, or_now, orb_now = osm_s[cur_i], osm_f[cur_i], osm_r[cur_i], osm_rb[cur_i]
        h_maj, h_fest = holiday_flags(day, city, holidays)
        for k, true_eta in valid_targets:
            if k not in etas_nc or k not in cum_hist:
                continue
            b_nc = max(5.0, etas_nc[k] - t)
            b_h = max(5.0, cum_hist[k])
            b_c = max(5.0, cum_corr[k])
            dist_ahead = max(1.0, ds[k] - s)
            dist_km = max(0.05, dist_ahead / 1000.0)
            n_ahead = k - next_k + 1
            path_v = dist_ahead / b_c
            kk = min(k, len(turns) - 1)
            turn_sum = max(0.0, turns[kk] - turn_now)
            n_sig = max(0, osm_s[kk] - os_now)
            n_fly = max(0, osm_f[kk] - of_now)
            n_rail = max(0, osm_r[kk] - or_now)
            n_rb = max(0, osm_rb[kk] - orb_now)
            live_ratio = b_c / b_h

            # Placeholders [0.0, 0.0, 0.0] for the 3 train-fit route/weekend residual priors
            feat = [
                city_idx, bc_idx, dist_ahead, n_ahead,
                b_nc, b_h, b_c, live_ratio,
                cur_i, k, n_stops, cur_i / max(n_stops, 1), k / max(n_stops, 1),
                cum_sinu[k], lat_now, lon_now,
                hour_ist, min_of_day, dow, is_wknd, sin_tod, cos_tod, sin_dow, cos_dow,
                nc_p, st[0], st[1], t - st[2], max(0.0, s - st[3]),
                cur_seg_v15, c_v5, c_v15, c_v30, c_n15, c_cv, c_sg,
                up_v15, dn_v15, path_v, cum_lcov[k], city_v,
                lead_hw, corr_hw,
            ] + wx_vals + [
                n_ahead / dist_km,
                cum_max_sinu[k],
                turn_sum,
                turn_sum / dist_km,
                n_sig, n_fly, n_rail, n_rb,
                n_sig / dist_km, n_fly / dist_km,
                h_maj, h_fest,
                0.0, 0.0, 0.0,  # filled after train-split Empirical-Bayes prior fitting
                precip_3h * is_peak,
                (st[0] - 1.0) * math.exp(-b_nc / 300.0),
                (live_ratio - 1.0) * math.log1p(b_nc / 60.0),
            ]
            X_rows.append(feat)
            route_ids.append(f"{city}:{code}")
            y_true.append(true_eta)
            y_base.append(b_nc)

    X_arr = np.asarray(X_rows, dtype=np.float32)
    y_arr = np.asarray(y_true, dtype=np.float64)
    b_arr = np.asarray(y_base, dtype=np.float64)
    if cache_dir is not None and len(X_arr):
        cp = cache_dir / f"v3_{city}_{day}_{int(sample_rate*1000)}_{fix_stride}_{max_queries}.npz"
        np.savez(cp, X=X_arr, rids=np.asarray(route_ids), y=y_arr, b=b_arr)
    return X_arr, route_ids, y_arr, b_arr


def agaci_calibrate_online(
    r_true: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    init_margin: float,
    alpha: float = 0.20,
    gammas: tuple[float, ...] = (0.002, 0.005, 0.015, 0.05),
) -> tuple[np.ndarray, np.ndarray]:
    """Online AgACI (Zaffran et al., ICML 2022) — adapts conformal margin online across test stream."""
    n = len(r_true)
    # Track multiple ACI experts over gamma grid and average their adaptive margins
    expert_m = np.full(len(gammas), init_margin, dtype=np.float64)
    out_lo = np.empty(n, dtype=np.float64)
    out_hi = np.empty(n, dtype=np.float64)
    # Chunked updates every 64 arrivals (simulates ~5 min batch feedback in production)
    step = 64
    for s in range(0, n, step):
        e = min(n, s + step)
        m_now = float(np.median(expert_m))
        out_lo[s:e] = r_lo[s:e] - m_now
        out_hi[s:e] = r_hi[s:e] + m_now
        # Observe empirical miscoverage on this chunk and update experts causally
        cov_chunk = (r_true[s:e] >= out_lo[s:e]) & (r_true[s:e] <= out_hi[s:e])
        err_rate = 1.0 - float(np.mean(cov_chunk))
        for gi, g in enumerate(gammas):
            expert_m[gi] = float(np.clip(expert_m[gi] + g * (err_rate - alpha), -0.15, 0.45))
    return out_lo, out_hi


def _build_city_bundle(args):
    city, city_idx, tr_days, va_day, te_day, tr_sr, te_sr, data_root_s, class_map = args
    data_root = Path(data_root_s)
    meta_dir = data_root / "meta" / city
    cache_dir = data_root / "v3_cache"
    hol_path = data_root / "holidays_india_2025_2026.json"
    holidays = json.loads(hol_path.read_text()) if hol_path.exists() else {}
    routes_tm = tm.Routes(str(meta_dir / "route_stops.json"), str(meta_dir / "stop_dist.json"))
    routes_ed = ed.Routes(meta_dir / "route_stops.json", meta_dir / "stop_dist.json")
    r_turns = precompute_route_turns(routes_ed)
    r_osm = precompute_route_osm(routes_ed, data_root / "osm" / f"{city}_osm_infra.json")
    m_json = meta_dir / "model.json"
    wx_map = v2.load_weather_map(data_root, city)
    max_gap = 600 if city in ("mtc", "mumbai") else 180

    tr_out, va_out, te_out = [], [], None
    for d in tr_days:
        if not list(tm.day_files(str(data_root / "raw" / city), "obs", 1, d)):
            continue
        X, rids, yt, yb = build_v3_day(city, city_idx, d, data_root, routes_tm, routes_ed, r_turns, r_osm, holidays, m_json, wx_map, class_map, tr_sr, max_gap, fix_stride=4, max_queries=220_000, cache_dir=cache_dir)
        if len(X):
            tr_out.append((X, rids, yt, yb))
            print(f"  [v3 train:{city}] {d}: {len(X):,} queries", flush=True)

    if list(tm.day_files(str(data_root / "raw" / city), "obs", 1, va_day)):
        X, rids, yt, yb = build_v3_day(city, city_idx, va_day, data_root, routes_tm, routes_ed, r_turns, r_osm, holidays, m_json, wx_map, class_map, te_sr, max_gap, fix_stride=3, max_queries=260_000, cache_dir=cache_dir)
        if len(X):
            va_out.append((X, rids, yt, yb))
            print(f"  [v3 val:{city}]   {va_day}: {len(X):,} queries", flush=True)

    if list(tm.day_files(str(data_root / "raw" / city), "obs", 1, te_day)):
        X, rids, yt, yb = build_v3_day(city, city_idx, te_day, data_root, routes_tm, routes_ed, r_turns, r_osm, holidays, m_json, wx_map, class_map, te_sr, max_gap, fix_stride=3, max_queries=1_500_000, cache_dir=cache_dir)
        if len(X):
            te_out = (te_day, X, rids, yt, yb)
            print(f"  [v3 test:{city}]  {te_day}: {len(X):,} queries", flush=True)
    return city, tr_out, va_out, te_out


def run_v3(data_root: Path, model_dir: Path, cities: list[str], gdrive_out: Path | None):
    t0 = time.time()
    city_map = {c: i for i, c in enumerate(cities)}
    class_vocab = ["", "u", "a", "n", "k", "dn", "ln", "dc", "lc", "ev", "ac", "e", "o", "s", "d"]
    class_map = {c: i for i, c in enumerate(class_vocab)}

    day_plan = {
        "mtc":    (["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.16, 0.35),
        "bmtc":   (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.35, 0.35),
        "mumbai": (["2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"], "2026-11-16", "2026-11-17", 0.50, 0.50),
        "ncr":    (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"], "2026-11-16", "2026-11-17", 1.00, 0.50),
        "delhi":  (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"], "2026-11-22", "2026-11-23", 0.04, 0.10),
    }

    jobs = [
        (c, city_map[c], *day_plan[c], str(data_root), class_map)
        for c in cities if c in day_plan
    ]
    X_tr_l, r_tr_ids, y_tr_l, b_tr_l = [], [], [], []
    X_va_l, r_va_ids, y_va_l, b_va_l = [], [], [], []
    test_data = {}

    with ProcessPoolExecutor(max_workers=min(5, len(jobs))) as ex:
        for city, tr_out, va_out, te_out in ex.map(_build_city_bundle, jobs):
            for X, rids, yt, yb in tr_out:
                X_tr_l.append(X); r_tr_ids.extend(rids); y_tr_l.append(yt); b_tr_l.append(yb)
            for X, rids, yt, yb in va_out:
                X_va_l.append(X); r_va_ids.extend(rids); y_va_l.append(yt); b_va_l.append(yb)
            if te_out is not None:
                test_data[city] = te_out

    X_tr = np.vstack(X_tr_l); y_tr = np.concatenate(y_tr_l); b_tr = np.concatenate(b_tr_l)
    X_va = np.vstack(X_va_l); y_va = np.concatenate(y_va_l); b_va = np.concatenate(b_va_l)
    r_tr = np.clip(np.log(y_tr / b_tr), -1.5, 1.5).astype(np.float32)
    r_va = np.clip(np.log(y_va / b_va), -1.5, 1.5).astype(np.float32)

    # Fit Empirical-Bayes Route & City-Hour-Weekend residual priors STRICTLY on train!
    idx_city = V3_FEATURES.index("city_idx")
    idx_hour = V3_FEATURES.index("hour_ist")
    idx_wknd = V3_FEATURES.index("is_weekend")
    idx_rb = V3_FEATURES.index("route_bias_prior")
    idx_rwb = V3_FEATURES.index("route_wknd_bias_prior")
    idx_chwb = V3_FEATURES.index("city_hour_wknd_bias_prior")

    tr_prior_df = pl.DataFrame({
        "rid": r_tr_ids,
        "city": X_tr[:, idx_city].astype(np.int32),
        "hour": X_tr[:, idx_hour].astype(np.int32),
        "wknd": X_tr[:, idx_wknd].astype(np.int32),
        "res": r_tr,
    })
    p_route = dict(
        tr_prior_df.group_by("rid")
        .agg((pl.col("res").sum() / (pl.len() + 30.0)).alias("b"))
        .iter_rows()
    )
    p_rw = {
        (str(r), int(w)): float(b)
        for r, w, b in tr_prior_df.group_by(["rid", "wknd"])
        .agg((pl.col("res").sum() / (pl.len() + 30.0)).alias("b"))
        .iter_rows()
    }
    p_chw = {
        (int(c), int(h), int(w)): float(b)
        for c, h, w, b in tr_prior_df.group_by(["city", "hour", "wknd"])
        .agg((pl.col("res").sum() / (pl.len() + 50.0)).alias("b"))
        .iter_rows()
    }

    def fill_priors(X_arr: np.ndarray, rids: list[str]):
        cs = X_arr[:, idx_city].astype(np.int32)
        hs = X_arr[:, idx_hour].astype(np.int32)
        ws = X_arr[:, idx_wknd].astype(np.int32)
        X_arr[:, idx_rb] = [p_route.get(r, 0.0) for r in rids]
        X_arr[:, idx_rwb] = [p_rw.get((r, int(w)), p_route.get(r, 0.0)) for r, w in zip(rids, ws)]
        X_arr[:, idx_chwb] = [p_chw.get((int(c), int(h), int(w)), 0.0) for c, h, w in zip(cs, hs, ws)]

    fill_priors(X_tr, r_tr_ids)
    fill_priors(X_va, r_va_ids)
    for city, (te_day, X_te, rids_te, y_te, b_te) in test_data.items():
        fill_priors(X_te, rids_te)

    # Export v3 compact dataset to Google Drive for Colab A100 Deep Residual challenger!
    if gdrive_out is not None:
        gdrive_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gdrive_out / "v3_val.npz", X=X_va, y=y_va, b=b_va, r=r_va)
        # Save a 1.5M train slice for ultra-fast A100 training
        rng_s = np.random.default_rng(42)
        s_idx = rng_s.choice(len(X_tr), size=min(1_500_000, len(X_tr)), replace=False)
        np.savez_compressed(gdrive_out / "v3_train_1.5m.npz", X=X_tr[s_idx], y=y_tr[s_idx], b=b_tr[s_idx], r=r_tr[s_idx])
        (gdrive_out / "v3_features.json").write_text(json.dumps(V3_FEATURES, indent=2))
        print(f"[GDrive] Exported v3 train/val arrays to {gdrive_out}")

    print(f"\n[v3 Ensemble] Train: {len(X_tr):,} | Val: {len(X_va):,} | Features: {len(V3_FEATURES)}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Model 1: LightGBM Huber (leaf-wise) ──
    dtr = lgb.Dataset(X_tr, label=r_tr, feature_name=V3_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    dva = lgb.Dataset(X_va, label=r_va, reference=dtr, feature_name=V3_FEATURES, categorical_feature=["city_idx", "bus_class_idx"])
    print("[v3] 1/3 Training LightGBM Huber ...")
    m_lgb = lgb.train(
        {
            "objective": "huber", "alpha": 0.25, "metric": "l1",
            "learning_rate": 0.04, "num_leaves": 191, "min_child_samples": 100,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
            "n_jobs": -1, "verbose": -1, "seed": 42,
        },
        dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(45, verbose=False), lgb.log_evaluation(250)],
    )
    m_lgb.save_model(str(model_dir / "v3_lgb_huber.txt"))
    p_va_lgb = m_lgb.predict(X_va, num_iteration=m_lgb.best_iteration)

    # ── Model 2: XGBoost Pseudo-Huber (depth-wise) ──
    print("[v3] 2/3 Training XGBoost Pseudo-Huber ...")
    m_xgb = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        huber_slope=0.25,
        eval_metric="mae",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=9,
        min_child_weight=80,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=40,
    )
    m_xgb.fit(X_tr, r_tr, eval_set=[(X_va, r_va)], verbose=200)
    m_xgb.save_model(str(model_dir / "v3_xgb_huber.json"))
    p_va_xgb = m_xgb.predict(X_va)

    # ── Model 3: CatBoost MAE (oblivious symmetric trees) ──
    print("[v3] 3/3 Training CatBoost MAE ...")
    m_cat = CatBoostRegressor(
        loss_function="MAE",
        iterations=1200,
        learning_rate=0.07,
        depth=8,
        random_seed=42,
        thread_count=-1,
        verbose=300,
    )
    m_cat.fit(X_tr, r_tr, eval_set=(X_va, r_va), early_stopping_rounds=60, use_best_model=True)
    m_cat.save_model(str(model_dir / "v3_cat_mae.cbm"))
    p_va_cat = m_cat.predict(X_va)

    # Optimal non-negative convex blend weights on Val MAE
    best_w, best_mae = (0.5, 0.25, 0.25), 1e9
    for w1 in np.linspace(0.2, 0.7, 11):
        for w2 in np.linspace(0.1, 0.6, 11):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            blend = w1 * p_va_lgb + w2 * p_va_xgb + w3 * p_va_cat
            mae = float(np.mean(np.abs(b_va * np.exp(blend) - y_va)))
            if mae < best_mae:
                best_mae = mae
                best_w = (round(float(w1), 3), round(float(w2), 3), round(float(w3), 3))
    print(f"\n[v3 Blend] Optimal weights (LGB, XGB, CAT) = {best_w} | Val MAE = {best_mae:.2f}s")
    print(f"  Singles on Val MAE: LGB={np.mean(np.abs(b_va*np.exp(p_va_lgb)-y_va)):.2f}s | XGB={np.mean(np.abs(b_va*np.exp(p_va_xgb)-y_va)):.2f}s | CAT={np.mean(np.abs(b_va*np.exp(p_va_cat)-y_va)):.2f}s")

    # Quantile P10 / P90 for Mondrian CQR + Online AgACI
    q_models, q_va = {}, {}
    for alpha in (0.10, 0.90):
        qm = lgb.train(
            {"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "verbose": -1, "seed": 42},
            dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        qm.save_model(str(model_dir / f"v3_lgb_q{int(alpha * 100)}.txt"))
        q_models[alpha] = qm
        q_va[alpha] = qm.predict(X_va, num_iteration=qm.best_iteration)

    hb_va = v2.horizon_bucket_idx(b_va)
    cqr_by_bucket = {}
    for b_id in range(5):
        msk = hb_va == b_id
        cqr_by_bucket[b_id] = float(np.quantile(np.maximum(q_va[0.10][msk] - r_va[msk], r_va[msk] - q_va[0.90][msk]), 0.80)) if np.sum(msk) >= 50 else 0.05

    report = {"weights": best_w, "cqr_by_bucket": cqr_by_bucket, "cities": {}}
    w1, w2, w3 = best_w
    for city, (te_day, X_te, _rids, y_te, b_te) in test_data.items():
        r_te_true = np.clip(np.log(y_te / b_te), -1.5, 1.5)
        r_hat = (
            w1 * m_lgb.predict(X_te, num_iteration=m_lgb.best_iteration)
            + w2 * m_xgb.predict(X_te)
            + w3 * m_cat.predict(X_te)
        )
        r_lo = q_models[0.10].predict(X_te, num_iteration=q_models[0.10].best_iteration)
        r_hi = q_models[0.90].predict(X_te, num_iteration=q_models[0.90].best_iteration)

        # Apply Online AgACI per horizon bucket!
        hb_te = v2.horizon_bucket_idx(b_te)
        adj_lo = np.empty_like(r_lo)
        adj_hi = np.empty_like(r_hi)
        for b_id in range(5):
            msk = hb_te == b_id
            if np.any(msk):
                alo, ahi = agaci_calibrate_online(r_te_true[msk], r_lo[msk], r_hi[msk], cqr_by_bucket[b_id])
                adj_lo[msk] = alo
                adj_hi[msk] = ahi

        pred_s = b_te * np.exp(r_hat)
        pred_lo = b_te * np.exp(adj_lo)
        pred_hi = b_te * np.exp(adj_hi)

        s_base = eval_eta.summarize_errors([(float(yt), float(yb), None, None) for yt, yb in zip(y_te, b_te)])
        s_v3 = eval_eta.summarize_errors([(float(yt), float(yp), float(pl_), float(ph_)) for yt, yp, pl_, ph_ in zip(y_te, pred_s, pred_lo, pred_hi)])
        eval_eta.print_table(f"{city.upper()} ({te_day}) — v3 Stacked (LGB+XGB+CAT) + ConSTGAT + WDR Priors + AgACI", s_v3)

        b_mae = s_base["ALL(1-30m)"]["mae_s"]
        v3_mae = s_v3["ALL(1-30m)"]["mae_s"]
        lift = (1.0 - v3_mae / b_mae) * 100.0
        print(f">>> {city.upper()} v3 LIFT vs nowcast: {b_mae:.1f}s -> {v3_mae:.1f}s ({lift:+.1f}% MAE reduction!) | Cov80={s_v3['ALL(1-30m)']['cov80_pct']:.1f}%")
        report["cities"][city] = {"day": te_day, "nowcast": s_base, "v3": s_v3, "mae_lift_pct": round(lift, 2)}

    (model_dir / "v3_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved v3 Ensemble + AgACI models & report to {model_dir} in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--model-dir", default=str(ROOT / "models/v3_ensemble_agaci"))
    ap.add_argument("--cities", default="mtc,bmtc,mumbai,ncr,delhi")
    ap.add_argument("--gdrive-out", default=None)
    a = ap.parse_args()
    run_v3(Path(a.data_root), Path(a.model_dir), [c.strip() for c in a.cities.split(",")], Path(a.gdrive_out) if a.gdrive_out else None)


if __name__ == "__main__":
    main()
