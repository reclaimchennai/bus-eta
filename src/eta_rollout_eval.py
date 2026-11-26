#!/usr/bin/env python3
"""End-to-End Multi-Stop ETA Rollout Evaluator for Unified Multi-City LightGBM+CQR+Kalman Model.

Replays a held-out test day in strict chronological order across the whole fleet (100% causal),
maintains shared-corridor multi-route live probes, headway, traffic-discipline proxies (speed CV,
stop-go ratio), per-bus online EWMA pace, and weather (pol.reclaimchennai.city + Open-Meteo),
then rolls out P10/P50/P90 arrival times to every upcoming stop on the route.

Evaluates on the EXACT same ground-truth stop-arrival protocol as eval_eta.py (nowcast.py baseline)
at 1-3m, 3-5m, 5-10m, 10-15m, and 15-30m horizons!
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

# Import shared helpers from sibling worktrees
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference" / "mumbai"))

import eval_eta  # type: ignore
import extract_dataset as ed  # type: ignore
import travel_model as tm  # type: ignore

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


class FastPriorLookup:
    """O(1) dict-backed hierarchical Empirical-Bayes prior lookup for real-time CPU serving."""

    def __init__(self, model_dir: Path, city: str):
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.global_v = float(meta["global_v"])
        self.city_idx = int(meta["city_map"].get(city, -1))
        self.class_map = meta["class_map"]
        self.feature_cols = meta["feature_cols"]
        self.q_cqr = float(meta["q_cqr"])

        def load_map(fname: str, key_cols: list[str], val_col: str) -> dict:
            df = pl.read_parquet(model_dir / fname).filter(pl.col("city") == city)
            if len(df) == 0:
                return {}
            if len(key_cols) == 1:
                return dict(zip(df[key_cols[0]].to_list(), df[val_col].to_list()))
            keys = zip(*(df[c].to_list() for c in key_cols))
            return dict(zip(keys, df[val_col].to_list()))

        self.p_city = load_map("prior_city.parquet", ["hour_ist", "is_weekend"], "prior_city_hw")
        self.p_corr = load_map("prior_corr.parquet", ["phys_link_id", "hour_ist", "is_weekend"], "prior_corr_hw")
        self.p_seg = load_map("prior_seg.parquet", ["seg_key", "hour_ist", "is_weekend"], "prior_seg_hw")
        self.p_offpeak = load_map("prior_offpeak.parquet", ["phys_link_id"], "offpeak_pothole_v")
        self.p_veh = load_map("prior_veh.parquet", ["reg"], "veh_hist_pace")

        self.m10 = lgb.Booster(model_file=str(model_dir / "lgb_q10.txt"))
        self.m50 = lgb.Booster(model_file=str(model_dir / "lgb_q50.txt"))
        self.m90 = lgb.Booster(model_file=str(model_dir / "lgb_q90.txt"))

    def lookup_priors(self, seg_key: str, phys_id: str, hour: int, is_wknd: int, reg: str):
        pc = self.p_city.get((hour, is_wknd), self.global_v)
        pcr = self.p_corr.get((phys_id, hour, is_wknd), pc)
        ps = self.p_seg.get((seg_key, hour, is_wknd), pcr)
        pop = self.p_offpeak.get(phys_id, -1.0)
        pv = self.p_veh.get(reg, 1.0)
        return ps, pcr, pc, pop, pv


def load_weather_lookup(data_root: Path, city: str) -> dict[int, dict]:
    wp = data_root / "weather" / "weather_city_hourly.parquet"
    if not wp.exists():
        return {}
    df = pl.read_parquet(wp).filter(pl.col("city") == city)
    out = {}
    for r in df.iter_rows(named=True):
        out[int(r["hour_ts"])] = r
    return out


def evaluate_gbdt_rollout(
    city: str,
    day: str,
    data_root: Path,
    model_dir: Path,
    sample_rate: float = 0.25,
    max_gap: int = 180,
):
    t0 = time.time()
    meta_dir = data_root / "meta" / city
    raw_dir = data_root / "raw" / city

    routes_tm = tm.Routes(str(meta_dir / "route_stops.json"), str(meta_dir / "stop_dist.json"))
    routes_ed = ed.Routes(meta_dir / "route_stops.json", meta_dir / "stop_dist.json")
    priors = FastPriorLookup(model_dir, city)
    wx_map = load_weather_lookup(data_root, city)

    tracks = eval_eta.load_day_tracks(routes_tm, raw_dir, day)
    trips_by_reg = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(max_gap, 300))
    print(f"[{city}] {day}: {len(tracks):,} buses, {sum(len(v) for v in trips_by_reg.values()):,} valid trips")

    rng = np.random.default_rng(42)
    picked = {reg for reg in trips_by_reg if rng.random() <= sample_rate}

    events = []
    for reg, trips in trips_by_reg.items():
        for t_idx, trip in enumerate(trips):
            for f_idx, (t, s, i, f) in enumerate(trip["fixes"]):
                events.append((t, reg, t_idx, f_idx))
    events.sort(key=lambda x: x[0])

    tracker = ed.LiveCorridorTracker()
    # Per-bus online state: pace_ewma, recent_v, trip_t0, trip_s0
    bus_state: dict[tuple[str, int], list[float]] = {}

    # Collect feature rows for batched LightGBM inference (100x faster than 1-by-1!)
    pending_feats: list[list[float]] = []
    pending_meta: list[tuple] = []  # (query_id, seg_dist_m, prior_v)
    query_targets: list[tuple[float, list[tuple[int, float]]]] = []  # (t0, [(k, true_eta_s), ...])
    query_seg_slices: list[tuple[int, int, float]] = []  # (start_row, end_row, pace_ewma)

    for t, reg, t_idx, f_idx in events:
        trip = trips_by_reg[reg][t_idx]
        code = trip["code"]
        if code not in routes_ed.ids:
            continue
        s, i, bus_class = trip["fixes"][f_idx][1], trip["fixes"][f_idx][2], trip["fixes"][f_idx][3]
        ids, ds, phys, sinu = routes_ed.ids[code], routes_ed.ds[code], routes_ed.phys[code], routes_ed.sinuosity[code]

        key_trip = (reg, t_idx)
        if key_trip not in bus_state:
            bus_state[key_trip] = [1.0, -1.0, t, s]
        st = bus_state[key_trip]

        # Strictly causal update from previous fix
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
                            tracker.record(sk, ph, t, d * share, dt_ * share)
                            tracker.seg_last_pass[sk] = t
                            tracker.corr_last_pass[ph] = t
                # Update online driver/bus pace EWMA against prior/live speed
                ref_v = tracker.query_corr(phys[min(i, len(phys) - 1)], t)[1]
                if ref_v <= 0:
                    dt_obj = dt.datetime.fromtimestamp(t, IST)
                    ref_v = priors.p_city.get((dt_obj.hour, int(dt_obj.weekday() >= 5)), 4.2)
                ratio = max(0.5, min(2.0, v_pair / max(ref_v, 0.5)))
                st[0] = st[0] + 0.25 * (ratio - st[0])
                st[1] = v_pair
            elif 0 < dt_ <= max_gap and abs(d) < 12.0 and i < len(phys):
                tracker.record(f"{ids[i]}|{ids[i + 1]}", phys[i], t, 0.0, dt_)
                st[1] = 0.0

        if reg not in picked or (f_idx % 3 != 0):
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
        dt_obj = dt.datetime.fromtimestamp(t, IST)
        hour_ist = dt_obj.hour
        min_of_day = hour_ist * 60 + dt_obj.minute
        dow = dt_obj.weekday()
        is_wknd = int(dow >= 5)
        sin_tod = math.sin(min_of_day * (2.0 * math.pi / 1440.0))
        cos_tod = math.cos(min_of_day * (2.0 * math.pi / 1440.0))
        sin_dow = math.sin(dow * (2.0 * math.pi / 7.0))
        cos_dow = math.cos(dow * (2.0 * math.pi / 7.0))
        city_v = tracker.query_city(t)
        bc_idx = priors.class_map.get(bus_class, -1)

        wx = wx_map.get((int(t) // 3600) * 3600, {})
        wx_vals = [
            float(wx.get("temp_c") or np.nan),
            float(wx.get("humidity") or np.nan),
            float(wx.get("precip_mm") or 0.0),
            float(wx.get("precip_3h_mm") or 0.0),
            float(wx.get("precip_6h_mm") or 0.0),
            float(wx.get("precip_24h_mm") or 0.0),
            float(wx.get("is_raining") or 0.0),
            float(wx.get("is_heavy_rain") or 0.0),
            float(wx.get("wind_kph") or np.nan),
            float(wx.get("gust_kph") or np.nan),
            float(wx.get("cloud_cover") or np.nan),
            float(wx.get("vis_km") or np.nan),
            float(wx.get("pm25") or np.nan),
            float(wx.get("aqi") or np.nan),
            float(wx.get("pol_precip_mean") or np.nan),
            float(wx.get("pol_rain_accum_mean") or np.nan),
        ]

        start_row = len(pending_feats)
        here = s
        n_stops = len(ds)
        for seg_i in range(next_k - 1, max_k):
            if seg_i < 0 or seg_i >= len(phys):
                continue
            seg_d = max(1.0, ds[seg_i + 1] - here)
            full_len = max(1.0, ds[seg_i + 1] - ds[seg_i])
            here = ds[seg_i + 1]

            sk = f"{ids[seg_i]}|{ids[seg_i + 1]}"
            ph = phys[seg_i]
            ps_pr, pcr_pr, pc_pr, pop_pr, pv_pr = priors.lookup_priors(sk, ph, hour_ist, is_wknd, reg)
            seg_v15, seg_n15 = tracker.query_seg(sk, t)
            c_v5, c_v15, c_v30, c_n15, c_cv, c_stopgo = tracker.query_corr(ph, t)
            up_v15 = tracker.query_seg(f"{ids[seg_i - 1]}|{ids[seg_i]}", t)[0] if seg_i >= 2 else -1.0
            dn_v15 = tracker.query_seg(f"{ids[seg_i + 1]}|{ids[seg_i + 2]}", t)[0] if seg_i + 2 < len(ids) else -1.0
            lead_hw = t - tracker.seg_last_pass.get(sk, -1e9)
            lead_hw = lead_hw if 0 < lead_hw <= 7200 else -1.0
            corr_hw = t - tracker.corr_last_pass.get(ph, -1e9)
            corr_hw = corr_hw if 0 < corr_hw <= 7200 else -1.0

            (la1, lo1), (la2, lo2) = routes_ed.ll[code][seg_i], routes_ed.ll[code][seg_i + 1]
            row = [
                priors.city_idx, bc_idx, seg_i, n_stops, seg_i / max(n_stops, 1),
                full_len, seg_d, sinu[seg_i], 0.5 * (la1 + la2), 0.5 * (lo1 + lo2),
                hour_ist, min_of_day, dow, is_wknd, sin_tod, cos_tod, sin_dow, cos_dow,
                ps_pr, pcr_pr, pc_pr, ps_pr, pop_pr, pv_pr,
                st[0], t - st[2], max(0.0, s - st[3]), st[1], -1.0,
                seg_v15, seg_n15, c_v5, c_v15, c_v30, c_n15, c_cv, c_stopgo,
                up_v15, dn_v15, city_v,
                (c_v15 / ps_pr) if c_v15 > 0 else -1.0,
                (seg_v15 / ps_pr) if seg_v15 > 0 else -1.0,
                lead_hw, corr_hw,
            ] + wx_vals
            pending_feats.append(row)
            pending_meta.append((seg_i + 1, seg_d, ps_pr))

        end_row = len(pending_feats)
        if end_row > start_row:
            query_targets.append((t, valid_targets))
            query_seg_slices.append((start_row, end_row, st[0]))

    print(f"[{city}] Built {len(pending_feats):,} segment rollout rows for {len(query_targets):,} bus queries in {time.time() - t0:.1f}s")
    if not pending_feats:
        return {}

    X = np.asarray(pending_feats, dtype=np.float32)
    t_inf = time.time()
    p10_lr = priors.m10.predict(X)
    p50_lr = priors.m50.predict(X)
    p90_lr = priors.m90.predict(X)
    inf_ms = (time.time() - t_inf) * 1000.0
    print(f"[{city}] Batched LightGBM P10/P50/P90 inference on {len(X):,} segments in {inf_ms:.0f} ms ({inf_ms / max(len(query_targets), 1):.3f} ms/bus)")

    prior_vs = np.array([m[2] for m in pending_meta], dtype=np.float64)
    seg_ds = np.array([m[1] for m in pending_meta], dtype=np.float64)
    stop_ks = [m[0] for m in pending_meta]

    v50 = np.clip(prior_vs * np.exp(p50_lr), 0.4, 22.0)
    # Correlated multi-stop uncertainty scaling: sqrt(N) underestimates corridor correlation;
    # 0.65 * CQR margin per segment aggregates cleanly over multi-stop horizons
    v_lo = np.clip(prior_vs * np.exp(p10_lr - 0.35 * priors.q_cqr), 0.4, 22.0)
    v_hi = np.clip(prior_vs * np.exp(p90_lr + 0.35 * priors.q_cqr), 0.4, 22.0)

    d50 = seg_ds / v50
    d_hi = seg_ds / v_lo
    d_lo = seg_ds / v_hi

    recs = []
    for (t_q, targets), (s_idx, e_idx, _pace) in zip(query_targets, query_seg_slices):
        cum50 = cum_lo = cum_hi = 0.0
        eta_by_k = {}
        for r_i in range(s_idx, e_idx):
            cum50 += d50[r_i]
            cum_lo += d_lo[r_i]
            cum_hi += d_hi[r_i]
            eta_by_k[stop_ks[r_i]] = (cum50, cum_lo, cum_hi)
        for k, true_eta in targets:
            if k in eta_by_k:
                e50, elo, ehi = eta_by_k[k]
                recs.append((true_eta, e50, elo, ehi))

    summary = eval_eta.summarize_errors(recs)
    eval_eta.print_table(f"{city.upper()} ({day}) — Unified Multi-City LightGBM + CQR + Weather", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["delhi", "mtc", "bmtc", "mumbai", "ncr"])
    ap.add_argument("--day", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--model-dir", default=str(ROOT / "models/gbdt_cqr_v1"))
    ap.add_argument("--sample", type=float, default=0.35)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()

    max_gap = 600 if a.city in ("mtc", "mumbai") else 180
    res = evaluate_gbdt_rollout(a.city, a.day, Path(a.data_root), Path(a.model_dir), a.sample, max_gap=max_gap)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
