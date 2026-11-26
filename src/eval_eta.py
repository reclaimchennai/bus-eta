#!/usr/bin/env python3
"""Unified, leak-free Stop-Arrival ETA Evaluator across Indian bus systems.

Evaluates ANY ETA predictor on held-out test day(s) by replaying the entire fleet in
strict chronological order and scoring predicted arrival times at upcoming stops against
actual observed stop arrival times.

Metrics reported per horizon bucket (1-3m, 3-5m, 5-10m, 10-15m, 15-30m) and overall:
  - MAE (seconds)
  - Median AE (seconds)
  - P80 AE (seconds)
  - P90 AE (seconds)
  - MAPE (%)
  - [P10, P90] empirical coverage (%) and mean interval width (seconds)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow importing existing reference nowcast/travel_model
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reference" / "mumbai"))

import nowcast  # type: ignore
import travel_model as tm  # type: ignore

HORIZON_BUCKETS = [
    ("1-3m", 60, 180),
    ("3-5m", 180, 300),
    ("5-10m", 300, 600),
    ("10-15m", 600, 900),
    ("15-30m", 900, 1800),
]


def load_day_tracks(routes: tm.Routes, archive_dir: Path, day: str):
    """Load one day's GPS archive into per-bus projected tracks: (t, code, s, i, bus_class)."""
    tracks = collections.defaultdict(list)
    last = {}
    day_iter = list(tm.day_files(str(archive_dir), "obs", 1, day))
    if not day_iter:
        raise FileNotFoundError(f"No archive file for {day} in {archive_dir}")
    path = day_iter[0][1]
    for r in tm.lines(path):
        reg = r.get("reg")
        code = r.get("r")
        if not reg or not code:
            continue
        prev = last.get(reg)
        near = prev[3] if prev and prev[1] == code else None
        pr = routes.project(code, r["la"], r["lo"], near)
        if not pr or pr[2] > 150:
            continue
        rec = (float(r["t"]), code, float(pr[0]), int(pr[1]), str(r.get("f") or ""))
        tracks[reg].append(rec)
        last[reg] = rec
    return tracks


def extract_ground_truth_stop_times(tracks, routes: tm.Routes, max_gap: int = 300):
    """For each bus track, compute observed arrival times at each stop crossed along each trip.

    Returns dict: reg -> list of trips, where each trip is:
      {"code": str, "fixes": [(t, s, i, bus_class), ...], "stop_times": {stop_idx: t_arr}}
    """
    trips_by_reg = {}
    for reg, tr in tracks.items():
        trips = []
        cur = None
        for idx, (t, code, s, i, f) in enumerate(tr):
            if (
                cur is None
                or cur["code"] != code
                or t - cur["fixes"][-1][0] > max_gap
                or s < cur["fixes"][-1][1] - 30.0
            ):
                if cur and len(cur["stop_times"]) >= 3:
                    trips.append(cur)
                cur = {"code": code, "fixes": [(t, s, i, f)], "stop_times": {}}
                continue
            pt, ps, pi, _pf = cur["fixes"][-1]
            ds = routes.ds[code]
            d = s - ps
            dt_ = t - pt
            if d > 5.0 and dt_ > 0 and d <= 25.0 * dt_:
                for k in range(pi + 1, i + 1):
                    if 0 < k < len(ds) - 1 and k not in cur["stop_times"]:
                        frac = (ds[k] - ps) / d
                        cur["stop_times"][k] = pt + frac * dt_
            cur["fixes"].append((t, s, i, f))
        if cur and len(cur["stop_times"]) >= 3:
            trips.append(cur)
        if trips:
            trips_by_reg[reg] = trips
    return trips_by_reg


def summarize_errors(records: list[tuple[float, float, float | None, float | None]]) -> dict:
    """Compute summary metrics from list of (true_eta_s, pred_eta_s, p10_s, p90_s)."""
    if not records:
        return {}
    out = {}
    for label, lo, hi in HORIZON_BUCKETS + [("ALL(1-30m)", 60, 1800)]:
        sub = [r for r in records if lo <= r[0] < hi]
        if not sub:
            continue
        trues = np.array([r[0] for r in sub], dtype=np.float64)
        preds = np.array([r[1] for r in sub], dtype=np.float64)
        abs_err = np.abs(preds - trues)
        mape = float(np.mean(abs_err / np.maximum(trues, 1.0)) * 100.0)
        entry = {
            "n": int(len(sub)),
            "mae_s": round(float(np.mean(abs_err)), 1),
            "med_s": round(float(np.median(abs_err)), 1),
            "p80_s": round(float(np.percentile(abs_err, 80)), 1),
            "p90_s": round(float(np.percentile(abs_err, 90)), 1),
            "mape_pct": round(mape, 2),
        }
        if sub[0][2] is not None and sub[0][3] is not None:
            p10s = np.array([r[2] for r in sub], dtype=np.float64)
            p90s = np.array([r[3] for r in sub], dtype=np.float64)
            cov = float(np.mean((trues >= p10s) & (trues <= p90s)) * 100.0)
            wid = float(np.mean(p90s - p10s))
            entry["cov80_pct"] = round(cov, 1)
            entry["width_s"] = round(wid, 1)
        out[label] = entry
    return out


def print_table(name: str, summary: dict):
    print(f"\n=== {name} ===")
    hdr = f"{'Horizon':<12} {'N':>8} {'MAE(s)':>8} {'Med(s)':>8} {'P80(s)':>8} {'P90(s)':>8} {'MAPE%':>8} {'Cov80%':>8} {'Wid(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k, v in summary.items():
        cov = f"{v['cov80_pct']:.1f}" if "cov80_pct" in v else "-"
        wid = f"{v['width_s']:.1f}" if "width_s" in v else "-"
        print(
            f"{k:<12} {v['n']:>8,} {v['mae_s']:>8.1f} {v['med_s']:>8.1f} "
            f"{v['p80_s']:>8.1f} {v['p90_s']:>8.1f} {v['mape_pct']:>8.2f} {cov:>8} {wid:>8}"
        )


def evaluate_nowcast_baseline(
    city: str,
    day: str,
    data_root: Path,
    sample_rate: float = 0.25,
    max_gap: int = 180,
):
    meta_dir = data_root / "meta" / city
    raw_dir = data_root / "raw" / city
    routes = tm.Routes(str(meta_dir / "route_stops.json"), str(meta_dir / "stop_dist.json"))
    model_path = meta_dir / "model.json"
    M = nowcast.Model(str(model_path)) if model_path.exists() else nowcast.Model(None)
    pace = nowcast.Pace()

    t0 = time.time()
    tracks = load_day_tracks(routes, raw_dir, day)
    trips_by_reg = extract_ground_truth_stop_times(tracks, routes, max_gap=max(max_gap, 300))
    print(f"[{city}] {day}: {len(tracks):,} buses, {sum(len(v) for v in trips_by_reg.values()):,} valid trips")

    # Sample buses deterministically
    rng = np.random.default_rng(42)
    picked = {reg for reg in trips_by_reg if rng.random() <= sample_rate}

    # Build chronological event list across whole fleet so live probes are strictly causal
    events = []
    for reg, trips in trips_by_reg.items():
        for t_idx, trip in enumerate(trips):
            for f_idx, (t, s, i, f) in enumerate(trip["fixes"]):
                events.append((t, reg, t_idx, f_idx))
    events.sort(key=lambda x: x[0])

    recs_const = []
    recs_nowcast = []
    recs_nowcast_nolive = []

    for t, reg, t_idx, f_idx in events:
        trip = trips_by_reg[reg][t_idx]
        code = trip["code"]
        s, i = trip["fixes"][f_idx][1], trip["fixes"][f_idx][2]
        ids, ds = routes.ids[code], routes.ds[code]
        hour = time.gmtime(t + 19800).tm_hour

        # Causal update of live probes & pace from previous fix
        if f_idx > 0:
            pt, ps, pi, _pf = trip["fixes"][f_idx - 1]
            dt_ = t - pt
            if 0 < dt_ <= max_gap and s - ps >= 12.0:
                v, span = (s - ps) / dt_, s - ps
                for j in range(pi, i + 1):
                    if j + 1 < len(ids):
                        share = max(0.0, min(s, ds[j + 1]) - max(ps, ds[j])) / span
                        if share > 0:
                            M.add_probe(f"{ids[j]}|{ids[j + 1]}", t, v, dt_ * share)
                pace.update(reg, M.expected_time(ids, ds, ps, s, hour, t), dt_)
            elif 0 < dt_ <= max_gap and abs(s - ps) < 12.0 and i + 1 < len(ids):
                M.add_probe(f"{ids[i]}|{ids[i + 1]}", t, 0.0, dt_)

        # Evaluate only on sampled buses and every 3rd fix to keep speed crisp
        if reg not in picked or (f_idx % 3 != 0):
            continue

        next_k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
        stop_times = trip["stop_times"]
        if not stop_times:
            continue

        # Predict with existing nowcast.py (live probes + pace)
        etas_nc = dict(M.eta(ids, ds, s, next_k, t, pace=pace.get(reg), n=30))
        # Constant speed baseline (4.2 m/s + 15s dwell per stop)
        for k, t_true in stop_times.items():
            if k < next_k or k not in etas_nc:
                continue
            true_eta = t_true - t
            if true_eta < 60.0 or true_eta >= 1800.0:
                continue
            pred_nc = etas_nc[k] - t
            recs_nowcast.append((true_eta, pred_nc, None, None))
            pred_const = max(0.0, ds[k] - s) / 4.2 + max(0, k - next_k) * 15.0
            recs_const.append((true_eta, pred_const, None, None))

    print(f"[{city}] Replayed {len(events):,} fixes in {time.time() - t0:.1f}s")
    s_const = summarize_errors(recs_const)
    s_nc = summarize_errors(recs_nowcast)
    print_table(f"{city.upper()} ({day}) — Constant (4.2 m/s + 15s dwell)", s_const)
    print_table(f"{city.upper()} ({day}) — Existing Production nowcast.py", s_nc)
    return {"city": city, "day": day, "constant": s_const, "nowcast": s_nc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["delhi", "mtc", "bmtc", "mumbai", "ncr"])
    ap.add_argument("--day", required=True, help="YYYY-MM-DD held-out test day")
    ap.add_argument("--data-root", default="~/projects/ETA/data")
    ap.add_argument("--sample", type=float, default=0.25)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()

    max_gap = 600 if a.city in ("mtc", "mumbai") else 180
    res = evaluate_nowcast_baseline(a.city, a.day, Path(a.data_root), a.sample, max_gap=max_gap)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
