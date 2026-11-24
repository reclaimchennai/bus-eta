#!/usr/bin/env python3
"""Shown-vs-v4 on an RTC map: what the public map showed for a bus at a fix (its etas.json) against what the v4
engine would have shown for the same bus at the same fix, both scored against when the bus really crossed the stop.

Input: a recording made by `rtc_adapt.py live <map>/live/vehicles.json DIR --shown` (DIR/obs-<day>.jsonl = the fixes the
map published, DIR/shown-<day>.jsonl.gz = its etas.json whenever it changed), the system's network in
ETA/data/meta/SYSTEM and its history in ETA/data/raw/SYSTEM (for the causal model.json: days before DAY only).

Truth = eval_eta.extract_ground_truth_stop_times on the recorded fixes. Scored: stops 60 s .. 30 min after the fix, only
(bus, fix, stop) triples that both the map and v4 gave a time for. The engine warms up on the first --warm s.

Usage: rtc_shadow_eval.py SYSTEM DIR DAY [--city-idx 3] [--warm 1800]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rtc_audit as RA  # noqa: E402
from rtc_audit import E, eval_eta, tm, DATA, MODELS  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system"); ap.add_argument("dir"); ap.add_argument("day")
    ap.add_argument("--city-idx", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=0)
    ap.add_argument("--warm", type=int, default=1800)
    ap.add_argument("--no-model-json", action="store_true", help="no history archive for this system (e.g. UP)")
    a = ap.parse_args()
    t0 = time.time()
    run = Path(a.dir)
    cfg = RA.SYSTEMS[a.system]
    max_gap = a.max_gap or cfg["max_gap"]
    meta = DATA / "meta" / a.system
    mj = None if a.no_model_json else RA.causal_model_json(a.system, a.day, max_gap)
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", RA.osm_path(a.system))
    wxm = RA.weather_map(a.system)
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    S = E.CityStream(a.system, R, str(mj) if mj else None, hol, (lambda c, h: wxm.get(h)), max_gap, a.city_idx)
    ens = E.Ensemble(str(MODELS), threads=8)

    # what the map showed: (bus, fix t) -> {k: epoch}, from the first etas.json that carried that fix
    names = {c: [norm(x) for x in R.ids[c]] for c in R.ids}
    rs = json.loads((meta / "route_stops.json").read_text())
    stop_names = {c: [norm(p[1]) for p in v["d"][0]["s"]] for c, v in rs.items()}
    raw_shown, n_snap = {}, 0                       # (bus, fix t) -> (rid, eta list), the first etas.json with that fix
    for p in sorted(run.glob(f"shown-{a.day}.jsonl.gz")):
        for ln in gzip.open(p, "rt"):
            try:
                j = json.loads(ln)
            except ValueError:
                continue
            n_snap += 1
            for bid, (fix, rid, lst) in j["b"].items():
                raw_shown.setdefault((bid, int(fix)), (str(rid), lst))
    n_unmatched = [0]

    def shown_for(bid, fix, code):
        """The map's times for this bus at this fix, placed on the model's working (by idx, name or stop id)."""
        got = raw_shown.get((bid, int(fix)))
        if not got or not (got[0] == code or code.startswith(got[0] + "~")) or code not in stop_names:
            return None
        sn, si, out = stop_names[code], names.get(code) or [], {}
        for e in got[1]:
            if not isinstance(e, list) or len(e) < 2 or not isinstance(e[1], (int, float)) or e[1] <= 0:
                continue
            nm = norm(e[0])
            k = e[2] if len(e) > 2 and isinstance(e[2], int) and 0 <= e[2] < len(sn) and nm in (sn[e[2]], si[e[2]] if e[2] < len(si) else None) else None
            if k is None:
                cand = [i for i, x in enumerate(sn) if x == nm] or [i for i, x in enumerate(si) if x == nm]
                k = cand[0] if len(cand) == 1 else None
            if k is None:
                n_unmatched[0] += 1
                continue
            out[k] = float(e[1])
        return out or None

    print(f"{n_snap} etas.json snapshots, {len(raw_shown):,} (bus, fix) with shown times", flush=True)

    routes_tm = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    tracks = eval_eta.load_day_tracks(routes_tm, run, a.day)
    gt = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(max_gap, 300))
    del tracks
    recs = list(tm.lines(next(p for _d, p in tm.day_files(str(run), "obs", 1, a.day))))
    t_start = min(r["t"] for r in recs) + a.warm
    rows = []   # (y, shown, v4, nowcast, lo, hi)
    batch, clock, last_flush = [], 0.0, 0.0

    def flush():
        res = E.score(S, ens, batch)
        for reg, (code, t, lst, _n) in res.items():
            sh = shown_for(reg, t, code)
            if not sh or t < t_start:
                continue
            j = next((tr for tr in gt.get(reg, ()) if tr["code"] == code and tr["fixes"][0][0] <= t <= tr["fixes"][-1][0]), None)
            if not j:
                continue
            st = j["stop_times"]
            for k, e, lo, hi, nc in lst:
                if k in st and k in sh:
                    y = st[k] - t
                    if 60.0 <= y < 1800.0:
                        rows.append((y, sh[k] - t, e - t, max(5.0, nc - t), lo - t, hi - t))
        batch.clear()

    for r in recs:
        S.ingest(r)
        clock = max(clock, float(r.get("t") or 0))
        for reg in S.take_dirty():
            if reg not in batch:
                batch.append(reg)
        if len(batch) >= 256 or (batch and clock - last_flush >= 10):
            flush(); last_flush = clock
    flush()
    A = np.asarray(rows, dtype=float).reshape(-1, 6)
    y, sh, v4, nc, lo, hi = (A[:, i] for i in range(6))
    print(f"{n_unmatched[0]:,} shown stops not placed on the working", flush=True)
    out = {"system": a.system, "day": a.day, "window": [int(t_start), int(clock)], "n": int(len(y)),
           "shown": RA.stats(sh - y), "v4": RA.stats(v4 - y), "nowcast": RA.stats(nc - y),
           "shown_bias_s": round(float(np.median(sh - y)), 1) if len(y) else None,
           "v4_bias_s": round(float(np.median(v4 - y)), 1) if len(y) else None,
           "cov80": round(float(np.mean((lo <= y) & (y <= hi)) * 100), 1) if len(y) else None, "by_horizon": {}}
    hb = np.array([RA.bucket(h) for h in y]) if len(y) else np.array([])
    for k in ("1-5m", "5-10m", "10-20m", "20-30m"):
        m = hb == k
        out["by_horizon"][k] = {"shown": RA.stats(sh[m] - y[m]), "v4": RA.stats(v4[m] - y[m])}
    print(f">>> {a.system} {a.day} shadow n={out['n']:,}: shown {out['shown'].get('mae_s')} | v4 {out['v4'].get('mae_s')} | "
          f"engine nowcast {out['nowcast'].get('mae_s')} | cov80 {out['cov80']} | bias shown {out['shown_bias_s']} v4 {out['v4_bias_s']} "
          f"| {time.time()-t0:.0f}s", flush=True)
    for k, v in out["by_horizon"].items():
        print(f"    {k:7s} n={v['shown'].get('n', 0):>6,} shown {v['shown'].get('mae_s')} v4 {v['v4'].get('mae_s')}")
    (DATA / "audit" / f"rtc_shadow_{a.system}_{a.day}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
