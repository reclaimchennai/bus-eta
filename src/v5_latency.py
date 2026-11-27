#!/usr/bin/env python3
"""Serving cost: replay a system-day up to a clock time through the streaming engine exactly as the service
does (ingest every fix, score every bus with a new fix each 10 s of feed time) and report feature and model
time per scored row, time per cycle, and the process's peak memory. One process per engine, so the numbers
are separate.

    v5_latency.py v4|v5 SYSTEM DAY [--models DIR] [--threads 2] [--until 9.0] [--score-from 8.5]
"""
from __future__ import annotations

import argparse
import calendar
import json
import resource
import time

import numpy as np

import v5_build as VB
import v5_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine"); ap.add_argument("system"); ap.add_argument("day")
    ap.add_argument("--models", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--use", default="")
    ap.add_argument("--until", type=float, default=9.0)        # IST hour
    ap.add_argument("--score-from", type=float, default=8.5)
    a = ap.parse_args()
    import eta_engine as E
    import eta_engine_v5 as E5
    import travel_model as tm
    DATA, s = VB.DATA, a.system
    mj, _hd, _kind = VB.model_json(s, a.day)
    meta = DATA / "meta" / s
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", DATA / "osm" / f"{s}_osm_infra.json")
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if a.engine == "v5":
        ens = E5.EnsembleV5(a.models or str(L.ROOT / "models" / "v5_ensemble"), threads=a.threads, use=tuple(a.use.split(",")) if a.use else None)
        S = E5.V5Stream(s, R, str(mj) if mj else None, hol, None, VB.RTC.get(s), VB.PLAN[s]["ci"])
    else:
        ens = E.Ensemble(a.models or str(L.MODELS / ("v4r_rtc" if s in VB.RTC else "v4_clean_ensemble")), threads=a.threads)
        S = E.CityStream(s, R, str(mj) if mj else None, hol, None, VB.RTC.get(s), VB.PLAN[s]["ci"])
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    day0 = calendar.timegm(time.strptime(a.day, "%Y-%m-%d")) - 19800
    t_from, t_until = day0 + a.score_from * 3600, day0 + a.until * 3600
    clock = last = 0.0
    t_ing = t_feat = t_pred = 0.0
    n_fix = n_rows = n_cyc = 0
    cyc = []
    dirty = set()
    orig_predict = ens.predict

    def timed_predict(X, b, margins=None):
        nonlocal t_pred, n_rows
        t0 = time.perf_counter()
        out = orig_predict(X, b, margins)
        t_pred += time.perf_counter() - t0
        n_rows += len(X)
        return out
    ens.predict = timed_predict
    for r in tm.lines(str(VB.raw_file(s, a.day))):
        t0 = time.perf_counter()
        S.ingest(r)
        t_ing += time.perf_counter() - t0
        n_fix += 1
        clock = max(clock, float(r.get("t") or 0))
        if clock >= t_until:
            break
        if clock - last >= 10:
            last = clock
            dirty.update(S.take_dirty())
            if clock >= t_from:
                t0 = time.perf_counter(); p0 = t_pred
                E.score(S, ens, list(dirty))
                dt = time.perf_counter() - t0
                t_feat += dt - (t_pred - p0)
                cyc.append(dt); n_cyc += 1
            dirty.clear()
        if n_fix % 500000 == 0:
            S.gc(clock)
    rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(json.dumps({"engine": a.engine, "system": s, "day": a.day, "threads": a.threads, "fixes": n_fix,
                      "ingest_us_per_fix": round(1e6 * t_ing / max(1, n_fix), 1), "cycles": n_cyc, "rows_scored": n_rows,
                      "features_us_per_row": round(1e6 * t_feat / max(1, n_rows), 1), "model_us_per_row": round(1e6 * t_pred / max(1, n_rows), 1),
                      "cycle_s_mean": round(float(np.mean(cyc)), 3) if cyc else None, "cycle_s_p95": round(float(np.percentile(cyc, 95)), 3) if cyc else None,
                      "rows_per_cycle": round(n_rows / max(1, n_cyc)), "rss_mb_imports": round(rss0), "rss_mb_models_loaded": round(rss1), "rss_mb_peak": round(rss2)}))


if __name__ == "__main__":
    main()
