#!/usr/bin/env python3
"""Run the live engine (eta_engine.CityStream + Ensemble) over one archived day and score it against the
arrivals actually observed, on exactly the same queries as nowcast.

--order file  : lines in the order the collector wrote them (what the live service sees: a fix reaches
                us late, and other buses' fixes arrive late too)
--order time  : sorted by fix time (what training did)

Truth = eval_eta.extract_ground_truth_stop_times (interpolated crossing of each stop between two fixes).
Usage: replay_eval.py CITY DAY --model-json M --sample 0.35 [--order file] [--every 3]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("~/projects")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(BASE / "ETA-wt-baseline" / "src"))
sys.path.insert(2, str(BASE / "ETA-wt-models" / "src"))
import eta_engine as E  # noqa: E402
import eval_eta  # type: ignore  # noqa: E402
import train_v2_deepreta_cqr as v2  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402

DATA = BASE / "ETA" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city"); ap.add_argument("day")
    ap.add_argument("--model-json", default="")
    ap.add_argument("--sample", type=float, default=0.35)
    ap.add_argument("--every", type=int, default=3, help="score a sampled bus at every Nth fix")
    ap.add_argument("--order", choices=["file", "time"], default="file")
    ap.add_argument("--models", default=str(BASE / "ETA" / "models" / "v4_clean_ensemble"))
    ap.add_argument("--use", default="lgb,xgb,cat")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--agaci", action="store_true", help="adapt the 80%% band online from observed arrivals")
    ap.add_argument("--out", default=str(DATA / "audit"))
    a = ap.parse_args()
    t0 = time.time()
    meta = DATA / "meta" / a.city
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", DATA / "osm" / f"{a.city}_osm_infra.json")
    print(f"routes {len(R.ll)} in {time.time()-t0:.0f}s", flush=True)
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    wxm = {} if a.no_weather else v2.load_weather_map(DATA, a.city)
    S = E.CityStream(a.city, R, a.model_json or None, hol, (lambda c, h: wxm.get(h)), None)
    ens = E.Ensemble(a.models, threads=8, use=tuple(a.use.split(",")))
    if a.agaci:
        S.cal = E.AgACI(ens.cqr)

    # truth, exactly as the evaluator builds it
    routes_tm = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    tracks = eval_eta.load_day_tracks(routes_tm, DATA / "raw" / a.city, a.day)
    gt = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(S.max_gap, 300))
    spans = {reg: [(tr["fixes"][0][0], tr["fixes"][-1][0], tr["code"], tr["stop_times"]) for tr in trips] for reg, trips in gt.items()}
    del tracks
    rng = np.random.default_rng(42)
    picked = {reg for reg in gt if rng.random() <= a.sample}

    path = next(p for _d, p in tm.day_files(str(DATA / "raw" / a.city), "obs", 1, a.day))
    recs = list(tm.lines(path))
    if a.order == "time":
        recs.sort(key=lambda r: r.get("t") or 0)
    print(f"{len(recs):,} lines, {len(picked)} sampled buses, {time.time()-t0:.0f}s", flush=True)

    nfix = collections.Counter()
    rows_y, rows_b, rows_p, rows_lo, rows_hi, rows_age = [], [], [], [], [], []
    batch = []
    clock = 0.0

    def flush():
        res = E.score(S, ens, batch)
        for reg, (code, t, lst, _n) in res.items():
            sp = spans.get(reg)
            if not sp:
                continue
            j = next((x for x in sp if x[2] == code and x[0] <= t <= x[1]), None)
            if not j:
                continue
            st = j[3]
            for k, e, lo, hi, nc in lst:
                if k in st:
                    y = st[k] - t
                    if 60.0 <= y < 1800.0:
                        rows_y.append(y); rows_p.append(e - t); rows_lo.append(lo - t); rows_hi.append(hi - t)
                        rows_b.append(max(5.0, nc - t)); rows_age.append(clock - t)
        batch.clear()

    n = 0
    for r in recs:
        S.ingest(r)
        clock = max(clock, float(r.get("t") or 0))
        for reg in S.take_dirty():
            if reg in picked:
                nfix[reg] += 1
                if nfix[reg] % a.every == 0 and reg not in batch:
                    batch.append(reg)
        if len(batch) >= 256:
            flush()
        n += 1
        if n % 500000 == 0:
            print(f"  {n:,} lines, {len(rows_y):,} scored, {time.time()-t0:.0f}s", flush=True)
            S.gc(clock)
    flush()
    y, b, p = np.array(rows_y), np.array(rows_b), np.array(rows_p)
    lo, hi = np.array(rows_lo), np.array(rows_hi)
    s_n = eval_eta.summarize_errors(list(zip(y, b, [None] * len(y), [None] * len(y))))
    s_v = eval_eta.summarize_errors(list(zip(y, p, lo, hi)))
    eval_eta.print_table(f"{a.city} {a.day} nowcast ({a.order} order)", s_n)
    eval_eta.print_table(f"{a.city} {a.day} v4 engine ({a.order} order)", s_v)
    A, B = s_n["ALL(1-30m)"], s_v["ALL(1-30m)"]
    print(f">>> {a.city} {a.day} [{a.order}, {a.use}] n={A['n']:,} nowcast {A['mae_s']} -> v4 {B['mae_s']} "
          f"({100*(1-B['mae_s']/A['mae_s']):+.1f}%) cov80 {B.get('cov80_pct')} width {B.get('width_s')} | {time.time()-t0:.0f}s", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"replay_{a.city}_{a.day}_{a.order}_{a.use.replace(',', '+')}.json").write_text(
        json.dumps({"nowcast": s_n, "v4": s_v, "args": vars(a)}, indent=1))
    np.savez_compressed(Path(a.out) / f"replay_{a.city}_{a.day}_{a.order}.npz", y=y, b=b, p=p, lo=lo, hi=hi)


if __name__ == "__main__":
    main()
