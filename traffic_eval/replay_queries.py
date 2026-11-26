#!/usr/bin/env python3
"""Replay one held-out day through the LIVE v4 engine and keep every scored query with its context.

    replay_queries.py SYSTEM DAY [--every 2] [--other 0.10]

SYSTEM is a bus system (tgsrtc, delhi, bmtc, mtc). Same machinery as serve/replay_eval.py and serve/rtc_audit.py:
  * model.json (historical segment speeds) from the 7 days BEFORE the day only (travel_model --until DAY-1);
  * eta_engine.CityStream ingests the day's archive in the collector's write order; eta_engine.score() is the
    function the live service calls; truth = the interpolated crossing time of each stop (eval_eta);
  * a query = (bus, fix, stop k) with 60 s <= true time-to-arrival < 30 min, as everywhere else in the repo.

Which buses are scored: every bus that runs a route passing within 500 m of any of the city's traffic events that
day (so incident-affected queries are not thinned by sampling), plus a random OTHER share of the rest.

Output: work/q_<SYSTEM>_<DAY>.npz with, per query: reg, code, t (fix time), s (bus metres along its route), k,
dk (stop k's metres along the route), y (true seconds to arrival), b (nowcast), p (v4 P50), lo, hi (v4 80% band).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import DATA, MODELS, ROOT, WORK, SYSTEM_OF, CITY_OF, day_bounds

import eta_engine as E  # type: ignore  # noqa: E402
import eval_eta  # type: ignore  # noqa: E402
import train_v2_deepreta_cqr as v2  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402

TRAVEL_MODEL = next(p for p in [ROOT / "reference" / "mumbai" / "travel_model.py"] if p.exists())
CFG = {   # max fix gap read as one movement, city index the models use, travel_model flags (as on raw)
    "tgsrtc": {"max_gap": 600, "city_idx": 3, "tm": ["--max-gap", "1800"]},
    "delhi": {"max_gap": 180, "city_idx": None, "tm": []},
    "bmtc": {"max_gap": 180, "city_idx": None, "tm": ["--max-gap", "1800", "--max-span", "40"]},
    "mtc": {"max_gap": 600, "city_idx": None, "tm": ["--max-gap", "1800", "--max-span", "40"]},
}
K = 111320.0


def causal_model_json(system: str, day: str) -> Path | None:
    import datetime as dt
    until = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    out = WORK / f"model_{system}_until_{until}.json"
    if out.exists():
        return out
    meta = DATA / "meta" / system
    rs = json.loads((meta / "route_stops.json").read_text())
    if any("a" in v for v in rs.values()):
        raise SystemExit("aliased network: build its model.json with serve/rtc_adapt.py canon first")
    cmd = [sys.executable, str(TRAVEL_MODEL), "--stops", str(meta / "route_stops.json"), "--stop-dist", str(meta / "stop_dist.json"),
           "--archive", str(DATA / "raw" / system), "--out", str(out), "--days", "7", "--until", until] + CFG[system]["tm"]
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if r.returncode != 0 or not out.exists():
        print(f"[{system}] no model.json until {until} (no history?)", flush=True)
        return None
    print(f"[{system}] model.json until {until}: {time.time()-t0:.0f}s", flush=True)
    return out


def weather(system: str):
    if system == "tgsrtc":
        p = DATA / "weather" / "rtc_tgsrtc.json"
        m = {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}
    else:
        m = v2.load_weather_map(DATA, system)
    return lambda c, h: m.get(h)


def near_codes(R, events, radius=500.0):
    """Route codes with a stop within `radius` m of any event point (a 0.01 degree grid of the events)."""
    grid = collections.defaultdict(list)
    for e in events:
        for lon, lat in e["pts"][:: max(1, len(e["pts"]) // 40)]:
            grid[(int(lat * 100), int(lon * 100))].append((lat, lon))
    out = set()
    for code, ll in R.ll.items():
        hit = False
        for lat, lon in ll:
            ci, cj = int(lat * 100), int(lon * 100)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for (la, lo) in grid.get((ci + di, cj + dj), ()):
                        if math.hypot((lat - la) * K, (lon - lo) * K * math.cos(math.radians(lat))) < radius:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            out.add(code)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system"); ap.add_argument("day")
    ap.add_argument("--every", type=int, default=2, help="score a selected bus at every Nth fix")
    ap.add_argument("--other", type=float, default=0.10, help="share of buses on routes nowhere near an event")
    a = ap.parse_args()
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    cfg = CFG[a.system]
    meta = DATA / "meta" / a.system
    osm = DATA / "osm" / f"{a.system}_osm_infra.json"
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", osm if osm.exists() else None)
    mj = causal_model_json(a.system, a.day)
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    S = E.CityStream(a.system, R, str(mj) if mj else None, hol, weather(a.system), cfg["max_gap"], cfg["city_idx"])
    ens = E.Ensemble(str(MODELS), threads=8)

    d0, d1 = day_bounds(a.day)
    city = CITY_OF[a.system]
    evs = [e for e in json.loads((WORK / "events.json").read_text())
           if e["city"] == city and e["t_start"] < d1 and e["t_end"] > d0]
    near = near_codes(R, evs)
    print(f"[{a.system} {a.day}] {len(evs)} events that day, {len(near)} of {len(R.ll)} routes pass near one; "
          f"{time.time()-t0:.0f}s", flush=True)

    routes_tm = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    tracks = eval_eta.load_day_tracks(routes_tm, DATA / "raw" / a.system, a.day)
    gt = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(S.max_gap, 300))
    del tracks
    rng = np.random.default_rng(42)
    picked = {reg for reg, trips in gt.items()
              if any(tr["code"] in near for tr in trips) or rng.random() <= a.other}
    path = next(p for _d, p in tm.day_files(str(DATA / "raw" / a.system), "obs", 1, a.day))
    print(f"  {len(gt)} buses with truth, {len(picked)} scored; {time.time()-t0:.0f}s", flush=True)

    cols = collections.defaultdict(list)
    nfix, batch = collections.Counter(), []

    def flush():
        res = E.score(S, ens, batch)
        for reg, (code, t, lst, _n) in res.items():
            last = S.last.get(reg)
            if not last or last[0] != t:
                continue
            s_now = last[2]
            j = next((tr for tr in gt.get(reg, ()) if tr["code"] == code and tr["fixes"][0][0] <= t <= tr["fixes"][-1][0]), None)
            if not j:
                continue
            st, dsr = j["stop_times"], R.ds[code]
            for k, e, lo, hi, nc in lst:
                if k in st:
                    y = st[k] - t
                    if 60.0 <= y < 1800.0:
                        for c, v in (("reg", reg), ("code", code), ("t", t), ("s", s_now), ("k", k), ("dk", dsr[k]),
                                     ("y", y), ("b", max(5.0, nc - t)), ("p", e - t), ("lo", lo - t), ("hi", hi - t)):
                            cols[c].append(v)
        batch.clear()

    clock = last_flush = 0.0
    n = 0
    for r in tm.lines(path):
        S.ingest(r)
        clock = max(clock, float(r.get("t") or 0))
        for reg in S.take_dirty():
            if reg in picked:
                nfix[reg] += 1
                if nfix[reg] % a.every == 0 and reg not in batch:
                    batch.append(reg)
        if len(batch) >= 256 or (batch and clock - last_flush >= 10):
            flush()
            last_flush = clock
        n += 1
        if n % 1_000_000 == 0:
            print(f"  {n:,} lines, {len(cols['y']):,} queries, {time.time()-t0:.0f}s", flush=True)
            S.gc(clock)
    flush()
    out = {c: np.asarray(v) for c, v in cols.items()}
    np.savez_compressed(WORK / f"q_{a.system}_{a.day}.npz", **out,
                        model_json=np.asarray(mj.name if mj else ""), n_events=np.asarray(len(evs)))
    y, b, p = out.get("y", np.zeros(0)), out.get("b", np.zeros(0)), out.get("p", np.zeros(0))
    print(f">>> {a.system} {a.day}: {len(y):,} queries, nowcast MAE {np.abs(b-y).mean():.1f} s, v4 {np.abs(p-y).mean():.1f} s "
          f"(model.json {mj.name if mj else None}) | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
