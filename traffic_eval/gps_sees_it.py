#!/usr/bin/env python3
"""Does the buses' own GPS already show the incident, and when? An event study of bus speeds at each incident.

    gps_sees_it.py SYSTEM [DAY ...]       (default days: every archived day of SYSTEM)

For every event (work/events.json) of the system's city that falls on an archived day:
  * the routes passing it (incident_penalty.py's geometry: 150 m of a stop-to-stop chord, 60 m for a road line),
    and where along each route the pass is (metres `at`);
  * every bus pass of that spot on every archived day: the time to cover the 600 m stretch [at-300, at+300]
    (both ends interpolated between consecutive fixes, gap <= the system's max gap) -> a speed and a pass time;
  * baseline = the median speed of passes at the same spot on OTHER days within +-60 min of the same time of day;
  * ratio = pass speed / baseline, against minutes since the alert's start (-120 .. +240 min);
  * placebo: the same spot and clock time on every OTHER archived day (no alert), scored the same way.

Writes results/gps_<SYSTEM>_passes.csv.gz (one row per pass on an event day) and prints the per-event summary.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import time

import numpy as np

from common import DATA, RESULTS, WORK, CITY_OF, day_of, IST

import eval_eta  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402
from match_incidents import on_path, prep  # noqa: E402

HALF = 300.0
MAXGAP = {"tgsrtc": 600, "delhi": 180, "bmtc": 180, "mtc": 600}


def cross_time(fx, x):
    """Interpolated time a trip's fixes [(t, s)] cross metres x; None outside or across a gap."""
    for (t1, s1), (t2, s2) in zip(fx, fx[1:]):
        if s1 <= x <= s2 and s2 > s1:
            return t1 + (x - s1) / (s2 - s1) * (t2 - t1)
    return None


def main():
    system = sys.argv[1]
    city = CITY_OF[system]
    meta = DATA / "meta" / system
    routes = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    events = json.loads((WORK / "events.json").read_text())
    items = prep(events)
    arch = DATA / "raw" / system
    days = sys.argv[2:] or sorted({d for d, _p in tm.day_files(str(arch), "obs", 400, "2026-12-31")})
    ev_days = {}
    for n, e in enumerate(events):
        if e["city"] == city and day_of(e["t_start"]) in days and e["type"] != "u_turn_change":
            ev_days[n] = day_of(e["t_start"])
    if not ev_days:
        print("no events on archived days")
        return
    # where each event sits on each route (the production geometry, ignoring time)
    spots = collections.defaultdict(list)          # code -> [(event n, at)]
    t0 = time.time()
    for code, ll_la in routes.ll.items():
        ll = [(lo, la) for la, lo in ll_la]
        ds = routes.ds[code]
        its = [(items[n][0], items[n][1], items[n][2], -1, 2**40) + items[n][5:] for n in ev_days]
        for at, n, _d in on_path(its, ll, ds, -1e9, 0, ahead_max=1e12):
            if HALF <= at <= ds[-1] - HALF:
                spots[code].append((n, at))
    print(f"{len(ev_days)} events on {len(set(ev_days.values()))} archived days, passed by {len(spots)} routes; "
          f"{time.time()-t0:.0f}s", flush=True)
    passes = []                                     # (event n, code, at, day, t_pass, speed)
    import gzip
    cache = WORK / f"gps_passes_{system}.json.gz"
    if cache.exists() and len(sys.argv) == 2:
        passes = [tuple(x) for x in json.load(gzip.open(cache, "rt"))]
        days = []
        print(f"{len(passes):,} passes from {cache.name}", flush=True)
    for day in days:
        try:
            tracks = eval_eta.load_day_tracks(routes, arch, day)
        except FileNotFoundError:
            continue
        gt = eval_eta.extract_ground_truth_stop_times(tracks, routes, max_gap=MAXGAP[system])
        for trips in gt.values():
            for tr in trips:
                sp = spots.get(tr["code"])
                if not sp:
                    continue
                fx = [(f[0], f[1]) for f in tr["fixes"]]
                for n, at in sp:
                    ta, tb, tc = cross_time(fx, at - HALF), cross_time(fx, at + HALF), cross_time(fx, at)
                    if ta is None or tb is None or tc is None or tb <= ta:
                        continue
                    v = 2 * HALF / (tb - ta)
                    if 0.2 <= v <= 25:
                        passes.append((n, tr["code"], at, day, tc, v))
        print(f"  {day}: {len(passes):,} passes so far, {time.time()-t0:.0f}s", flush=True)
    if days:
        json.dump(passes, gzip.open(cache, "wt"))
    # baseline per event at the same time of day on other days
    by = collections.defaultdict(list)            # pooled over the routes passing the spot: it is the road's speed
    for p in passes:
        by[p[0]].append(p)
    rows = []
    tod = lambda t: (t + 19800) % 86400  # noqa: E731
    for n, ps in by.items():
        e = events[n]
        eday = ev_days[n]
        st_tod = tod(e["t_start"])
        T = np.array([tod(p[4]) for p in ps], float)
        V = np.array([p[5] for p in ps], float)
        Dy = np.array([p[3] for p in ps])
        for j, p in enumerate(ps):
            # the event day itself, and every other archived day as a placebo (same spot, same clock time, no alert)
            placebo = p[3] != eday
            day_start = p[4] - tod(p[4])
            tau = (p[4] - (day_start + st_tod)) / 60.0 if placebo else (p[4] - e["t_start"]) / 60.0
            if not (-120 <= tau <= 240):
                continue
            near = np.abs(((T - T[j] + 43200) % 86400) - 43200) <= 3600
            base = V[near & (Dy != p[3]) & (Dy != eday)]
            if len(base) < 3:
                continue
            bm = float(np.median(base))
            rows.append({"event": n, "id": e["id"], "type": e["type"], "code": p[1], "placebo": int(placebo),
                         "day": p[3], "tau_min": round(tau, 1),
                         "dur_min": round((e["t_end"] - e["t_start"]) / 60), "speed": round(p[5], 2),
                         "base": round(bm, 2), "n_base": int(len(base)), "ratio": round(p[5] / bm, 3)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    import gzip
    with gzip.open(RESULTS / f"gps_{system}_passes.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["event"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows):,} event-day passes with a baseline -> results/gps_{system}_passes.csv.gz")


if __name__ == "__main__":
    main()
