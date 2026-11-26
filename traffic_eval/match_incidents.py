#!/usr/bin/env python3
"""For every replayed query, which active traffic incidents lie on the bus's remaining path to the target stop.

    match_incidents.py [SYSTEM ...] [--window alert|pre|post]   (default: every work/q_*.npz, the alert window)

Geometry is serve/incident_penalty.py's, unchanged: an incident is "on the path" when its point lies within 150 m
(60 m for a line drawn on the road) of one of the trip's stop-to-stop chords between the bus and the target stop,
more than 100 m ahead of the bus (a bus already at the jam carries it in its own pace). Only incidents active at the
fix time count (t_start <= t < t_end).

Adds, per query (work/m_<SYSTEM>_<DAY>.npz, aligned with q_*.npz):
  n_inc      active incidents between the bus and stop k
  inc        index (into work/events.json) of the nearest one, -1 if none
  d_inc      metres from the bus to it along the route
  age_min    minutes since that incident's start at the fix time
  pen        what serve/incident_penalty.py adds to stop k (its priors, caps and easing)
  near_any   an incident is active within 1 km of the route ahead but NOT on the path (for a "near-miss" control)
"""
from __future__ import annotations

import collections
import json
import math
import sys

import numpy as np

from common import WORK

import incident_penalty as IP  # type: ignore  # noqa: E402

K = IP.K


def prep(events):
    """incident_penalty.Incidents.load()'s item tuples, built from work/events.json."""
    items = []
    for n, e in enumerate(events):
        pts = [tuple(p) for p in e["pts"]]
        line = bool(e["is_line"])
        sev = min(1.0, max(0.0, float(e.get("severity") or 0.5)))
        d = min(IP.CAP_ONE, IP.BASE.get(e["type"], IP.BASE["other"]) * (0.5 + sev))
        lim = (IP.NEAR_LINE if line else IP.NEAR_PT) * 1.2
        la0, la1 = min(y for _, y in pts), max(y for _, y in pts)
        dlon = lim / (K * max(0.2, math.cos(math.radians(max(abs(la0), abs(la1))))))
        box = (min(x for x, _ in pts) - dlon, la0 - lim / K, max(x for x, _ in pts) + dlon, la1 + lim / K)
        items.append((n, e["type"], d, int(e["t_start"]), int(e["t_end"]), pts, line, box))
    return items


def on_path(items, ll, ds, s, now, radius=None, ahead_max=IP.AHEAD_MAX):
    """[(metres along, event index, delay at full strength x easing)] for active incidents ahead, nearest pass each.
    radius=None: the production thresholds (150 m point / 60 m line); else a single radius in metres."""
    best = {}
    for iid, typ, d0, st, en, pts, line, (x0, y0, x1, y1) in items:
        if not (st <= now < en):
            continue
        lim = radius if radius is not None else (IP.NEAR_LINE if line else IP.NEAR_PT)
        pad = 0.0 if radius is None else (radius / K) * 1.5
        for i in range(len(ll) - 1):
            if ds[i + 1] < s - IP.PASSED:
                continue
            (ax, ay), (bx, by) = ll[i], ll[i + 1]
            if max(ax, bx) < x0 - pad or min(ax, bx) > x1 + pad or max(ay, by) < y0 - pad or min(ay, by) > y1 + pad:
                continue
            for lon, lat in pts:
                cx = math.cos(math.radians(lat)) * K
                vx, vy = (bx - ax) * cx, (by - ay) * K
                px, py = (lon - ax) * cx, (lat - ay) * K
                L2 = vx * vx + vy * vy
                f = max(0.0, min(1.0, (px * vx + py * vy) / L2)) if L2 else 0.0
                if math.hypot(px - f * vx, py - f * vy) > lim:
                    continue
                at = ds[i] + f * (ds[i + 1] - ds[i])
                if at < s + IP.PASSED or at - s > ahead_max:
                    continue
                if iid not in best or at < best[iid][0]:
                    best[iid] = (at, d0 * IP.Incidents.strength(st, en, now))
    return sorted(((at, iid, d) for iid, (at, d) in best.items()))


def production_penalty(hits, dk):
    """incident_penalty.Incidents.ahead()'s caps, then the caller's rule: add each delay at every stop beyond it."""
    tot, pen = 0.0, 0.0
    for at, _iid, d in hits:
        d = min(d, IP.CAP_BUS - tot)
        if d < 30:
            break
        tot += int(d)
        if at <= dk:
            pen += int(d)
    return pen


def run(qpath, events, items_by_city, R_cache, tag=""):
    import eta_engine as E  # type: ignore
    from common import DATA, CITY_OF
    system = qpath.stem.split("_")[1]
    z = np.load(qpath)
    if "y" not in z.files or not len(z["y"]):
        return None
    if system not in R_cache:
        meta = DATA / "meta" / system
        R_cache.clear()
        R_cache[system] = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", None)
    R = R_cache[system]
    items = items_by_city[CITY_OF[system]]
    n = len(z["y"])
    out = {"n_inc": np.zeros(n, np.int16), "inc": np.full(n, -1, np.int32), "d_inc": np.full(n, np.nan),
           "age_min": np.full(n, np.nan), "pen": np.zeros(n), "near_any": np.zeros(n, bool)}
    groups = collections.defaultdict(list)
    for i, key in enumerate(zip(z["reg"], z["t"])):
        groups[key].append(i)
    codes, ss, dks, ts = z["code"], z["s"], z["dk"], z["t"]
    for (reg, t), idx in groups.items():
        code = codes[idx[0]]
        ll = [(lo, la) for la, lo in R.ll[code]]
        ds = R.ds[code]
        s = float(ss[idx[0]])
        act = [x for x in items if x[3] <= t < x[4]]
        if not act:
            continue
        hits = on_path(act, ll, ds, s, t)
        wide = on_path(act, ll, ds, s, t, radius=1000.0) if not hits else []
        for i in idx:
            dk = float(dks[i])
            h = [x for x in hits if x[0] <= dk]
            if h:
                out["n_inc"][i] = len(h)
                out["inc"][i] = h[0][1]
                out["d_inc"][i] = h[0][0] - s
                out["age_min"][i] = (t - events[h[0][1]]["t_start"]) / 60.0
                out["pen"][i] = production_penalty(hits, dk)
            elif any(x[0] <= dk for x in wide):
                out["near_any"][i] = True
    np.savez_compressed(WORK / qpath.name.replace("q_", f"m{tag}_", 1), **out)
    return int((out["n_inc"] > 0).sum()), n


def shifted(events, window):
    """The same incidents in a window where no alert was out yet (pre: the hour before the start) or any more
    (post: the hour after the end): if v4 is just as late there, the slowdown was not the alert's news."""
    out = []
    for e in events:
        e = dict(e)
        if window == "pre":
            e["t_start"], e["t_end"] = e["t_start"] - 3600, e["t_start"]
        elif window == "post":
            e["t_start"], e["t_end"] = e["t_end"], e["t_end"] + 3600
        out.append(e)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--window", choices=["alert", "pre", "post"], default="alert")
    a = ap.parse_args()
    events = json.loads((WORK / "events.json").read_text())
    ev = events if a.window == "alert" else shifted(events, a.window)
    tag = "" if a.window == "alert" else "_" + a.window
    by_city = collections.defaultdict(list)
    for n, e in enumerate(ev):
        by_city[e["city"]].append(n)
    all_items = prep(ev)
    items_by_city = {c: [all_items[n] for n in ns] for c, ns in by_city.items()}
    want = set(a.systems)
    R_cache = {}
    for q in sorted(WORK.glob("q_*.npz"), key=lambda p: p.stem.split("_")[1]):
        if want and q.stem.split("_")[1] not in want:
            continue
        r = run(q, ev, items_by_city, R_cache, tag)
        if r:
            print(f"{q.stem} [{a.window}]: {r[0]:,} of {r[1]:,} queries have an incident on the path", flush=True)


if __name__ == "__main__":
    main()
