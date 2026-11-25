#!/usr/bin/env python3
"""Honest check of the v4 engine on a bus system it was not trained on (the RTC-standard maps).

For one held-out DAY of SYSTEM (archive built by rtc_adapt.py hist|live in ETA/data/raw/SYSTEM, network in
ETA/data/meta/SYSTEM):
  1. model.json from the 7 days BEFORE the day only (travel_model --until DAY-1), as the live server would have it;
  2. the live engine (eta_engine.CityStream + Ensemble) replayed over the day in the collector's write order,
     scored at every Nth fix of a sampled set of buses;
  3. truth = eval_eta.extract_ground_truth_stop_times (crossing of each stop interpolated between two fixes);
  4. nowcast (the engine's own base) vs v4 on exactly the same (bus, fix, stop) queries, stops 60 s .. 30 min ahead.

Weather: Open-Meteo hourly at the system's centre, derived exactly as eta_ml_service does live (cached in
ETA/data/weather/rtc_<system>.json). OSM counts: Overpass over the system's bbox (ETA/data/osm/<system>_osm_infra.json).
city_idx: which trained city's index the system borrows; choose it on a validation day, report on a later one.

Usage: rtc_audit.py SYSTEM DAY [--city-idx 3] [--max-gap 180] [--sample 0.5] [--every 2]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("~/projects")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(BASE / "ETA-wt-baseline" / "src"))
sys.path.insert(2, str(BASE / "ETA-wt-features" / "src"))
import eta_engine as E  # noqa: E402
import eval_eta  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402

DATA = BASE / "ETA" / "data"
MODELS = BASE / "ETA" / "models" / "v4_clean_ensemble"
TRAVEL_MODEL = BASE / "ETA" / "reference" / "mumbai" / "travel_model.py"

# centre (weather), bbox s,w,n,e (OSM), max fix gap read as one movement
SYSTEMS = {
    "tricity": {"c": (30.7333, 76.7794), "bbox": (30.60, 76.62, 30.82, 76.98), "max_gap": 180},
    "amts": {"c": (23.0225, 72.5714), "bbox": (22.90, 72.44, 23.20, 72.75), "max_gap": 300},
    "osrtc": {"c": (20.2961, 85.8245), "bbox": (19.75, 85.55, 20.55, 86.05), "max_gap": 900},
    "osrtc_city": {"c": (20.2961, 85.8245), "bbox": (19.75, 85.55, 20.55, 86.05), "max_gap": 300},
    "up": {"c": (26.8467, 80.9462), "bbox": (26.70, 80.80, 27.00, 81.10), "max_gap": 600},
    "tgsrtc": {"c": (17.3850, 78.4867), "bbox": (17.22, 78.22, 17.58, 78.68), "max_gap": 600},
    "apsrtc": {"c": (16.5062, 80.6480), "bbox": (16.40, 80.50, 16.62, 80.75), "max_gap": 600},
    "apsrtc2": {"c": (16.5062, 80.6480), "bbox": (16.40, 80.50, 16.62, 80.75), "max_gap": 600},   # the cumulative network (25 Nov)
    "kerala": {"c": (9.9312, 76.2673), "bbox": (9.85, 76.20, 10.10, 76.40), "max_gap": 900},
    "hr": {"c": (29.0588, 76.0856), "bbox": (28.40, 76.80, 28.50, 76.90), "max_gap": 900},
    "gsrtc": {"c": (23.2156, 72.6369), "bbox": (23.15, 72.55, 23.30, 72.72), "max_gap": 300},
}
OM_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "weather_code", "cloud_cover",
           "wind_speed_10m", "wind_gusts_10m"]


def get_json(url, data=None, timeout=120):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "reclaim-bus-maps-eta/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def weather_map(system):
    p = DATA / "weather" / f"rtc_{system}.json"
    if not p.exists():
        la, lo = SYSTEMS[system]["c"]
        j = get_json(f"https://api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&past_days=14&forecast_days=1"
                     f"&hourly={','.join(OM_VARS)}&timezone=UTC")
        h = j.get("hourly") or {}
        rows = {}
        for n, ts in enumerate(h.get("time") or []):
            t = int(dt.datetime.fromisoformat(ts + "+00:00").timestamp())
            g = lambda v: (h.get(v) or [None] * (n + 1))[n]  # noqa: E731
            rows[t] = {"temp_c": g("temperature_2m"), "humidity": g("relative_humidity_2m"), "precip_mm": g("precipitation"),
                       "cloud_cover": g("cloud_cover"), "wind_kph": g("wind_speed_10m"), "gust_kph": g("wind_gusts_10m")}
        keys = sorted(rows)
        for i, t in enumerate(keys):                 # eta_ml_service.Weather.fetch
            pr = [float(rows[keys[j]].get("precip_mm") or 0.0) for j in range(max(0, i - 23), i + 1)]
            r = rows[t]
            r["precip_mm"] = pr[-1]
            r["precip_3h_mm"], r["precip_6h_mm"], r["precip_24h_mm"] = sum(pr[-3:]), sum(pr[-6:]), sum(pr)
            r["is_raining"], r["is_heavy_rain"] = float(pr[-1] > 0.2), float(pr[-1] >= 5.0)
        p.write_text(json.dumps({str(k): v for k, v in rows.items() if k <= time.time()}))
    return {int(k): v for k, v in json.loads(p.read_text()).items()}


def osm_path(system):
    p = DATA / "osm" / f"{system}_osm_infra.json"
    if not p.exists():
        import osm_infra_fetch as of  # type: ignore
        s_, w_, n_, e_ = SYSTEMS[system]["bbox"]
        try:
            of.CITY_BBOX[system] = (s_, w_, n_, e_)
            of.fetch_city_osm(system, DATA / "osm")
        except RuntimeError:                       # a big city times Overpass out: four quarters, merged
            merged = {}
            ms, mw = (s_ + n_) / 2, (w_ + e_) / 2
            for i, bb in enumerate([(s_, w_, ms, mw), (s_, mw, ms, e_), (ms, w_, n_, mw), (ms, mw, n_, e_)]):
                q = f"{system}_q{i}"
                of.CITY_BBOX[q] = bb
                part = json.loads(of.fetch_city_osm(q, DATA / "audit").read_text())
                for k, v in part.items():
                    merged.setdefault(k, []).extend(v)
            p.write_text(json.dumps(merged))
    return p


def has_alias(meta):
    rs = json.loads((meta / "route_stops.json").read_text())
    return any("a" in v for v in rs.values())


def causal_model_json(system, day, max_gap):
    until = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    meta = DATA / "meta" / system
    out = DATA / "audit" / f"model_{system}_until_{until}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        stops, dist, arch = meta / "route_stops.json", meta / "stop_dist.json", DATA / "raw" / system
        with tempfile.TemporaryDirectory(dir=DATA / "audit") as td:
            if has_alias(meta):                    # travel_model reads full entries only: the canonical view
                subprocess.run([sys.executable, str(HERE / "rtc_adapt.py"), "canon", str(meta), td, "--archive", str(arch),
                                "--days", "7", "--until", until], check=True)
                stops, dist, arch = Path(td) / "route_stops.json", Path(td) / "stop_dist.json", Path(td) / "archive"
            subprocess.run([sys.executable, str(TRAVEL_MODEL), "--stops", str(stops), "--stop-dist", str(dist),
                            "--archive", str(arch), "--out", str(out), "--days", "7", "--until", until,
                            "--max-gap", str(max(1800, max_gap))], check=True, stdout=subprocess.DEVNULL)
    return out


def tm_routes(meta):
    """travel_model.Routes, with aliases ({"a": chain}) sharing their chain's lists (it reads full entries only)."""
    rs = json.loads((meta / "route_stops.json").read_text())
    alias = {c: v["a"] for c, v in rs.items() if "a" in v}
    if not alias:
        return tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=DATA / "audit", delete=False) as fh:
        json.dump({c: v for c, v in rs.items() if "a" not in v}, fh)
    try:
        R = tm.Routes(fh.name, str(meta / "stop_dist.json"))
    finally:
        os.unlink(fh.name)
    for c, a in alias.items():
        if a in R.ll:
            R.ll[c], R.ds[c], R.ids[c] = R.ll[a], R.ds[a], R.ids[a]
    return R


def bucket(h):
    return "1-5m" if h < 300 else "5-10m" if h < 600 else "10-20m" if h < 1200 else "20-30m"


def stats(err):
    e = np.abs(np.asarray(err, dtype=float))
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mae_s": round(float(e.mean()), 1), "med_s": round(float(np.median(e)), 1),
            "p90_s": round(float(np.percentile(e, 90)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system"); ap.add_argument("day")
    ap.add_argument("--city-idx", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=0)
    ap.add_argument("--sample", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--no-model-json", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    cfg = SYSTEMS[a.system]
    max_gap = a.max_gap or cfg["max_gap"]
    meta = DATA / "meta" / a.system
    mj = None if a.no_model_json else causal_model_json(a.system, a.day, max_gap)
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", osm_path(a.system))
    wxm = weather_map(a.system)
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    S = E.CityStream(a.system, R, str(mj) if mj else None, hol, (lambda c, h: wxm.get(h)), max_gap, a.city_idx)
    ens = E.Ensemble(str(MODELS), threads=8)

    routes_tm = tm_routes(meta)
    tracks = eval_eta.load_day_tracks(routes_tm, DATA / "raw" / a.system, a.day)
    gt = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(max_gap, 300))
    del tracks
    rng = np.random.default_rng(42)
    picked = {reg for reg in gt if rng.random() <= a.sample}
    path = next(p for _d, p in tm.day_files(str(DATA / "raw" / a.system), "obs", 1, a.day))
    recs = list(tm.lines(path))

    rows, codes = [], []                           # (horizon, y, nowcast, v4, lo, hi), working
    gfile = meta / "groups.json"                   # optional {working: group}, e.g. city vs intercity services
    groups = json.loads(gfile.read_text()) if gfile.exists() else {}
    sfile = meta / "sched.json"                    # optional {working: [seconds after IST midnight per stop]}: the timetable
    sched = json.loads(sfile.read_text()) if sfile.exists() else {}
    day0 = dt.datetime.fromisoformat(a.day + "T00:00:00+05:30").timestamp()
    sch_rows = []
    nfix, batch = collections.Counter(), []

    dbg = collections.Counter()

    def flush():
        res = E.score(S, ens, batch)
        dbg["batches"] += 1; dbg["scored"] += len(res)
        for reg, (code, t, lst, _n) in res.items():
            j = next((tr for tr in gt.get(reg, ()) if tr["code"] == code and tr["fixes"][0][0] <= t <= tr["fixes"][-1][0]), None)
            if not j:
                dbg["no_trip"] += 1
                continue
            st = j["stop_times"]
            for k, e, lo, hi, nc in lst:
                if k in st:
                    y = st[k] - t
                    if 60.0 <= y < 1800.0:
                        rows.append((y, y, max(5.0, nc - t), e - t, lo - t, hi - t))
                        codes.append(code)
                        ss = sched.get(code)
                        sch_rows.append(day0 + ss[k] - t if ss and k < len(ss) and ss[k] is not None else np.nan)
        batch.clear()

    clock = last_flush = 0.0
    for r in recs:
        S.ingest(r)
        clock = max(clock, float(r.get("t") or 0))
        for reg in S.take_dirty():
            if reg in picked:
                nfix[reg] += 1
                if nfix[reg] % a.every == 0 and reg not in batch:
                    batch.append(reg)
        if len(batch) >= 256 or (batch and clock - last_flush >= 10):   # the live service scores every 10 s tick
            flush()
            last_flush = clock
    flush()
    A = np.asarray(rows, dtype=float).reshape(-1, 6)
    y, b, p, lo, hi = A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]
    out = {"system": a.system, "day": a.day, "city_idx": a.city_idx, "max_gap": max_gap, "scoring": dict(dbg),
           "model_json": mj.name if mj else None, "buses_with_truth": len(gt), "sampled": len(picked),
           "all": {"nowcast": stats(b - y), "v4": stats(p - y),
                   "v4_bias_s": round(float(np.median(p - y)), 1) if len(y) else None,
                   "cov80": round(float(np.mean((lo <= y) & (y <= hi)) * 100), 1) if len(y) else None},
           "by_horizon": {}}
    hb = np.array([bucket(h) for h in y]) if len(y) else np.array([])
    for k in ("1-5m", "5-10m", "10-20m", "20-30m"):
        m = hb == k
        out["by_horizon"][k] = {"nowcast": stats(b[m] - y[m]), "v4": stats(p[m] - y[m])}
    if sched:
        sc = np.asarray(sch_rows, dtype=float)
        m = ~np.isnan(sc)
        out["timetable"] = {"timetable": stats(sc[m] - y[m]), "v4_same": stats(p[m] - y[m]), "nowcast_same": stats(b[m] - y[m])}
        print(f"    timetable (what /{a.system} shows) n={int(m.sum()):,}: timetable {out['timetable']['timetable'].get('mae_s')} "
              f"| v4 {out['timetable']['v4_same'].get('mae_s')} | nowcast {out['timetable']['nowcast_same'].get('mae_s')}")
    if groups:
        ga = np.array([groups.get(c, "other") for c in codes])
        out["by_group"] = {g: {"nowcast": stats(b[ga == g] - y[ga == g]), "v4": stats(p[ga == g] - y[ga == g])}
                           for g in sorted(set(ga))}
    n_, v_ = out["all"]["nowcast"], out["all"]["v4"]
    gain = 100 * (1 - v_["mae_s"] / n_["mae_s"]) if n_.get("n") else float("nan")
    print(f">>> {a.system} {a.day} ci={a.city_idx} n={n_.get('n', 0):,} nowcast {n_.get('mae_s')} -> v4 {v_.get('mae_s')} "
          f"({gain:+.1f}%) cov80 {out['all']['cov80']} bias {out['all']['v4_bias_s']} | {time.time()-t0:.0f}s", flush=True)
    for k, v in out["by_horizon"].items():
        print(f"    {k:7s} n={v['nowcast'].get('n', 0):>7,} nowcast {v['nowcast'].get('mae_s')} v4 {v['v4'].get('mae_s')}")
    for g, v in (out.get("by_group") or {}).items():
        print(f"    [{g}] n={v['nowcast'].get('n', 0):>7,} nowcast {v['nowcast'].get('mae_s')} v4 {v['v4'].get('mae_s')}")
    (DATA / "audit").mkdir(exist_ok=True)
    (DATA / "audit" / f"rtc_{a.system}_{a.day}_ci{a.city_idx}{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
