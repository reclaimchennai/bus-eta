#!/usr/bin/env python3
"""v5 feature caches: one npz per system-day, built by replaying the day through the streaming engine
(serve/eta_engine_v5.V5Stream) in the collector's write order, exactly as a live service would see it.

Leakage rules (the history model, model.json, is the only thing that looks at other days):
  validation and test days   model.json from the 7 days BEFORE the day (travel_model --until day-1), as live.
  training days              model.json from the 7 nearest OTHER days that are not later than the last training
                             day ("leave the day out"): a training row never sees its own day in the history, nor
                             the validation day, nor any test day. This mimics serving (a week of other days) even
                             for the first training days, whose archive has little before them.
v4's own caches (data/v3_cache) used one model.json built after all of these days; that is the leak audit_v4.py
found for the test day, and it is also a train/serve mismatch for the training days.

Rows: every stop ahead of a sampled bus at every Nth fix whose true arrival is 60 s .. 30 min ahead (truth =
eval_eta.extract_ground_truth_stop_times). Columns: v4's 74 features, then eta_engine_v5.V5_EXTRA.
Saved with each row: fix time t, stop index k, bus, working, the bus's sampling draw u (u <= the audit's sample
rate reproduces the audit's bus set), and for RTC systems the service group.

    v5_build.py all [--jobs 8]            every day of the plan
    v5_build.py day SYSTEM DAY            one day

ETA_DATA: the data folder to read (default <repo>/data).  ETA_V5: where to write (default <repo>/data/v5).
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
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(1, str(ROOT / "src"))
sys.path.insert(2, str(ROOT / "reference" / "mumbai"))

DATA = Path(os.environ.get("ETA_DATA", ROOT / "data"))
OUT = Path(os.environ.get("ETA_V5", ROOT / "data" / "v5"))
TRAVEL_MODEL = ROOT / "reference" / "mumbai" / "travel_model.py"


def days(a, b):
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(days=n)).isoformat() for n in range((d1 - d0).days + 1)]


# system: city_idx, train days, validation day, test days, bus sample, score every Nth fix, the audit's sample
PLAN = {
    "mtc":     dict(ci=0, train=days("2026-11-14", "2026-11-21"), val="2026-11-22", test=["2026-11-23", "2026-11-24"], sample=0.35, every=3, std=0.35),
    "bmtc":    dict(ci=1, train=days("2026-11-17", "2026-11-21"), val="2026-11-22", test=["2026-11-23", "2026-11-24"], sample=1.0, every=3, std=0.35),
    "mumbai":  dict(ci=2, train=days("2026-11-08", "2026-11-15"), val="2026-11-16", test=["2026-11-17"], sample=0.5, every=3, std=0.5),
    "ncr":     dict(ci=3, train=days("2026-11-13", "2026-11-15"), val="2026-11-16", test=["2026-11-17"], sample=1.0, every=3, std=0.5),
    "delhi":   dict(ci=4, train=days("2026-11-18", "2026-11-21"), val="2026-11-22", test=["2026-11-23", "2026-11-24"], sample=0.10, every=3, std=0.10),
    "tricity": dict(ci=5, train=days("2026-11-18", "2026-11-22"), val="2026-11-23", test=["2026-11-24"], sample=1.0, every=2, std=0.5),
    "tgsrtc":  dict(ci=6, train=days("2026-11-18", "2026-11-22"), val="2026-11-23", test=["2026-11-24"], sample=1.0, every=2, std=0.5),
    "apsrtc2": dict(ci=7, train=days("2026-11-18", "2026-11-22"), val="2026-11-23", test=["2026-11-24"], sample=1.0, every=2, std=0.5),
}
RTC = {"tricity": 180, "tgsrtc": 600, "apsrtc2": 600}          # max fix gap (serve/rtc_audit.SYSTEMS)
TM_ARGS = {"mtc": ["--max-gap", "1800", "--max-span", "40"], "bmtc": ["--max-gap", "1800", "--max-span", "40"],
           "mumbai": ["--max-gap", "1800", "--max-span", "40"], "delhi": [], "ncr": [],
           "tricity": ["--max-gap", "1800"], "tgsrtc": ["--max-gap", "1800"], "apsrtc2": ["--max-gap", "1800"]}


def raw_file(system, day):
    for p in (DATA / "raw" / system / f"obs-{day}.jsonl", DATA / "raw" / system / f"obs-{day}.jsonl.gz"):
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def canon_dir(system):
    """APSRTC's service keys are aliases of stop chains; travel_model needs the chains and an archive keyed by them."""
    import fcntl
    d = OUT / "canon" / system
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (d / ".done").exists():
            subprocess.run([sys.executable, str(ROOT / "serve" / "rtc_adapt.py"), "canon", str(DATA / "meta" / system), str(d),
                            "--archive", str(DATA / "raw" / system), "--days", "40", "--until", "2026-11-24"], check=True)
            (d / ".done").write_text("ok")
    return d


def has_alias(system):
    rs = json.loads((DATA / "meta" / system / "route_stops.json").read_text())
    return any("a" in v for v in rs.values())


def hist_days(system, day):
    """The days model.json may be built from for this day (see the module docstring)."""
    p = PLAN[system]
    if day in p["train"]:
        pool = [d for d in days((dt.date.fromisoformat(p["train"][0]) - dt.timedelta(days=10)).isoformat(), p["train"][-1])
                if d != day and raw_file(system, d)]
        d0 = dt.date.fromisoformat(day)
        pool.sort(key=lambda d: (abs((dt.date.fromisoformat(d) - d0).days), d))
        return sorted(pool[:7]), "lodo"
    prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    return [d for d in days((dt.date.fromisoformat(day) - dt.timedelta(days=7)).isoformat(), prev) if raw_file(system, d)], "causal"


def model_json(system, day):
    hd, kind = hist_days(system, day)
    if not hd:
        return None, hd, kind
    until = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    if kind == "causal":                      # the audits already built these
        old = DATA / "audit" / f"model_{system}_until_{until}.json"
        if old.exists():
            return old, hd, kind
    out = OUT / "model" / f"model_{system}_{day}_{kind}.json"
    if out.exists():
        return out, hd, kind
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = DATA / "meta" / system
    stops, dist = meta / "route_stops.json", meta / "stop_dist.json"
    with tempfile.TemporaryDirectory(dir=OUT / "model") as td:
        if has_alias(system):
            cd = canon_dir(system)
            stops, dist = cd / "route_stops.json", cd / "stop_dist.json"
            src = {d: cd / "archive" / f"obs-{d}.jsonl.gz" for d in hd}
        else:
            src = {d: raw_file(system, d) for d in hd}
        for d, p in src.items():
            os.symlink(p, Path(td) / p.name)
        span = (dt.date.fromisoformat(hd[-1]) - dt.date.fromisoformat(hd[0])).days + 1
        tmp = out.with_suffix(".tmp")
        subprocess.run([sys.executable, str(TRAVEL_MODEL), "--stops", str(stops), "--stop-dist", str(dist), "--archive", td,
                        "--out", str(tmp), "--days", str(span), "--until", hd[-1]] + TM_ARGS[system],
                       check=True, stdout=subprocess.DEVNULL)
        os.replace(tmp, out)
    return out, hd, kind


def build_day(system, day, force=False):
    import eta_engine as E
    import eta_engine_v5 as E5
    import eval_eta
    import travel_model as tm
    t0 = time.time()
    p = PLAN[system]
    out = OUT / "cache" / f"{system}_{day}.npz"
    if out.exists() and not force:
        return f"{system} {day}: cached"
    if not raw_file(system, day):
        return f"{system} {day}: no archive"
    out.parent.mkdir(parents=True, exist_ok=True)
    mj, hd, kind = model_json(system, day)
    meta = DATA / "meta" / system
    R = E.Routes(meta / "route_stops.json", meta / "stop_dist.json", DATA / "osm" / f"{system}_osm_infra.json")
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    if system in RTC:
        import rtc_audit
        rtc_audit.DATA = DATA
        wxm = rtc_audit.weather_map(system)
        routes_tm = rtc_audit.tm_routes(meta)
        max_gap = RTC[system]
    else:
        import train_v2_deepreta_cqr as v2
        wxm = v2.load_weather_map(DATA, system)
        routes_tm = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
        max_gap = None
    S = E5.V5Stream(system, R, str(mj) if mj else None, hol, (lambda c, h: wxm.get(h)), max_gap, p["ci"])
    tracks = eval_eta.load_day_tracks(routes_tm, DATA / "raw" / system, day)
    gt = eval_eta.extract_ground_truth_stop_times(tracks, routes_tm, max_gap=max(S.max_gap, 300))
    del tracks
    spans = {reg: [(tr["fixes"][0][0], tr["fixes"][-1][0], tr["code"], tr["stop_times"]) for tr in trips] for reg, trips in gt.items()}
    rng = np.random.default_rng(42)
    draw = {reg: rng.random() for reg in gt}                 # the audits pick a bus when its draw <= sample
    picked = {reg for reg, u in draw.items() if u <= p["sample"]}
    del gt
    gfile = meta / "groups.json"
    groups = json.loads(gfile.read_text()) if gfile.exists() else {}
    gvocab = sorted(set(groups.values())) + ["other"]

    X, Y, B, T, Kk, RG, CD, U, G = [], [], [], [], [], [], [], [], []
    reg_id, code_id = {}, {}
    nfix, batch = collections.Counter(), []

    def flush():
        for reg in batch:
            if reg not in S.last or reg not in S.trip:
                continue
            t, code, rows, ks, bases, etas_nc = S.features(reg)
            sp = spans.get(reg)
            j = next((x for x in sp if x[2] == code and x[0] <= t <= x[1]), None) if sp else None
            if not j:
                continue
            st, n_stops = j[3], len(R.ds[code])
            for row, k, b in zip(rows, ks, bases):
                if k in st and k < n_stops - 1:
                    y = st[k] - t
                    if 60.0 <= y < 1800.0:
                        X.append(row); Y.append(y); B.append(b); T.append(t); Kk.append(k)
                        RG.append(reg_id.setdefault(reg, len(reg_id))); CD.append(code_id.setdefault(code, len(code_id)))
                        U.append(draw[reg]); G.append(gvocab.index(groups.get(code, "other")))
        batch.clear()

    # The feed clock is the median time of the last 301 records, not the newest time seen: the archives hold
    # a few records dated hours or days ahead (APSRTC 23/24 Nov open with one), and a clock that jumps to them
    # makes every gc() drop the whole live state. The service itself runs on wall time.
    clock = last_flush = 0.0
    recent = collections.deque(maxlen=301)
    n = 0
    for r in tm.lines(str(raw_file(system, day))):
        S.ingest(r)
        recent.append(float(r.get("t") or 0))
        if n % 50 == 0:
            clock = max(clock, statistics.median(recent))
        for reg in S.take_dirty():
            if reg in picked:
                nfix[reg] += 1
                if nfix[reg] % p["every"] == 0 and reg not in batch:
                    batch.append(reg)
        if len(batch) >= 256 or (batch and clock - last_flush >= 10):    # the live service scores every 10 s
            flush()
            last_flush = clock
        n += 1
        if n % 500000 == 0:
            S.gc(clock)
    flush()
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, X=np.asarray(X, dtype=np.float32).reshape(-1, len(E5.V5_FEATURES)), y=np.asarray(Y, dtype=np.float32),
             b=np.asarray(B, dtype=np.float32), t=np.asarray(T, dtype=np.float64), k=np.asarray(Kk, dtype=np.int16),
             reg=np.asarray(RG, dtype=np.int32), code=np.asarray(CD, dtype=np.int32), u=np.asarray(U, dtype=np.float32),
             g=np.asarray(G, dtype=np.int8), regs=np.asarray(list(reg_id)), codes=np.asarray(list(code_id)),
             groups=np.asarray(gvocab), hist=np.asarray([kind, mj.name if mj else ""] + hd))
    os.replace(tmp, out)
    return f"{system} {day}: {len(Y):,} rows, {len(reg_id)} buses, history {kind} {hd[0] if hd else '-'}..{hd[-1] if hd else '-'} ({len(hd)} d), {time.time()-t0:.0f}s"


def _job(a):
    try:
        return build_day(*a)
    except Exception as e:  # noqa: BLE001
        import traceback
        return f"{a}: FAILED {e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "day"])
    ap.add_argument("system", nargs="?"); ap.add_argument("day", nargs="?")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--systems", default=",".join(PLAN))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "day":
        print(build_day(a.system, a.day, a.force), flush=True)
        return
    tasks = []
    for s in a.systems.split(","):
        p = PLAN[s]
        tasks += [(s, d, a.force) for d in p["train"] + [p["val"]] + p["test"]]
    size = lambda t: -(raw_file(t[0], t[1]).stat().st_size * (8 if str(raw_file(t[0], t[1])).endswith(".gz") else 1)) if raw_file(t[0], t[1]) else 0  # noqa: E731
    tasks.sort(key=size)                                      # the big days first
    for s in a.systems.split(","):
        if has_alias(s):
            canon_dir(s)
    with ProcessPoolExecutor(a.jobs) as ex:
        for f in as_completed([ex.submit(_job, t) for t in tasks]):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
