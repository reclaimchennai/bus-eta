#!/usr/bin/env python3
"""Serving parity: replay a held-out day through the live service's own code path (eta_ml_service.City with
"engine": "v5": V5Stream, eta_engine.score, EnsembleV5, a tick every 10 s of feed time) and compare the P50 it
would publish with the offline v5 prediction for the same bus, fix and stop (the cache row scored by the same
models). The two differ only in when a fix is scored relative to other buses' fixes (the offline builder
flushes every 256 buses or 10 s, the service at its tick), in rounding to whole seconds, and in the service
keeping a bus's times in stop order.

    v5_parity.py SYSTEM DAY [--models models/v5_ensemble]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import tempfile

import numpy as np

import v5_build as VB
import v5_lib as L
import eta_engine as E  # noqa: E402
import eta_engine_v5 as E5  # noqa: E402
import eta_ml_service as SVC  # noqa: E402
import travel_model as tm  # noqa: E402


class Feed:
    """Stands in for the archive tail: the service reads what the replay queued for this tick."""

    def __init__(self):
        self.q = []

    def open(self, warm_from=None):
        return False

    def read(self):
        q, self.q = self.q, []
        return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system"); ap.add_argument("day")
    ap.add_argument("--models", default=str(L.ROOT / "models" / "v5_ensemble"))
    a = ap.parse_args()
    s, day, DATA = a.system, a.day, VB.DATA
    ens = E5.EnsembleV5(a.models, threads=8)
    z = np.load(VB.OUT / "cache" / f"{s}_{day}.npz")
    p_off = ens.predict(z["X"], z["b"].astype(float))[0]
    regs = z["regs"]
    off = {(str(regs[r]), int(t), int(k)): (float(p), float(b)) for r, t, k, p, b in zip(z["reg"], z["t"], z["k"], p_off, z["b"])}
    mj, _hd, _kind = VB.model_json(s, day)
    meta = DATA / "meta" / s
    if s in VB.RTC:
        import rtc_audit
        rtc_audit.DATA = DATA
        wxm = rtc_audit.weather_map(s)
    else:
        import train_v2_deepreta_cqr as v2
        wxm = v2.load_weather_map(DATA, s)
    tmp = tempfile.mkdtemp()
    cfg = {"city": s, "engine": "v5", "archive": tmp, "out": f"{tmp}/out.json", "route_stops": str(meta / "route_stops.json"),
           "stop_dist": str(meta / "stop_dist.json"), "osm": str(DATA / "osm" / f"{s}_osm_infra.json"),
           "model_json": str(mj) if mj else None, "max_gap": VB.RTC.get(s), "city_idx": VB.PLAN[s]["ci"], "log_share": 0}
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    feed = Feed()
    city = SVC.City(cfg, hol, (lambda c, h: wxm.get(h)), ens, tail=feed)
    diffs, clamped, n_pub = [], [], 0
    recent, clock, last, n = collections.deque(maxlen=301), 0.0, 0.0, 0

    def tick(now):
        nonlocal n_pub
        city.step(now)
        for reg, at in city.last_scored.items():
            if at != now:
                continue
            _code, fix, flat = city.cur[reg]
            for i in range(0, len(flat), 4):
                n_pub += 1
                o = off.get((reg, fix, flat[i]))
                if o and o[1] <= E.HORIZON_MAX:
                    diffs.append(flat[i + 1] - o[0])
                    # the service never lets a later stop come before an earlier one: e = max(previous + 1, model)
                    clamped.append(i > 0 and flat[i + 1] <= flat[i - 3] + 1)

    for r in tm.lines(str(VB.raw_file(s, day))):
        feed.q.append(r)
        recent.append(float(r.get("t") or 0)); n += 1
        if n % 50 == 0:
            clock = max(clock, statistics.median(recent))
        if clock - last >= 10:
            last = clock
            tick(clock)
        if n % 500000 == 0:
            city.S.gc(clock)
    tick(clock + 10)
    d, c = np.abs(np.asarray(diffs)), np.asarray(clamped, dtype=bool)

    def stats(x):
        return {"n": int(len(x)), "median": round(float(np.median(x)), 2), "mean": round(float(x.mean()), 2), "p90": round(float(np.percentile(x, 90)), 2),
                "p99": round(float(np.percentile(x, 99)), 2), "max": round(float(x.max()), 1), "within_1s": round(float(np.mean(x <= 1.0)), 4),
                "within_5s": round(float(np.mean(x <= 5.0)), 4)}
    print(json.dumps({"system": s, "day": day, "published_stop_times": n_pub, "offline_rows": len(off),
                      "abs_diff_s_all": stats(d), "abs_diff_s_model_time_published_as_is": stats(d[~c]),
                      "abs_diff_s_moved_to_keep_stop_order": stats(d[c]) if c.any() else None}))


if __name__ == "__main__":
    main()
