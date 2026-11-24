"""Put the v4 model's arrival times (eta_ml.json, written by eta_ml_service.py) into a collector's
vehicles.json snapshot, just before it is written. Stdlib only; a copy sits beside each collector.

Row format (fmt 2) is unchanged: [reg, code, lat, lon, bearing, fix, flags, eta, (route no.)] with eta a
flat [i, dt, i, dt, ...] (dt = seconds after the fix); only the times change. The snapshot gains
"eta": {"v4": rows given v4 times, "of": rows with any times}. The 80% ranges go to a sidecar next to
vehicles.json, eta_range.json = {"t": snapshot t, "u": 10, "b": {reg: [fix, lo, hi, lo, hi, ...]}}: for the row's
first RANGE_N stops in its eta order, the 80% range as offsets from the P50 time in units of "u" seconds
(range = [P50 + u*lo, P50 + u*hi]), which the cards fetch only while open: no reader of vehicles.json sees a change
(share_api, gtfs_rt, rtc_compact and the dashboards index rows by position).

A row keeps the collector's own times (nowcast, or the operator's) when:
  - eta_ml.json is missing or more than STALE s old (service down),
  - the model has nothing for this bus on this working, or scored an older fix than ~MAX_LAG s,
  - the bus is waiting to start (u), in a depot (p), off its route (g) or its progress is elsewhere (o),
  - mode "ours" and the row's times are the operator's (MTC/BMTC rows without flag m or h).
"""
import json
import os
import time

SKIP = set("upgo")
STALE = 120
MAX_LAG = 120
RANGE_N = 12
UNIT = 10


class Merge:
    def __init__(self, path, mode="all", keep=25, range_out=None):
        self.path, self.mode, self.keep, self.range_out = path, mode, keep, range_out
        self.b, self.mtime, self.t = {}, 0.0, 0

    def load(self):
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            self.b = {}
            return self.b
        if m != self.mtime:
            try:
                with open(self.path) as fh:
                    j = json.load(fh)
                self.b, self.t, self.mtime = j.get("b") or {}, j.get("t") or 0, m
            except (OSError, ValueError):
                pass
        if time.time() - self.t > STALE:
            return {}
        return self.b

    def apply(self, snap, now=None):
        if not self.path or self.mode == "off":
            return snap
        now = now or time.time()
        b = self.load()
        n = have = 0
        ranges = {}
        for row in snap.get("v") or []:
            if len(row) < 8 or not row[7]:
                continue
            have += 1
            fl = set(row[6] or "")
            if fl & SKIP or (self.mode == "ours" and not fl & {"m", "h"}):
                continue
            m = b.get(row[0])
            if not m or m[0] != row[1]:
                continue
            fix, mfix = row[5], m[1]
            if not 0 <= fix - mfix <= MAX_LAG:
                continue
            first = row[7][0]                       # the collector knows which stops are already passed
            flat, rng, q = [], [fix], m[2]
            for j in range(0, len(q) - 3, 4):
                k = q[j]
                if k < first:
                    continue
                t = mfix + q[j + 1]
                if t < now - 120:
                    continue
                flat += [k, int(t - fix)]
                if len(rng) < 1 + 2 * RANGE_N:
                    rng += [round((q[j + 2] - q[j + 1]) / UNIT), round((q[j + 3] - q[j + 1]) / UNIT)]
                if len(flat) >= 2 * self.keep:
                    break
            if not flat:
                continue
            row[7] = flat
            ranges[row[0]] = rng
            n += 1
        snap["eta"] = {"v4": n, "of": have}
        if self.range_out:
            try:
                tmp = self.range_out + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"t": snap.get("t"), "u": UNIT, "b": ranges}, fh, separators=(",", ":"))
                os.replace(tmp, self.range_out)
            except OSError:
                pass
        return snap
