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
import sys
import time

SKIP = set("upgo")
STALE = 120
MAX_LAG = 120
RANGE_N = 12
UNIT = 10
FETCH_EVERY = 5


class Merge:
    def __init__(self, path, mode="all", keep=25, range_out=None):
        self.path, self.mode, self.keep, self.range_out = path, mode, keep, range_out
        self.b, self.mtime, self.t = {}, 0.0, 0
        # ETA_ML_HMAX: use the model only for arrivals up to this many seconds ahead (the collector's own beyond)
        self.hmax = float(os.environ.get("ETA_ML_HMAX") or 1e9)

    def load(self):
        if self.path.startswith("http"):
            return self.load_url()
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

    def load_url(self):
        """eta_ml.json from another host (an RTC collector on SVR reading raw's eta-ml over the internal network):
        fetched at most every FETCH_EVERY s, with If-Modified-Since."""
        import urllib.request
        import urllib.error
        now = time.time()
        if now - getattr(self, "fetched", 0) >= FETCH_EVERY:
            self.fetched = now
            req = urllib.request.Request(self.path, headers={"User-Agent": "eta-ml-merge/1.0"})
            if getattr(self, "lastmod", None):
                req.add_header("If-Modified-Since", self.lastmod)
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    j = json.loads(r.read())
                    self.b, self.t, self.lastmod = j.get("b") or {}, j.get("t") or 0, r.headers.get("Last-Modified")
            except urllib.error.HTTPError as e:
                if e.code != 304:
                    self.b = {}
            except (OSError, ValueError):
                pass
        if time.time() - self.t > STALE:
            return {}
        return self.b

    def times(self, bid, code, fix, now=None):
        """v4 for one bus, for writers that build their own rows: {stop index k: (t, lo, hi)} in epoch seconds, or
        None when the model has nothing current for it (service down, other working, older fix)."""
        if not self.path or self.mode == "off":
            return None
        now = now or time.time()
        m = self.load().get(str(bid))
        if not m or not (m[0] == str(code) or m[0].startswith(f"{code}~")) or not fix or not 0 <= fix - m[1] <= MAX_LAG:
            return None
        q = m[2]
        out = {q[j]: (m[1] + q[j + 1], m[1] + q[j + 2], m[1] + q[j + 3]) for j in range(0, len(q) - 3, 4)
               if now - 120 <= m[1] + q[j + 1] <= now + self.hmax}
        return out or None

    def write_ranges(self, snap_t, ranges):
        """The 80% range sidecar (eta_range.json): {busId: [fix, lo, hi, lo, hi, ...]} as offsets from P50 in UNIT s."""
        if not self.range_out:
            return
        try:
            tmp = self.range_out + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"t": snap_t, "u": UNIT, "b": ranges}, fh, separators=(",", ":"))
            os.replace(tmp, self.range_out)
        except OSError:
            pass

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


class MergeRTC(Merge):
    """The same, for the RTC-standard maps (/tricity, /amts, /tgsrtc, ...), whose times live in live/etas.json:
    {t, buses: {busId: [rid, trip, nextIdx, eta, [[stop, epoch, idx?], ...]]}}. Each listed stop whose route index
    the model scored gets the v4 time; the rest keep the collector's. A stop is placed on the working by its idx
    (when given and the names agree) or by its name / stop id. `fixes` = {busId: fix t} from the same snapshot.

    The model's working must be the row's rid, or (TGSRTC, whose rid is the route number) start with "rid~".
    route_stops: the service's route_stops.json, or its slim copy route_names.json (stop ids and names per working)."""

    def __init__(self, path, route_stops, mode="all", keep=25, range_out=None):
        super().__init__(path, mode, keep, range_out)
        self.rs_path, self.rs_mtime, self.w, self.nn, self.ni, self.nr = route_stops, 0.0, {}, [], [], []

    @staticmethod
    def norm(s):
        return "".join(ch for ch in str(s or "").lower() if ch.isalnum())

    def stops(self, code):
        """(normalised names, normalised stop ids) of a working's stops, or None. The network is held as tables of
        unique names / ids and one tuple of indices per working (TGSRTC: 21k workings in ~10 MB, not ~100)."""
        try:
            m = os.path.getmtime(self.rs_path)
        except OSError:
            return None
        if m != self.rs_mtime:
            try:
                with open(self.rs_path) as fh:
                    rs = json.load(fh)
                if "w" not in rs:                     # route_stops.json: to the same tables
                    nix, iix, w = {}, {}, {}
                    for c, v in rs.items():
                        if v.get("d"):
                            st = v["d"][0]["s"]
                            w[c] = (",".join(str(nix.setdefault(p[1], len(nix))) for p in st) + "|"
                                    + ",".join(str(iix.setdefault(str(p[0]), len(iix))) for p in st))
                    rs = {"n": list(nix), "i": list(iix), "w": w}
                self.nr = rs["n"]
                self.nn = [self.norm(x) for x in rs["n"]]
                self.ni = [self.norm(x) for x in rs["i"]]
                self.w = rs["w"]
                self.rs_mtime = m
                del rs
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass
        ix = self._ix(code)
        return ([self.nn[i] for i in ix[0]], [self.ni[i] for i in ix[1]]) if ix else None

    def _ix(self, code):
        v = self.w.get(code)
        if not v:
            return None
        a, _, b = v.partition("|")
        return [int(x) for x in a.split(",")], [int(x) for x in b.split(",")]

    def raw_names(self, code):
        ix = self._ix(code)
        return [self.nr[i] for i in ix[0]] if ix else []

    def apply_rtc(self, etas, fixes, now=None):
        if not self.path or self.mode == "off":
            return etas
        now = now or time.time()
        b = self.load()
        n = have = 0
        ranges = {}
        for bid, row in (etas.get("buses") or {}).items():
            if not isinstance(row, list) or len(row) < 5 or not row[4]:
                continue
            have += 1
            m = b.get(str(bid))
            if not m or not (m[0] == str(row[0]) or m[0].startswith(f"{row[0]}~")):
                continue
            fix = fixes.get(str(bid)) or fixes.get(bid)
            if not fix or not 0 <= fix - m[1] <= MAX_LAG:
                continue
            sn = self.stops(m[0])
            if not sn:
                continue
            names, ids = sn
            v4 = {m[2][j]: m[2][j + 1:j + 4] for j in range(0, len(m[2]) - 3, 4)}
            rng, done = [int(fix)], 0
            first0 = row[4][0][1] if isinstance(row[4][0], list) and len(row[4][0]) > 1 else None
            for e in row[4]:
                if not isinstance(e, list) or len(e) < 2 or not isinstance(e[1], (int, float)) or e[1] <= 0:
                    continue
                nm = self.norm(e[0])
                k = e[2] if len(e) > 2 and isinstance(e[2], int) and 0 <= e[2] < len(names) and nm in (names[e[2]], ids[e[2]]) else None
                if k is None:
                    c = [i for i, x in enumerate(names) if x == nm] or [i for i, x in enumerate(ids) if x == nm]
                    k = c[0] if len(c) == 1 else None
                q = v4.get(k) if k is not None else None
                if not q or m[1] + q[0] < now - 120 or m[1] + q[0] > now + self.hmax:
                    if len(rng) < 1 + 2 * RANGE_N:
                        rng += [None, None]               # the sidecar follows the list's order: no range here
                    continue
                t = int(m[1] + q[0])
                e[1] = t
                done += 1
                if len(rng) < 1 + 2 * RANGE_N:
                    rng += [round((q[1] - q[0]) / UNIT), round((q[2] - q[0]) / UNIT)]
            if done:
                if first0 is not None and row[3] == first0:  # row[3] was the next stop's time (not TGSRTC's metres)
                    row[3] = row[4][0][1]
                ranges[str(bid)] = rng
                n += 1
        etas["eta"] = {"v4": n, "of": have}
        if self.range_out:
            try:
                tmp = self.range_out + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"t": etas.get("t"), "u": UNIT, "b": ranges}, fh, separators=(",", ":"))
                os.replace(tmp, self.range_out)
            except OSError:
                pass
        return etas

    def for_stops(self, bid, code, fix, names, next_k=None, now=None):
        """v4 for a writer that lists the working's stops itself (names in order, index = the model's stop index):
        {index: (t, lo, hi)} only where the model's stop name at that index is the same stop. Mode "ours" leaves
        the operator's own next-stop time (index next_k) alone."""
        v = self.times(bid, code, fix, now)
        if not v:
            return None
        sn = self.stops(self.load().get(str(bid), [code])[0])
        if not sn:
            return None
        mine = sn[0]
        out = {k: x for k, x in v.items()
               if k < len(names) and k < len(mine) and mine[k] == self.norm(names[k]) and not (self.mode == "ours" and k == next_k)}
        return out or None

    @staticmethod
    def range_row(fix, ordered):
        """[fix, lo, hi, ...] for the sidecar, one pair per entry of the bus's etas list in its order (first RANGE_N):
        offsets from that entry's time in UNIT s; None, None where the entry has no v4 time ((t, lo, hi) or None)."""
        r = [int(fix)]
        for x in ordered[:RANGE_N]:
            r += [round((x[1] - x[0]) / UNIT), round((x[2] - x[0]) / UNIT)] if x else [None, None]
        return r

    def build_etas(self, buses, now=None):
        """An etas.json from the model alone, for a map whose collector makes no ETAs (APSRTC): buses = [(busId,
        working, fix t)] -> {t, cols, buses: {busId: [working, "", first index, first time, [[stop, t, index], ...]]}};
        the 80% ranges go to the sidecar as usual. Stop names come from route_stops."""
        now = now or time.time()
        out, ranges = {}, {}
        for bid, code, fix in buses:
            v = self.times(bid, code, fix, now)
            sn = self.stops(code) if v else None
            if not v or not sn:
                continue
            rs_names = self.raw_names(code)
            ks = [k for k in sorted(v) if k < len(rs_names)][:self.keep]
            if not ks:
                continue
            out[str(bid)] = [code, "", ks[0], int(v[ks[0]][0]), [[rs_names[k], int(v[k][0]), k] for k in ks]]
            ranges[str(bid)] = self.range_row(fix, [v[k] for k in ks])
        self.write_ranges(int(now), ranges)
        return {"t": int(now), "cols": ["stop", "eta", "idx"], "buses": out, "eta": {"v4": len(out), "of": len(buses)}}
