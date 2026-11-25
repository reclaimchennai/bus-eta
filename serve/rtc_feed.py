#!/usr/bin/env python3
"""eta-rtc-feed: the RTC maps' live feeds turned into the archives eta-ml reads, without touching their collectors.

For each configured system it reads the map's own vehicles.json (on SVR's feed server, read-only) every `every` s and
appends each bus's new fixes to <out>/<system>/obs-<IST day>.jsonl as {t, reg, r, la, lo, f}, r = the route working:
  mode "rid"  the vehicles.json rid column (a services.json key: /tricity, /amts, /upsrtc, ...)
  mode "ap"   APSRTC: the bus id is the serviceDocId DDMMYYYY_<dcp>_<trip>_<depot>; the working is what follows the date.
              Every `net_every` s (300) live/trips.json (stop lists of the services running now) is read and each
              service not met before, or whose stop list changed, appended to new_workings.jsonl for rtc_adapt ap-net
  mode "tg"   TGSRTC: routelink.json says which trip each bus runs; live/trip/<bus>.json gives that trip's station
              chain (live/trip/<serviceId>.json), fetched once per new trip (at most `trip_rate` a second); working = route no. + station sequence
At the first fix of a new IST day the previous day's file is gzipped (travel_model and eta-ml read either form).

Usage: rtc_feed.py --config rtc_feed.json
  {"base": "http://RTC_SERVER_LAN:8091", "out": "/home/user/projects/eta-ml/rtc",
   "systems": [{"name": "apsrtc", "mode": "ap", "every": 30}, {"name": "tgsrtc", "mode": "tg", "every": 20}, ...]}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import threading
import time
import urllib.request
import zlib

import rtc_adapt as A

UA = "reclaim-bus-maps-eta/1.0"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            b = gzip.decompress(b)
        return json.loads(b)


class Feed:
    def __init__(self, base, out, cfg):
        self.name, self.mode, self.every = cfg["name"], cfg.get("mode", "rid"), float(cfg.get("every", 20))
        self.page = cfg.get("page", self.name)                  # the map's folder on the feed server
        self.url = f"{base}/{self.page}/live/"
        self.dir = os.path.join(out, self.name)
        os.makedirs(self.dir, exist_ok=True)
        self.last, self.fh, self.day = {}, None, None
        self.trip_rate = float(cfg.get("trip_rate", 4))
        self.tg = {}                                            # bus -> (routelink key, working)
        self.tg_known = set()
        if self.mode == "tg":                                   # workings already in the network
            for p in (cfg.get("net"), os.path.join(self.dir, "new_workings.jsonl")):
                try:
                    if p and p.endswith(".json"):
                        self.tg_known |= set(json.load(open(p)))
                    elif p and os.path.exists(p):
                        self.tg_known |= {json.loads(ln)["code"] for ln in open(p)}
                except (OSError, ValueError, KeyError):
                    pass
        self.ap_seen, self.ap_next = {}, 0.0                     # APSRTC: service key -> crc of its stop list
        self.net_every = float(cfg.get("net_every", 300))
        if self.mode == "ap":
            try:
                for ln in open(os.path.join(self.dir, "new_workings.jsonl")):
                    w = json.loads(ln)
                    self.ap_seen[w["code"]] = zlib.crc32(json.dumps(w["stops"]).encode())
            except (OSError, ValueError, KeyError):
                pass
        self.n = 0

    def ap_workings(self):
        """APSRTC: the services in trips.json not met before (or with a new stop list) to new_workings.jsonl."""
        tj = get(self.url + "trips.json", timeout=60)
        new = []
        for key, chain in (tj.get("trips") or {}).items():
            stops = [[p[0], p[1], p[2]] for p in chain if isinstance(p, list) and len(p) >= 3]
            h = zlib.crc32(json.dumps(stops).encode())
            if self.ap_seen.get(key) != h:
                self.ap_seen[key] = h
                new.append(json.dumps({"code": key, "stops": stops}, separators=(",", ":")))
        if new:
            with open(os.path.join(self.dir, "new_workings.jsonl"), "a") as fh:
                fh.write("\n".join(new) + "\n")
            log(f"[{self.name}] {len(new)} new services' stop lists ({len(self.ap_seen)} known)")

    def rotate(self, day):
        if self.fh:
            self.fh.close()
            old = os.path.join(self.dir, f"obs-{self.day}.jsonl")
            if os.path.exists(old) and not os.path.exists(old + ".gz"):
                with open(old, "rb") as src, gzip.open(old + ".gz.tmp", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(old + ".gz.tmp", old + ".gz")
                os.remove(old)                                  # the .gz is the complete copy now
        self.fh, self.day = open(os.path.join(self.dir, f"obs-{day}.jsonl"), "a"), day

    def working_tg(self, bid, link):
        """TGSRTC: the working for this bus's current trip, from its trip file (fetched once per new trip)."""
        key = tuple(link[:2]) if isinstance(link, list) else None
        if key is None:
            return None
        have = self.tg.get(bid)
        if have and have[0] == key:
            return have[1]
        if self._trip_budget <= 0:
            return None
        self._trip_budget -= 1
        try:
            tj = get(f"{self.url}trip/{link[1]}.json", timeout=15)      # trip files are named by serviceId
        except Exception:  # noqa: BLE001
            return None
        cols = {c: i for i, c in enumerate(tj.get("cols") or [])}
        names = [s[cols.get("stop", 0)] for s in tj.get("stops") or []]
        if len(names) < 2:
            return None
        code = A.tg_code(tj.get("route") or link[0], names)
        self.tg[bid] = (key, code)
        if code not in self.tg_known:            # a working the network lacks: kept for rtc_adapt tg-extend
            self.tg_known.add(code)
            I, O = cols.get("lat", 1), cols.get("lon", 2)
            with open(os.path.join(self.dir, "new_workings.jsonl"), "a") as fh:
                fh.write(json.dumps({"code": code, "rno": tj.get("route") or link[0], "type": tj.get("type"),
                                     "stops": [[s[cols.get("stop", 0)], s[I], s[O]] for s in tj.get("stops") or []]},
                                    separators=(",", ":")) + "\n")
        return code

    def step(self):
        j = get(self.url + "vehicles.json")
        ix = {c: i for i, c in enumerate(j.get("cols") or [])}
        link = {}
        if self.mode == "tg":
            rl = get(self.url + "routelink.json")
            link = rl.get("buses", rl) if isinstance(rl.get("buses"), dict) else rl
            self._trip_budget = int(self.trip_rate * self.every)
        recs = []
        for b in j.get("buses") or []:
            bid, t = str(b[ix["id"]]), int(b[ix["t"]] or 0)
            if self.mode == "rid":
                code = b[ix["rid"]] if "rid" in ix else None
            elif self.mode == "ap":
                code = A.ap_key(bid)
            else:
                code = self.working_tg(bid, link.get(bid))
            if not code or not t or self.last.get(bid) == (t, code):
                continue
            self.last[bid] = (t, code)
            recs.append({"t": t, "reg": bid, "r": code, "la": b[ix["lat"]], "lo": b[ix["lon"]],
                         "f": A.bus_class(b[ix["type"]] if "type" in ix else "")})
        recs.sort(key=lambda r: r["t"])
        day = A.ist_day(time.time())
        if day != self.day:
            self.rotate(day)
        for r in recs:
            self.fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.fh.flush()
        self.n += len(recs)
        if len(self.last) > 60000:
            cut = time.time() - 6 * 3600
            self.last = {k: v for k, v in self.last.items() if v[0] >= cut}

    def loop(self):
        while True:
            t0 = time.time()
            if self.mode == "ap" and t0 >= self.ap_next:
                self.ap_next = t0 + self.net_every
                try:
                    self.ap_workings()
                except Exception as e:  # noqa: BLE001
                    log(f"[{self.name}] trips.json: {e}")
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                log(f"[{self.name}] {e}")
            time.sleep(max(1.0, self.every - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    feeds = [Feed(cfg["base"].rstrip("/"), cfg["out"], s) for s in cfg["systems"] if s.get("enabled", True)]
    for f in feeds:
        threading.Thread(target=f.loop, daemon=True, name=f.name).start()
    while True:
        time.sleep(300)
        log(" | ".join(f"{f.name}: {f.n} fixes" + (f", {len(f.tg)} trips" if f.mode == "tg" else "") for f in feeds))
        for f in feeds:
            f.n = 0


if __name__ == "__main__":
    main()
