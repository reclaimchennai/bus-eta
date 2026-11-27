#!/usr/bin/env python3
"""eta-ml: live v4 arrival times for the bus maps.

One process for every configured city (the models are loaded once). For each city it tails the
collector's own archive (archive/obs-<IST date>.jsonl, one line per new GPS fix, in the order the
collector wrote them), keeps the fleet-wide live state exactly as training did (eta_engine.CityStream),
and every --tick seconds scores the buses that have a new fix: the next 30 stops, P50 and an 80%
range. It writes <live>/eta_ml.json, which the collector merges into vehicles.json (and so into the
cards and the GTFS-RT trip_updates):

  {"t": now, "v": "v4", "b": {reg: [code, fix, [k, dt, dlo, dhi, k, dt, dlo, dhi, ...]]}}
  dt / dlo / dhi = seconds after the fix: P50 arrival, 80% range low / high.

A sample of buses (--log-share) is logged with the nowcast alongside, for the live accuracy check
(<archive>/eta_ml-<IST date>.jsonl.gz).

A city with "engine": "v5" in the config is scored by the v5 models (--models-v5 DIR) on the v5 live state
(eta_engine_v5.V5Stream: last-pass segment times, leader travel time, bus state, today-vs-history ratios);
every other city stays on v4/v4r (--models). Each model set is loaded only if a city uses it, so a system
goes to v5, or back, by changing its "engine" and restarting. docs/V5_RESULTS.md has the memory of each mix.

Usage: eta_ml_service.py --config cities.json [--models DIR] [--use lgb,xgb,cat] [--models-v5 DIR] [--use-v5 lgb,xgb] [--tick 10]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_engine as E  # noqa: E402
import eta_engine_v5 as E5  # noqa: E402

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
UA = "reclaim-bus-maps-eta/1.0"
KEEP_S = int(os.environ.get("ETA_KEEP_S", 1200))   # a bus whose newest fix is older than this leaves the file
WARM_S = int(os.environ.get("ETA_WARM_S", 2700))   # replay this much of today's archive at start (trackers look back 30 min)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ── weather: the same hourly fields as training (Open-Meteo, + pol.reclaimchennai.city for Chennai) ──
COORDS = {"mtc": (13.0827, 80.2707), "delhi": (28.6139, 77.2090), "bmtc": (12.9716, 77.5946),
          "mumbai": (19.0760, 72.8777), "ncr": (28.4595, 77.0266)}
OM_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "weather_code", "cloud_cover",
           "wind_speed_10m", "wind_gusts_10m"]
POL = "https://pol.reclaimchennai.city/api"


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Weather:
    def __init__(self, cities):
        self.cities, self.by, self.lock = cities, {}, threading.Lock()
        threading.Thread(target=self.loop, daemon=True).start()

    def __call__(self, city, hour_ts):
        with self.lock:
            m = self.by.get(city) or {}
        # the newest complete hour at or before this one (a live fetch can lag an hour)
        for h in (hour_ts, hour_ts - 3600, hour_ts - 7200):
            if h in m:
                return m[h]
        return None

    def loop(self):
        while True:
            for c in self.cities:
                try:
                    self.fetch(c)
                except Exception as e:  # noqa: BLE001
                    log(f"[weather {c}] {e}")
            time.sleep(1800)

    def fetch(self, city):
        if city not in COORDS:
            return
        la, lo = COORDS[city]
        j = get_json(f"https://api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&past_days=2&forecast_days=1"
                     f"&hourly={','.join(OM_VARS)}&timezone=UTC")
        h = j.get("hourly") or {}
        rows = {}
        for n, ts in enumerate(h.get("time") or []):
            t = int(dt.datetime.fromisoformat(ts + "+00:00").timestamp())
            g = lambda v: (h.get(v) or [None] * (n + 1))[n]  # noqa: E731
            rows[t] = {"temp_c": g("temperature_2m"), "humidity": g("relative_humidity_2m"), "precip_mm": g("precipitation"),
                       "cloud_cover": g("cloud_cover"), "wind_kph": g("wind_speed_10m"), "gust_kph": g("wind_gusts_10m")}
        if city == "mtc":
            self.pol(rows)
        keys = sorted(rows)
        for i, t in enumerate(keys):
            p = [float(rows[keys[j]].get("precip_mm") or 0.0) for j in range(max(0, i - 23), i + 1)]
            r = rows[t]
            r["precip_mm"] = p[-1]
            r["precip_3h_mm"], r["precip_6h_mm"], r["precip_24h_mm"] = sum(p[-3:]), sum(p[-6:]), sum(p)
            r["is_raining"], r["is_heavy_rain"] = float(p[-1] > 0.2), float(p[-1] >= 5.0)
        now = time.time()
        rows = {t: r for t, r in rows.items() if t <= now}      # never a forecast hour
        with self.lock:
            self.by[city] = rows

    def pol(self, rows):
        """Chennai: city means over pol's stations per hour (training: weather_fetch.build_unified_weather)."""
        agg = {}
        for metric in ("precip_mm", "rain_accum_mm", "temp_c", "humidity", "vis_km", "pm25", "aqi"):
            try:
                j = get_json(f"{POL}/history?uid=all&metric={metric}&hours=30", timeout=60)
            except Exception as e:  # noqa: BLE001
                log(f"[weather pol {metric}] {e}")
                continue
            per = {}
            for s in j.get("series", []):
                for ts, v in s.get("points", []):
                    if v is None:
                        continue
                    try:
                        h = int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()) // 3600 * 3600
                    except ValueError:
                        continue
                    key = (s.get("uid"), h)
                    if metric in ("precip_mm", "rain_accum_mm"):
                        per[key] = max(per.get(key, -1e9), float(v))
                    else:
                        a = per.setdefault(key, [0.0, 0])
                        a[0] += float(v); a[1] += 1
            byh = {}
            for (uid, h), v in per.items():
                val = v if not isinstance(v, list) else v[0] / v[1]
                byh.setdefault(h, []).append(val)
            for h, vs in byh.items():
                agg.setdefault(h, {})[metric] = sum(vs) / len(vs)
        for h, a in agg.items():
            r = rows.get(h)
            if r is None:
                continue
            if "precip_mm" in a:
                r["pol_precip_mean"] = a["precip_mm"]
                r["precip_mm"] = max(float(r.get("precip_mm") or 0.0), a["precip_mm"])
            if "rain_accum_mm" in a:
                r["pol_rain_accum_mean"] = a["rain_accum_mm"]
            for m in ("vis_km", "pm25", "aqi"):
                if m in a:
                    r[m] = a[m]
            if "temp_c" in a:
                r["temp_c"] = a["temp_c"]
            if "humidity" in a:
                r["humidity"] = a["humidity"]


# ── archive tailing ──
def obs_path(archive, day):
    return os.path.join(archive, f"obs-{day}.jsonl")


def ist_day(t=None):
    return dt.datetime.fromtimestamp(t or time.time(), IST).date().isoformat()


class Tail:
    """Follows archive/obs-<IST day>.jsonl, moving to the next day's file once it appears."""

    def __init__(self, archive):
        self.archive, self.day, self.fh, self.buf = archive, ist_day(), None, b""

    def open(self, warm_from=None):
        p = obs_path(self.archive, self.day)
        if not os.path.exists(p):
            return False
        self.fh = open(p, "rb")
        if warm_from:
            self.seek_time(warm_from)
        else:
            self.fh.seek(0, 2)
        return True

    def seek_time(self, t0):
        """Binary search on byte offset for the first line with t >= t0 (lines are written in time order,
        give or take a fix's lag)."""
        size = os.fstat(self.fh.fileno()).st_size
        lo, hi = 0, size
        while hi - lo > 1 << 20:
            mid = (lo + hi) // 2
            self.fh.seek(mid); self.fh.readline()
            ln = self.fh.readline()
            try:
                t = json.loads(ln).get("t") or 0
            except ValueError:
                t = 0
            if t < t0:
                lo = mid
            else:
                hi = mid
        self.fh.seek(lo)
        if lo:
            self.fh.readline()

    def read(self):
        """New archive lines, parsed one at a time (a generator: the warm-up reads ~1 GB of Delhi's archive,
        which as one list of dicts took the service to its memory cap)."""
        if self.fh is None:
            if not self.open(warm_from=1):          # a new day's file: from its start
                return
        while True:
            chunk = self.fh.read(8 << 20)
            if not chunk:
                break
            lines = (self.buf + chunk).split(b"\n")
            self.buf = lines.pop()
            for ln in lines:
                try:
                    yield json.loads(ln)
                except ValueError:
                    pass
        today = ist_day()
        if today != self.day and os.path.exists(obs_path(self.archive, today)):
            self.fh.close()
            self.day, self.fh, self.buf = today, None, b""
            self.open(warm_from=1)


def trim():
    """Give freed heap back to the OS after a big burst (glibc keeps it otherwise)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Pipe:
    """Archive lines from a command (e.g. ssh raw tail -F ...) instead of a local file: a shadow run that
    touches nothing on the server."""

    def __init__(self, cmd, tee=None):
        self.q, self.lock = [], threading.Lock()
        self.tee = open(tee, "a") if tee else None
        self.p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1 << 20)
        threading.Thread(target=self.pump, daemon=True).start()

    def pump(self):
        for ln in self.p.stdout:
            with self.lock:
                self.q.append(ln)

    def open(self, warm_from=None):
        return True

    def read(self):
        with self.lock:
            q, self.q = self.q, []
        out = []
        for ln in q:
            if self.tee:
                self.tee.write(ln.decode("utf-8", "replace"))
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        if self.tee:
            self.tee.flush()
        return out


class City:
    def __init__(self, cfg, hol, weather, ens, tail=None):
        self.cfg, self.name = cfg, cfg["city"]
        self.engine, self.ens = engine_of(cfg), ens
        t0 = time.time()
        R = E.Routes(cfg["route_stops"], cfg["stop_dist"], cfg.get("osm"))
        stream = E5.V5Stream if self.engine == "v5" else E.CityStream
        self.S = stream(self.name, R, cfg.get("model_json"), hol, weather, cfg.get("max_gap"), cfg.get("city_idx"))
        self.S.cal = E.AgACI(cfg.get("cqr_v5" if self.engine == "v5" else "cqr") or ens.cqr)
        self.tail = tail or (Tail(cfg["archive"]) if not cfg.get("source_cmd") else Pipe(cfg["source_cmd"], cfg.get("tee")))
        self.out = cfg["out"]
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        self.vurl, self.vlast, self.shown_seen = cfg.get("vehicles_url"), 0.0, {}
        self.cur = {}                  # reg -> [code, fix, flat]
        self.last_scored = {}
        self.min_gap = cfg.get("min_gap", 0)
        self.log_share = cfg.get("log_share", 0.05)
        self.logf, self.logday = None, None
        self.stats = {"lines": 0, "scored": 0, "rows": 0}
        self.routes_mtime = os.path.getmtime(cfg["route_stops"])
        log(f"[{self.name}] {len(R.ll)} workings in {time.time()-t0:.0f}s")
        if isinstance(self.tail, Pipe):
            time.sleep(cfg.get("warm_wait", 60))       # let the pipe deliver its back-log first
        if self.tail.open(warm_from=time.time() - WARM_S):
            n = 0
            for r in self.tail.read():
                self.S.ingest(r); n += 1
            self.S.take_dirty()
            trim()
            log(f"[{self.name}] warmed on {n:,} archive lines")

    def logged(self, reg):
        return (zlib.crc32(reg.encode()) % 1000) < self.log_share * 1000

    def write_log(self, recs):
        day = ist_day()
        if day != self.logday:
            if self.logf:
                self.logf.close()
            self.logf = gzip.open(os.path.join(self.cfg["archive"], f"eta_ml-{day}.jsonl.gz"), "at")
            self.logday = day
        for r in recs:
            self.logf.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.logf.flush()

    def step(self, now):
        ens = self.ens
        for r in self.tail.read():
            self.S.ingest(r)
            self.stats["lines"] += 1
        dirty = self.S.take_dirty()
        todo = []
        for reg in dirty:
            if self.min_gap and now - self.last_scored.get(reg, 0) < self.min_gap:
                self.S.dirty[reg] = self.S.last[reg][0] if reg in self.S.last else now   # keep it queued
                continue
            todo.append(reg)
        if todo:
            res = E.score(self.S, ens, todo)
            recs = []
            for reg, (code, fix, lst, n_model) in res.items():
                flat = []
                for k, e, lo, hi, _nc in lst:
                    flat += [k, int(round(e - fix)), int(round(lo - fix)), int(round(hi - fix))]
                self.cur[reg] = [code, int(fix), flat]
                self.last_scored[reg] = now
                self.stats["scored"] += 1; self.stats["rows"] += n_model
                if self.logged(reg):
                    recs.append({"reg": reg, "r": code, "fix": int(fix), "at": int(now),
                                 "e": [[k, round(e - fix), round(lo - fix), round(hi - fix), round(nc - fix)] for k, e, lo, hi, nc in lst]})
            if recs:
                self.write_log(recs)
        if self.vurl and now - self.vlast >= 20:
            self.vlast = now
            self.log_shown(now)

    def log_shown(self, now):
        """Shadow check: what the map shows right now for the logged buses (operator ETAs or nowcast)."""
        try:
            j = get_json(self.vurl, timeout=20)
        except Exception as e:  # noqa: BLE001
            log(f"[{self.name}] vehicles: {e}")
            return
        recs = []
        for v in j.get("v") or []:
            reg = v[0]
            if not self.logged(reg) or self.shown_seen.get(reg) == v[5]:
                continue
            self.shown_seen[reg] = v[5]
            recs.append({"s": 1, "reg": reg, "r": v[1], "fix": v[5], "f": v[6], "at": int(now), "shown": v[7] if len(v) > 7 else []})
        if recs:
            self.write_log(recs)

    def write(self, now):
        keep = {r: v for r, v in self.cur.items() if now - v[1] < KEEP_S}
        self.cur = keep
        tmp = self.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"t": int(now), "v": self.engine, "b": keep}, fh, separators=(",", ":"))
        os.replace(tmp, self.out)

    def maintain(self, now):
        self.S.M.reload()
        self.S.gc(now)
        for reg in [r for r, t in self.last_scored.items() if now - t > 3 * 3600]:
            del self.last_scored[reg]
        try:
            m = os.path.getmtime(self.cfg["route_stops"])
        except OSError:
            return
        if m != self.routes_mtime:      # network republished: rebuild routes, keep the live state
            self.routes_mtime = m
            try:
                self.S.R = E.Routes(self.cfg["route_stops"], self.cfg["stop_dist"], self.cfg.get("osm"))
                log(f"[{self.name}] routes reloaded: {len(self.S.R.ll)}")
            except Exception as e:  # noqa: BLE001
                log(f"[{self.name}] routes reload failed: {e}")


def engine_of(cfg):
    e = cfg.get("engine", "v4")
    if e not in ("v4", "v5"):
        raise ValueError(f"{cfg.get('city')}: engine must be v4 or v5, not {e!r}")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    ap.add_argument("--use", default="lgb,xgb,cat")
    ap.add_argument("--models-v5", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models-v5"))
    ap.add_argument("--use-v5", default="", help="e.g. lgb: LightGBM only (least memory); default: the blend of v5_report.json")
    ap.add_argument("--tick", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--holidays", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "holidays_india_2025_2026.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    hol = json.load(open(a.holidays)) if os.path.exists(a.holidays) else {}
    names = [c["city"] for c in cfg["cities"] if c.get("enabled", True) is not False]
    for c in cfg["cities"]:
        if c.get("coords"):                       # systems beyond the trained five: where to ask Open-Meteo
            COORDS[c["city"]] = tuple(c["coords"])
    weather = Weather(names)
    on = [c for c in cfg["cities"] if c.get("enabled", True) is not False]
    ens = {}
    if any(engine_of(c) == "v4" for c in on):
        ens["v4"] = E.Ensemble(a.models, threads=a.threads, use=tuple(a.use.split(",")))
        log(f"v4 models {sorted(ens['v4'].m)} weights {ens['v4'].w}")
    if any(engine_of(c) == "v5" for c in on):
        ens["v5"] = E5.EnsembleV5(a.models_v5, threads=a.threads, use=tuple(a.use_v5.split(",")) if a.use_v5 else None)
        log(f"v5 models {[(n, round(w, 2)) for n, w, _ in ens['v5'].m]} for {[c['city'] for c in on if engine_of(c) == 'v5']}")
    trim()
    cities = []
    for c in cfg["cities"]:
        if c.get("enabled", True) is False:
            log(f"[{c['city']}] disabled in config")
            continue
        try:
            cities.append(City(c, hol, weather, ens[engine_of(c)]))
        except Exception as e:  # noqa: BLE001
            log(f"[{c['city']}] not started: {e}")
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    last_maint = last_stat = time.time()
    while not stop:
        t0 = time.time()
        for c in cities:
            try:
                c.step(t0)
                c.write(t0)
            except Exception:  # noqa: BLE001
                log(f"[{c.name}] {traceback.format_exc()}")
        if t0 - last_maint > 300:
            for c in cities:
                c.maintain(t0)
            trim()
            last_maint = t0
        if t0 - last_stat > 60:
            log(" | ".join(f"{c.name} {c.engine}: {c.stats['lines']} fixes, {c.stats['scored']} scored, {len(c.cur)} buses, "
                           f"80% band miss {sum(c.S.cal.miss)}/{sum(c.S.cal.n)} margins {[round(m, 3) for m in c.S.cal.margins()]}" for c in cities)
                + f" | {time.time()-t0:.1f}s")
            for c in cities:
                c.stats = {"lines": 0, "scored": 0, "rows": 0}
            last_stat = t0
        time.sleep(max(0.5, a.tick - (time.time() - t0)))


if __name__ == "__main__":
    main()
