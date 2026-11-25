#!/usr/bin/env python3
"""Adapter between the RTC-standard live maps (/tgsrtc, /apsrtc, /hr, /tricity, /up, /kerala, /amts, /osrtc,
/rjrtc, /gsrtc ...) and the v4 engine, without touching their collectors.

The engine wants (eta_engine / eta_ml_service):
  archive/obs-<IST day>.jsonl   one line per new GPS fix {t, reg, r, la, lo, f}, r = the route working
  route_stops.json              {r: {"d": [{"s": [[stop_id, name, lat, lon], ...]}]}}
  stop_dist.json                {r: [metres along the route at each stop]}

The RTC maps share one network format (live/net/services.json, stops.json, segs.json, built by each
system's network.py) and one feed format (live/vehicles.json {t, cols, buses}, rid = a services.json key).

  rtc_adapt.py net  NET_DIR OUT_DIR [--min-stops 3]
        services + stops + segs -> OUT_DIR/route_stops.json, stop_dist.json (distances along each
        stop-to-stop road shape in segs.json; crow-fly x 1.25 where a leg has none)
  rtc_adapt.py hist SYSTEM LIVE_CSV_DIR ARCHIVE [--from YYYYMMDD] [--to YYYYMMDD] [--net OUT_DIR]
        a collector's own raw/live/YYYYMMDD.csv.gz -> ARCHIVE/obs-<day>.jsonl (only systems whose rows
        carry the route working: see HIST)
  rtc_adapt.py live VEHICLES_JSON ARCHIVE [--every 15] [--shown]
        follows the public vehicles.json (or its file on disk) and appends each bus's new fixes to
        ARCHIVE/obs-<IST day>.jsonl. Works for every RTC map, on any host, read-only.

Bus class f: "ac" for AC types, "ev" for electric, else "n" (the trained vocabulary).
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.request

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
K = 111320.0
UA = "reclaim-bus-maps-eta/1.0"


def save_net(out_dir, rs, sd):
    """route_stops.json + stop_dist.json, each replaced atomically (eta-ml reloads them when they change), and
    route_names.json, each working's stop names and ids only, indexed ({"n": [names], "i": [stop ids], "w": {working:
    "name index,name index,...|id index,id index,..."}}): what the collectors' merge reads (TGSRTC's 21k workings
    in a few MB of memory)."""
    os.makedirs(out_dir, exist_ok=True)
    nix, iix, w = {}, {}, {}
    for c, v in rs.items():
        if v.get("d"):
            st = v["d"][0]["s"]
            w[c] = ",".join(str(nix.setdefault(p[1], len(nix))) for p in st) + "|" + ",".join(str(iix.setdefault(str(p[0]), len(iix))) for p in st)
    for c, v in rs.items():                       # an alias ({"a": chain code}) reads as its chain
        if "a" in v and v["a"] in w:
            w[c] = w[v["a"]]
    names = {"n": list(nix), "i": list(iix), "w": w}
    for f, obj in (("stop_dist.json", sd), ("route_stops.json", rs), ("route_names.json", names)):
        tmp = os.path.join(out_dir, f + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
        os.replace(tmp, os.path.join(out_dir, f))


def hav(la1, lo1, la2, lo2):
    p = math.pi / 180
    a = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 12742000 * math.asin(math.sqrt(min(1.0, a)))


def decode(poly):
    """segs.json shape: [lon0*1e5, lat0*1e5, dlon, dlat, ...] -> [(lat, lon), ...]."""
    if not poly or len(poly) < 2:
        return []
    lo, la = poly[0], poly[1]
    out = [(la / 1e5, lo / 1e5)]
    for j in range(2, len(poly) - 1, 2):
        lo += poly[j]; la += poly[j + 1]
        out.append((la / 1e5, lo / 1e5))
    return out


def length(pts):
    return sum(hav(a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))


def bus_class(typ):
    t = (typ or "").lower()
    if "electric" in t or t.startswith("e ") or t.startswith("e-") or t.startswith("ev") or "ev " in t:
        return "ev"
    if "non ac" in t or "non-ac" in t or "nonac" in t:
        return "n"
    if " ac" in f" {t}" or "a/c" in t or "volvo" in t or "garuda" in t or "amaravati" in t or "rajdhani" in t:
        return "ac"
    return "n"


# ── network ──
def cmd_net(a):
    sv = json.load(open(os.path.join(a.net_dir, "services.json")))
    stops = json.load(open(os.path.join(a.net_dir, "stops.json")))
    segs_j = json.load(open(os.path.join(a.net_dir, "segs.json")))
    seg_len = {}
    for s in segs_j.get("segs") or []:
        if len(s) >= 4 and s[3]:
            seg_len[(s[0], s[1])] = length(decode(s[3]))
    rs, sd, n_road, n_leg, skipped = {}, {}, 0, 0, 0
    id_list = sv.get("ids")                       # Kerala: stops given as indices into "ids"
    for sid, row in (sv.get("services") or {}).items():
        ids = row[6] if len(row) > 6 else []
        if id_list:
            ids = [id_list[x] if isinstance(x, int) and 0 <= x < len(id_list) else x for x in ids]
        ids = [x for x in ids if x in stops]
        if len(ids) < a.min_stops:
            skipped += 1
            continue
        d, acc = [0.0], 0.0
        for x, y in zip(ids, ids[1:]):
            n_leg += 1
            L = seg_len.get((x, y))
            crow = hav(stops[x][1], stops[x][2], stops[y][1], stops[y][2])
            if L and crow * 0.9 <= L <= max(crow * 4.0, crow + 3000):
                n_road += 1
            else:
                L = crow * 1.25
            acc += max(L, 1.0)
            d.append(round(acc))
        rs[sid] = {"sn": row[0], "d": [{"o": row[4] if len(row) > 4 else "", "e": row[5] if len(row) > 5 else "",
                                        "s": [[x, stops[x][0], stops[x][1], stops[x][2]] for x in ids]}]}
        sd[sid] = d
    save_net(a.out_dir, rs, sd)
    gaps = sorted(d[-1] / max(1, len(d) - 1) for d in sd.values())
    print(f"{len(rs)} workings ({skipped} with < {a.min_stops} stops left out); legs on a road shape {n_road}/{n_leg}; "
          f"median stop spacing {gaps[len(gaps)//2] if gaps else 0:.0f} m", flush=True)


# ── history: per-system raw/live CSV columns -> obs ──
# (id column, route-working column, time column, lat, lon, filter(row) -> bool, class column or None)
HIST = {
    "tricity": ("id", "routeId", "t", "lat", "lon", lambda r: r.get("status") == "Running", None),
    "hr": ("id", "rid", "t", "lat", "lon", lambda r: r.get("status") == "1", None),
    "amts": ("id", "routeId", "t", "lat", "lon", lambda r: True, "type"),
    "kerala": ("id", "svc", "t", "lat", "lon", lambda r: True, None),
    "osrtc": ("id", "routeId", "gpsTime", "lat", "lon", lambda r: True, None),
}


def ist_day(t):
    return dt.datetime.fromtimestamp(t, IST).date().isoformat()


def cmd_hist(a):
    idc, rc, tc, lac, loc, keep, cc = HIST[a.system]
    valid = None
    if a.net:
        valid = set(json.load(open(os.path.join(a.net, "stop_dist.json"))))
    os.makedirs(a.archive, exist_ok=True)
    files = sorted(f for f in os.listdir(a.live_dir) if f.endswith(".csv.gz"))
    files = [f for f in files if (not a.frm or f[:8] >= a.frm) and (not a.to or f[:8] <= a.to)]
    outs, last = {}, {}
    n_in = n_out = n_norte = 0
    for f in files:
        with gzip.open(os.path.join(a.live_dir, f), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                n_in += 1
                rid = r.get(rc) or ""
                if not rid or not keep(r):
                    continue
                if valid is not None and rid not in valid:
                    n_norte += 1
                    continue
                try:
                    t, la, lo = int(float(r[tc])), float(r[lac]), float(r[loc])
                except (TypeError, ValueError, KeyError):
                    continue
                reg = str(r[idc])
                if last.get(reg) == (t, rid):
                    continue                          # the same fix read again
                last[reg] = (t, rid)
                day = ist_day(t)
                if day not in outs:
                    outs[day] = open(os.path.join(a.archive, f"obs-{day}.jsonl"), "w")
                outs[day].write(json.dumps({"t": t, "reg": reg, "r": rid, "la": la, "lo": lo,
                                            "f": bus_class(r.get(cc)) if cc else "n"}, separators=(",", ":")) + "\n")
                n_out += 1
    for fh in outs.values():
        fh.close()
    print(f"{a.system}: {n_in:,} rows -> {n_out:,} fixes on a known working ({n_norte:,} on workings not in the net) "
          f"over {len(outs)} days", flush=True)


# ── live: vehicles.json -> obs ──
def read_vehicles(src):
    if src.startswith("http"):
        req = urllib.request.Request(src + ("&" if "?" in src else "?") + f"_={int(time.time())}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    with open(src) as fh:
        return json.load(fh)


def save_shown(a, j, etas_t_seen):
    """What the map shows: etas.json next to vehicles.json, kept as {t, b: {id: [fix t, rid, eta list]}} whenever it
    changes, for the shown-vs-v4 check (ARCHIVE/shown-<IST day>.jsonl.gz)."""
    src = a.vehicles.replace("vehicles.json", "etas.json")
    try:
        e = read_vehicles(src)
    except Exception as x:  # noqa: BLE001
        print(time.strftime("%H:%M:%S"), "etas read failed:", x, flush=True)
        return etas_t_seen
    if not e or e.get("t") == etas_t_seen:
        return etas_t_seen
    ix = {c: i for i, c in enumerate(j.get("cols") or [])}
    fix = {str(b[ix["id"]]): b[ix["t"]] for b in j.get("buses") or []}
    out = {"t": e.get("t"), "cols": e.get("cols"), "b": {}}
    for bid, row in (e.get("buses") or {}).items():
        if bid in fix and isinstance(row, list) and len(row) >= 5 and row[4]:
            out["b"][bid] = [fix[bid], row[0], row[4]]
    with gzip.open(os.path.join(a.archive, f"shown-{ist_day(time.time())}.jsonl.gz"), "at") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    return e.get("t")


def cmd_live(a):
    os.makedirs(a.archive, exist_ok=True)
    last, fh, day, etas_t = {}, None, None, None
    while True:
        try:
            j = read_vehicles(a.vehicles)
            if a.shown:
                etas_t = save_shown(a, j, etas_t)
            cols = j.get("cols") or []
            ix = {c: i for i, c in enumerate(cols)}
            if "rid" not in ix and not a.shown_only:
                raise SystemExit("vehicles.json has no rid column: this map's buses are not linked to a route working")
            if a.shown_only:                      # TGSRTC / APSRTC: fixes come from rtc_feed.py; keep only what the map shows
                time.sleep(a.every)
                continue
            lines = []
            for b in j.get("buses") or []:
                rid = b[ix["rid"]]
                if not rid:
                    continue
                reg, t = str(b[ix["id"]]), int(b[ix["t"]] or 0)
                if not t or last.get(reg) == (t, rid):
                    continue
                last[reg] = (t, rid)
                lines.append((t, {"t": t, "reg": reg, "r": rid, "la": b[ix["lat"]], "lo": b[ix["lon"]],
                                  "f": bus_class(b[ix["type"]] if "type" in ix else "")}))
            lines.sort(key=lambda x: x[0])
            today = ist_day(time.time())
            if today != day:
                if fh:
                    fh.close()
                fh, day = open(os.path.join(a.archive, f"obs-{today}.jsonl"), "a"), today
            for _t, r in lines:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            fh.flush()
            if len(last) > 50000:
                cut = time.time() - 6 * 3600
                last = {k: v for k, v in last.items() if v[0] >= cut}
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M:%S"), "read failed:", e, flush=True)
        time.sleep(a.every)


# ── TGSRTC: no route working in its feed; the trip logs (raw/trips, live/trip/<bus>.json) carry the station chain ──
import re as _re
import zlib as _zlib


def tg_code(rno, names):
    """A TGSRTC working = route number + its station sequence (serviceId is new for every trip)."""
    key = "|".join(_re.sub(r"[^a-z0-9]", "", (n or "").lower()) for n in names)
    return f"{rno}~{_zlib.crc32(key.encode()) & 0xffffff:06x}"


def proj_along(pl, cum, la, lo, start_i=0):
    """Distance along polyline pl (cumulative cum) of the point nearest (la, lo), searching from start_i on."""
    best, bi = None, start_i
    cx = math.cos(math.radians(la)) * K
    for i in range(start_i, len(pl) - 1):
        (ay, ax), (by, bx) = pl[i], pl[i + 1]
        vx, vy, px, py = (bx - ax) * cx, (by - ay) * K, (lo - ax) * cx, (la - ay) * K
        L2 = vx * vx + vy * vy
        t = max(0.0, min(1.0, (px * vx + py * vy) / L2)) if L2 else 0.0
        off = math.hypot(px - t * vx, py - t * vy)
        if best is None or off < best[1]:
            best, bi = (cum[i] + t * (cum[i + 1] - cum[i]), off), i
    return (best[0], best[1], bi) if best else (None, None, start_i)


def tg_records(idx_dir):
    for f in sorted(os.listdir(idx_dir)):
        if f.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(idx_dir, f), "rt") as fh:
                for ln in fh:
                    yield json.loads(ln)


def cmd_tg_net(a):
    best = {}                                     # code -> record with a polyline (newest day wins)
    for r in tg_records(a.idx_dir):
        pts = r.get("pts") or []
        if len(pts) < 2:
            continue
        code = tg_code(r.get("rno"), [p[1] for p in pts])
        o = best.get(code)                        # a record with a polyline beats one without; then the newest day
        if o is None or (bool(r.get("pl")), r["day"]) >= (bool(o.get("pl")), o["day"]):
            best[code] = r
    rs, sd, n_off = {}, {}, 0
    for code, r in best.items():
        pl = [tuple(x) for x in (r.get("pl") or [])]
        if len(pl) < 2:                           # no line kept for this working: crow-fly x 1.25 between stations
            pl = [(p[2], p[3]) for p in r["pts"]]
        cum = [0.0]
        for p1, p2 in zip(pl, pl[1:]):
            cum.append(cum[-1] + hav(p1[0], p1[1], p2[0], p2[1]))
        d, si, prev = [], 0, None
        for p in r["pts"]:
            s_, off, si = proj_along(pl, cum, p[2], p[3], si)
            if s_ is None or off > 1500 or (prev is not None and s_ <= d[-1]):
                n_off += 1
                s_ = (d[-1] if d else 0.0) + (hav(prev[2], prev[3], p[2], p[3]) * 1.25 if prev else 0.0)
            d.append(round(s_)); prev = p
        d = [x - d[0] for x in d]
        if d[-1] <= 0:
            continue
        rs[code] = {"sn": r.get("rno"), "type": r.get("type"),
                    "d": [{"o": r["pts"][0][1], "e": r["pts"][-1][1], "s": [[str(p[0]), p[1], p[2], p[3]] for p in r["pts"]]}]}
        sd[code] = d
    save_net(a.out_dir, rs, sd)
    print(f"{len(rs)} TGSRTC workings; {n_off} stations placed by crow-fly (off the trip's line)", flush=True)


def cmd_tg_hist(a):
    id2reg = {}
    for f in (a.vehicles, a.lastseen):
        j = json.load(open(f))
        ix = {c: i for i, c in enumerate(j["cols"])}
        for b in j["buses"]:
            id2reg[str(b[ix["id"]])] = b[ix["veh"]]
    valid = set(json.load(open(os.path.join(a.net, "stop_dist.json"))))
    types = {c: v.get("type") for c, v in json.load(open(os.path.join(a.net, "route_stops.json"))).items()}
    os.makedirs(a.archive, exist_ok=True)
    for f in sorted(os.listdir(a.live_dir)):
        day = f[:8]
        if not f.endswith(".csv.gz") or (a.frm and day < a.frm) or (a.to and day > a.to):
            continue
        spans = collections.defaultdict(list)     # reg -> [(start, end, code)]
        for r in tg_records(a.idx_dir):
            if r["day"] != day or len(r.get("pts") or []) < 2 or not r.get("reg"):
                continue
            code = tg_code(r.get("rno"), [p[1] for p in r["pts"]])
            if code not in valid:
                continue
            p0, pn = r["pts"][0], r["pts"][-1]
            st = p0[5] or p0[4] or r.get("ps")
            en = pn[4] or r.get("fin") or r.get("pe")
            if st and en and en > st:
                spans[r["reg"]].append((st - 120, en + 120, code))
        for v in spans.values():
            v.sort()
        out = collections.defaultdict(list)
        n_in = n_out = 0
        last = {}
        with gzip.open(os.path.join(a.live_dir, f), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                n_in += 1
                reg = id2reg.get(r["id"])
                if not reg or reg not in spans:
                    continue
                try:
                    t, la, lo = int(float(r["gpsTime"])), float(r["lat"]), float(r["lon"])
                except (TypeError, ValueError):
                    continue
                code = next((c for s0, s1, c in spans[reg] if s0 <= t <= s1), None)
                if not code or last.get(r["id"]) == t:
                    continue
                last[r["id"]] = t
                out[ist_day(t)].append((t, {"t": t, "reg": r["id"], "r": code, "la": la, "lo": lo, "f": bus_class(types.get(code))}))
                n_out += 1
        for d, lst in out.items():               # the collector writes in poll order; a bus's fix is new when first read
            with open(os.path.join(a.archive, f"obs-{d}.jsonl"), "a") as fh:
                for _t, rec in lst:
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        print(f"{day}: {n_in:,} rows -> {n_out:,} fixes on a trip ({sum(len(v) for v in spans.values()):,} trips)", flush=True)


# ── APSRTC: fixes carry serviceDocId (DDMMYYYY_<dcp>_<trip>_<depot>); live/trips.json gives each working's stop chain ──
def ap_key(doc):
    return doc.split("_", 1)[1] if doc and "_" in doc and doc[:8].isdigit() else doc


def ap_sources(paths):
    """(service key, [[name, lat, lon, ...], ...]) from trips.json-shaped files ({"trips": {key: chain}}, .gz or not)
    and from eta-rtc-feed's new_workings.jsonl ({"code", "stops"} per line)."""
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        try:
            if ".jsonl" in p:
                with op(p, "rt") as fh:
                    for ln in fh:
                        try:
                            w = json.loads(ln)
                        except ValueError:
                            continue
                        yield w.get("code"), w.get("stops") or []
            else:
                with op(p, "rt") as fh:
                    j = json.load(fh)
                yield from (j.get("trips") or j).items()
        except (OSError, ValueError, EOFError) as e:          # a missing or half-written trips.json: skipped
            print(f"ap-net: {p}: {e}", file=sys.stderr)


def ap_stop_id(name, la, lo):
    """A physical stop: its name at its place (11 m grid). Services passing the same stops share segment speeds."""
    return f"{name}@{la:.4f},{lo:.4f}"


def cmd_ap_net(a):
    """APSRTC network, cumulative. Every service key ever met (trips.json lists only the services running at that
    moment) is kept in OUT/ap_chains.json.gz, newest stop list winning. The ~76k service keys share ~15k distinct
    stop chains: each chain is written once (code c<crc32>), each service key as {"a": chain code}, an alias
    eta_engine.Routes and save_net's route_names.json resolve (travel_model does not: see `canon`)."""
    import zlib
    state = os.path.join(a.out_dir, "ap_chains.json.gz")
    chains = {}
    if not a.fresh and os.path.exists(state):
        with gzip.open(state, "rt") as fh:
            chains = json.load(fh)
    n0, n_chg = len(chains), 0
    for key, chain in ap_sources(a.src):
        if not key:
            continue
        pts = []
        for p in chain:
            try:
                if isinstance(p, list) and len(p) >= 3 and p[1] not in (None, "") and p[2] not in (None, ""):
                    pts.append([str(p[0]), round(float(p[1]), 5), round(float(p[2]), 5)])
            except (TypeError, ValueError):
                pass
        if len(pts) < a.min_stops:
            continue
        old = chains.get(key)
        if old != pts:
            n_chg += old is not None
            chains[key] = pts
    os.makedirs(a.out_dir, exist_ok=True)
    with gzip.open(state + ".tmp", "wt", compresslevel=6) as fh:
        json.dump(chains, fh, separators=(",", ":"))
    os.replace(state + ".tmp", state)
    canon, rs, sd = {}, {}, {}
    for key in sorted(chains):
        pts = chains[key]
        sig = json.dumps(pts, separators=(",", ":"))
        c = canon.get(sig)
        if c is None:
            c, n = "c%08x" % zlib.crc32(sig.encode()), 0
            while c in rs:                                     # crc collision: a suffix
                n += 1
                c = "c%08x_%d" % (zlib.crc32(sig.encode()), n)
            canon[sig] = c
            d = [0.0]
            for x, y in zip(pts, pts[1:]):
                d.append(d[-1] + max(1.0, hav(x[1], x[2], y[1], y[2]) * 1.25))
            rs[c] = {"d": [{"o": pts[0][0], "e": pts[-1][0], "s": [[ap_stop_id(*p), p[0], p[1], p[2]] for p in pts]}]}
            sd[c] = [round(x) for x in d]
        rs[key] = {"a": c}
    save_net(a.out_dir, rs, sd)
    gaps = sorted(v[-1] / max(1, len(v) - 1) for v in sd.values())
    print(f"APSRTC: {len(chains)} services ({len(chains) - n0} new, {n_chg} changed) on {len(sd)} stop chains; "
          f"median stop spacing {gaps[len(gaps) // 2]:.0f} m (crow-fly x 1.25)", flush=True)


def cmd_canon(a):
    """For travel_model, which reads only full route entries: the network's chains alone (OUT/route_stops.json,
    stop_dist.json) and, with --archive, the archive days it would read with each fix's working mapped to its chain
    (OUT/archive/obs-<day>.jsonl.gz; fixes on workings the network lacks dropped)."""
    rs = json.load(open(os.path.join(a.net_dir, "route_stops.json")))
    alias = {c: v["a"] for c, v in rs.items() if "a" in v}
    full = {c: v for c, v in rs.items() if "a" not in v}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "route_stops.json"), "w") as fh:
        json.dump(full, fh, separators=(",", ":"))
    shutil.copyfile(os.path.join(a.net_dir, "stop_dist.json"), os.path.join(a.out_dir, "stop_dist.json"))
    if not a.archive:
        return
    until = a.until or ist_day(time.time())
    days = sorted(f for f in os.listdir(a.archive) if f.startswith("obs-") and f[4:14] <= until
                  and (f.endswith(".jsonl") or f.endswith(".jsonl.gz")))
    days = days[-(a.days + 1):]
    os.makedirs(os.path.join(a.out_dir, "archive"), exist_ok=True)
    n_in = n_out = 0
    for f in days:
        op = gzip.open if f.endswith(".gz") else open
        dst = os.path.join(a.out_dir, "archive", f[:14] + ".jsonl.gz")
        with op(os.path.join(a.archive, f), "rt") as src, gzip.open(dst, "wt", compresslevel=1) as out:
            for ln in src:
                n_in += 1
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                c = alias.get(r.get("r"), r.get("r") if r.get("r") in full else None)
                if c is None:
                    continue
                r["r"] = c
                out.write(json.dumps(r, separators=(",", ":")) + "\n")
                n_out += 1
    print(f"canon: {len(full)} chains, {len(alias)} aliases; {len(days)} archive days, {n_in:,} fixes -> {n_out:,}", flush=True)


def cmd_ap_hist(a):
    valid = set(json.load(open(os.path.join(a.net, "route_stops.json"))))   # chains and their service-key aliases
    link = json.load(open(a.routelink)) if a.routelink else {}
    link = link.get("buses", link) if isinstance(link.get("buses"), dict) else link
    types = {}
    for doc, row in link.items():
        if isinstance(row, list) and len(row) >= 6:
            types[ap_key(doc)] = row[5]
    os.makedirs(a.archive, exist_ok=True)
    for f in sorted(os.listdir(a.live_dir)):
        day = f[:8]
        if not f.endswith(".csv.gz") or (a.frm and day < a.frm) or (a.to and day > a.to):
            continue
        out, last, n_in, n_out = collections.defaultdict(list), {}, 0, 0
        with gzip.open(os.path.join(a.live_dir, f), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                n_in += 1
                doc = r.get("serviceDocId") or ""
                code = ap_key(doc)
                if code not in valid:
                    continue
                try:
                    t, la, lo = int(float(r["locationTime"]) / 1000), float(r["latitude"]), float(r["longitude"])
                except (TypeError, ValueError):
                    continue
                if last.get(doc) == t:
                    continue
                last[doc] = t
                out[ist_day(t)].append({"t": t, "reg": doc, "r": code, "la": la, "lo": lo, "f": bus_class(types.get(code))})
                n_out += 1
        for d, lst in out.items():
            with open(os.path.join(a.archive, f"obs-{d}.jsonl"), "a") as fh:
                for rec in lst:
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        print(f"{day}: {n_in:,} rows -> {n_out:,} fixes on a known working", flush=True)


def cmd_tg_extend(a):
    """New TGSRTC workings seen live (rtc_feed's new_workings.jsonl) added to the network: stations as the trip file
    places them, distances crow-fly x 1.25 (no line is kept for them)."""
    rs = json.load(open(os.path.join(a.net_dir, "route_stops.json")))
    sd = json.load(open(os.path.join(a.net_dir, "stop_dist.json")))
    n = 0
    for ln in open(a.jsonl):
        try:
            w = json.loads(ln)
        except ValueError:
            continue
        pts = [p for p in w.get("stops") or [] if p[1] and p[2]]
        if w["code"] in rs or len(pts) < 2:
            continue
        d = [0.0]
        for x, y in zip(pts, pts[1:]):
            d.append(d[-1] + max(1.0, hav(x[1], x[2], y[1], y[2]) * 1.25))
        rs[w["code"]] = {"sn": w.get("rno"), "type": w.get("type"),
                         "d": [{"o": pts[0][0], "e": pts[-1][0], "s": [[f"{w['code']}:{i}", p[0], p[1], p[2]] for i, p in enumerate(pts)]}]}
        sd[w["code"]] = [round(x) for x in d]
        n += 1
    if n:
        save_net(a.net_dir, rs, sd)
    print(f"{n} new TGSRTC workings added ({len(rs)} in all)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("net"); p.add_argument("net_dir"); p.add_argument("out_dir"); p.add_argument("--min-stops", type=int, default=3)
    p = sub.add_parser("hist"); p.add_argument("system", choices=sorted(HIST)); p.add_argument("live_dir"); p.add_argument("archive")
    p.add_argument("--from", dest="frm"); p.add_argument("--to"); p.add_argument("--net")
    p = sub.add_parser("live"); p.add_argument("vehicles"); p.add_argument("archive"); p.add_argument("--every", type=float, default=15)
    p.add_argument("--shown", action="store_true", help="also keep what the map shows (etas.json) for the comparison")
    p.add_argument("--shown-only", action="store_true", help="record only etas.json (no fixes)")
    p = sub.add_parser("tg-net"); p.add_argument("idx_dir"); p.add_argument("out_dir")
    p = sub.add_parser("tg-hist"); p.add_argument("idx_dir"); p.add_argument("live_dir"); p.add_argument("archive")
    p.add_argument("--net", required=True); p.add_argument("--vehicles", required=True); p.add_argument("--lastseen", required=True)
    p.add_argument("--from", dest="frm"); p.add_argument("--to")
    p = sub.add_parser("ap-net"); p.add_argument("out_dir"); p.add_argument("src", nargs="+", help="trips.json(.gz) / new_workings.jsonl")
    p.add_argument("--min-stops", type=int, default=3); p.add_argument("--fresh", action="store_true", help="ignore OUT/ap_chains.json.gz")
    p = sub.add_parser("canon"); p.add_argument("net_dir"); p.add_argument("out_dir"); p.add_argument("--archive")
    p.add_argument("--days", type=int, default=7); p.add_argument("--until")
    p = sub.add_parser("ap-hist"); p.add_argument("live_dir"); p.add_argument("archive"); p.add_argument("--net", required=True)
    p.add_argument("--routelink"); p.add_argument("--from", dest="frm"); p.add_argument("--to")
    p = sub.add_parser("tg-extend"); p.add_argument("net_dir"); p.add_argument("jsonl")
    a = ap.parse_args()
    {"tg-extend": cmd_tg_extend, "ap-net": cmd_ap_net, "canon": cmd_canon, "ap-hist": cmd_ap_hist, "net": cmd_net, "hist": cmd_hist, "live": cmd_live, "tg-net": cmd_tg_net, "tg-hist": cmd_tg_hist}[a.cmd](a)


if __name__ == "__main__":
    main()
