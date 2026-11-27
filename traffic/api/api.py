#!/usr/bin/env python3
"""Traffic API: police alerts, observed bus speeds and official congestion spots, city by city.

Served at https://api.reclaim.city/traffic/v1/ (Caddy -> container traffic-api:5150, run.sh). Stdlib only.
Read-only: every answer is built from files the maps already publish (SOURCES below), read from raw's disk when
the file is there (LOCAL_ROOTS) and otherwise fetched from its public URL, then cached in memory (live files for
LIVE_TTL s, road shapes and stop lists for STATIC_TTL s). Nothing is stored and no request is logged here.

  python3 api.py                 # listens on :5150
  python3 api.py --port 8790     # local testing (fetches everything from the public URLs)

Endpoints (all GET, JSON, CORS open):  see OPENAPI below or /traffic/v1/openapi.json
"""
import argparse, gzip, json, math, os, sys, threading, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IST = timezone(timedelta(hours=5, minutes=30))
PREFIX = "/traffic/v1"
LIVE_TTL, STATIC_TTL = 60, 6 * 3600
MIN_HOURS = 6                  # hours of typical speeds a road needs before it gets a free-flow speed
UA = "reclaim-traffic-api/1 (+https://api.reclaim.city/docs)"
MAPS = "https://maps.reclaim.city/"
# On raw the maps' files are on disk (maps-site checkout mounted read-only); elsewhere everything comes over HTTPS.
LOCAL_ROOTS = {MAPS: os.environ.get("MAPS_SITE", "/maps-site/")}

# city -> what it has. speeds: the map's own observed-speed file and its road shapes ("geo" = seg_geo.json GeoJSON,
# "segs" = RTC delta-encoded segs.json), stops for naming a stretch. alerts: incidents feed (TGSRTC format) or Delhi's
# closures. hotspots: official congestion places.
CITIES = {
    "hyderabad": {
        "name": "Hyderabad", "state": "Telangana", "bbox": [78.05, 17.00, 78.80, 17.75],
        "alerts": {"url": MAPS + "tgsrtc/live/incidents.json", "fmt": "incidents",
                   "agencies": ["Hyderabad Traffic Police", "Cyberabad Traffic Police"],
                   "from": "@HYDTP and @CYBTRAFFIC on X, the Hyderabad Traffic Police e-challan site, the Cyberabad WhatsApp channel",
                   "history": MAPS + "traffic/history/hyderabad.json", "history_from": "Cyberabad and Hyderabad posts since 2020"},
        "speeds": {"url": MAPS + "tgsrtc/rt/speeds.json", "geo": MAPS + "tgsrtc/data/segs.json", "geo_fmt": "segs",
                   "stops": MAPS + "tgsrtc/data/stops.json", "stops_fmt": "rtc", "fleet": "TGSRTC buses (Telangana-wide)"},
        "hotspots": {"url": MAPS + "tgsrtc/traffic/htp_congestion.json", "fmt": "htp",
                     "from": "Hyderabad Traffic Police's list of major congested areas (echallan.tspolice.gov.in/htp/tcongestion.html)"},
    },
    "bengaluru": {
        "name": "Bengaluru", "state": "Karnataka", "bbox": [77.30, 12.70, 77.90, 13.30],
        "alerts": {"url": MAPS + "bengaluru/live/incidents.json", "fmt": "incidents", "agencies": ["Bengaluru Traffic Police"],
                   "from": "@blrcitytraffic and 56 station, ACP and DCP accounts on X",
                   "history": MAPS + "traffic/history/bengaluru.json", "history_from": "posts since 26 Oct 2026"},
        "speeds": {"url": MAPS + "bengaluru/live/speeds.json", "geo": MAPS + "bengaluru/data/seg_geo.json", "geo_fmt": "geo",
                   "stops": MAPS + "bengaluru/data/bus_stops.geojson", "stops_fmt": "geojson", "fleet": "BMTC buses"},
        "hotspots": {"url": MAPS + "bengaluru/live/btp_congestion.json", "fmt": "btp",
                     "from": "Bengaluru Traffic Police congestion dashboard (btp.karnataka.gov.in/188/congestion-map): ASTraM's 15-minute checks"},
    },
    "chennai": {
        "name": "Chennai", "state": "Tamil Nadu", "bbox": [79.95, 12.75, 80.35, 13.35],
        "alerts": {"url": MAPS + "mapsv2/live/incidents.json", "fmt": "incidents", "agencies": ["Greater Chennai Traffic Police"],
                   "from": "@ChennaiTraffic on X",
                   "history": MAPS + "traffic/history/chennai.json", "history_from": "road posts since Jul 2026"},
        "speeds": {"url": MAPS + "mapsv2/live/speeds.json", "geo": MAPS + "mapsv2/data/seg_geo.json", "geo_fmt": "geo",
                   "stops": MAPS + "mapsv2/data/bus_stops.geojson", "stops_fmt": "geojson", "fleet": "MTC buses"},
    },
    "delhi": {
        "name": "Delhi", "state": "Delhi", "bbox": [76.80, 28.35, 77.45, 28.90],
        "alerts": {"url": MAPS + "delhi/route-dodging/data/closures.json", "fmt": "closures", "agencies": ["Delhi Traffic Police"],
                   "from": "traffic.delhipolice.gov.in advisories and @dtptraffic on X",
                   "history_from": "the advisories still in Delhi's closures file (about the last few weeks)"},
        "speeds": {"url": MAPS + "delhi/live/speeds.json", "geo": MAPS + "delhi/data/seg_geo.json", "geo_fmt": "geo",
                   "stops": MAPS + "delhi/data/bus_stops.geojson", "stops_fmt": "geojson", "fleet": "DTC and cluster buses"},
    },
    "mumbai": {
        "name": "Mumbai region", "state": "Maharashtra", "bbox": [72.75, 18.85, 73.25, 19.50],
        "speeds": {"url": MAPS + "mumbai/live/speeds.json", "geo": MAPS + "mumbai/data/seg_geo.json", "geo_fmt": "geo",
                   "stops": MAPS + "mumbai/data/bus_stops.geojson", "stops_fmt": "geojson", "fleet": "BEST, TMT, KDMT and VVMT buses"},
    },
    "andhra-pradesh": {
        "name": "Andhra Pradesh (APSRTC)", "state": "Andhra Pradesh", "bbox": [76.75, 12.60, 84.80, 19.95],
        "speeds": {"url": MAPS + "apsrtc/rt/speeds.json", "geo": MAPS + "apsrtc/data/segs.json", "geo_fmt": "segs",
                   "stops": MAPS + "apsrtc/data/stops.json", "stops_fmt": "rtc", "fleet": "APSRTC buses (state-wide)"},
    },
    "chandigarh": {
        "name": "Chandigarh tricity", "state": "Chandigarh, Punjab, Haryana", "bbox": [76.60, 30.60, 76.95, 30.85],
        "alerts": {"url": MAPS + "tricity/live/incidents.json", "fmt": "incidents", "agencies": ["Chandigarh Traffic Police"],
                   "from": "@trafficchd on X (read twice a day: it posts advisories a few times a week)",
                   "history": MAPS + "traffic/history/chandigarh.json", "history_from": "advisories since 10 Nov 2026"},
        "speeds": {"url": MAPS + "tricity/rt/speeds.json", "geo": MAPS + "tricity/data/segs.json", "geo_fmt": "segs",
                   "stops": MAPS + "tricity/data/stops.json", "stops_fmt": "rtc", "fleet": "CTU buses"},
    },
    "ahmedabad": {
        "name": "Ahmedabad", "state": "Gujarat", "bbox": [72.40, 22.90, 72.75, 23.20],
        "speeds": {"url": MAPS + "amts/rt/speeds.json", "geo": MAPS + "amts/data/segs.json", "geo_fmt": "segs",
                   "stops": MAPS + "amts/data/stops.json", "stops_fmt": "rtc", "fleet": "AMTS and Janmarg BRTS buses"},
    },
}

# ---------------------------------------------------------------- fetching and caching
_cache, _locks, _glock = {}, {}, threading.Lock()


def _read(url):
    for pre, root in LOCAL_ROOTS.items():
        if url.startswith(pre):
            p = os.path.join(root, url[len(pre):])
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return f.read(), os.path.getmtime(p)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        b = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            b = gzip.decompress(b)
    return b, time.time()


def get(url, ttl, parse=None):
    """The parsed file (json, then parse(obj) once per refresh); a failed refresh keeps the last good copy."""
    with _glock:
        lk = _locks.setdefault(url, threading.Lock())
    with lk:
        c = _cache.get(url)
        if c and time.time() - c["at"] < ttl:
            return c["val"], c
        try:
            b, mt = _read(url)
            obj = json.loads(b)
            val = parse(obj) if parse else obj
            c = {"val": val, "at": time.time(), "mtime": mt, "ok": True, "url": url}
        except Exception as e:                                  # keep serving the last good copy
            if c:
                c = {**c, "at": time.time() - ttl + 30, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
            else:
                c = {"val": None, "at": time.time() - ttl + 30, "mtime": None, "ok": False, "url": url,
                     "error": f"{type(e).__name__}: {e}"[:200]}
        _cache[url] = c
        return c["val"], c


# ---------------------------------------------------------------- geometry helpers
def hav(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(a))


def line_len(c):
    return sum(hav(c[i - 1][1], c[i - 1][0], c[i][1], c[i][0]) for i in range(1, len(c)))


def coords_of(g):
    t = g.get("type")
    if t == "Point":
        return [g["coordinates"]]
    if t == "LineString":
        return g["coordinates"]
    if t in ("MultiLineString", "Polygon"):
        return [p for part in g["coordinates"] for p in part]
    if t == "MultiPolygon":
        return [p for poly in g["coordinates"] for part in poly for p in part]
    if t == "GeometryCollection":
        return [p for gg in g.get("geometries", []) for p in coords_of(gg)]
    return []


def in_bbox(cs, bb):
    return any(bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3] for x, y in cs)


def near(cs, lat, lon, r):
    """Distance in m from the point to the nearest vertex (shapes are dense enough at road scale), or None."""
    best = None
    for x, y in cs:
        if abs(y - lat) > r / 111000 + 0.01:
            continue
        d = hav(lat, lon, y, x)
        best = d if best is None or d < best else best
    return best if best is not None and best <= r else None


# ---------------------------------------------------------------- stops (to name a stretch)
class StopGrid:
    def __init__(self, pts):                    # pts: [(name, lat, lon)]
        self.g = {}
        for n, la, lo in pts:
            self.g.setdefault((int(la * 200), int(lo * 200)), []).append((n, la, lo))

    def nearest(self, lat, lon, r=80):
        best, bd = None, r
        gy, gx = int(lat * 200), int(lon * 200)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for n, la, lo in self.g.get((gy + dy, gx + dx), ()):
                    d = hav(lat, lon, la, lo)
                    if d <= bd:
                        best, bd = n, d
        return best


def parse_stops(fmt):
    def p(obj):
        if fmt == "rtc":
            return {"ids": {k: v[0] for k, v in obj.items()}, "grid": StopGrid([(v[0], v[1], v[2]) for v in obj.values()])}
        pts = []
        for f in obj.get("features", []):
            x, y = f["geometry"]["coordinates"][:2]
            pts.append((f["properties"].get("Stop Name") or f["properties"].get("name") or "", y, x))
        return {"ids": {}, "grid": StopGrid(pts)}
    return p


def parse_geo(fmt):
    def p(obj):
        out = {}
        if fmt == "segs":
            for s in obj.get("segs", []):
                if not s[3]:
                    continue
                d = s[3]; x, y = d[0], d[1]; c = [[x / 1e5, y / 1e5]]
                for i in range(2, len(d) - 1, 2):
                    x += d[i]; y += d[i + 1]; c.append([x / 1e5, y / 1e5])
                out[f"{s[0]}|{s[1]}"] = c
        else:
            for f in obj.get("features", []):
                out[f["properties"]["k"]] = f["geometry"]["coordinates"]
        return out
    return p


# ---------------------------------------------------------------- speeds
def speeds(city):
    S = CITIES[city].get("speeds")
    if not S:
        return None
    sp, c_sp = get(S["url"], LIVE_TTL)
    geo, c_geo = get(S["geo"], STATIC_TTL, parse_geo(S["geo_fmt"]))
    stops, _ = get(S["stops"], STATIC_TTL, parse_stops(S["stops_fmt"]))
    return sp, geo or {}, stops, c_sp, c_geo


def clean(v):
    return None if v is None or v < 0 else v


_rows = {}


def segment_rows(city):
    """Every stretch with geometry: key, coords, now, buses, hour profile, free-flow, names, length. Built once per
    refresh of the speeds file (names and lengths once per refresh of the road shapes)."""
    r = speeds(city)
    if not r or not r[0]:
        return [], None
    sp, geo, stops, c_sp, c_geo = r
    memo = _rows.get(city)
    if memo and memo["sp"] is sp:
        return memo["rows"], sp
    names = memo["names"] if memo and memo["geo"] is geo and memo["stops"] is stops else {}
    rows = []
    for k, v in sp.get("s", {}).items():
        c = geo.get(k)
        if not c or len(c) < 2:
            continue
        prof = [clean(x) for x in v[2:26]]
        known = [x for x in prof if x is not None]
        # free flow: 90th percentile of the hourly medians, only when at least MIN_HOURS hours are known
        free = round(sorted(known)[int(0.9 * (len(known) - 1))], 1) if len(known) >= MIN_HOURS else None
        if k not in names:
            a, b = (k.split("|") + [""])[:2]
            na = (stops["ids"].get(a) or stops["grid"].nearest(c[0][1], c[0][0])) if stops else None
            nb = (stops["ids"].get(b) or stops["grid"].nearest(c[-1][1], c[-1][0])) if stops else None
            names[k] = (na, nb, round(line_len(c)))
        na, nb, ln = names[k]
        rows.append({"k": k, "c": c, "now": clean(v[0]), "buses": v[1] if len(v) > 1 else None, "h": prof,
                     "free": free, "from": na, "to": nb, "len": ln})
    _rows[city] = {"sp": sp, "geo": geo, "stops": stops, "names": names, "rows": rows}
    return rows, sp


def seg_props(city, r, hour, when="now"):
    typ = r["h"][hour] if hour is not None else None
    cur = r["now"]
    ref = cur if when == "now" else typ
    return {"id": f"{city}:{r['k']}", "from_stop": r["from"], "to_stop": r["to"], "length_m": r["len"],
            "speed_kmph_now": cur, "buses_last_hour": r["buses"],
            "speed_kmph_typical": typ, "typical_hour": hour, "free_flow_kmph": r["free"],
            "congestion": (round(max(0.0, 1 - ref / r["free"]), 2) if ref is not None and r["free"] else None),
            "congestion_basis": when}


# ---------------------------------------------------------------- alerts
def norm_incident(f, city, agencies):
    p = f.get("properties") or {}
    return {"type": "Feature", "geometry": f.get("geometry"), "properties": {
        "id": p.get("id"), "city": city, "agency": p.get("agency") or (agencies[0] if agencies else None),
        "type": p.get("type"), "kind": p.get("kind"), "severity": p.get("severity"), "status": p.get("status"),
        "place": p.get("place"), "direction": p.get("direction"), "text": p.get("text_en"),
        "start": p.get("start"), "end": p.get("end"), "source_url": p.get("src"), "source": p.get("source"),
        "route_ids": p.get("route_ids") or [], "stop_ids": p.get("stop_ids") or [], "tags": p.get("tags") or [],
        "vehicles": p.get("vehicles") or []}}


def norm_closure(it, city, agencies):
    return {"type": "Feature", "geometry": it.get("geom"), "properties": {
        "id": f"dtp:{it.get('id')}", "city": city, "agency": agencies[0], "type": it.get("kind") or "advisory",
        "kind": "scheduled" if it.get("when") not in (None, "default") else "unplanned", "severity": None,
        "status": None, "place": it.get("title"), "direction": None, "text": it.get("text"),
        "start": it.get("from"), "end": it.get("to"), "source_url": it.get("link"), "source": "delhi_traffic_police",
        "route_ids": [], "stop_ids": [], "tags": it.get("names") or [], "vehicles": [], "confidence": it.get("conf")}}


def alerts(city, history=False):
    A = CITIES[city].get("alerts")
    if not A:
        return [], None
    if history and A.get("history"):
        d, c = get(A["history"], 900)
    else:
        d, c = get(A["url"], LIVE_TTL)
    if not d:
        return [], c
    if A["fmt"] == "closures":
        feats = [norm_closure(it, city, A["agencies"]) for it in d.get("items", []) if it.get("geom")]
    else:
        # one alert can be a point plus its road line: merge them into one feature with a GeometryCollection
        by = {}
        for f in d.get("features", []):
            pid = str((f.get("properties") or {}).get("id", "")).replace(":line", "")
            n = norm_incident(f, city, A["agencies"]); n["properties"]["id"] = pid
            if pid in by:
                g0 = by[pid]["geometry"]
                gs = g0["geometries"] if g0["type"] == "GeometryCollection" else [g0]
                gs = gs + [n["geometry"]]
                gs.sort(key=lambda g: g["type"] != "Point")              # the point first, then its road lines
                if f["geometry"]["type"] == "Point":
                    by[pid]["properties"] = n["properties"]
                by[pid]["geometry"] = {"type": "GeometryCollection", "geometries": gs}
            else:
                by[pid] = n
        feats = list(by.values())
    return feats, c


# ---------------------------------------------------------------- official hotspots
def hotspots(city, how):
    H = CITIES[city].get("hotspots")
    if not H:
        return [], None, {}
    d, c = get(H["url"], STATIC_TTL)
    if not d:
        return [], c, {}
    out, meta = [], {}
    if H["fmt"] == "htp":
        for i, s in enumerate(d.get("spots", [])):
            g = ({"type": "Point", "coordinates": s["p"]} if "p" in s else {"type": "LineString", "coordinates": s["c"]} if "c" in s
                 else {"type": "MultiLineString", "coordinates": s["m"]})
            out.append({"type": "Feature", "geometry": g, "properties": {
                "id": f"htp:{i}", "city": city, "name": s.get("n"), "agency": d.get("by"), "source_url": d.get("src"),
                "kind": "listed", "note": "listed by the police as a major congested area; no hourly data"}})
        meta = {"source": H["from"]}
    elif H["fmt"] == "btp":
        for i, c5 in enumerate(d.get("cells", [])):
            lat, lon, name, ps, cor, n, ext, sev, rec, p = c5
            peak = max(range(168), key=lambda h: p[h])
            out.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {
                "id": f"btp:{i}", "city": city, "name": name or None, "agency": "Bengaluru Traffic Police",
                "police_station": ps or None, "corridor": cor or None, "kind": "measured",
                "jam_share_pct": p[how] if how is not None else None, "hour_of_week": how,
                "peak_jam_share_pct": p[peak], "peak_hour_of_week": peak, "typical_queue_m": ext,
                "severe_or_high_pct": sev, "recurring_pct": rec, "alerts": n}})
        meta = {"source": H["from"], "period": d.get("period"), "snapshots": d.get("snapshots"),
                "model_refreshed_utc": d.get("model_refreshed_utc"),
                "hour_of_week": "0 = Monday 00:00-00:59 IST ... 167 = Sunday 23:00-23:59",
                "jam_share_pct": "share of BTP's 15-minute checks in that hour of the week that found congestion at this place"}
    return out, c, meta


# ---------------------------------------------------------------- request handling
class Bad(Exception):
    pass


def num(q, k, lo=None, hi=None, default=None, cast=float):
    v = q.get(k)
    if v in (None, ""):
        return default
    try:
        x = cast(v)
    except ValueError:
        raise Bad(f"{k} must be a number")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise Bad(f"{k} must be between {lo} and {hi}")
    return x


def bbox_of(q):
    if not q.get("bbox"):
        return None
    try:
        b = [float(x) for x in q["bbox"].split(",")]
        assert len(b) == 4 and b[0] < b[2] and b[1] < b[3]
        return b
    except (ValueError, AssertionError):
        raise Bad("bbox is min_lon,min_lat,max_lon,max_lat")


def now_ist():
    return datetime.now(IST)


def how_of(q):
    """hour of week from day (mon..sun or 0-6) and hour, default now in IST."""
    t = now_ist()
    d = q.get("day")
    if d not in (None, ""):
        days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        d = days.index(d[:3].lower()) if d[:3].lower() in days else int(num(q, "day", 0, 6, cast=int))
    else:
        d = t.weekday()
    h = num(q, "hour", 0, 23, t.hour, int)
    return d * 24 + h


def freshness(c, field_t=None):
    if not c:
        return None
    return {"source_updated": int(field_t or c.get("mtime") or 0) or None, "fetched": int(c["at"]), "ok": c["ok"],
            **({"error": c["error"]} if not c["ok"] else {})}


def city_or_404(city):
    if city not in CITIES:
        raise LookupError(f"unknown city '{city}'; see {PREFIX}/cities")
    return CITIES[city]


def city_of_point(lat, lon):
    for k, v in CITIES.items():
        b = v["bbox"]
        if b[0] <= lon <= b[2] and b[1] <= lat <= b[3] and k != "andhra-pradesh":
            return k
    b = CITIES["andhra-pradesh"]["bbox"]
    return "andhra-pradesh" if b[0] <= lon <= b[2] and b[1] <= lat <= b[3] else None


def ep_cities(q):
    out = []
    for k, v in CITIES.items():
        out.append({"id": k, "name": v["name"], "state": v["state"], "bbox": v["bbox"],
                    "alerts": bool(v.get("alerts")), "speeds": bool(v.get("speeds")), "hotspots": bool(v.get("hotspots")),
                    "links": {"summary": f"{PREFIX}/{k}"}})
    return {"cities": out}


def ep_city(city, q):
    C = city_or_404(city)
    out = {"id": city, "name": C["name"], "state": C["state"], "bbox": C["bbox"], "datasets": {}}
    if C.get("alerts"):
        a, c = alerts(city)
        t = int(time.time())
        act = [f for f in a if (f["properties"]["end"] or t + 1) > t and (f["properties"]["start"] or 0) <= t + 60]
        out["datasets"]["alerts"] = {"from": C["alerts"]["from"], "agencies": C["alerts"]["agencies"], "active": len(act),
                                     "in_feed": len(a), "freshness": freshness(c), "url": f"{PREFIX}/{city}/alerts",
                                     "history": f"{PREFIX}/{city}/alerts/history", "history_from": C["alerts"].get("history_from")}
    if C.get("speeds"):
        r = speeds(city)
        if r and r[0]:
            sp, geo, _, c_sp, _ = r
            out["datasets"]["speeds"] = {"from": f"GPS of {C['speeds']['fleet']}, our own tracking", "window_days": sp.get("days"),
                                         "now_window_min": sp.get("now_min"), "segments": len(sp.get("s", {})),
                                         "segments_with_shape": sum(1 for k in sp.get("s", {}) if k in geo),
                                         "city_median_kmph_now": sp.get("city", {}).get("now"),
                                         "city_median_kmph_by_hour": [clean(x) for x in sp.get("city", {}).get("h", [])],
                                         "freshness": freshness(c_sp, sp.get("t")), "url": f"{PREFIX}/{city}/speeds"}
    if C.get("hotspots"):
        h, c, meta = hotspots(city, how_of({}))
        out["datasets"]["hotspots"] = {"from": C["hotspots"]["from"], "count": len(h), "freshness": freshness(c),
                                       **({k: meta[k] for k in ("period",) if k in meta}), "url": f"{PREFIX}/{city}/hotspots"}
    return out


def ep_alerts(city, q):
    C = city_or_404(city)
    if not C.get("alerts"):
        return {"type": "FeatureCollection", "city": city, "features": [], "note": "no police alert source for this city yet"}
    feats, c = alerts(city)
    t = int(time.time())
    state = q.get("state", "active")
    if state not in ("active", "upcoming", "all"):
        raise Bad("state is active, upcoming or all")
    if state == "active":
        feats = [f for f in feats if (f["properties"]["end"] or t + 1) > t and (f["properties"]["start"] or 0) <= t + 60]
    elif state == "upcoming":
        feats = [f for f in feats if (f["properties"]["start"] or 0) > t + 60]
    if q.get("type"):
        ts = set(q["type"].split(","))
        feats = [f for f in feats if f["properties"]["type"] in ts]
    if q.get("route"):
        feats = [f for f in feats if q["route"] in f["properties"]["route_ids"]]
    bb = bbox_of(q)
    if bb:
        feats = [f for f in feats if f["geometry"] and in_bbox(coords_of(f["geometry"]), bb)]
    lat, lon = num(q, "lat", -90, 90), num(q, "lon", -180, 180)
    if (lat is None) != (lon is None):
        raise Bad("give both lat and lon")
    if lat is not None:
        r = num(q, "radius", 10, 20000, 1000)
        keep = []
        for f in feats:
            d = near(coords_of(f["geometry"]), lat, lon, r) if f["geometry"] else None
            if d is not None:
                f = {**f, "properties": {**f["properties"], "distance_m": round(d)}}
                keep.append(f)
        feats = sorted(keep, key=lambda f: f["properties"]["distance_m"])
    if lat is None:
        feats.sort(key=lambda f: -(f["properties"]["start"] or 0))
    lim = num(q, "limit", 1, 1000, 500, int)
    return {"type": "FeatureCollection", "city": city, "state": state, "count": len(feats[:lim]), "freshness": freshness(c),
            "features": feats[:lim]}


def when_arg(q, k):
    v = q.get(k)
    if v in (None, ""):
        return None
    try:
        return int(v) if v.isdigit() else int(datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=IST).timestamp())
    except ValueError:
        raise Bad(f"{k} is a unix time or YYYY-MM-DD (IST)")


def ep_history(city, q):
    C = city_or_404(city)
    if not C.get("alerts"):
        raise LookupError(f"no police alert source for {city}")
    feats, c = alerts(city, history=True)
    since, until = when_arg(q, "since"), when_arg(q, "until")
    if until is not None and q.get("until", "").count("-") == 2:
        until += 86400                                    # a date means the whole day
    st = lambda f: f["properties"]["start"] or 0
    if since is not None:
        feats = [f for f in feats if st(f) >= since]
    if until is not None:
        feats = [f for f in feats if st(f) < until]
    if q.get("type"):
        ts = set(q["type"].split(","))
        feats = [f for f in feats if f["properties"]["type"] in ts]
    if q.get("agency"):
        feats = [f for f in feats if q["agency"].lower() in (f["properties"]["agency"] or "").lower()]
    if q.get("route"):
        feats = [f for f in feats if q["route"] in f["properties"]["route_ids"]]
    bb = bbox_of(q)
    if bb:
        feats = [f for f in feats if f["geometry"] and in_bbox(coords_of(f["geometry"]), bb)]
    lat, lon = num(q, "lat", -90, 90), num(q, "lon", -180, 180)
    if (lat is None) != (lon is None):
        raise Bad("give both lat and lon")
    if lat is not None:
        r = num(q, "radius", 10, 20000, 1000)
        feats = [f for f in feats if f["geometry"] and near(coords_of(f["geometry"]), lat, lon, r) is not None]
    feats.sort(key=lambda f: -st(f))
    total = len(feats)
    off, lim = num(q, "offset", 0, 10 ** 7, 0, int), num(q, "limit", 1, 5000, 500, int)
    page = feats[off:off + lim]
    return {"type": "FeatureCollection", "city": city, "from": C["alerts"].get("history_from"), "total": total,
            "offset": off, "count": len(page), "next_offset": off + lim if off + lim < total else None,
            "freshness": freshness(c), "features": page}


def ep_speeds(city, q):
    C = city_or_404(city)
    if not C.get("speeds"):
        raise LookupError(f"no bus speeds for {city}")
    hour = num(q, "hour", 0, 23, now_ist().hour, int)
    rows, sp = segment_rows(city)
    if sp is None:
        raise RuntimeError("speeds file not available right now")
    bb = bbox_of(q)
    if bb:
        rows = [r for r in rows if in_bbox(r["c"], bb)]
    lat, lon = num(q, "lat", -90, 90), num(q, "lon", -180, 180)
    if lat is not None and lon is not None:
        rad = num(q, "radius", 10, 5000, 500)
        rows = [dict(r, dist=near(r["c"], lat, lon, rad)) for r in rows]
        rows = [r for r in rows if r["dist"] is not None]
    when = q.get("when", "now")
    if when not in ("now", "typical"):
        raise Bad("when is now or typical")
    key = (lambda r: r["now"]) if when == "now" else (lambda r: r["h"][hour])
    rows = [r for r in rows if key(r) is not None]
    mn, mx = num(q, "min_kmph", 0, 200), num(q, "max_kmph", 0, 200)
    if mn is not None:
        rows = [r for r in rows if key(r) >= mn]
    if mx is not None:
        rows = [r for r in rows if key(r) <= mx]
    minb = num(q, "min_buses", 0, 10000, None, int)
    if minb is not None:
        rows = [r for r in rows if (r["buses"] or 0) >= minb]
    order = q.get("sort", "")
    if order == "slowest":
        rows.sort(key=key)
    elif order == "congested":
        rows.sort(key=lambda r: -(1 - key(r) / r["free"]) if r["free"] else 0)
    lim = num(q, "limit", 1, 20000, 5000, int)
    rows = rows[:lim]
    feats = []
    for r in rows:
        pr = seg_props(city, r, hour, when)
        if "dist" in r:
            pr["distance_m"] = round(r["dist"])
        if q.get("profile") in ("1", "true"):
            pr["typical_kmph_by_hour"] = r["h"]
        feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": r["c"]}, "properties": pr})
    _, _, _, c_sp, _ = speeds(city)
    return {"type": "FeatureCollection", "city": city, "when": when, "hour": hour, "window_days": sp.get("days"),
            "now_window_min": sp.get("now_min"), "count": len(feats), "freshness": freshness(c_sp, sp.get("t")),
            "notes": "speed = median of buses over the stop-to-stop road, stops and signals included; now = last "
                     f"{sp.get('now_min')} min; typical = median at that hour over the last {sp.get('days')} days; "
                     "free_flow = 90th percentile of the hourly medians; congestion = 1 - speed / free_flow",
            "features": feats}


def ep_city_speed(city, q):
    C = city_or_404(city)
    if not C.get("speeds"):
        raise LookupError(f"no bus speeds for {city}")
    r = speeds(city)
    if not r or not r[0]:
        raise RuntimeError("speeds file not available right now")
    sp, _, _, c_sp, _ = r
    h = [clean(x) for x in sp.get("city", {}).get("h", [])]
    valid = [(v, i) for i, v in enumerate(h) if v is not None]
    return {"city": city, "fleet": C["speeds"]["fleet"], "median_kmph_now": sp.get("city", {}).get("now"),
            "now_window_min": sp.get("now_min"), "median_kmph_by_hour": h, "window_days": sp.get("days"),
            "slowest_hour": min(valid)[1] if valid else None, "fastest_hour": max(valid)[1] if valid else None,
            "freshness": freshness(c_sp, sp.get("t"))}


def ep_hotspots(city, q):
    C = city_or_404(city)
    if not C.get("hotspots"):
        return {"type": "FeatureCollection", "city": city, "features": [],
                "note": "no official congestion list for this city; see /speeds?sort=congested for measured slow roads"}
    how = how_of(q)
    feats, c, meta = hotspots(city, how)
    mn = num(q, "min_pct", 0, 100)
    if mn is not None:
        feats = [f for f in feats if (f["properties"].get("jam_share_pct") or 0) >= mn]
    bb = bbox_of(q)
    if bb:
        feats = [f for f in feats if in_bbox(coords_of(f["geometry"]), bb)]
    if any("jam_share_pct" in f["properties"] for f in feats):
        feats.sort(key=lambda f: -(f["properties"]["jam_share_pct"] or 0))
    if q.get("profile") in ("1", "true") and C["hotspots"]["fmt"] == "btp":
        d, _ = get(C["hotspots"]["url"], STATIC_TTL)
        cells = d.get("cells", [])
        for f in feats:
            f["properties"]["jam_share_pct_by_hour_of_week"] = cells[int(f["properties"]["id"].split(":")[1])][9]
    return {"type": "FeatureCollection", "city": city, "hour_of_week": how if C["hotspots"]["fmt"] == "btp" else None,
            "count": len(feats), "meta": meta, "freshness": freshness(c), "features": feats}


def ep_near(q):
    lat, lon = num(q, "lat", -90, 90), num(q, "lon", -180, 180)
    if lat is None or lon is None:
        raise Bad("lat and lon are required")
    city = q.get("city") or city_of_point(lat, lon)
    if not city:
        raise LookupError("that point is outside every city this API covers; see /cities")
    city_or_404(city)
    r = num(q, "radius", 10, 5000, 500)
    base = {"lat": str(lat), "lon": str(lon), "radius": str(r)}
    out = {"city": city, "lat": lat, "lon": lon, "radius_m": r}
    C = CITIES[city]
    if C.get("alerts"):
        out["alerts"] = ep_alerts(city, base)["features"]
    if C.get("hotspots"):
        hs = ep_hotspots(city, {})["features"]
        keep = []
        for f in hs:
            d = near(coords_of(f["geometry"]), lat, lon, r)
            if d is not None:
                keep.append({**f, "properties": {**f["properties"], "distance_m": round(d)}})
        out["hotspots"] = sorted(keep, key=lambda f: f["properties"]["distance_m"])
    if C.get("speeds"):
        s = ep_speeds(city, {**base, "limit": "200"})
        out["speeds"] = sorted(s["features"], key=lambda f: f["properties"]["distance_m"])
        cs = ep_city_speed(city, {})
        out["city_median_kmph_now"] = cs["median_kmph_now"]
    return out


def ep_health(q):
    out = {}
    for k, C in CITIES.items():
        st = {}
        for ds in ("alerts", "speeds", "hotspots"):
            if C.get(ds):
                c = _cache.get(C[ds]["url"])
                st[ds] = {"cached": bool(c), "ok": c["ok"] if c else None, "age_s": int(time.time() - c["at"]) if c else None}
        out[k] = st
    return {"ok": True, "time": int(time.time()), "cities": out}


# ---------------------------------------------------------------- OpenAPI (also the source for the docs page)
def openapi():
    cities = list(CITIES)
    P = lambda name, where, desc, typ="string", req=False, **kw: {"name": name, "in": where, "required": req, "description": desc,
                                                                    "schema": {"type": typ, **kw}}
    city = P("city", "path", "City id from /cities", req=True, enum=cities)
    bbox = P("bbox", "query", "min_lon,min_lat,max_lon,max_lat")
    lat, lon = P("lat", "query", "Latitude (with lon and radius)", "number"), P("lon", "query", "Longitude", "number")
    rad = P("radius", "query", "Metres around lat/lon", "number")
    lim = P("limit", "query", "Maximum features", "integer")
    return {"openapi": "3.0.3", "info": {"title": "reclaim.city traffic API", "version": "1.0",
            "description": "Police traffic alerts, observed bus speeds and official congestion spots, city by city. "
                           "No key, CORS open, read-only. Docs: https://api.reclaim.city/docs/#traffic"},
            "servers": [{"url": "https://api.reclaim.city/traffic/v1"}],
            "paths": {
                "/cities": {"get": {"summary": "Cities and which datasets each has", "responses": {"200": {"description": "JSON"}}}},
                "/{city}": {"get": {"summary": "What a city has, with counts and freshness", "parameters": [city],
                                    "responses": {"200": {"description": "JSON"}}}},
                "/{city}/alerts": {"get": {"summary": "Police traffic alerts (GeoJSON)", "parameters": [
                    city, P("state", "query", "active (default), upcoming or all", enum=["active", "upcoming", "all"]),
                    P("type", "query", "Comma-separated types, e.g. breakdown,diversion,congestion"),
                    P("route", "query", "Only alerts on this bus route number"), bbox, lat, lon, rad, lim],
                    "responses": {"200": {"description": "GeoJSON FeatureCollection"}}}},
                "/{city}/alerts/history": {"get": {"summary": "Every placed past alert, newest first (GeoJSON, paged)", "parameters": [
                    city, P("since", "query", "Unix time or YYYY-MM-DD (IST)"), P("until", "query", "Unix time or YYYY-MM-DD (IST), that day included"),
                    P("type", "query", "Comma-separated types"), P("agency", "query", "Part of the agency name, e.g. Cyberabad"),
                    P("route", "query", "Bus route number"), bbox, lat, lon, rad,
                    P("offset", "query", "Paging offset", "integer"), P("limit", "query", "Page size, up to 5000 (default 500)", "integer")],
                    "responses": {"200": {"description": "GeoJSON FeatureCollection with total and next_offset"}}}},
                "/{city}/speeds": {"get": {"summary": "Observed bus speed per stop-to-stop road (GeoJSON)", "parameters": [
                    city, P("when", "query", "now (last hour, default) or typical (median at hour over the window)", enum=["now", "typical"]),
                    P("hour", "query", "Hour of day 0-23 IST for typical (default: now)", "integer"),
                    P("sort", "query", "slowest or congested", enum=["slowest", "congested"]),
                    P("min_kmph", "query", "Lowest speed to include", "number"), P("max_kmph", "query", "Highest speed to include", "number"),
                    P("min_buses", "query", "Fewest buses in the last hour", "integer"),
                    P("profile", "query", "1 adds the 24-hour typical profile per road", enum=["1"]), bbox, lat, lon, rad, lim],
                    "responses": {"200": {"description": "GeoJSON FeatureCollection"}}}},
                "/{city}/speeds/city": {"get": {"summary": "City-wide median bus speed now and by hour", "parameters": [city],
                                                "responses": {"200": {"description": "JSON"}}}},
                "/{city}/hotspots": {"get": {"summary": "Official congestion places (GeoJSON)", "parameters": [
                    city, P("day", "query", "mon..sun or 0-6 (Bengaluru; default today IST)"),
                    P("hour", "query", "0-23 IST (Bengaluru; default now)", "integer"),
                    P("min_pct", "query", "Bengaluru: only places jammed in at least this % of checks", "number"),
                    P("profile", "query", "1 adds the 168-hour profile (Bengaluru)", enum=["1"]), bbox],
                    "responses": {"200": {"description": "GeoJSON FeatureCollection"}}}},
                "/near": {"get": {"summary": "Everything near a point: alerts, official spots, bus speeds", "parameters": [
                    P("lat", "query", "Latitude", "number", True), P("lon", "query", "Longitude", "number", True), rad,
                    P("city", "query", "Force a city (default: from the point)", enum=cities)],
                    "responses": {"200": {"description": "JSON"}}}},
                "/health": {"get": {"summary": "Cache state per source", "responses": {"200": {"description": "JSON"}}}},
            }}


# 120 requests a minute per client IP (as the geo and police lookups); in memory only, never written anywhere
RATE, WINDOW = 120, 60
_hits, _hlock = {}, threading.Lock()


def limited(ip):
    now = time.time()
    with _hlock:
        t0, n = _hits.get(ip, (now, 0))
        if now - t0 >= WINDOW:
            t0, n = now, 0
        n += 1
        _hits[ip] = (t0, n)
        if len(_hits) > 50000:                    # forget old windows
            for k in [k for k, (t, _) in _hits.items() if now - t >= WINDOW]:
                _hits.pop(k, None)
        return n > RATE


ROUTES = [
    ("cities", ep_cities), ("near", ep_near), ("health", ep_health), ("openapi.json", lambda q: openapi()),
]


class H(BaseHTTPRequestHandler):
    server_version = "traffic-api"
    sys_version = ""

    def log_message(self, *a):                  # no access log (privacy promise, as the rest of the API host)
        pass

    def send(self, code, obj, cache=30):
        b = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(b) > 1024
        if gz:
            b = gzip.compress(b, 5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", f"public, max-age={cache}" if code == 200 else "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        ip = (self.headers.get("CF-Connecting-IP") or (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
              or self.client_address[0])
        if limited(ip):
            return self.send(429, {"error": f"over {RATE} requests a minute; please cache (responses say for how long)"})
        u = urllib.parse.urlsplit(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        path = u.path.rstrip("/")
        if not path.startswith(PREFIX):
            return self.send(404, {"error": "not found", "see": "https://api.reclaim.city/docs/#traffic"})
        parts = [p for p in path[len(PREFIX):].split("/") if p]
        try:
            if not parts:
                return self.send(200, {"name": "reclaim.city traffic API", "version": 1, "docs": "https://api.reclaim.city/docs/#traffic",
                                       "openapi": f"{PREFIX}/openapi.json", "cities": f"{PREFIX}/cities"}, 300)
            for name, fn in ROUTES:
                if parts == [name]:
                    return self.send(200, fn(q), 300 if name in ("cities", "openapi.json") else 30)
            city = parts[0]
            if len(parts) == 1:
                return self.send(200, ep_city(city, q))
            sub = "/".join(parts[1:])
            fn = {"alerts": ep_alerts, "alerts/history": ep_history, "speeds": ep_speeds, "speeds/city": ep_city_speed,
                  "hotspots": ep_hotspots}.get(sub)
            if not fn:
                return self.send(404, {"error": f"unknown endpoint '{sub}'", "see": f"{PREFIX}/openapi.json"})
            return self.send(200, fn(city, q), 300 if sub in ("hotspots", "alerts/history") else 30)
        except Bad as e:
            return self.send(400, {"error": str(e)})
        except LookupError as e:
            return self.send(404, {"error": str(e).strip("'\"")})
        except Exception as e:                    # never a stack trace to the client
            print("error", self.path[:200], type(e).__name__, e, file=sys.stderr, flush=True)
            return self.send(503, {"error": "source not available right now, try again shortly"})


def warm():
    """Load every source once at start, so the first callers don't wait for 8 MB road files."""
    for k, C in CITIES.items():
        try:
            if C.get("speeds"):
                speeds(k)
            if C.get("alerts"):
                alerts(k)
            if C.get("hotspots"):
                hotspots(k, 0)
        except Exception as e:
            print("warm", k, type(e).__name__, e, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5150)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    threading.Thread(target=warm, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    print(f"traffic-api on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
