"""Turn one traffic post (tweet or WhatsApp message) into a normalized event record."""
import hashlib, json, math, re
from datetime import datetime, timezone
from pathlib import Path
from .extract import extract
from .classify import classify, scrub, IST
from .gazetteer import Gazetteer, hav
from . import route

from .gazetteer import TGSRTC_DATA as SITE, MTC_DATA as MTC, BMTC_DATA as BMTC   # map stop ids: ref/ copies (mss) or the Mac checkouts
STOP_R = 75   # metres, per the TGSRTC map session

class StopIndex:
    """Map stop ids and route numbers near a point/line (maps-site/tgsrtc ids)."""
    def __init__(self, city="hyd"):
        if city in ("chn", "blr"):           # MTC / BMTC map ids (route_stops.json, same format): stop id, route number
            self.stops, self.routes = {}, {}
            for r in json.load(open((MTC if city == "chn" else BMTC) / "route_stops.json")).values():
                for d in r["d"]:
                    for sid, name, la, lo in d["s"]:
                        self.stops[sid] = (name, la, lo); self.routes.setdefault(sid, set()).add(r["sn"])
            self._grid(); return
        self.stops = json.load(open(SITE / "stops.json"))
        self.routes = {}
        sv = json.load(open(SITE / "services.json"))["services"]
        for s in sv.values():
            for sid in s[6]:
                self.routes.setdefault(sid, set()).add(s[0])
        self._grid()
    def _grid(self):
        self.grid = {}
        for sid, (_, la, lo) in self.stops.items():
            self.grid.setdefault((int(la * 500), int(lo * 500)), []).append(sid)   # ~200 m cells
    def near(self, pts, r=STOP_R):
        out = set()
        for la, lo in pts:
            gy, gx = int(la * 500), int(lo * 500)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for sid in self.grid.get((gy + dy, gx + dx), ()):
                        _, sla, slo = self.stops[sid]
                        if hav((la, lo), (sla, slo)) <= r: out.add(sid)
        return sorted(out)
    def routes_for(self, sids):
        return sorted({r for s in sids for r in self.routes.get(s, ())}, key=lambda x: (len(x), x))

def densify(line, step=40):
    out = [line[0]]
    for p in line[1:]:
        q = out[-1]; d = hav(q, p); n = int(d // step)
        for i in range(1, n + 1):
            f = i / (n + 1); out.append([q[0] + (p[0] - q[0]) * f, q[1] + (p[1] - q[1]) * f])
        out.append(p)
    return out

class Builder:
    def __init__(self, web=True, city="hyd"):
        """web: True (always look up unknown places online), False, or a set of normalized phrases allowed.
        city: "hyd" (Cyberabad + Hyderabad, TGSRTC stops), "chn" (Greater Chennai, MTC stops) or "blr" (Bengaluru, BMTC stops)."""
        self.gaz = Gazetteer(city); self.stops = StopIndex(city); self.web = web
        self.unresolved = {}

    def build(self, text, posted, source, src_url=None, post_id=None, region=None, hint=None):
        """posted: aware datetime. region: 'cyb' (Cyberabad) or 'hyd' (Hyderabad city), default from source.
        hint: (lat, lon) the post is most likely near (Bengaluru: the posting station's area); a station named in
        the text wins. Returns event dict (always; lat/lon may be None)."""
        region = region or region_of(source)
        text = scrub(text)
        c = classify(text, posted)
        ex = extract(text, arrows=region in ("chn", "blr"))
        if ex["traffic_ps"]:
            h = self.gaz.resolve(ex["traffic_ps"].split("&")[0].split("/")[0], allow_web=False, region=region)
            hint = (h["lat"], h["lon"]) if h else hint
        places, line, primary, direction = [], None, None, ""
        paths, pdirs = [], []
        for m in ex["mentions"]:
            res = [self.gaz.resolve(p, hint, allow_web=self.web, region=region) for p in m["places"]]
            for p, r in zip(m["places"], res):
                if r: places.append({"q": p, "name": r["name"], "lat": r["lat"], "lon": r["lon"], "src": r["src"], "score": r["score"]})
                else: self.unresolved[p.lower()] = self.unresolved.get(p.lower(), 0) + 1
            if m["kind"] == "path":          # 📍 chain of waypoints: route through every resolved one, in order
                wp = [(r["lat"], r["lon"]) for r in res if r]
                if len(wp) >= 2:
                    paths += route.road_path(wp, max_km=12)
                pdirs.append(" -> ".join(m["places"]))
                if primary is None and wp:
                    primary = next(r for r in res if r)
                continue
            a = res[0]
            if a and primary is None and m["kind"] == "at_from_to" and not res[1] and len(res) > 2 and res[2]:
                primary = a; C = res[2]
                line = route.road_line((a["lat"], a["lon"]), (C["lat"], C["lon"]), max_km=10)
                direction = f"from {m['places'][1]} towards {m['places'][2]}"
            elif a and primary is None:
                primary = a
                if len(res) > 1 and res[1]:
                    b = res[1]; A, B = (a["lat"], a["lon"]), (b["lat"], b["lon"])
                    if m["kind"] == "at_from_to":
                        C = res[2] if len(res) > 2 and res[2] else None
                        if C:
                            line = route.road_line(B, (C["lat"], C["lon"]), max_km=12, via=A)
                            if not line: line = route.road_line(A, (C["lat"], C["lon"]), max_km=10)
                        direction = f"from {m['places'][1]} towards {m['places'][2]}"
                    elif m["kind"] == "from_to":
                        V = None
                        if m.get("via"):
                            rv = self.gaz.resolve(m["via"], hint, allow_web=self.web, region=region)
                            if rv:
                                V = (rv["lat"], rv["lon"])
                                places.append({"q": m["via"], "name": rv["name"], "lat": rv["lat"], "lon": rv["lon"], "src": rv["src"], "score": rv["score"]})
                        line = route.road_line(A, B, max_km=12, via=V)
                        direction = f"{m['places'][0]} -> {m['places'][1]}" + (f" via {m['via']}" if m.get("via") else "")
                    elif m["kind"] == "at_towards":
                        line = route.road_line(A, B, max_km=10, cap_km=1.5)
                        direction = f"towards {m['places'][1]}"
                    elif m["kind"] == "at_from":
                        full = route.road_line(B, A, max_km=10)
                        if full:                                   # keep the last 1.5 km before A
                            acc, k = 0.0, len(full) - 1
                            while k > 0 and acc < 1500:
                                acc += hav(full[k], full[k - 1]); k -= 1
                            line = full[k:]
                        direction = f"from {m['places'][1]}"
            elif primary is None and len(res) > 1 and res[1]:
                primary = res[1]
        if paths:
            line = line or paths[0]
            if direction == "": direction = "; ".join(pdirs)
        lines = ([line] if line and line not in paths else []) + paths
        geo_pts = [p for l in lines for p in densify(l)] if lines else ([(primary["lat"], primary["lon"])] if primary else [])
        sids = self.stops.near(geo_pts) if geo_pts else []
        if primary and not sids:   # nothing within 75 m: report the nearest stops within 300 m as "nearby"
            near = self.stops.near([(primary["lat"], primary["lon"])], r=300)
        else:
            near = []
        utc = lambda d: int(d.timestamp()) if d else None
        pid = post_id or hashlib.sha1((source + norm_text(text) + posted.strftime("%Y%m%d")).encode()).hexdigest()[:16]
        return {
            "id": f"{source}:{pid}", "source": source, "agency": AGENCY.get(region, region), "src": src_url,
            "post_time": utc(posted), "post_time_ist": posted.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
            "type": c["type"], "tags": c["tags"], "sidecar_type": c["sidecar_type"], "kind": c["kind"],
            "status": c["status"], "severity": c["severity"], "speed_multiplier": c["speed_multiplier"],
            "default_duration_min": c["default_duration_min"],
            "t_start": utc(c["start"]), "t_end": utc(c["end"]), "t_expires": utc(c["expires"]),
            "t_start_ist": c["start"].astimezone(IST).strftime("%Y-%m-%d %H:%M"),
            "lat": primary["lat"] if primary else None, "lon": primary["lon"] if primary else None,
            "place": primary["name"] if primary else None, "place_query": primary["query"] if primary else None,
            "geo_src": primary["src"] if primary else None, "geo_score": primary["score"] if primary else None,
            "line": line, "lines": lines if len(lines) > 1 else None, "direction": direction, "places": places,
            "police_station": ex["traffic_ps"], "stop_ids": sids, "near_stop_ids": near,
            "routes": self.stops.routes_for(sids or near),
            "text": text,
        }

    def save(self):
        self.gaz.save(); route.save()

AGENCY = {"cyb": "Cyberabad Traffic Police", "hyd": "Hyderabad Traffic Police", "chn": "Greater Chennai Traffic Police",
          "blr": "Bengaluru Traffic Police"}

def region_of(source):
    return ("hyd" if source.startswith("hydtp") else "chn" if source.startswith("chennai")
            else "blr" if source.startswith("btp") else "cyb")

def norm_text(t):
    return re.sub(r"\W+", " ", t.lower()).strip()
