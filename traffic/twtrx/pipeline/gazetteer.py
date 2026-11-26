"""Location index for Cyberabad / Hyderabad traffic posts.

Sources, best first:
  1. data/aliases.csv           hand-curated names -> lat/lon or -> another name (edit freely)
  2. OSM named features          data/osm_*.geojsonseq (cut from the India PBF)
  3. TGSRTC stops                maps-site/tgsrtc/data/stops.json (the map's own stop ids)
  4. Nominatim fallback          cached in data/nominatim_cache.json, 1 req/s, Hyderabad viewbox only

resolve(phrase, hint=(lat,lon)) -> {"name","lat","lon","src","kind","score"} or None
"""
import csv, json, math, re, time, urllib.parse, urllib.request
from pathlib import Path
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# TGSRTC and MTC map data (stop ids, route numbers): ref/ when present (mss, refreshed daily from the live sites by
# pipeline.refdata), else the local maps-site and mtc-bus-map checkouts (the Mac)
def _ref(name, mac):
    r = ROOT / "ref" / name
    return r if r.is_dir() else Path.home() / mac
TGSRTC_DATA = _ref("tgsrtc", "Desktop/projects/maps-site/tgsrtc/data")
MTC_DATA = _ref("mtc", "Desktop/projects/mtc-bus-map/web/data")
BMTC_DATA = _ref("bmtc", "Desktop/projects/maps-site/bengaluru/data")
STOPS = TGSRTC_DATA / "stops.json"
BBOX = (17.00, 78.05, 17.75, 78.80)  # Cyberabad incl. Shadnagar/Shamshabad; OSM extract covers 17.15-17.70           # lat0, lon0, lat1, lon1
CENTER = (17.445, 78.375)                     # Cyberabad (HITEC City) centroid
# jurisdiction used to break ties between same-named places.
#   cyb: Cyberabad Traffic Police (west/north/south; east of ~78.53 is Rachakonda)
#   hyd: Hyderabad Traffic Police (core city, Secunderabad, Amberpet/Uppal side up to ~78.60)
REGIONS = {"cyb": {"center": CENTER, "lon_max": 78.53}, "hyd": {"center": (17.400, 78.475), "lon_max": 78.60},
           "chn": {"center": (13.050, 80.235), "lon_max": 80.40},   # Greater Chennai Traffic Police
           "blr": {"center": (12.972, 77.594), "lon_max": 77.90}}   # Bengaluru Traffic Police
# One gazetteer per city: its own OSM cut, aliases, Nominatim cache, bus stops and search box.
CITIES = {
    "hyd": {"data": DATA, "bbox": BBOX, "suffix": ", Hyderabad, Telangana", "stops": ("tgsrtc", STOPS)},
    "chn": {"data": DATA / "chn", "bbox": (12.75, 79.95, 13.35, 80.35), "suffix": ", Chennai, Tamil Nadu",
            "stops": ("mtc", MTC_DATA / "bus_stops.geojson")},
    "blr": {"data": DATA / "blr", "bbox": (12.70, 77.30, 13.30, 77.90), "suffix": ", Bengaluru, Karnataka",
            "stops": ("bmtc", BMTC_DATA / "bus_stops.geojson")},
}

ABBR = [
    (r"&", " and "), (r"\bjn\.?|\bjnc\.?|\bjct\.?|\bjunc\.?\b", " junction "),
    (r"\bx[\s-]*roads?\b|\bcross[\s-]*roads?\b|\bx[\s-]*rd\b|\bxroads?\b", " x roads "),
    (r"\brd\b", " road "), (r"\btwds?\b", " towards "), (r"\bopp\.?\b|\bopposite\b", " "),
    (r"\bcybertowers?\b", " cyber towers "), (r"\bfly[\s-]*over\b", " flyover "),
    (r"\bu/?pass\b", " underpass "), (r"\bstn\b", " station "), (r"\bhosp\b", " hospital "),
    (r"\bl\s*and\s*t\b", " lt "), (r"\bpolice station\b|\bps\b", " ps "),
    (r"\bcentre\b", " center "), (r"\bu[\s-]*turn\b", " u turn "), (r"\bsignals\b", " signal "), (r"\brotory\b", " rotary "),
]
GENERIC = r"\b(route|routes|stretch|junction|rotary|signal|circle|x roads|flyover|underpass|bridge|rob|metro station|metro|bus stop|bus stand|bus station|main road|road|area|village|colony|center|towers?|ps|the|near|at|stop|point|chowrasta|chowrastha|u turn|pillar no|pillar|rub|no|landing|starting point)\b"

def norm(s):
    s = s.lower()
    for a, b in ABBR:
        s = re.sub(a, b, s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def core(s):
    c = re.sub(GENERIC, " ", norm(s))
    return re.sub(r"\s+", " ", c).strip()

def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(h))

def inbox(lat, lon, bbox=BBOX):
    return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]

# priority: lower is better
def osm_prio(props, layer):
    tags = props.get("other_tags") or ""
    if layer == "points":
        if re.search(r'"place"=>"(suburb|neighbourhood|locality|village|town|quarter|hamlet)"', tags): return 1, "place"
        if re.search(r'"highway"=>"(traffic_signals|motorway_junction)"|"junction"=>', tags): return 1, "junction"
        if re.search(r'"(railway|public_transport)"=>"(station|stop_position|platform|halt)"|"highway"=>"bus_stop"', tags): return 2, "stop"
        if re.search(r'"amenity"=>"(hospital|college|university|school|police|bus_station|marketplace|place_of_worship|cinema)"|"tourism"|"shop"=>"mall"|"office"', tags): return 3, "landmark"
        return 6, "poi"
    if layer == "lines":
        if props.get("highway") in ("motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link"):
            return 4, "road"
        return 5, "road"
    return 3, "landmark"

class Gazetteer:
    def __init__(self, city="hyd"):
        C = CITIES[city]; self.city, self.data, self.bbox = city, C["data"], C["bbox"]
        self.suffix, self.stops_src = C["suffix"], C["stops"]
        self.full, self.core = {}, {}
        self.alias_to = {}
        self._load_aliases(); self._load_junctions(); self._load_osm(); self._load_stops()
        self.core_keys = list(self.core)
        cp = self.data / "nominatim_cache.json"
        self.nom = json.loads(cp.read_text()) if cp.exists() else {}
        self.nom_calls = 0

    def add(self, name, lat, lon, prio, kind, src, region=None):
        if not name or not inbox(lat, lon, self.bbox): return
        c = {"name": name, "lat": round(lat, 6), "lon": round(lon, 6), "prio": prio, "kind": kind, "src": src}
        if region: c["region"] = region          # alias valid only for that police jurisdiction
        n, k = norm(name), core(name)
        if n: self.full.setdefault(n, []).append(c)
        if k and len(k) >= 3: self.core.setdefault(k, []).append(c)

    def _load_aliases(self):
        p = self.data / "aliases.csv"
        if not p.exists(): return
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("lat") and r.get("lon"):
                self.add(r["alias"], float(r["lat"]), float(r["lon"]), 0, "alias", "alias", r.get("region") or None)
            elif r.get("target"):
                self.alias_to[(norm(r["alias"]), r.get("region") or None)] = r["target"]

    def _load_junctions(self):
        """The police's own junction names with their position (Bengaluru: data/blr/junctions.csv, from BTP's
        congestion alerts, pipeline.btp_congestion), ranked just below the hand aliases."""
        p = self.data / "junctions.csv"
        if not p.exists(): return
        for r in csv.DictReader(open(p, encoding="utf-8")):
            self.add(r["name"], float(r["lat"]), float(r["lon"]), 1, "junction", "btp")

    def _load_osm(self):
        for layer in ("points", "lines", "polys"):
            p = self.data / f"osm_{layer}.geojsonseq"
            if not p.exists(): continue
            for line in open(p, encoding="utf-8"):
                try: f = json.loads(line.strip().lstrip("\x1e"))
                except ValueError: continue
                g, pr = f["geometry"], f["properties"]
                if not g: continue
                lat, lon = _centroid(g)
                prio, kind = osm_prio(pr, layer)
                names = {pr.get("name")}
                ot = pr.get("other_tags") or ""
                for k in ("name:en", "alt_name", "old_name", "short_name"):
                    m = re.search(r'"%s"=>"([^"]+)"' % k, ot)
                    if m: names.update(m.group(1).split(";"))
                for n in names - {None}:
                    self.add(n, lat, lon, prio, kind, "osm")

    def _load_stops(self):
        kind, path = self.stops_src
        if not path.exists(): return
        if kind in ("mtc", "bmtc"):             # MTC stop names are upper case (as on the boards); BMTC's are not
            for f in json.load(open(path))["features"]:
                lon, lat = f["geometry"]["coordinates"][:2]
                n = f["properties"]["Stop Name"]
                self.add(n.title() if kind == "mtc" else n, lat, lon, 2, f"{kind}_stop", kind)
            return
        for sid, (name, lat, lon) in json.load(open(path)).items():
            base = re.split(r"\s+(?:twd|twds|towards)\b", name, flags=re.I)[0]
            self.add(base, lat, lon, 2, "tgsrtc_stop", "tgsrtc")

    def _pick(self, cands, hint, region="cyb"):
        """Choose one candidate, or None when it would be a guess."""
        R = REGIONS[region]; CENTER = R["center"]
        cands = [c for c in cands if c.get("region") in (None, region)]
        cands = [c for c in cands if c["lon"] <= R["lon_max"] or c["prio"] == 0]
        if not cands: return None
        if hint:
            near = [c for c in cands if hav(hint, (c["lat"], c["lon"])) < 12000]
            if not near: return None
            return min(near, key=lambda c: (c["prio"], hav(hint, (c["lat"], c["lon"]))))
        best = min(c["prio"] for c in cands)
        pool = [c for c in cands if c["prio"] == best]
        if len(pool) > 1:
            far = max(hav((a["lat"], a["lon"]), (b["lat"], b["lon"])) for a in pool for b in pool)
            if far > 5000:                      # same name in several places
                pool.sort(key=lambda c: hav(CENTER, (c["lat"], c["lon"])))
                p0 = (pool[0]["lat"], pool[0]["lon"]); d0 = hav(CENTER, p0)
                others = [c for c in pool if hav(p0, (c["lat"], c["lon"])) > 1000]   # distinct places, not duplicates
                d1 = min(hav(CENTER, (c["lat"], c["lon"])) for c in others)
                return pool[0] if d0 < 9000 and d1 > 2 * d0 + 3000 else None   # one is clearly the local one
        return min(pool, key=lambda c: hav(CENTER, (c["lat"], c["lon"])))

    def resolve(self, phrase, hint=None, allow_web=True, region="cyb"):
        if not phrase: return None
        parts = re.split(r"\s+(?:and|&)\s+", phrase, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) >= 3:     # "Hafeezpet and Botanical signal" -> try the first
            r = self.resolve(parts[0], hint, allow_web, region)
            if r: return r
        nm = re.split(r"\s+near\s+(?:opp\.?\s+|opposite\s+)?", phrase, maxsplit=1, flags=re.I)
        if len(nm) == 2 and len(nm[1]) >= 3:            # "Palle Cheruvu Road near Opp. Mehfil": landmark first
            r = self.resolve(nm[1], hint, allow_web, region) or self.resolve(nm[0], hint, allow_web, region)
            if r: return r
        n = norm(phrase)
        tgt = self.alias_to.get((n, region), self.alias_to.get((n, None)))
        if tgt:
            if tgt == "-": return None                       # deliberately ambiguous
            phrase = tgt; n = norm(phrase)
        k = core(phrase)
        def hit(cs, score):
            c = self._pick(cs, hint, region)
            return {**c, "score": score, "query": phrase} if c else None
        if n in self.full:
            r = hit(self.full[n], 100)
            if r: return r
        # looser matches must land on a place/junction/stop/road/landmark, never a shop
        good = lambda cs: [c for c in cs if c["prio"] <= 4]
        if k and good(self.core.get(k, [])):
            r = hit(good(self.core[k]), 95)
            if r: return r
        if k and len(k) >= 5:
            for key, sc, _ in process.extract(k, self.core_keys, scorer=fuzz.ratio, score_cutoff=90, limit=5):
                if good(self.core[key]):
                    r = hit(good(self.core[key]), int(sc))
                    if r: return r
        if allow_web and k and len(k) >= 4 and (allow_web is True or norm(phrase) in allow_web):
            return self._nominatim(phrase, hint, region)
        return None

    def _nominatim(self, phrase, hint, region="cyb"):
        key = norm(phrase)
        if key not in self.nom:
            q = urllib.parse.urlencode({"q": phrase + self.suffix, "format": "jsonv2", "limit": 5,
                 "viewbox": f"{self.bbox[1]},{self.bbox[2]},{self.bbox[3]},{self.bbox[0]}", "bounded": 1, "countrycodes": "in"})
            req = urllib.request.Request("https://nominatim.openstreetmap.org/search?" + q,
                                         headers={"User-Agent": "twtrx-traffic-gazetteer/1.0"})
            time.sleep(1.1); self.nom_calls += 1
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    res = json.load(r)
            except Exception as e:
                print("nominatim error", phrase, e); return None
            self.nom[key] = [{"name": x["display_name"].split(",")[0], "lat": float(x["lat"]), "lon": float(x["lon"]),
                              "prio": 3, "kind": x.get("type", "nominatim"), "src": "nominatim"} for x in res]
        cands = [c for c in self.nom[key] if inbox(c["lat"], c["lon"], self.bbox)]
        c = self._pick(cands, hint, region) if cands else None
        return {**c, "score": 80, "query": phrase} if c else None

    def save(self):
        (self.data / "nominatim_cache.json").write_text(json.dumps(self.nom, ensure_ascii=False, indent=0))

def _centroid(g):
    t, c = g["type"], g["coordinates"]
    if t == "Point": return c[1], c[0]
    pts = []
    def walk(x):
        if isinstance(x[0], (int, float)): pts.append(x)
        else:
            for y in x: walk(y)
    walk(c)
    if t in ("LineString", "MultiLineString"):
        p = pts[len(pts) // 2]; return p[1], p[0]           # a point on the road, not off it
    return sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts)
