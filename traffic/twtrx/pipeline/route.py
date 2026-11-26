"""Road geometry between two points via OSRM (driving). Cached; never a straight join.

Uses OSRM_URL (default: public demo router). If you start a local OSRM on
~/Desktop/projects/einfra/data/osm/osrm/india.osrm, set OSRM_URL=http://localhost:5000.
Returns [[lat, lon], ...] or None (caller then publishes the point only).
"""
import json, os, time, urllib.request
from pathlib import Path
from .gazetteer import hav, DATA

OSRM = os.environ.get("OSRM_URL", "https://router.project-osrm.org")
CACHE_P = DATA / "osrm_cache.json"
_cache = json.loads(CACHE_P.read_text()) if CACHE_P.exists() else {}
calls = 0

def road_line(a, b, max_km=8.0, cap_km=None, via=None):
    """a, b: (lat, lon). via: optional waypoint. cap_km trims the line to that length from a."""
    global calls
    d = hav(a, b) if not via else hav(a, via) + hav(via, b)
    if d < 80 or d > max_km * 1000: return None
    pts = [a] + ([via] if via else []) + [b]
    key = ";".join(f"{p[0]:.5f},{p[1]:.5f}" for p in pts)
    if key not in _cache:
        url = f"{OSRM}/route/v1/driving/" + ";".join(f"{p[1]:.6f},{p[0]:.6f}" for p in pts) + "?overview=full&geometries=geojson"
        try:
            time.sleep(0.4 if "project-osrm" in OSRM else 0); calls += 1
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "twtrx/1.0"}), timeout=30) as r:
                j = json.load(r)
            rt = j["routes"][0] if j.get("routes") else None
            _cache[key] = {"dist": rt["distance"], "coords": [[round(y, 5), round(x, 5)] for x, y in rt["geometry"]["coordinates"]]} if rt else None
        except Exception as e:
            print("osrm error", e); return None
    r = _cache[key]
    if not r: return None
    if r["dist"] > 2.5 * d + 800:       # implausible detour: geocode is probably wrong
        return None
    line = r["coords"]
    # endpoints must be on the road near the geocoded points (snap tolerance 300 m)
    if hav(a, line[0]) > 300 or hav(b, line[-1]) > 300: return None
    if cap_km:
        out, acc = [line[0]], 0.0
        for p in line[1:]:
            acc += hav(out[-1], p); out.append(p)
            if acc >= cap_km * 1000: break
        line = out
    return line if len(line) >= 2 else None

def save():
    CACHE_P.write_text(json.dumps(_cache))

def road_path(pts, max_km=25.0, loop=False):
    """Road geometry through several waypoints [(lat, lon), ...] in order (e.g. a procession route or a
    one-way ring). Each leg is routed and checked like road_line; a leg that fails is skipped rather than
    joined straight. Returns a list of lines (one per contiguous routed stretch), possibly empty."""
    pts = list(pts) + ([pts[0]] if loop else [])
    out, cur = [], None
    for a, b in zip(pts, pts[1:]):
        leg = road_line(a, b, max_km=max_km)
        if not leg:
            if cur: out.append(cur)
            cur = None; continue
        cur = leg if cur is None else cur + leg[1:]
    if cur: out.append(cur)
    return out
