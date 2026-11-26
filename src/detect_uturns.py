#!/usr/bin/env python3
"""Empirical Bus U-Turn Detector & Interactive Map Generator from Indian Bus GPS Logs.

Indian arterial roads (Chennai OMR/GST/Anna Salai, Bengaluru ORR, Delhi Ring Rd, Mumbai WEH/EEH,
Gurugram Sohna Rd) have median barriers that force buses to execute 150-180 deg U-turns across
mixed traffic, adding 30-150s of queue + weaving delay that static OSM tags rarely capture.

Algorithm (per city):
  1. Stream recent GPS archives (obs-*.jsonl*) per bus `reg`.
  2. Track moving vectors (step >= 25m, speed 1.5..22 m/s, gap <= 90s).
  3. Flag a U-turn when the heading reverses by 140..180 deg across consecutive/near-consecutive
     moving vectors within <= 180s and <= 300m displacement, EXCLUDING route termini (within 250m
     of route start/end stops in route_stops.json).
  4. Cluster detections into 60m spatial cells; keep recurring hotspots (n_uturns >= min_hits and
     n_buses >= 2) to eliminate GPS glitches.
  5. Export:
     - ETA/data/uturns/{city}_uturns.geojson   (standard GeoJSON FeatureCollection)
     - ETA/data/uturns/all_cities_uturns.geojson
     - ETA/maps/indian_bus_uturns_map.html     (interactive Leaflet map with city switcher & popups)
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

K = 111320.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    cx = math.cos(math.radians(0.5 * (lat1 + lat2))) * K
    return math.hypot((lon2 - lon1) * cx, (lat2 - lat1) * K)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlo = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def load_termini(stops_path: Path) -> np.ndarray:
    """Return Nx2 array of (lat, lon) for all route start/end termini to exclude turnaround loops."""
    if not stops_path.exists():
        return np.empty((0, 2))
    rs = json.loads(stops_path.read_text())
    pts = []
    for r in rs.values():
        d_list = r.get("d") or []
        if not d_list:
            continue
        s = d_list[0].get("s") or []
        if len(s) >= 2:
            pts.append((float(s[0][2]), float(s[0][3])))
            pts.append((float(s[-1][2]), float(s[-1][3])))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2))


def detect_city_uturns(args) -> tuple[str, list[dict]]:
    city, data_root_s, max_days = args
    t0 = time.time()
    root = Path(data_root_s)
    raw_dir = root / "raw" / city
    stops_path = root / "meta" / city / "route_stops.json"

    termini = load_termini(stops_path)
    # Build fast spatial cell set of termini (250m grid)
    term_cells = set()
    if len(termini) > 0:
        dlat_t = 250.0 / K
        dlon_t = 250.0 / (K * math.cos(math.radians(float(np.mean(termini[:, 0])))))
        for la, lo in termini:
            cy, cx = int(round(la / dlat_t)), int(round(lo / dlon_t))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    term_cells.add((cy + dy, cx + dx))
    else:
        dlat_t = dlon_t = 0.0025

    # Support TGSRTC CSV archives (id,lat,lon,heading,speedKph,gpsTime,status) for Hyderabad metro!
    csv_files = sorted(raw_dir.glob("*.csv.gz"))[-max_days:]
    if city == "hyderabad" and csv_files:
        import csv
        grid_m = 60.0
        dlat_g = grid_m / K
        dlon_g = grid_m / (K * math.cos(math.radians(17.42)))
        cells = collections.defaultdict(list)
        for path in csv_files:
            state = {}
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        la = float(row["lat"])
                        lo = float(row["lon"])
                        if not (17.22 <= la <= 17.58 and 78.22 <= lo <= 78.68):
                            continue
                        reg = row["id"]
                        t = float(row["gpsTime"])
                        spd = float(row.get("speedKph") or 0.0)
                    except (ValueError, KeyError, TypeError):
                        continue
                    prev = state.get(reg)
                    if prev is None:
                        state[reg] = (t, la, lo, -1.0, -1.0, la, lo)
                        continue
                    pt, pla, plo, lvt, lvb, lvla, lvlo = prev
                    dt_ = t - pt
                    if dt_ <= 0 or dt_ > 120.0:
                        state[reg] = (t, la, lo, -1.0, -1.0, la, lo)
                        continue
                    dist = haversine_m(pla, plo, la, lo)
                    if dist < 25.0:
                        continue
                    if dist / dt_ > 23.0:
                        state[reg] = (t, la, lo, -1.0, -1.0, la, lo)
                        continue
                    b = bearing_deg(pla, plo, la, lo)
                    if lvt > 0 and (t - lvt) <= 210.0:
                        adiff = angle_diff_deg(lvb, b)
                        if adiff >= 145.0 and haversine_m(lvla, lvlo, la, lo) <= 350.0:
                            apex_la = 0.5 * (lvla + pla)
                            apex_lo = 0.5 * (lvlo + plo)
                            gc = (int(round(apex_la / dlat_g)), int(round(apex_lo / dlon_g)))
                            cells[gc].append((apex_la, apex_lo, min(210.0, t - lvt), reg, "TGSRTC", adiff))
                            state[reg] = (t, la, lo, -1.0, -1.0, la, lo)
                            continue
                    state[reg] = (t, la, lo, t, b, la, lo)
        features = []
        for gc, evs in cells.items():
            buses = {e[3] for e in evs}
            if len(evs) < 6 or len(buses) < 3:
                continue
            durs = [e[2] for e in evs]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(np.median([e[1] for e in evs])), 6), round(float(np.median([e[0] for e in evs])), 6)]},
                "properties": {
                    "city": "hyderabad", "n_uturns": len(evs), "n_buses": len(buses), "n_routes": 1,
                    "top_routes": ["TGSRTC-City"], "median_dur_s": round(float(np.median(durs)), 1),
                    "p90_dur_s": round(float(np.percentile(durs, 90)), 1), "mean_angle_deg": round(float(np.mean([e[5] for e in evs])), 1),
                },
            })
        features.sort(key=lambda f: f["properties"]["n_uturns"], reverse=True)
        out_dir = root / "uturns"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hyderabad_uturns.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
        print(f"[uturns:hyderabad] {len(csv_files)}d -> {len(features):,} recurring U-turn hotspots in {time.time() - t0:.1f}s", flush=True)
        return "hyderabad", features

    day_re = re.compile(r"^obs-(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")
    by_day: dict[str, Path] = {}
    for p in sorted(raw_dir.glob("obs-*.jsonl*")):
        m = day_re.match(p.name)
        if m:
            d = m.group(1)
            if d not in by_day or p.suffix == ".jsonl":
                by_day[d] = p
    days = sorted(by_day.items())[-max_days:]

    # Cluster grid (~60m)
    grid_m = 60.0
    dlat_g = grid_m / K
    # cell -> list of (lat, lon, dur_s, reg, route_id, turn_deg)
    cells: dict[tuple[int, int], list[tuple]] = collections.defaultdict(list)
    mean_cos = 0.95

    for day_str, path in days:
        # reg -> (t, lat, lon, route_id, last_vec_t, last_vec_bearing, last_vec_lat, last_vec_lon)
        state: dict[str, tuple] = {}
        op = gzip.open if path.suffix == ".gz" else open
        with op(path, "rt", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                reg = r.get("reg")
                la = r.get("la")
                lo = r.get("lo")
                t = r.get("t")
                route = str(r.get("r") or "")
                if not reg or la is None or lo is None or t is None:
                    continue
                la, lo, t = float(la), float(lo), float(t)
                mean_cos = math.cos(math.radians(la))

                prev = state.get(reg)
                if prev is None:
                    state[reg] = (t, la, lo, route, -1.0, -1.0, la, lo)
                    continue
                pt, pla, plo, proute, lvt, lvb, lvla, lvlo = prev
                dt_ = t - pt
                if dt_ <= 0 or dt_ > 120.0 or proute != route:
                    state[reg] = (t, la, lo, route, -1.0, -1.0, la, lo)
                    continue

                dist = haversine_m(pla, plo, la, lo)
                if dist < 25.0:
                    # Bus is slow/queuing at the U-turn; keep last moving bearing!
                    continue
                v = dist / dt_
                if v > 23.0:
                    state[reg] = (t, la, lo, route, -1.0, -1.0, la, lo)
                    continue

                b = bearing_deg(pla, plo, la, lo)
                if lvt > 0 and (t - lvt) <= 210.0:
                    adiff = angle_diff_deg(lvb, b)
                    if adiff >= 145.0:
                        # Candidate U-turn apex is midpoint between previous vector end and current start
                        apex_la = 0.5 * (lvla + pla)
                        apex_lo = 0.5 * (lvlo + plo)
                        disp = haversine_m(lvla, lvlo, la, lo)
                        if disp <= 350.0:
                            # Exclude route termini turnarounds
                            tc = (int(round(apex_la / dlat_t)), int(round(apex_lo / dlon_t)))
                            if tc not in term_cells:
                                dlon_g = grid_m / (K * max(0.2, mean_cos))
                                gc = (int(round(apex_la / dlat_g)), int(round(apex_lo / dlon_g)))
                                uturn_dur = min(210.0, t - lvt)
                                cells[gc].append((apex_la, apex_lo, uturn_dur, reg, route, adiff))
                                # Reset vector so we don't double-count same U-turn
                                state[reg] = (t, la, lo, route, -1.0, -1.0, la, lo)
                                continue

                state[reg] = (t, la, lo, route, t, b, la, lo)

    min_hits = 3 if city == "ncr" else 6
    features = []
    for gc, evs in cells.items():
        buses = {e[3] for e in evs}
        if len(evs) < min_hits or len(buses) < 2:
            continue
        lats = [e[0] for e in evs]
        lons = [e[1] for e in evs]
        durs = [e[2] for e in evs]
        routes = collections.Counter(e[4] for e in evs)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(np.median(lons)), 6), round(float(np.median(lats)), 6)],
            },
            "properties": {
                "city": city,
                "n_uturns": len(evs),
                "n_buses": len(buses),
                "n_routes": len(routes),
                "top_routes": [k for k, _ in routes.most_common(6)],
                "median_dur_s": round(float(np.median(durs)), 1),
                "p90_dur_s": round(float(np.percentile(durs, 90)), 1),
                "mean_angle_deg": round(float(np.mean([e[5] for e in evs])), 1),
            },
        })

    features.sort(key=lambda f: f["properties"]["n_uturns"], reverse=True)
    out_dir = root / "uturns"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{city}_uturns.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
    )
    print(f"[uturns:{city}] {len(days)}d -> {len(features):,} recurring mid-route U-turn hotspots in {time.time() - t0:.1f}s", flush=True)
    return city, features


def build_interactive_map(all_features: list[dict], out_html: Path):
    """Generate a self-contained Leaflet interactive map of all discovered Indian bus U-turns."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    city_centers = {
        "hyderabad": [17.42, 78.45, 12],
        "mtc": [13.06, 80.23, 12],
        "bmtc": [12.97, 77.59, 12],
        "delhi": [28.63, 77.21, 11],
        "mumbai": [19.08, 72.88, 11],
        "ncr": [28.46, 77.04, 12],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Indian Bus U-Turn Bottlenecks (GPS-Discovered)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family: system-ui, -apple-system, sans-serif; background:#0d1117; color:#e6edf3; }}
  #map {{ width:100vw; height:100vh; }}
  .panel {{ position:fixed; top:14px; left:54px; z-index:999; background:rgba(13,17,23,.92);
            border:1px solid #30363d; border-radius:10px; padding:12px 14px; max-width:340px; }}
  .panel h2 {{ margin:0 0 6px; font-size:15px; }}
  .panel p {{ margin:0 0 8px; font-size:12px; color:#9ba3b0; }}
  .btns {{ display:flex; flex-wrap:wrap; gap:6px; }}
  button {{ background:#1f2937; color:#e6edf3; border:1px solid #374151; border-radius:6px;
            padding:5px 9px; font-size:12px; cursor:pointer; }}
  button:hover, button.active {{ background:#2563eb; border-color:#3b82f6; }}
  .legend {{ margin-top:8px; font-size:11px; color:#9ba3b0; display:flex; gap:10px; align-items:center; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }}
</style>
</head>
<body>
<div class="panel">
  <h2>Indian Bus U-Turn Hotspots</h2>
  <p>Empirically discovered mid-route 145°–180° bus U-turns from live GPS archives (excluding termini). Size = frequency, Color = median delay.</p>
  <div class="btns" id="cityBtns"></div>
  <div class="legend">
    <span><span class="dot" style="background:#22c55e"></span>&lt;45s</span>
    <span><span class="dot" style="background:#f59e0b"></span>45–80s</span>
    <span><span class="dot" style="background:#ef4444"></span>&gt;80s</span>
  </div>
</div>
<div id="map"></div>
<script>
const DATA = {json.dumps({"type": "FeatureCollection", "features": all_features})};
const CENTERS = {json.dumps(city_centers)};
const map = L.map('map').setView([13.06, 80.23], 12);
const esriStreets = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Tiles &copy; Esri | Reclaim City ETA', maxZoom: 19
}}).addTo(map);
const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Tiles &copy; Esri World Imagery', maxZoom: 19
}});
const esriGray = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Tiles &copy; Esri Gray Canvas', maxZoom: 16
}});
L.control.layers({{'Esri Streets': esriStreets, 'Esri Satellite': esriSat, 'Esri Light Gray': esriGray}}, null, {{position: 'topright'}}).addTo(map);

function colorFor(s) {{ return s >= 80 ? '#ef4444' : (s >= 45 ? '#f59e0b' : '#22c55e'); }}

const layer = L.geoJSON(DATA, {{
  pointToLayer: (f, ll) => {{
    const p = f.properties;
    const r = Math.min(14, Math.max(4, Math.sqrt(p.n_uturns) * 1.1));
    return L.circleMarker(ll, {{
      radius: r, fillColor: colorFor(p.median_dur_s), color: '#fff',
      weight: 1, opacity: 0.9, fillOpacity: 0.78
    }});
  }},
  onEachFeature: (f, l) => {{
    const p = f.properties;
    l.bindPopup(`<b>${{p.city.toUpperCase()}} U-Turn Hotspot</b><br/>
      Events: <b>${{p.n_uturns}}</b> (${{p.n_buses}} buses, ${{p.n_routes}} routes)<br/>
      Median delay: <b>${{p.median_dur_s}}s</b> | P90: <b>${{p.p90_dur_s}}s</b><br/>
      Routes: <code>${{(p.top_routes || []).join(', ')}}</code>`);
  }}
}}).addTo(map);

const counts = {{}};
DATA.features.forEach(f => {{ const c = f.properties.city; counts[c] = (counts[c] || 0) + 1; }});
const btns = document.getElementById('cityBtns');
Object.entries(CENTERS).forEach(([c, [la, lo, z]]) => {{
  const b = document.createElement('button');
  b.textContent = `${{c.toUpperCase()}} (${{counts[c] || 0}})`;
  b.onclick = () => map.setView([la, lo], z);
  btns.appendChild(b);
}});
</script>
</body>
</html>
"""
    out_html.write_text(html)
    print(f"[uturns] Wrote interactive map ({len(all_features):,} U-turn hotspots) -> {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/projects/ETA/data")
    ap.add_argument("--cities", default="mtc,bmtc,mumbai,delhi,ncr")
    ap.add_argument("--max-days", type=int, default=4)
    a = ap.parse_args()

    cities = [c.strip() for c in a.cities.split(",")]
    jobs = [(c, a.data_root, a.max_days) for c in cities]
    all_feats = []
    with ProcessPoolExecutor(max_workers=min(5, len(jobs))) as ex:
        for city, feats in ex.map(detect_city_uturns, jobs):
            all_feats.extend(feats)

    root = Path(a.data_root)
    (root / "uturns" / "all_cities_uturns.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": all_feats}, indent=2)
    )
    build_interactive_map(all_feats, root.parent / "maps" / "indian_bus_uturns_map.html")


if __name__ == "__main__":
    main()
