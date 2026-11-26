#!/usr/bin/env python3
"""Fetch fresh OpenStreetMap road infrastructure via Overpass API for all 5 Indian cities:
  - Traffic signals (highway=traffic_signals)
  - Flyovers / Bridges (highway~trunk|primary|secondary|tertiary + bridge=yes) — captures NEW Chennai flyovers!
  - Railway level crossings (railway=level_crossing)
  - Roundabouts (junction=roundabout)
  - Traffic calming / speed breakers (traffic_calming)

Caches raw JSON per city under ETA/data/osm/{city}_infra.json and builds a fast spatial
KD-Tree lookup so any bus route segment (stopA -> stopB) gets exact counts of signals,
flyovers, rail crossings, and roundabouts along its corridor!
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]  # repo root

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Tight bounding boxes (south, west, north, east) covering each city's bus network
CITY_BBOX = {
    "mtc":    (12.78, 79.95, 13.30, 80.36),  # Greater Chennai (incl. new flyovers on OMR/GST/Mount Rd/Maduravoyal)
    "bmtc":   (12.75, 77.38, 13.20, 77.82),  # Bengaluru
    "mumbai": (18.88, 72.76, 19.32, 73.06),  # Greater Mumbai + Thane
    "delhi":     (28.42, 76.85, 28.88, 77.36),  # Delhi NCT
    "ncr":       (28.35, 76.90, 28.56, 77.16),  # Gurugram
    "hyderabad": (17.22, 78.22, 17.58, 78.68),  # Greater Hyderabad (HITEC City / Gachibowli / Cyberabad)
}


def query_overpass(bbox: tuple[float, float, float, float]) -> dict:
    s, w, n, e = bbox
    q = f"""
    [out:json][timeout:90];
    (
      node["highway"="traffic_signals"]({s},{w},{n},{e});
      node["railway"="level_crossing"]({s},{w},{n},{e});
      node["traffic_calming"]({s},{w},{n},{e});
      way["junction"="roundabout"]({s},{w},{n},{e});
      way["bridge"="yes"]["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});
    );
    out center tags;
    """
    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=120, headers={"User-Agent": "ReclaimCity-ETA-Research/1.0"})
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"Overpass query failed: {last_err}")


def fetch_city_osm(city: str, out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city}_osm_infra.json"
    if out_path.exists() and not force:
        print(f"[osm:{city}] Reusing cached {out_path}")
        return out_path

    print(f"[osm:{city}] Querying Overpass API {CITY_BBOX[city]} ...", end=" ", flush=True)
    raw = query_overpass(CITY_BBOX[city])
    elems = raw.get("elements", [])
    buckets: dict[str, list[tuple[float, float]]] = {
        "signals": [],
        "rail_crossings": [],
        "speed_bumps": [],
        "roundabouts": [],
        "flyovers": [],
    }
    for el in elems:
        tags = el.get("tags") or {}
        lat = el.get("lat") if "lat" in el else (el.get("center") or {}).get("lat")
        lon = el.get("lon") if "lon" in el else (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        pt = (round(float(lat), 5), round(float(lon), 5))
        if tags.get("highway") == "traffic_signals":
            buckets["signals"].append(pt)
        elif tags.get("railway") == "level_crossing":
            buckets["rail_crossings"].append(pt)
        elif "traffic_calming" in tags:
            buckets["speed_bumps"].append(pt)
        elif tags.get("junction") == "roundabout":
            buckets["roundabouts"].append(pt)
        elif tags.get("bridge") == "yes":
            buckets["flyovers"].append(pt)

    out_path.write_text(json.dumps(buckets))
    counts = {k: len(v) for k, v in buckets.items()}
    print(f"DONE: {counts}")
    return out_path


class CityOSMIndex:
    """Fast KD-Tree spatial index for querying OSM infrastructure near route stop chords."""

    def __init__(self, osm_json_path: Path):
        data = json.loads(osm_json_path.read_text()) if osm_json_path.exists() else {}
        self.trees: dict[str, cKDTree | None] = {}
        self.K = 111320.0
        for kind in ("signals", "rail_crossings", "speed_bumps", "roundabouts", "flyovers"):
            pts = data.get(kind) or []
            if pts:
                arr = np.asarray(pts, dtype=np.float64)
                # Convert to approximate local metres for Euclidean KD-Tree queries
                mean_lat = float(np.mean(arr[:, 0]))
                cx = math.cos(math.radians(mean_lat)) * self.K
                self._cx = cx
                xy = np.column_stack([arr[:, 0] * self.K, arr[:, 1] * cx])
                self.trees[kind] = cKDTree(xy)
            else:
                self.trees[kind] = None
        if not hasattr(self, "_cx"):
            self._cx = 100000.0

    def count_near_segment(self, lat1: float, lon1: float, lat2: float, lon2: float, radius_m: float = 65.0) -> tuple[int, int, int, int]:
        """Count (signals, flyovers, rail_crossings, roundabouts) within radius_m of segment midpoint/endpoints."""
        pts = np.array([
            [lat1 * self.K, lon1 * self._cx],
            [0.5 * (lat1 + lat2) * self.K, 0.5 * (lon1 + lon2) * self._cx],
            [lat2 * self.K, lon2 * self._cx],
        ])
        out = []
        for kind in ("signals", "flyovers", "rail_crossings", "roundabouts"):
            tr = self.trees.get(kind)
            if tr is None:
                out.append(0)
            else:
                hits = set()
                for idx_list in tr.query_ball_point(pts, r=radius_m):
                    hits.update(idx_list)
                out.append(len(hits))
        return (out[0], out[1], out[2], out[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data/osm"))
    ap.add_argument("--cities", default="mtc,bmtc,mumbai,delhi,ncr")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for c in [x.strip() for x in a.cities.split(",")]:
        try:
            fetch_city_osm(c, Path(a.out_dir), force=a.force)
            time.sleep(2.0)
        except Exception as exc:
            print(f"[osm:{c}] ERROR: {exc}")


if __name__ == "__main__":
    main()
