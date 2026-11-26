"""Refresh ref/: the TGSRTC, MTC and BMTC map data the pipeline matches stops and routes against, from the live sites.

  python -m pipeline.refdata          # mss cron, daily

Fetches each file to a temp name, checks it loads as the pipeline reads it, then swaps it in (os.replace), so a bad
or half download never replaces a good copy. ref/ is used whenever it exists (pipeline.gazetteer._ref); the Mac
without ref/ keeps reading its maps-site and mtc-bus-map checkouts.
"""
import json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {"tgsrtc": {"stops.json": "https://maps.reclaim.city/tgsrtc/data/stops.json",
                    "services.json": "https://maps.reclaim.city/tgsrtc/data/services.json"},
         "mtc": {"route_stops.json": "https://mtc.reclaimchennai.city/data/route_stops.json",
                 "bus_stops.geojson": "https://mtc.reclaimchennai.city/data/bus_stops.geojson"},
         "bmtc": {"route_stops.json": "https://maps.reclaimchennai.city/bengaluru/data/route_stops.json",
                  "bus_stops.geojson": "https://maps.reclaimchennai.city/bengaluru/data/bus_stops.geojson"}}

def check(name, d):
    if name == "stops.json":
        sid, (n, la, lo) = next(iter(d.items())); assert 17 < la < 18.5 and len(d) > 1000
    elif name == "services.json":
        s = next(iter(d["services"].values())); assert isinstance(s[6], list) and len(d["services"]) > 100
    elif name == "route_stops.json":
        r = next(iter(d.values())); sid, n, la, lo = r["d"][0]["s"][0]; assert r["sn"] and 12 < la < 14 and len(d) > 500   # Chennai and Bengaluru
    elif name == "bus_stops.geojson":
        f = d["features"][0]; assert f["properties"]["Stop Name"] and len(d["features"]) > 1000

def main():
    bad = 0
    for sub, files in FILES.items():
        dest = ROOT / "ref" / sub; dest.mkdir(parents=True, exist_ok=True)
        for name, url in files.items():
            tmp = dest / f".{name}.tmp"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "twtrx-refdata/1"})
                with urllib.request.urlopen(req, timeout=120) as r: body = r.read()
                check(name, json.loads(body))
                tmp.write_bytes(body); os.replace(tmp, dest / name)
                print(f"{sub}/{name}: {len(body)} bytes")
            except Exception as e:
                bad += 1; print(f"{sub}/{name}: kept the old copy ({type(e).__name__}: {e})")
                tmp.unlink(missing_ok=True)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
