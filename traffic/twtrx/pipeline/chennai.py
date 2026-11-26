"""Greater Chennai Traffic Police (@ChennaiTraffic on X) -> live incidents on the MTC map (mtc.reclaimchennai.city).

  python -m pipeline.chennai              # build new posts, write out/chennai/incidents_live.json, publish if changed
  python -m pipeline.chennai --no-publish
  python -m pipeline.chennai --rebuild    # rebuild every event from the posts (keeps the Nominatim cache)

Posts: out/chennai/chennaitraffic.jsonl (live, forwarded by pipeline.x_live from the Nitter poller) and
raw/nitter/chennaitraffic_backfill.jsonl (history: a Nitter search for its road posts, scripts/nitter_search_pages.py).
Most of the account's posts are road-safety messages; only road updates are built (x_live's "update" flag, or the
same test for backfilled posts). Places resolve in the Chennai gazetteer (data/chn: OSM cut, MTC stop names, aliases,
Nominatim within Chennai); stop ids and route numbers are the MTC map's (mtc-bus-map web/data/route_stops.json).
The output has the TGSRTC map's format (assets/busmap/incidents.js reads both). Store: data/chn/events.json.
Published to raw:projects/maps-site/mapsv2/live/incidents.json (tmp file, then mv) when it changes.
"""
import argparse, json, os, subprocess, time
from datetime import datetime
from pathlib import Path
from .classify import IST
from .x_live import CHENNAI_UPDATE, CHENNAI_NOT
from . import build

ROOT = Path(__file__).resolve().parent.parent
POSTS = [ROOT / "raw/nitter/chennaitraffic_backfill.jsonl", ROOT / "out/chennai/chennaitraffic.jsonl"]
STORE = ROOT / "data/chn/events.json"
OUT = ROOT / "out/chennai"
HOST = os.environ.get("MAP_SERVER", "user@map-server")   # ssh target that serves the bus maps
DEST = "projects/maps-site/mapsv2/live"
SOURCE = "chennai_x"

def is_update(p):
    return p.get("update") if "update" in p else bool(CHENNAI_UPDATE.search(p["text"])) and not CHENNAI_NOT.search(p["text"])

def posts():
    seen = {}
    for fp in POSTS:
        if fp.exists():
            for ln in open(fp):
                if ln.strip():
                    p = json.loads(ln); seen[str(p["id"])] = p
    return sorted(seen.values(), key=lambda p: p["t"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-publish", action="store_true"); ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    ev = {} if a.rebuild or not STORE.exists() else json.loads(STORE.read_text())
    new = [p for p in posts() if is_update(p) and f"{SOURCE}:{p['id']}" not in ev]
    if new:
        from .events import Builder
        b = Builder(web=True, city="chn")
        for p in new:
            e = b.build(p["text"], datetime.fromtimestamp(p["t"], IST), SOURCE, p.get("url"), p["id"], region="chn")
            ev[e["id"]] = e
        b.save()
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(ev, ensure_ascii=False))
    now = int(time.time())
    rows = sorted(ev.values(), key=lambda e: e["t_start"] or 0)
    live = {"type": "FeatureCollection", "generated": now,
            "features": [f for e in rows if build.is_live(e, now) for f in build.feature(e)]}
    OUT.mkdir(parents=True, exist_ok=True)
    allf = [f for e in rows for f in build.feature(e, full=True)]
    (OUT / "incidents_all.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": allf}, ensure_ascii=False))
    body = json.dumps(live, ensure_ascii=False, separators=(",", ":"))
    (OUT / "incidents_live.json").write_text(body)
    n_live = sum(1 for f in live["features"] if f["geometry"]["type"] == "Point")
    msg = f"chennai: {len(new)} new, {len(ev)} events, {sum(e['lat'] is not None for e in ev.values())} placed, {n_live} live"
    # publish: every run while anything is live (the map hides a file over 30 min old), else only on change
    stamp = OUT / ".published"
    key = json.dumps(live["features"], sort_keys=True)
    if not a.no_publish and os.environ.get("TWTRX_NO_PUSH") != "1" and (n_live or not stamp.exists() or stamp.read_text() != key):
        cmd = f"mkdir -p {DEST} && cat > {DEST}/.incidents.json.tmp && mv {DEST}/.incidents.json.tmp {DEST}/incidents.json"
        try:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd],
                               input=body.encode(), capture_output=True, timeout=60)
            if r.returncode == 0: stamp.write_text(key); msg += ", published"
            else: msg += ", publish failed: " + r.stderr.decode(errors="replace").strip()[:200]
        except subprocess.TimeoutExpired:
            msg += ", publish timed out"
    print(msg)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
