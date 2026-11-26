"""Bengaluru Traffic Police on X -> live incidents on the BMTC map (maps.reclaim.city/bmtc, maps.reclaimchennai.city/bengaluru).

  python -m pipeline.bengaluru              # build new posts, write out/bengaluru/incidents_live.json, publish if changed
  python -m pipeline.bengaluru --no-publish
  python -m pipeline.bengaluru --rebuild    # rebuild every event from the posts (keeps the Nominatim cache)

Accounts: data/blr/btp_accounts.csv (scripts/blr_accounts.py): @blrcitytraffic, the Joint CP, DCPs, ACPs and the
station accounts. The Nitter poller reads @blrcitytraffic's timeline and, in turn, searches of the others' own posts
(no replies, no reposts: a repost repeats a post already collected from its source); pipeline.x_live appends them to
out/bengaluru/btp.jsonl with a road-update flag. History: raw/nitter/blr_backfill.jsonl (scripts/nitter_search_pages.py
over the same searches). Only road updates are built. Places resolve in the Bengaluru gazetteer (data/blr: OSM cut,
BTP's junction names from the congestion dashboard, BMTC stop names, aliases, Nominatim within Bengaluru); a station
account's post is placed near that station's area (data/blr/station_areas.json) unless the text names another
station. Stop ids and route numbers are the BMTC map's (maps-site/bengaluru/data/route_stops.json).
Output has the TGSRTC map's format (assets/busmap/incidents.js). Store: data/blr/events.json.
Hand-curated Bengaluru events ("region": "blr" in data/manual_events.json, see pipeline.manual) are applied when
that file changes or one of them is missing from the store (e.g. the dated windows of an advisory whose dates are
only in its images).
Published to raw:projects/maps-site/bengaluru/live/incidents.json (tmp file, then mv): every run while anything is
live (the layer hides a file over 30 min old), else when it changes.
"""
import argparse, csv, json, os, subprocess, time
from datetime import datetime
from pathlib import Path
from .classify import IST
from .x_live import blr_update
from . import build

ROOT = Path(__file__).resolve().parent.parent
POSTS = [ROOT / "raw/nitter/blr_backfill.jsonl", ROOT / "out/bengaluru/btp.jsonl"]
STORE = ROOT / "data/blr/events.json"
OUT = ROOT / "out/bengaluru"
HOST = os.environ.get("MAP_SERVER", "user@map-server")   # ssh target that serves the bus maps
DEST = "projects/maps-site/bengaluru/live"
SOURCE = "btp_x"


def is_update(p):
    return p.get("update") if "update" in p else blr_update(p["text"])


def posts():
    seen = {}
    for fp in POSTS:
        if fp.exists():
            for ln in open(fp, encoding="utf-8"):
                if ln.strip():
                    p = json.loads(ln); seen[str(p["id"])] = p
    return sorted(seen.values(), key=lambda p: p["t"])


def hints():
    """account (lower case) -> (lat, lon) of its station's area, for station accounts only."""
    areas = json.loads((ROOT / "data/blr/station_areas.json").read_text()) if (ROOT / "data/blr/station_areas.json").exists() else {}
    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    by = {norm(k): v for k, v in areas.items()}
    alias = {"chickpet": "chamarajpet"}          # BTP's station list says Chickpet, its alerts Chamarajpet (the account: Chamarajapete)
    out = {}
    acc = ROOT / "data/blr/btp_accounts.csv"
    if acc.exists():
        for r in csv.DictReader(open(acc, encoding="utf-8")):
            if r["role"] != "station":
                continue
            k = norm(r["unit"]); k = alias.get(k, k)
            a = by.get(k) or next((v for kk, v in by.items() if kk.startswith(k[:6])), None)
            if a:
                out[r["handle"].lower()] = (a["lat"], a["lon"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-publish", action="store_true"); ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    ev = {} if a.rebuild or not STORE.exists() else json.loads(STORE.read_text())
    new = [p for p in posts() if is_update(p) and f"{SOURCE}:{p['id']}" not in ev]
    if new:
        from .events import Builder
        b = Builder(web=True, city="blr")
        hs = hints()
        for p in new:
            e = b.build(p["text"], datetime.fromtimestamp(p["t"], IST), SOURCE, p.get("url"), p["id"], region="blr",
                        hint=hs.get((p.get("user") or "").lower()))
            e["account"] = p.get("user")
            ev[e["id"]] = e
        b.save()
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(ev, ensure_ascii=False))
    from . import manual
    mine = manual.entries("blr")
    mstamp = OUT / ".manual_applied"
    mkey = json.dumps(mine, sort_keys=True)
    if mine and (any(not m.get("patch") and m["id"] not in ev for m in mine)
                 or not mstamp.exists() or mstamp.read_text() != mkey):
        from .events import Builder
        mb = Builder(web=False, city="blr")
        n_man = manual.apply(ev, mb, region="blr")
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(ev, ensure_ascii=False))
        OUT.mkdir(parents=True, exist_ok=True)
        mstamp.write_text(mkey)
        print(f"bengaluru: {n_man} manual entries applied")
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
    msg = f"bengaluru: {len(new)} new, {len(ev)} events, {sum(e['lat'] is not None for e in ev.values())} placed, {n_live} live"
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
