"""Chandigarh Traffic Police (@trafficchd on X) -> live incidents on the Chandigarh tricity map (maps.reclaim.city/tricity).

  python -m pipeline.chandigarh              # build new posts, write out/chandigarh/incidents_live.json, publish if changed
  python -m pipeline.chandigarh --no-publish
  python -m pipeline.chandigarh --rebuild    # rebuild every event from the posts (keeps the Nominatim cache)

Posts: out/chandigarh/trafficchd.jsonl (forwarded by pipeline.x_live; the Nitter poller reads the account only in the
09:00 and 18:00 IST hours, as it posts a few times a day) and raw/nitter/trafficchd_backfill.jsonl (a Nitter search for
its own posts, back to 10 Nov 2026). Only road updates are built (x_live.chd_update: #TrafficAdvisory / #TrafficAlert,
diversions, processions, marathons; not awareness drives).

Chandigarh Traffic Police write junctions in shorthand, so the text is normalised first: "Sec-33/34 Lp" -> "Sector 33/34
Light Point" (sector numbers ascending, matching data/chd/aliases.csv, generated from the sector grid by
scripts/chd_aliases.py), "Mkt." -> "Market", "Tpt" -> "Transport". A route given as "from A to B via C, D, E" becomes a
📍 line A – C – D – E – B, which the extractor routes through every point. Places resolve in the Chandigarh gazetteer
(data/chd: aliases, OSM cut, CTU stop names, Nominatim within the tricity); stop ids and route numbers are the tricity
map's (maps-site tricity/data/stops.json, services.json). Output has the TGSRTC map's format (assets/busmap/incidents.js).
Store: data/chd/events.json. Published to SVR /data/rtc/web/tricity/live/incidents.json (the tricity map's live/ folder),
tmp file then mv: every run while anything is live (the layer hides a file over 30 min old), else when it changes.
"""
import argparse, json, os, re, subprocess, time
from datetime import datetime
from pathlib import Path
from .classify import IST
from .x_live import chd_update
from . import build

ROOT = Path(__file__).resolve().parent.parent
POSTS = [ROOT / "raw/nitter/trafficchd_backfill.jsonl", ROOT / "out/chandigarh/trafficchd.jsonl"]
STORE = ROOT / "data/chd/events.json"
OUT = ROOT / "out/chandigarh"
HOST = os.environ.get("PUBLISH_HOST", "user@rtc-map-server")      # the RTC maps' server (SVR), as pipeline.publish
DEST = "/data/rtc/web/tricity/live"
SOURCE = "chd_x"

SECTORS = re.compile(r"\bSec(?:tor)?\s*[-.]?\s*(\d{1,2}(?:\s*/\s*\d{1,2})*)\b", re.I)
VIA = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)\s*[(,]?\s*\bvia\b\s*(.+?)\s*\)?\s*(?:[.;]\s|\.$|\n|$)", re.I | re.S)


def normalise(text):
    t = SECTORS.sub(lambda m: "Sector " + "/".join(str(n) for n in sorted({int(x) for x in re.findall(r"\d+", m.group(1))})), text)
    t = re.sub(r"\b(?:L\.P\.?|Lp|LP)\b", "Light Point", t)
    t = re.sub(r"\bMkt\b\.?", "Market", t, flags=re.I)
    t = re.sub(r"\bTpt\b\.?", "Transport", t, flags=re.I)
    t = re.sub(r"\bGmch\b", "GMCH", t, flags=re.I)
    t = re.sub(r"\b([A-Za-z]{3,})-(\d{1,2})\b", r"\1 Sector \2", t)          # "Rajasthan Bhawan-33", "GMCH-32 Chowk"
    m = VIA.search(t)
    if m:
        a, b, via = m.group(1), m.group(2), m.group(3)
        stops = [x.strip(" .") for x in re.split(r",|\band\b|&", via) if x.strip(" .")]
        if 1 <= len(stops) <= 15:
            t += "\n📍 " + " – ".join([a.strip(), *stops, b.strip()])
    return t


def is_update(p):
    return p.get("update") if "update" in p else chd_update(p["text"])


def posts():
    seen = {}
    for fp in POSTS:
        if fp.exists():
            for ln in open(fp, encoding="utf-8"):
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
        b = Builder(web=True, city="chd")
        for p in new:
            e = b.build(normalise(p["text"]), datetime.fromtimestamp(p["t"], IST), SOURCE, p.get("url"), p["id"], region="chd")
            e["text"] = p["text"]                       # keep the police's own wording for display
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
    msg = f"chandigarh: {len(new)} new, {len(ev)} events, {sum(e['lat'] is not None for e in ev.values())} placed, {n_live} live"
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
