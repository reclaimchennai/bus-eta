"""Every placed traffic alert, city by city, for the traffic API's alerts/history endpoint.

  python -m pipeline.publish_history          # run by pipeline.collect in its slow pass; pushes at most hourly, only on change

Reads the full-history GeoJSON the builds already write (out/map/incidents_all.geojson for Hyderabad + Cyberabad,
out/chennai/, out/bengaluru/ and out/chandigarh/incidents_all.geojson), drops the view counts, and pushes each city's file to
<map server>:projects/maps-site/traffic/history/<city>.json (tmp file, then mv), served at
maps.reclaim.city/traffic/history/<city>.json and read by traffic-api (api.reclaim.city/traffic/v1/<city>/alerts/history).
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {"hyderabad": ROOT / "out/map/incidents_all.geojson",
         "chennai": ROOT / "out/chennai/incidents_all.geojson",
         "bengaluru": ROOT / "out/bengaluru/incidents_all.geojson",
         "chandigarh": ROOT / "out/chandigarh/incidents_all.geojson"}
STATE = ROOT / "out/.history_pushed.json"
HOST, DEST = os.environ.get("MAP_SERVER", "user@map-server"), "projects/maps-site/traffic/history"   # ssh target that serves the bus maps
EVERY = 3600


def main():
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if time.time() - st.get("_at", 0) < EVERY and "--now" not in sys.argv:
        return 0
    st["_at"] = time.time()
    notes = []
    for city, fp in FILES.items():
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        for f in d.get("features", []):
            f["properties"].pop("views", None)
        d = {"type": "FeatureCollection", "city": city, "generated": int(time.time()), "features": d.get("features", [])}
        core = json.dumps(d["features"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        h = hashlib.sha1(core.encode()).hexdigest()
        if st.get(city) == h:
            continue
        body = json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode()
        if os.environ.get("TWTRX_NO_PUSH") == "1":
            notes.append(f"{city} not pushed (dry run)"); continue
        cmd = f"mkdir -p {DEST} && cat > {DEST}/.{city}.json.tmp && mv {DEST}/.{city}.json.tmp {DEST}/{city}.json"
        try:
            r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd], input=body,
                               capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            notes.append(f"{city} push timed out"); continue
        if r.returncode == 0:
            st[city] = h; notes.append(f"{city} {len(d['features'])} features {len(body) // 1024} KB")
        else:
            notes.append(f"{city} push failed: {r.stderr.decode(errors='replace').strip()[:150]}")
    STATE.write_text(json.dumps(st))
    if notes:
        print("history:", ", ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
