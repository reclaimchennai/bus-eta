"""Build all outputs from the event store.

  python -m pipeline.build --historic      # (re)import cybtraffic_road_incidents.json
  python -m pipeline.build                 # just regenerate outputs from out/events.jsonl

Outputs (out/):
  events.jsonl               master store, one event per line (all sources)
  eta/incidents.jsonl|.parquet   v4 ETA contract (point events, UTC epoch, not snapped)
  map/incidents_live.json    TGSRTC map layer: active events only, GeoJSON, 5-dp coords
  map/incidents_all.geojson  every geocoded event (history map)
  priors/hotspots.csv        place x type x hour-of-day counts (ETA priors)
  priors/unresolved.csv      place phrases the index could not find -> add to data/aliases.csv
"""
import argparse, csv, gzip, html, json, shutil, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .classify import IST

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STORE = OUT / "events.jsonl"
SITE_TRAFFIC = Path.home() / "Desktop/projects/maps-site/tgsrtc/traffic"

def load_store():
    ev = {}
    if STORE.exists():
        for l in open(STORE, encoding="utf-8"):
            e = json.loads(l); ev[e["id"]] = e
    return ev

def save_store(ev):
    OUT.mkdir(exist_ok=True)
    rows = sorted(ev.values(), key=lambda e: e["t_start"] or 0, reverse=True)
    with open(STORE, "w", encoding="utf-8") as f:
        for e in rows: f.write(json.dumps(e, ensure_ascii=False) + "\n")

def import_historic(ev, builder):
    from .relevance import is_road_event
    src = json.load(open(ROOT / "cybtraffic_road_incidents.json"))
    for i, r in enumerate(src):
        eid = f"twitter:{r['id']}"
        if eid in ev or not is_road_event(r["text"]): continue
        posted = datetime.strptime(r["datetime_ist"], "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        ev[eid] = builder.build(r["text"], posted, "twitter", r["url"], r["id"])
        if i % 100 == 0:
            print(f"{i}/{len(src)}", flush=True); builder.save()
    builder.save()

def hydtp_posts():
    """@HYDTP road-traffic posts from the cached Sorsa pages in raw/hydtp/ (no requests)."""
    from .relevance import is_road_event
    seen = {}
    for f in sorted((ROOT / "raw/hydtp").glob("*_*.json")):
        for t in json.load(open(f))["response"].get("tweets") or []:
            if (t.get("user") or {}).get("username", "").lower() == "hydtp" and not t["full_text"].lstrip().startswith("@"):
                seen[t["id"]] = t
    rows = []
    for t in seen.values():
        x = html.unescape(t["full_text"])
        if not is_road_event(x): continue
        dt = datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S %z %Y").astimezone(IST)
        rows.append({"id": t["id"], "datetime_ist": dt.strftime("%Y-%m-%d %H:%M"), "text": x,
                     "views": t.get("view_count"), "url": f"https://x.com/HYDTP/status/{t['id']}"})
    rows.sort(key=lambda r: r["datetime_ist"], reverse=True)
    json.dump(rows, open(ROOT / "hydtp_road_incidents.json", "w"), ensure_ascii=False, indent=1)
    return rows

def import_hydtp(ev, builder):
    rows = hydtp_posts(); n = 0
    for r in rows:
        eid = f"hydtp_x:{r['id']}"
        if eid in ev: continue
        posted = datetime.strptime(r["datetime_ist"], "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        e = builder.build(r["text"], posted, "hydtp_x", r["url"], r["id"]); e["views"] = r["views"]
        ev[eid] = e; n += 1
        if n % 100 == 0: print(f"hydtp {n}/{len(rows)}", flush=True); builder.save()
    builder.save(); print(f"hydtp: {n} new events from {len(rows)} road posts")

def r5(line): return [[round(lo, 5), round(la, 5)] for la, lo in line]

OPTIONAL = ("vehicles", "operator", "time_precision", "note")   # only present on some events

def feature(e, full=False):
    if e["lat"] is None: return []
    props = {"id": e["id"], "type": e["type"], "severity": e["severity"], "start": e["t_start"],
             "end": e["t_end"] or e["t_expires"], "text_en": e["text"][:600], "src": e["src"],
             "route_ids": e["routes"], "stop_ids": e["stop_ids"], "place": e["place"], "direction": e["direction"],
             "status": e["status"], "kind": e["kind"], "agency": e.get("agency") or "Cyberabad Traffic Police"}
    props.update({k: e[k] for k in OPTIONAL if e.get(k)})
    if full:
        props.update({"t_start_ist": e["t_start_ist"] or (e["post_time_ist"] + " (start unknown)"),
                      "police_station": e["police_station"], "tags": e["tags"],
                      "views": e.get("views"), "source": e["source"]})
    fs = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(e["lon"], 5), round(e["lat"], 5)]},
           "properties": props}]
    if e.get("lines") and len(e["lines"]) > 1:     # several routed stretches (ring / procession route)
        fs.append({"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": [r5(l) for l in e["lines"]]},
                   "properties": {**props, "id": e["id"] + ":line"}})
    elif e["line"]:
        fs.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": r5(e["line"])},
                   "properties": {**props, "id": e["id"] + ":line"}})
    return fs

def is_live(e, now):
    if e["status"] != "active": return False
    if e["kind"] == "standing":                 # long-running arrangement (one-way system, works): live until ended
        return (e["t_start"] or 0) <= now + 6 * 3600 and (e["t_end"] is None or e["t_end"] > now)
    if not e["t_start"] or e["t_start"] > now + 6 * 3600: return False
    end = e["t_end"] or e["t_expires"]
    if end: return end > now
    return now - e["t_start"] < 7 * 86400      # open-ended (e.g. U-turn change): show for a week

def write_outputs(ev, publish=False):
    now = int(time.time())
    for d in ("eta", "map", "priors"): (OUT / d).mkdir(parents=True, exist_ok=True)
    rows = sorted(ev.values(), key=lambda e: e["t_start"] or 0)

    # ---- ETA contract
    eta = []
    for e in rows:
        if e["lat"] is None: continue
        ln = e["line"]
        eta.append({"id": e["id"], "type": e["type"], "sidecar_type": e["sidecar_type"], "severity": e["severity"],
                    "speed_multiplier": e["speed_multiplier"], "status": e["status"], "kind": e["kind"],
                    "t_start": e["t_start"], "t_end": e["t_end"], "t_expires_default": e["t_expires"],
                    "default_duration_min": e["default_duration_min"], "t_start_ist": e["t_start_ist"],
                    "lat": e["lat"], "lon": e["lon"],
                    "lat2": ln[-1][0] if ln else None, "lon2": ln[-1][1] if ln else None, "line": ln,
                    "direction": e["direction"], "police_station": e["police_station"], "place": e["place"],
                    "source": e["source"], "agency": e.get("agency") or "Cyberabad Traffic Police",
                    "post_time": e["post_time"], "lines": e.get("lines"),
                    **{k: e.get(k) for k in OPTIONAL}})
    with open(OUT / "eta/incidents.jsonl", "w") as f:
        for r in eta: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        t = pa.Table.from_pylist([{**r, "line": json.dumps(r["line"]) if r["line"] else None,
                                   "lines": json.dumps(r["lines"]) if r["lines"] else None} for r in eta])
        pq.write_table(t, OUT / "eta/incidents.parquet")
    except ImportError:
        pass

    # ---- map: live (active now) and all
    live = [f for e in rows if is_live(e, now) for f in feature(e)]
    lj = {"type": "FeatureCollection", "generated": now, "features": live}
    (OUT / "map/incidents_live.json").write_text(json.dumps(lj, ensure_ascii=False, separators=(",", ":")))
    allf = [f for e in rows for f in feature(e, full=True)]
    (OUT / "map/incidents_all.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": allf},
                                                              ensure_ascii=False, separators=(",", ":")))
    if publish:
        SITE_TRAFFIC.mkdir(exist_ok=True)
        shutil.copy(OUT / "map/incidents_live.json", SITE_TRAFFIC / "incidents.json")

    # ---- priors
    hot = Counter(); hour = defaultdict(Counter)
    for e in rows:
        if e["place"] is None or e["t_start"] is None or e["kind"] == "standing": continue
        k = (e["place"], round(e["lat"], 4), round(e["lon"], 4), e["type"])
        hot[k] += 1
        h = datetime.fromtimestamp(e["t_start"], IST).hour
        hour[k][h] += 1
    with open(OUT / "priors/hotspots.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["place", "lat", "lon", "type", "count"] + [f"h{h:02d}" for h in range(24)])
        for k, n in hot.most_common():
            w.writerow(list(k) + [n] + [hour[k][h] for h in range(24)])
    geo = sum(1 for e in rows if e["lat"] is not None); lines = sum(1 for e in rows if e["line"])
    gz = len(gzip.compress(json.dumps(lj).encode()))
    from . import make_map
    make_map.main()
    print(f"events {len(rows)} | geocoded {geo} | road segments {lines} | live now {len(live)} features ({gz/1024:.1f} KB gz)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--historic", action="store_true"); ap.add_argument("--publish", action="store_true")
    ap.add_argument("--hydtp", action="store_true", help="(re)import @HYDTP posts from raw/hydtp/")
    ap.add_argument("--no-web", action="store_true", help="no Nominatim/OSRM calls")
    ap.add_argument("--rebuild", action="store_true", help="re-geocode every event in the store")
    a = ap.parse_args()
    old = load_store()
    ev = {} if a.rebuild else old
    if a.historic or a.rebuild or a.hydtp:
        from .events import Builder
        web = False
        if not a.no_web:   # only phrases seen at least twice go to Nominatim (polite + useful)
            from collections import Counter as C
            from .extract import extract
            from .gazetteer import norm
            texts = [r["text"] for r in json.load(open(ROOT / "cybtraffic_road_incidents.json"))] + \
                    [r["text"] for r in hydtp_posts()]
            cnt = C(norm(p) for t in texts for m in extract(t)["mentions"] for p in m["places"])
            web = {p for p, n in cnt.items() if n >= 2}
        b = Builder(web=web)
        if a.historic or a.rebuild: import_historic(ev, b)
        if a.hydtp or a.rebuild: import_hydtp(ev, b)
        if a.rebuild:     # re-process non-X events (WhatsApp) from their stored text and post time
            for e in old.values():
                # everything not re-imported above (WhatsApp, e-challan, X posts that came live via Nitter),
                # re-processed from its stored text and post time; curated entries are re-applied below
                if e["id"] not in ev and e["source"] != "manual":
                    n = b.build(e["text"], datetime.fromtimestamp(e["post_time"], IST), e["source"], e.get("src"),
                                region=e.get("region"))
                    n["id"] = e["id"]
                    if e.get("views") is not None: n["views"] = e["views"]
                    if e.get("t_end") and not n["t_end"]: n["t_end"] = e["t_end"]; n["t_expires"] = e["t_end"]
                    ev[e["id"]] = n
            b.save()
            from . import manual            # hand-curated events are rebuilt from data/manual_events.json
            manual.apply(ev, b)
        (OUT / "priors").mkdir(parents=True, exist_ok=True)
        with open(OUT / "priors/unresolved.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["phrase", "count"])
            for p, n in sorted(b.unresolved.items(), key=lambda x: -x[1]): w.writerow([p, n])
        print("nominatim calls", b.gaz.nom_calls)
    save_store(ev); write_outputs(ev, a.publish)
