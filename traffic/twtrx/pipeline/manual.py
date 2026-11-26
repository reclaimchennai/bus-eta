"""Hand-curated events and corrections: data/manual_events.json.

  python -m pipeline.manual          # apply to the store and refresh outputs

Each entry is either
  * a patch ("patch": true, "id" of an existing event): override times, point, type, kind, vehicles, ...
    e.g. GPS-confirmed start/end for a bus breakdown the police posted 20 min after it cleared; or
  * a new event ("text", "posted_ist", ...), built through the normal pipeline, then overridden.
region: "hyd" (default) or "blr"; each city's pipeline applies only its own entries
(pipeline.build: hyd, pipeline.bengaluru: blr).
Fields: t_start_ist / t_end_ist ("YYYY-MM-DD HH:MM" IST, or null = not known; never guessed),
point [lat, lon], ring [place, ...] (+ loop) routed on roads, paths [[[lat, lon], ...], ...] (each a list of
waypoints routed on roads: one stretch per road, when place names are not reliable), type, kind ("standing" = long-running
arrangement with no end), vehicles [registration as on the map], operator, direction, place, time_precision.
Rebuilt automatically by `build --rebuild`.
"""
import json
from datetime import datetime
from pathlib import Path
from .classify import IST, TYPES
from .gazetteer import hav
from . import route

FILE = Path(__file__).resolve().parent.parent / "data/manual_events.json"
TYPE_INFO = {t: (s, sev, d, sm) for t, _, s, sev, d, sm in TYPES}

def _ts(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=IST).timestamp()) if s else None

def _override(e, m, builder):
    if "type" in m and m["type"] in TYPE_INFO:
        s, sev, d, sm = TYPE_INFO[m["type"]]
        e.update({"type": m["type"], "sidecar_type": s, "severity": sev, "default_duration_min": d, "speed_multiplier": sm})
        if m["type"] not in e["tags"]: e["tags"] = [m["type"]] + e["tags"]
    for k in ("kind", "direction", "place", "vehicles", "operator", "time_precision", "src"):
        if k in m: e[k] = m[k]
    if "t_start_ist" in m:
        e["t_start"] = _ts(m["t_start_ist"]); e["t_start_ist"] = m["t_start_ist"]
    if "t_end_ist" in m:
        e["t_end"] = _ts(m["t_end_ist"])
    if e.get("kind") == "standing":
        e["t_expires"] = e["t_end"]                 # no default expiry for a standing arrangement
    elif e["t_end"]:
        e["t_expires"] = e["t_end"]
    elif e["t_start"] and e["default_duration_min"]:
        e["t_expires"] = e["t_start"] + 60 * e["default_duration_min"]
    if m.get("ring"):
        pts = []
        for p in m["ring"]:
            r = builder.gaz.resolve(p, allow_web=False, region=m.get("region", "hyd"))
            if r: pts.append((r["lat"], r["lon"]))
        lines = route.road_path(pts, max_km=6, loop=m.get("loop", False))
        e["line"] = lines[0] if lines else None
        e["lines"] = lines if len(lines) > 1 else None
    if m.get("paths"):
        lines = [l for wp in m["paths"] for l in route.road_path([tuple(x) for x in wp], max_km=6)]
        e["line"] = lines[0] if lines else None
        e["lines"] = lines if len(lines) > 1 else None
    if m.get("point"):
        e["lat"], e["lon"] = m["point"]
        e["geo_src"] = "manual"
    pts = [p for l in (e.get("lines") or ([e["line"]] if e.get("line") else [])) for p in l] or [(e["lat"], e["lon"])]
    if e["lat"] is not None:
        from .events import densify
        lines = e.get("lines") or ([e["line"]] if e.get("line") else [])
        geo = [p for l in lines for p in densify(l)] or [(e["lat"], e["lon"])]
        e["stop_ids"] = builder.stops.near(geo)
        e["routes"] = builder.stops.routes_for(e["stop_ids"] or e.get("near_stop_ids") or [])
    e["curated"] = m.get("why") or "manual entry"
    return e

def entries(region="hyd"):
    return [m for m in json.load(open(FILE)) if m.get("region", "hyd") == region] if FILE.exists() else []

def apply(ev, builder, region="hyd"):
    n = 0
    for m in entries(region):
        if m.get("patch"):
            if m["id"] not in ev:
                print("manual patch: event not in store yet:", m["id"]); continue
            _override(ev[m["id"]], m, builder)
        else:
            posted = datetime.strptime(m["posted_ist"], "%Y-%m-%d %H:%M").replace(tzinfo=IST)
            e = builder.build(m["text"], posted, m.get("source", "manual"), m.get("src"), region=m.get("region"))
            e["id"] = m["id"]
            ev[m["id"]] = _override(e, m, builder)
        n += 1
    builder.save()
    route.save()
    return n

if __name__ == "__main__":
    from . import build
    from .events import Builder
    ev = build.load_store(); b = Builder(web=False)
    print("applied", apply(ev, b), "manual entries")
    build.save_store(ev); build.write_outputs(ev)
