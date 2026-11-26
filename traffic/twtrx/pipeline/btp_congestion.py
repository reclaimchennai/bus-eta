"""BTP congestion alerts (data/blr/powerbi, pipeline.btp_powerbi) -> what the Bengaluru map, the gazetteer and the
v4 arrival model use.

  python -m pipeline.btp_congestion             # rebuild from the dump, publish to raw if changed
  python -m pipeline.btp_congestion --no-publish

The alerts are 15-minute snapshots (ASTraM) of every congested stretch: position of its head, length ("extent", m),
severity, BTP's grid cell, junction, police station, corridor; 1 Aug - 7 Oct 2026 in the current dump. Hours are IST
(the model stores wall time: alerts peak 6-8 pm). "how" is the hour of the week, 0 = Monday 00:00-00:59 IST.

Writes
  data/blr/junctions.csv            BTP's junction names (spaced: "KogilluCrossJunc" -> "Kogillu Cross Junction") at the
                                    median of their alerts; the Bengaluru gazetteer ranks them below hand aliases
  data/blr/station_areas.json       traffic police station -> median alert position (a post's location hint)
  out/bengaluru/congestion_typical.json   map: per BTP grid cell, how often it is congested in each hour of the week
                                    (p: % of that hour's snapshots with an alert in the cell) and its typical length
  out/bengaluru/eta_ctx/stop_congestion_how.json   v4 model input: per BMTC stop (map stop id), the highest p of the
                                    cells within STOP_R m, per hour of the week
  out/bengaluru/eta_ctx/corridor_speed_how.json    BTP's corridor travel speeds (Sep 2023 - Jul 2025): median km/h
                                    per corridor, direction and hour of the week, with the corridor's ends
The map file goes to raw:projects/maps-site/bengaluru/live/btp_congestion.json, the two model files to
raw:projects/eta-ml/ctx/blr/, each only when it changed (tmp file, then mv).
"""
import argparse, csv, gzip, json, math, os, re, statistics, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from .gazetteer import BMTC_DATA

ROOT = Path(__file__).resolve().parent.parent
PBI = ROOT / "data/blr/powerbi"
DATA = ROOT / "data/blr"
OUT = ROOT / "out/bengaluru"
HOST = os.environ.get("MAP_SERVER", "user@map-server")   # ssh target that serves the bus maps
MIN_JUNCTION = 20          # alerts before a junction name is trusted as a place
MIN_CELL = 30              # alerts before a cell goes on the map
STOP_R = 200               # m: a stop takes the congestion of cells this close
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(h))


def spaced(n):
    """BTP's junction ids -> words: 'KanakapuraRd-RingRdJunction' -> 'Kanakapura Road - Ring Road Junction'."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", n)
    s = re.sub(r"\s*([-/])\s*", r" \1 ", s)
    s = re.sub(r"\s*\(\s*", " (", re.sub(r"\s*\)\s*", ") ", s)).replace("&", " and ")
    s = re.sub(r"\bJunc\b\.?|\bJn\b\.?|\bJct\b\.?", "Junction", s)
    s = re.sub(r"\bRd\b\.?", "Road", s)
    s = re.sub(r"\bCir\b\.?", "Circle", s)
    return re.sub(r"\s+", " ", s).strip()


def how(ts):
    d = datetime.strptime(ts[:13], "%Y-%m-%d %H")
    return d.weekday() * 24 + d.hour


def load_alerts():
    return list(csv.DictReader(gzip.open(PBI / "congestion_alerts.csv.gz", "rt", encoding="utf-8")))


def junctions(rows):
    pts = defaultdict(list)
    for r in rows:
        j = r["junction_name"].strip()
        if j and j != "Not Assigned":
            pts[j].append((float(r["latitude"]), float(r["longitude"])))
    out = []
    for j, p in pts.items():
        if len(p) < MIN_JUNCTION:
            continue
        lat, lon = statistics.median(x for x, _ in p), statistics.median(y for _, y in p)
        out.append({"name": spaced(j), "btp_id": j, "lat": round(lat, 6), "lon": round(lon, 6), "alerts": len(p)})
    out.sort(key=lambda r: -r["alerts"])
    with open(DATA / "junctions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def station_areas(rows):
    pts = defaultdict(list)
    for r in rows:
        if r["traffic_police_station"]:
            pts[r["traffic_police_station"]].append((float(r["latitude"]), float(r["longitude"])))
    areas = {s: {"lat": round(statistics.median(x for x, _ in p), 5), "lon": round(statistics.median(y for _, y in p), 5),
                 "alerts": len(p)} for s, p in pts.items()}
    (DATA / "station_areas.json").write_text(json.dumps(areas, indent=1, ensure_ascii=False, sort_keys=True))
    return areas


def profile(rows):
    snaps = defaultdict(set)                       # how -> snapshot timestamps seen (the denominator)
    for r in rows:
        snaps[how(r["timestamp"])].add(r["timestamp"][:16])
    cells = defaultdict(lambda: {"pts": [], "hit": defaultdict(set), "ext": [], "names": Counter(), "roads": Counter(),
                                 "ps": Counter(), "cor": Counter(), "sev": Counter(), "rec": 0})
    for r in rows:
        c = cells[r["grid_id"] or f"{float(r['latitude']):.3f},{float(r['longitude']):.3f}"]
        c["pts"].append((float(r["latitude"]), float(r["longitude"])))
        c["hit"][how(r["timestamp"])].add(r["timestamp"][:16])
        if r["extent"]:
            c["ext"].append(int(float(r["extent"])))
        if r["junction_name"] not in ("", "Not Assigned"):
            c["names"][spaced(r["junction_name"])] += 1
        for k, f in (("roads", "road"), ("ps", "traffic_police_station"), ("cor", "corridor"), ("sev", "severity")):
            if r[f]:
                c[k][r[f]] += 1
        c["rec"] += r["recurring"] == "1"
    out = []
    for gid, c in cells.items():
        n = len(c["pts"])
        if n < MIN_CELL:
            continue
        p = [round(100 * len(c["hit"].get(h, ())) / len(snaps[h])) if snaps.get(h) else 0 for h in range(168)]
        name = (c["names"].most_common(1) or c["roads"].most_common(1) or [("", 0)])[0][0]
        out.append({"grid": gid, "lat": round(statistics.median(x for x, _ in c["pts"]), 5),
                    "lon": round(statistics.median(y for _, y in c["pts"]), 5), "name": name,
                    "ps": c["ps"].most_common(1)[0][0] if c["ps"] else "", "corridor": c["cor"].most_common(1)[0][0] if c["cor"] else "",
                    "alerts": n, "ext_m": int(statistics.median(c["ext"])) if c["ext"] else None,
                    "severe_pct": round(100 * (c["sev"]["Severe"] + c["sev"]["High"]) / n), "recurring_pct": round(100 * c["rec"] / n),
                    "p": p})
    out.sort(key=lambda c: -c["alerts"])
    days = sorted({r["timestamp"][:10] for r in rows})
    return out, days, sum(len(v) for v in snaps.values())


def stop_profile(cells):
    stops = json.load(open(BMTC_DATA / "bus_stops.geojson"))["features"]
    grid = defaultdict(list)
    for c in cells:
        grid[(int(c["lat"] * 500), int(c["lon"] * 500))].append(c)
    out = {}
    for f in stops:
        lon, lat = f["geometry"]["coordinates"][:2]
        sid = f["properties"].get("stop_id")
        near = [c for dy in (-1, 0, 1) for dx in (-1, 0, 1) for c in grid.get((int(lat * 500) + dy, int(lon * 500) + dx), ())
                if hav((lat, lon), (c["lat"], c["lon"])) <= STOP_R]
        if sid and near:
            p = [max(c["p"][h] for c in near) for h in range(168)]
            if max(p) >= 5:
                out[sid] = p
    return out


def corridor_speeds():
    corr = {r["Corridor Name"]: r for r in csv.DictReader(open(PBI / "corridors.csv", encoding="utf-8"))}
    sp = defaultdict(list)
    first, last = None, None
    for r in csv.DictReader(gzip.open(PBI / "travel_time.csv.gz", "rt", encoding="utf-8")):
        if not r["Speed_Kmph"]:
            continue
        sp[(r["Corridor Name"], r["direction"], how(r["Date Time"]))].append(float(r["Speed_Kmph"]))
        first = min(first or r["Date Time"], r["Date Time"]); last = max(last or r["Date Time"], r["Date Time"])
    out = {}
    for (name, d, h), v in sp.items():
        k = f"{name}|{d}"
        if k not in out:
            c = corr.get(name, {})
            out[k] = {"corridor": name, "direction": d, "from": c.get("Start Point"), "to": c.get("End Point"),
                      "km": float(c["Corridor Length"]) if c.get("Corridor Length") else None, "kmph": [None] * 168, "n": [0] * 168}
        out[k]["kmph"][h] = round(statistics.median(v), 1); out[k]["n"][h] = len(v)
    return {"period": [first, last], "how": "0 = Monday 00:00-00:59 IST",
            "note": "direction 1/2 as BTP numbers them; 1 is assumed to run from 'from' to 'to' (not stated by BTP)",
            "corridors": list(out.values())}


def push(body, dest, stamp):
    if os.environ.get("TWTRX_NO_PUSH") == "1":
        return "not pushed (TWTRX_NO_PUSH)"
    if stamp.exists() and stamp.read_bytes() == body:
        return "unchanged"
    d = os.path.dirname(dest)
    cmd = f"mkdir -p {d} && cat > {dest}.tmp && mv {dest}.tmp {dest}"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", HOST, cmd], input=body, capture_output=True, timeout=120)
    if r.returncode:
        return "push failed: " + r.stderr.decode(errors="replace").strip()[:200]
    stamp.write_bytes(body)
    return "pushed"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-publish", action="store_true"); a = ap.parse_args()
    meta = json.loads((PBI / "meta.json").read_text())
    rows = load_alerts()
    js = junctions(rows); areas = station_areas(rows)
    cells, days, nsnap = profile(rows)
    (OUT / "eta_ctx").mkdir(parents=True, exist_ok=True)
    src = {"source": "Bengaluru Traffic Police congestion map (btp.karnataka.gov.in/188/congestion-map, Power BI)",
           "model_refreshed_utc": meta.get("last_refresh_utc"), "period": [days[0], days[-1]], "snapshots": nsnap,
           "how": "index 0 = Monday 00:00-00:59 IST ... 167 = Sunday 23:00-23:59"}
    typical = {**src, "fields": "lat, lon, name, police station, corridor, alerts, typical length m, % severe or high, "
                                "% recurring, p[168]: % of that hour's 15-min snapshots with congestion here",
               "cells": [[c["lat"], c["lon"], c["name"], c["ps"], c["corridor"], c["alerts"], c["ext_m"], c["severe_pct"],
                          c["recurring_pct"], c["p"]] for c in cells]}
    body_map = json.dumps(typical, ensure_ascii=False, separators=(",", ":")).encode()
    (OUT / "congestion_typical.json").write_bytes(body_map)
    stops = stop_profile(cells)
    body_stops = json.dumps({**src, "radius_m": STOP_R, "stop_ids": "BMTC map stop ids (maps-site/bengaluru/data/bus_stops.geojson stop_id)",
                             "stops": stops}, separators=(",", ":")).encode()
    (OUT / "eta_ctx/stop_congestion_how.json").write_bytes(body_stops)
    body_corr = json.dumps({"source": src["source"], "model_refreshed_utc": src["model_refreshed_utc"], **corridor_speeds()},
                           ensure_ascii=False, separators=(",", ":")).encode()
    (OUT / "eta_ctx/corridor_speed_how.json").write_bytes(body_corr)
    msg = (f"btp_congestion: {len(rows)} alerts, {len(days)} days, {len(js)} junctions, {len(areas)} stations, "
           f"{len(cells)} cells on the map ({len(body_map) // 1024} KB), {len(stops)} stops with a profile")
    if not a.no_publish:
        msg += "; map " + push(body_map, "projects/maps-site/bengaluru/live/btp_congestion.json", OUT / ".pushed_congestion")
        msg += "; eta stops " + push(body_stops, "projects/eta-ml/ctx/blr/stop_congestion_how.json", OUT / ".pushed_stop_ctx")
        msg += "; eta corridors " + push(body_corr, "projects/eta-ml/ctx/blr/corridor_speed_how.json", OUT / ".pushed_corr_ctx")
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
