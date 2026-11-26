#!/usr/bin/env python3
"""Normalise every city's traffic-police feed into one event list (traffic_eval/work/events.json).

One event = {id, city, source, agency, type, severity, kind, t_start, t_end, end_rule, pts [[lon, lat]...],
             is_line, place, text}

Sources (snapshots in traffic_eval/src_data/, copied from the live pipelines):
  hyderabad  twtrx out/eta/incidents.jsonl      @HYDTP / @CYBTRAFFIC posts, HTP e-challan alerts (geocoded, routed)
  bengaluru  twtrx out/bengaluru/incidents_all.geojson   @blrcitytraffic and the BTP division accounts
  chennai    twtrx out/chennai/incidents_all.geojson     @ChennaiTraffic road updates
  delhi      maps-site delhi/route-dodging/data/closures.json   Delhi Traffic Police site + @dtptraffic (dtp_closures.py)

Rules:
  * time window: the post's stated start/end when it has one; otherwise start = post time and end = start + the
    type's default duration (DEFAULT_MIN, the same defaults the live layer uses);
  * "standing" advisories (a one-way system, months-long works) are dropped: they are part of every day's speeds;
  * geometry: the routed road line when the pipeline drew one, else the geocoded point;
  * an event with no time or no geometry is dropped (counted in the summary).
"""
from __future__ import annotations

import ast
import collections
import json
import re

from common import SRC_DATA, WORK

DEFAULT_MIN = {"accident": 60, "breakdown": 45, "spillage": 60, "waterlogging": 180, "fire": 90, "tree_fall": 90,
               "closure": 240, "diversion": 240, "u_turn_change": 0, "works": 480, "event": 240, "congestion": 45,
               "other": 45}
TYPE_WORDS = [  # Delhi items have no type: the first match in the title + text
    ("accident", r"accident|collision|overturn"), ("breakdown", r"break ?down|broken down|breakdown"),
    ("waterlogging", r"water ?logg|water-logg|flood"), ("tree_fall", r"tree (fall|uproot)|fallen tree"),
    ("fire", r"\bfire\b"), ("works", r"pipeline|construction|repair|work by|metro work|dug|laying"),
    ("event", r"procession|rally|kirtan|exhibition|match|wedding|festival|marathon|visit|vvip|vip|parade|protest"),
    ("closure", r"closed|closure|restriction"), ("diversion", r"divert|diversion"),
    ("congestion", r"congestion|heavy traffic|slow"),
]
SEV = {"accident": 0.65, "breakdown": 0.5, "waterlogging": 0.75, "tree_fall": 0.55, "fire": 0.6, "works": 0.5,
       "event": 0.6, "closure": 0.9, "diversion": 0.8, "congestion": 0.5, "other": 0.3, "spillage": 0.5,
       "u_turn_change": 0.2}


def delhi_type(text: str) -> str:
    t = text.lower()
    for typ, rx in TYPE_WORDS:
        if re.search(rx, t):
            return typ
    return "other"


def hyderabad():
    out = []
    for ln in open(SRC_DATA / "hyderabad_incidents.jsonl"):
        r = json.loads(ln)
        if r.get("kind") in ("standing", "network_change"):
            continue
        t0 = r.get("t_start")
        if not t0:
            continue
        t1 = r.get("t_end") or r.get("t_expires_default")
        rule = "stated" if r.get("t_end") else "default"
        if not t1:
            t1 = t0 + 60 * (r.get("default_duration_min") or DEFAULT_MIN.get(r["type"], 45))
        if r.get("lines"):
            pts, line = [[p[1], p[0]] for part in r["lines"] for p in part], True
        elif r.get("line"):
            pts, line = [[p[1], p[0]] for p in r["line"]], True
        elif r.get("lat") is not None:
            pts, line = [[r["lon"], r["lat"]]], False
        else:
            continue
        out.append({"id": r["id"], "city": "hyderabad", "source": r.get("source"), "agency": r.get("agency"),
                    "type": r["type"], "severity": float(r.get("severity") or SEV.get(r["type"], 0.5)),
                    "kind": r.get("kind"), "t_start": int(t0), "t_end": int(t1), "end_rule": rule,
                    "pts": pts, "is_line": line, "place": r.get("place"), "text": None})
    return out


def geojson(city: str, fname: str):
    g = json.load(open(SRC_DATA / fname))["features"]
    lines = {}
    for f in g:
        pid = str(f["properties"].get("id"))
        if pid.endswith(":line"):
            c = f["geometry"]["coordinates"]
            lines[pid[:-5]] = [x[:2] for part in c for x in part] if f["geometry"]["type"] == "MultiLineString" else [x[:2] for x in c]
    out = []
    for f in g:
        p = f["properties"]
        pid = str(p.get("id"))
        if pid.endswith(":line") or p.get("kind") == "standing":
            continue
        if not p.get("start"):
            continue
        t0 = int(p["start"])
        t1 = int(p.get("end") or t0 + 60 * DEFAULT_MIN.get(p.get("type"), 45))
        if pid in lines:
            pts, line = lines[pid], True
        elif f["geometry"]["type"] == "Point":
            pts, line = [f["geometry"]["coordinates"][:2]], False
        else:
            continue
        out.append({"id": pid, "city": city, "source": p.get("source"), "agency": p.get("agency"), "type": p.get("type") or "other",
                    "severity": float(p.get("severity") or 0.5), "kind": p.get("kind"), "t_start": t0, "t_end": t1,
                    "end_rule": "pipeline", "pts": pts, "is_line": line, "place": p.get("place"), "text": (p.get("text_en") or "")[:280]})
    return out


def delhi():
    out = []
    for x in json.load(open(SRC_DATA / "delhi_closures.json"))["items"]:
        if not x.get("geom") or not x.get("from"):
            continue
        g = ast.literal_eval(x["geom"]) if isinstance(x["geom"], str) else x["geom"]
        c = g.get("coordinates")
        if g["type"] == "Point":
            pts, line = [c[:2]], False
        elif g["type"] == "LineString":
            pts, line = [p[:2] for p in c], True
        elif g["type"] == "MultiLineString":
            pts, line = [p[:2] for part in c for p in part], True
        elif g["type"] == "MultiPoint":
            pts, line = [p[:2] for p in c], False
        else:
            continue
        typ = delhi_type((x.get("title") or "") + " " + (x.get("text") or ""))
        t0, t1 = int(x["from"]), int(x.get("to") or x["from"] + 60 * DEFAULT_MIN[typ])
        # the site's own default windows (alert 6 h, advisory 24 h) are far longer than most jams; when the post
        # gives no hours, an alert keeps the type's default duration instead
        rule = x.get("when") or "default"
        if rule == "default" and x.get("kind") == "alert":
            t1 = t0 + 60 * DEFAULT_MIN[typ]
        out.append({"id": "dtp:" + x["id"], "city": "delhi", "source": "dtp", "agency": "Delhi Traffic Police",
                    "type": typ, "severity": SEV.get(typ, 0.5), "kind": x.get("kind"), "t_start": t0, "t_end": t1,
                    "end_rule": rule, "pts": pts, "is_line": line, "place": ", ".join(x.get("names") or [])[:120],
                    "text": ((x.get("title") or "") + " " + (x.get("text") or ""))[:280], "conf": x.get("conf")})
    return out


def main():
    ev = hyderabad() + geojson("bengaluru", "bengaluru_incidents.geojson") + geojson("chennai", "chennai_incidents.geojson") + delhi()
    seen, uniq = set(), []
    for e in ev:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        uniq.append(e)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "events.json").write_text(json.dumps(uniq))
    c = collections.Counter((e["city"], e["type"]) for e in uniq)
    print(len(uniq), "events;", dict(collections.Counter(e["city"] for e in uniq)))
    for k in sorted(c):
        print("  ", k, c[k])


if __name__ == "__main__":
    main()
