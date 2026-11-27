#!/usr/bin/env python3
"""Chandigarh junction aliases from the sector grid -> data/chd/aliases.csv (keeps the hand-written rows below the marker).

Chandigarh Traffic Police name junctions by the sectors around them: "Sec-33/34 Lp" is the light point midway along the
road between sectors 33 and 34; "Sec-19/20/26/27" (or "... roundabout / chowk") is the corner where four sectors meet.
OSM gives each sector's outline (data/chd/osm_polys.geojsonseq, name "Sector N"); Panchkula and Mohali reuse the numbers,
so each number takes the outline nearest central Chandigarh. A pair of sectors whose centres are under PAIR_M apart
shares a side, and its light point is the midpoint of the two centres; four sectors that all lie within QUAD_M of their
common centre meet at that point. pipeline/chandigarh.py rewrites the police's spellings to "Sector a/b Light Point"
and "Sector a/b/c/d" with the numbers in ascending order, which is how these aliases are named.
  python3 scripts/chd_aliases.py
"""
import csv, itertools, json, math, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/chd/aliases.csv"
CENTRE = (30.735, 76.780)
PAIR_M, QUAD_M = 1300, 1000
MARK = "# hand-written below (kept on regeneration)"


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(h))


def main():
    best = {}
    for line in open(ROOT / "data/chd/osm_polys.geojsonseq", encoding="utf-8"):
        f = json.loads(line.strip().lstrip("\x1e"))
        m = re.fullmatch(r"Sector[- ]?(\d{1,2})", (f["properties"].get("name") or "").strip(), re.I)
        if not m:
            continue
        g = f["geometry"]
        rings = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
        pts = [c for r in rings[:1] for c in r] if g["type"] == "Polygon" else [c for r in rings for c in r]
        c = (sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts))
        n, d = int(m.group(1)), hav(c, CENTRE)
        if d < 7000 and (n not in best or d < best[n][1]):
            best[n] = (c, d)
    sec = {n: v[0] for n, v in best.items()}
    rows = []
    for a, b in itertools.combinations(sorted(sec), 2):
        if hav(sec[a], sec[b]) < PAIR_M:
            mid = ((sec[a][0] + sec[b][0]) / 2, (sec[a][1] + sec[b][1]) / 2)
            rows.append((f"Sector {a}/{b} Light Point", mid, "between the two sectors' centres"))
    for n, c in sorted(sec.items()):            # a sector's market and inner area: its centre (the markets sit mid-sector)
        for name in (f"Sector {n} Market", f"Market Sector {n}", f"Inner Area Sector {n}"):
            rows.append((name, c, "the sector's centre"))
    for q in itertools.combinations(sorted(sec), 4):
        c = (sum(sec[n][0] for n in q) / 4, sum(sec[n][1] for n in q) / 4)
        if all(hav(sec[n], c) < QUAD_M for n in q):
            rows.append(("Sector " + "/".join(map(str, q)), c, "where the four sectors meet"))
    hand = []
    if OUT.exists():
        txt = OUT.read_text(encoding="utf-8").split(MARK, 1)
        if len(txt) == 2:
            hand = [l for l in txt[1].strip().splitlines() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alias", "target", "lat", "lon", "note", "region"])
        for name, (la, lo), note in rows:
            w.writerow([name, "", f"{la:.5f}", f"{lo:.5f}", f"generated: {note}", ""])
        f.write(MARK + "\n")
        for l in hand:
            f.write(l + "\n")
    print(f"{len(sec)} sectors, {sum(1 for r in rows if 'Light' in r[0])} light points, "
          f"{sum(1 for r in rows if r[0].count('/') == 3)} corners, {len(hand)} hand rows")


if __name__ == "__main__":
    main()
