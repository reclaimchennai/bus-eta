"""Traffic incidents in live ETAs: a delay for every bus whose remaining trip runs through an active incident.

Stdlib only; a copy sits beside each collector that uses it (like eta_ml_merge.py). Reads a map's live incidents
file (GeoJSON from the twtrx feed: Point per incident, plus a LineString "<id>:line" on the road when known;
properties id, type, severity 0-1, start, end (epoch s), text_en, place, route_ids, stop_ids) and, for one bus,
answers which incidents lie ahead of it on its stop chain and how many seconds each one adds.

    inc = Incidents('/data/rtc/web/tgsrtc/live/incidents.json')
    for d_at, delay, iid in inc.ahead(stops_lonlat, stops_metres, bus_metres, now): ...

The caller adds `delay` to every listed stop at or beyond `d_at` metres along the trip, after any other ETA step
(nowcast, v4), so the penalty is on top of whatever the map shows. A bus already within PASSED m of the incident
gets none: its own live pace already carries the jam.

The delays are priors, not yet calibrated (too few incidents overlap the archived GPS so far):
  delay = BASE[type] x (0.5 + severity), at most CAP_ONE s per incident and CAP_BUS s per bus; it eases to 30% over
  the last 40% of the incident's window (most are cleared before their stated or default end).
The file is ignored when its "generated" is older than STALE s (the feed stopped: Mac asleep, WhatsApp unlinked).
Standing advisories (kind "standing": a one-way system, long works) add nothing: the buses' observed speeds carry them.
"""
import json
import math
import os

BASE = {   # seconds at severity 0.5
    'accident': 300, 'breakdown': 240, 'spillage': 240, 'waterlogging': 480, 'fire': 360, 'tree_fall': 360,
    'closure': 600, 'diversion': 360, 'u_turn_change': 90, 'works': 240, 'event': 300, 'congestion': 240, 'other': 120,
}
CAP_ONE, CAP_BUS = 900, 1500
STALE = 1800
NEAR_PT, NEAR_LINE = 150, 60     # m from a stop-to-stop chord: a point is a place, a line is drawn on the road
PASSED = 100
AHEAD_MAX = 25000                # m: incidents further along the trip than this are ignored
K = 111320.0


class Incidents:
    def __init__(self, path):
        self.path, self.mtime, self.gen, self.items = path, 0.0, 0, []

    def load(self, now):
        """Active incidents: [(id, type, delay at full strength, start, end, [(lon, lat), ...], is_line)]."""
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            self.items = []
            return []
        if m != self.mtime:
            try:
                with open(self.path) as fh:
                    j = json.load(fh)
                items = []
                for f in j.get('features') or []:
                    p, g = f.get('properties') or {}, f.get('geometry') or {}
                    c = g.get('coordinates')
                    if p.get('kind') == 'standing':     # a one-way system or long works: already in the observed speeds
                        continue
                    if g.get('type') == 'Point' and c:
                        pts, line = [tuple(c[:2])], False
                    elif g.get('type') == 'LineString' and c:
                        pts, line = [tuple(x[:2]) for x in c], True
                    elif g.get('type') == 'MultiLineString' and c:
                        pts, line = [tuple(x[:2]) for part in c for x in part], True
                    else:
                        continue
                    sev = min(1.0, max(0.0, float(p.get('severity') or 0.5)))
                    d = min(CAP_ONE, BASE.get(p.get('type'), BASE['other']) * (0.5 + sev))
                    # its box, widened by how near a trip must pass: ahead() looks closer only at the trip's
                    # stop-to-stop chords whose own box meets it (a march's line has hundreds of points)
                    lim = (NEAR_LINE if line else NEAR_PT) * 1.2
                    la0, la1 = min(y for _, y in pts), max(y for _, y in pts)
                    dlon = lim / (K * max(0.2, math.cos(math.radians(max(abs(la0), abs(la1))))))
                    box = (min(x for x, _ in pts) - dlon, la0 - lim / K, max(x for x, _ in pts) + dlon, la1 + lim / K)
                    items.append((str(p.get('id') or '').split(':line')[0], p.get('type') or 'other', d,
                                  int(p.get('start') or 0), int(p.get('end') or 0), pts, line, box))
                self.items, self.gen, self.mtime = items, int(j.get('generated') or 0), m
            except (OSError, ValueError, TypeError):
                pass
        if not self.gen or now - self.gen > STALE:
            return []
        return [x for x in self.items if x[3] <= now and (not x[4] or now < x[4])]

    @staticmethod
    def strength(start, end, now):
        if not end or end <= start:
            return 1.0
        k = (now - start) / (end - start)
        return 1.0 if k < 0.6 else max(0.3, 1.0 - (k - 0.6) / 0.4 * 0.7)

    def ahead(self, ll, ds, s, now):
        """[(metres along the trip, delay s, incident id)] for the incidents ahead of a bus at s metres on a trip whose
        stops are ll [(lon, lat)] at ds metres; one entry per incident (the nearest pass of its point or line)."""
        act = self.load(now)
        if not act or len(ll) < 2:
            return []
        best = {}
        for iid, typ, d0, st, en, pts, line, (x0, y0, x1, y1) in act:
            lim = NEAR_LINE if line else NEAR_PT
            for i in range(len(ll) - 1):
                if ds[i + 1] < s - PASSED:
                    continue
                (ax, ay), (bx, by) = ll[i], ll[i + 1]
                if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
                    continue
                for lon, lat in pts:
                    cx = math.cos(math.radians(lat)) * K
                    vx, vy = (bx - ax) * cx, (by - ay) * K
                    px, py = (lon - ax) * cx, (lat - ay) * K
                    L2 = vx * vx + vy * vy
                    f = max(0.0, min(1.0, (px * vx + py * vy) / L2)) if L2 else 0.0
                    if math.hypot(px - f * vx, py - f * vy) > lim:
                        continue
                    at = ds[i] + f * (ds[i + 1] - ds[i])
                    if at < s + PASSED or at - s > AHEAD_MAX:
                        continue
                    if iid not in best or at < best[iid][0]:
                        best[iid] = (at, d0 * self.strength(st, en, now))
        out, tot = [], 0.0
        for iid, (at, d) in sorted(best.items(), key=lambda kv: kv[1][0]):
            d = min(d, CAP_BUS - tot)
            if d < 30:
                break
            out.append((at, int(d), iid))
            tot += d
        return out
