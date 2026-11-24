#!/usr/bin/env python3
"""Route dodging: where Mumbai's BEST buses leave their route and rejoin it further on.

(Port of the Delhi detector, delhi_bus/server/detect_dodges.py, for a much sparser feed:
the Chalo collector sees each bus about every 8 minutes, against Delhi's 15 seconds.
So one fix well off the line, between two on it, is enough to call a bus away, gaps up
to MAX_GAP are allowed, and passes are counted from consecutive on-line fixes that
straddle a stretch rather than from fixes landing in it. BEST publishes no route
shapes: the lines are the ones the map drew through each route's timetabled stops,
so a detour most buses take is our line being wrong, which the USUAL rule removes.)

    detect_dodges.py --routes mtc_routes.geojson --stops route_stops.json --archive DIR
                     --cache DIR --out DIR [--model model.json] [--days 30]

For every bus, every archived GPS fix is measured against the official shape of
the route it is running (the feed's route_id; mtc_routes.geojson). A dodge is:

  leaves    the last fix within ON_M of the line, then
  away      at least MIN_AWAY consecutive fixes more than OFF_M from it, at
            least one more than PEAK_M,
            no gap between fixes over MAX_GAP s, while the route is unchanged,
  rejoins   back within ON_M, further along the route than where it left
            (forward progress SKIP_MIN..SKIP_MAX metres of the planned line),
  and not   within END_M of either terminus (depot runs and turn-arounds) and
            not longer than MAX_DUR (a long absence is a breakdown or a garage).

Each dodge keeps the bus, the route, the times, the path driven (the fixes) and
what it skipped: the planned metres between leaving and rejoining, and how long
those usually take at that hour (model.json, the nowcast's travel model), so
the time a dodge saved or lost can be estimated. Dodges that leave and rejoin
near the same places on the same route are one "alternative route".

An alternative taken by most buses that pass (share >= USUAL) is marked
`usual`: the published route is not the road the buses use, which is a finding
about the route data rather than dodging.

Days are processed once and cached (CACHE/dodges-YYYY-MM-DD.json); today is
redone each run. Writes OUT/dodges.json (every dodge, lean: its geometry is its alternative's) and
OUT/alternatives.json (lines thinned to ~8 m).
"""
import argparse
import collections
import datetime as dt
import glob
import gzip
import json
import math
import os
import statistics
import time

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
ON_M, OFF_M, PEAK_M = 80, 250, 400
MIN_AWAY, MAX_GAP, MAX_DUR = 1, 900, 45 * 60
SKIP_MIN, SKIP_MAX, END_M = 300, 15000, 400
USUAL = 0.6          # share of passing buses above which a detour is the road in use, not a dodge
# our line between two stops running over SUSPECT_X times their straight distance (and over
# SUSPECT_M) is probably a wrong road from the snapping, not the bus's: a detour across it is
# set aside as `suspect` rather than counted as dodging
SUSPECT_X, SUSPECT_M = 2.2, 800
CELL = 200.0
LAT0 = 19.07
KX, KY = 111320 * math.cos(math.radians(LAT0)), 111320.0


def xy(lon, lat):
    return lon * KX, lat * KY


class Line:
    """A route's planned line in local metres, with a grid of its segments."""
    __slots__ = ('x', 'y', 'cum', 'grid', 'L')

    def __init__(self, coords):
        pts = [xy(lo, la) for lo, la in coords]
        self.x = [p[0] for p in pts]; self.y = [p[1] for p in pts]
        self.cum = [0.0]
        for i in range(1, len(pts)):
            self.cum.append(self.cum[-1] + math.hypot(self.x[i] - self.x[i - 1], self.y[i] - self.y[i - 1]))
        self.L = self.cum[-1]
        self.grid = collections.defaultdict(list)
        for i in range(len(pts) - 1):
            x0, x1 = sorted((self.x[i], self.x[i + 1])); y0, y1 = sorted((self.y[i], self.y[i + 1]))
            for gx in range(int(x0 // CELL), int(x1 // CELL) + 1):
                for gy in range(int(y0 // CELL), int(y1 // CELL) + 1):
                    self.grid[(gx, gy)].append(i)

    def near(self, px, py, reach=2, s_hint=None):
        """(distance m, metres along) of the nearest point within `reach` cells; with
        s_hint, among equally near points (a road used twice) the one closest along."""
        gx, gy = int(px // CELL), int(py // CELL)
        best = None
        seen = set()
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for i in self.grid.get((gx + dx, gy + dy), ()):
                    if i in seen:
                        continue
                    seen.add(i)
                    ax, ay, bx, by = self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]
                    vx, vy = bx - ax, by - ay
                    L2 = vx * vx + vy * vy
                    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2)) if L2 else 0.0
                    d = math.hypot(ax + t * vx - px, ay + t * vy - py)
                    s = self.cum[i] + t * math.sqrt(L2)
                    if best is None or d < best[0] - 5 or (abs(d - best[0]) <= 5 and s_hint is not None and abs(s - s_hint) < abs(best[1] - s_hint)):
                        best = (d, s)
        return best if best else (reach * CELL + CELL, None)

    def point(self, s):
        """lon, lat at metres along."""
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.cum[m] <= s: lo = m
            else: hi = m
        t = (s - self.cum[lo]) / ((self.cum[hi] - self.cum[lo]) or 1)
        return [round((self.x[lo] + t * (self.x[hi] - self.x[lo])) / KX, 5), round((self.y[lo] + t * (self.y[hi] - self.y[lo])) / KY, 5)]

    def cut(self, s0, s1):
        out = [self.point(s0)]
        for i, c in enumerate(self.cum):
            if s0 < c < s1:
                out.append([round(self.x[i] / KX, 5), round(self.y[i] / KY, 5)])
        out.append(self.point(s1))
        return out


def lines(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as fh:
        for ln in fh:
            try:
                yield json.loads(ln)
            except ValueError:
                pass


def detect_day(path, routes):
    """All dodges in one day's archive."""
    st = {}                       # reg -> state
    out = []
    visits = collections.defaultdict(set)     # (code, 300 m bucket) -> {(bus, half hour)} seen on the line
    for r in lines(path):
        code = r.get('r')
        L = routes.get(code)
        reg, t = r['reg'], r['t']
        if L is None:
            st.pop(reg, None); continue
        px, py = xy(r['lo'], r['la'])
        s0 = st.get(reg)
        if s0 and (s0['code'] != code or t - s0['t'] > MAX_GAP):
            s0 = None
        hint = s0['s'] if s0 and s0.get('s') is not None else None
        d, s = L.near(px, py, 2, hint)
        if s0 is None:
            st[reg] = {'code': code, 't': t, 's': s if d <= ON_M else None, 'tl': t if d <= ON_M else None,
                       'pl': [r['lo'], r['la']] if d <= ON_M else None, 'away': None}
            continue
        s0['t'] = t
        if d <= ON_M:
            sp = s0['s'] if s0['away'] is None and s0['s'] is not None else s
            for b in range(int(min(sp, s) // 300), int(max(sp, s) // 300) + 1):
                visits[(code, b)].add((reg, t // 1800))
            a = s0['away']
            if a and s0['s'] is not None and len(a['pts']) >= MIN_AWAY and a['peak'] >= PEAK_M:
                skip = s - s0['s']
                dur = t - s0['tl']
                if SKIP_MIN <= skip <= SKIP_MAX and dur <= MAX_DUR and s0['s'] >= END_M and s <= L.L - END_M:
                    path = [s0['pl']] + a['pts'] + [[r['lo'], r['la']]]
                    plen = sum(math.hypot((path[i + 1][0] - path[i][0]) * KX, (path[i + 1][1] - path[i][1]) * KY) for i in range(len(path) - 1))
                    out.append({'reg': reg, 'code': code, 't0': s0['tl'], 't1': t, 's0': round(s0['s']), 's1': round(s),
                                'peak': round(a['peak']), 'len': round(plen),
                                'path': [[round(x, 5), round(y, 5)] for x, y in path]})
            s0.update(s=s, tl=t, pl=[r['lo'], r['la']], away=None)
        elif d > OFF_M:
            if s0['s'] is None:
                continue
            a = s0['away'] or {'pts': [], 'peak': 0}
            a['pts'].append([r['lo'], r['la']]); a['peak'] = max(a['peak'], d)
            s0['away'] = a
            if len(a['pts']) > 400:                       # far too long: not a dodge
                s0.update(s=None, away=None)
        # between ON_M and OFF_M: neither on nor clearly off; keep waiting
    return out, {f'{c}:{b}': len(v) for (c, b), v in visits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--routes', required=True)
    ap.add_argument('--stops', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='')
    ap.add_argument('--days', type=int, default=30)
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(a.cache, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    fc = json.load(open(a.routes))
    routes, meta = {}, {}
    for f in fc['features']:
        p, g = f['properties'], f['geometry']
        cs = g['coordinates'] if g['type'] == 'LineString' else [c for part in g['coordinates'] for c in part]
        if len(cs) >= 2 and p.get('snapped', True):        # straight stop-to-stop lines say nothing about the road
            routes[str(p['code'])] = Line(cs)
            meta[str(p['code'])] = (p.get('sn') or '', p.get('category') or '', p.get('source') or '', p.get('destinatio') or '')
    rs = json.load(open(a.stops))
    stopd = {}
    for c, r in rs.items():
        if c in routes:
            L = routes[c]
            stopd[c] = [(L.near(*xy(s[3], s[2]), 1)[1], s[1], round(s[3], 6), round(s[2], 6)) for s in r['d'][0]['s']]
    suspect = collections.defaultdict(list)     # code -> [(s_from, s_to)] legs our line probably gets wrong
    for c, sd_ in stopd.items():
        for (s_a, _, lo_a, la_a), (s_b, _, lo_b, la_b) in zip(sd_, sd_[1:]):
            if s_a is None or s_b is None:
                continue
            straight = math.hypot((lo_b - lo_a) * KX, (la_b - la_a) * KY)
            if s_b - s_a > SUSPECT_M and s_b - s_a > SUSPECT_X * max(straight, 1):
                suspect[c].append((s_a, s_b))
    model = json.load(open(a.model)) if a.model and os.path.exists(a.model) else {}
    print(f'{len(routes)} route lines indexed in {time.time() - t_start:.0f}s', flush=True)

    today = dt.datetime.now(IST).date().isoformat()
    days = sorted(glob.glob(os.path.join(a.archive, 'obs-*.jsonl*')))[-a.days:]
    allout = []
    visits = collections.Counter()           # how many bus passes each stretch of each route saw
    for p in days:
        day = os.path.basename(p)[4:14]
        cp = os.path.join(a.cache, f'dodges-{day}.json')
        if day != today and os.path.exists(cp):
            c_ = json.load(open(cp))
            allout += c_['d']
            for k, v in c_['v'].items():
                visits[k] += v
            continue
        t0 = time.time()
        got, vis = detect_day(p, routes)
        json.dump({'d': got, 'v': vis}, open(cp, 'w'), separators=(',', ':'))
        print(f'{day}: {len(got)} dodges in {time.time() - t0:.0f}s', flush=True)
        allout += got
        for k, v in vis.items():
            visits[k] += v

    def stop_name(code, s):
        sd = stopd.get(code) or []
        best = min(sd, key=lambda q: abs((q[0] or 0) - s), default=(0, ''))
        return best[1]

    def usual_seconds(code, s0, s1, hour):
        """How long the skipped planned section usually takes at this hour (travel model)."""
        sd, ids = stopd.get(code) or [], [x[0] for x in rs[code]['d'][0]['s']] if code in rs else []
        seg = (model.get('seg') or {})
        tsum, covered = 0.0, 0.0
        for i in range(len(sd) - 1):
            a_, b_ = sd[i][0], sd[i + 1][0]
            if a_ is None or b_ is None or b_ <= a_:
                continue
            lo, hi = max(a_, s0), min(b_, s1)
            if hi <= lo:
                continue
            row = seg.get(f'{ids[i]}|{ids[i + 1]}')
            v = (row[2][hour] if row and row[2][hour] > 0 else (row[1] if row and row[1] > 0 else None)) if row else None
            v = v or ((model.get('city') or {}).get('v') or [None] * 24)[hour] or 4.2
            tsum += (hi - lo) / max(v, 0.5); covered += hi - lo
        return round(tsum) if covered > 0 else None

    def thin(cs, tol=8.0):
        """Douglas-Peucker in metres, so lines stay true at street zoom but light."""
        if len(cs) < 3:
            return cs
        P = [(x * KX, y * 110574) for x, y in cs]
        keep, stack = {0, len(P) - 1}, [(0, len(P) - 1)]
        while stack:
            i, j = stack.pop()
            (ax, ay), (bx, by) = P[i], P[j]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy or 1e-9
            best, bk = 0.0, -1
            for k in range(i + 1, j):
                px, py = P[k]
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d = math.hypot(px - ax - t * dx, py - ay - t * dy)
                if d > best:
                    best, bk = d, k
            if best > tol:
                keep.add(bk); stack += [(i, bk), (bk, j)]
        return [cs[k] for k in sorted(keep)]

    def stops_between(code, s0, s1):
        """The route's stops a detour affects: [name, lon, lat, skipped?] - every stop it
        passed by (between leaving and rejoining), and the nearest stop at each end."""
        sd = [q for q in stopd.get(code) or [] if q[0] is not None]
        if not sd:
            return []
        ends = {min(sd, key=lambda q: abs(q[0] - s0)), min(sd, key=lambda q: abs(q[0] - s1))}
        return [[q[1], q[2], q[3], 1 if s0 < q[0] < s1 and q not in ends else 0]
                for q in sd if q in ends or s0 < q[0] < s1]

    day_names = [os.path.basename(p)[4:14] for p in days]
    day_of = lambda t: dt.datetime.fromtimestamp(t, IST).date().isoformat()

    # group into alternatives: same route, leaving and rejoining within ~300 m
    groups = collections.defaultdict(list)
    for e in allout:
        groups[(e['code'], round(e['s0'] / 300), round(e['s1'] / 300))].append(e)
    alts, dodges = [], []
    for gi, ((code, k0, k1), es) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        sn, cat, src, dst = meta.get(code, ('', '', '', ''))
        L = routes[code]
        s0 = statistics.median(e['s0'] for e in es); s1 = statistics.median(e['s1'] for e in es)
        rep = sorted(es, key=lambda e: e['len'])[len(es) // 2]
        saved = []
        for e in es:
            h = dt.datetime.fromtimestamp(e['t0'], IST).hour
            u = usual_seconds(code, e['s0'], e['s1'], h)
            e['usual'] = u
            e['saved'] = (u - (e['t1'] - e['t0'])) if u is not None else None
            if e['saved'] is not None:
                saved.append(e['saved'])
            dodges.append([e['t0'], e['t1'] - e['t0'], e['reg'], e['len'], e['saved'], gi])
        # how many of the buses that passed this point took the detour: near all of
        # them means the published route is not the road buses use, not a dodge
        passes = max(len(es), visits.get(f'{code}:{int(s0 // 300)}', 0))
        share = len(es) / passes if passes else 1.0
        hours = [0] * 24
        for e in es:
            hours[dt.datetime.fromtimestamp(e['t0'], IST).hour] += 1
        alts.append({'id': gi, 'key': f'{code}-{k0}-{k1}', 'code': code, 'sn': sn, 'cat': cat,
                     'leave': stop_name(code, s0), 'rejoin': stop_name(code, s1), 's0': round(s0), 's1': round(s1),
                     'n': len(es), 'passes': passes, 'share': round(share, 2), 'usual': share >= USUAL and len(es) >= 3,
                     'suspect': any(a_ < s1 and b_ > s0 for a_, b_ in suspect.get(code, ())),
                     'buses': len({e['reg'] for e in es}), 'regs': sorted({e['reg'] for e in es})[:40],
                     'first': min(e['t0'] for e in es), 'last': max(e['t0'] for e in es), 'hours': hours,
                     'saved_med': round(statistics.median(saved)) if saved else None,
                     'skipped': round(s1 - s0), 'driven_med': round(statistics.median(e['len'] for e in es)),
                     'path': thin(rep['path']), 'planned': thin(L.cut(s0, s1)),
                     'stops': stops_between(code, s0, s1),
                     'byday': [sum(1 for e in es if day_of(e['t0']) == d) for d in day_names]})
    dodges.sort(key=lambda d: d[0])
    json.dump({'t': int(time.time()), 'days': [os.path.basename(p)[4:14] for p in days], 'n': len(dodges),
               'cols': ['t0', 'dur_s', 'reg', 'driven_m', 'saved_s', 'alt'],
               'd': dodges}, open(os.path.join(a.out, 'dodges.json'), 'w'), separators=(',', ':'))
    json.dump({'t': int(time.time()), 'alts': alts}, open(os.path.join(a.out, 'alternatives.json'), 'w'), separators=(',', ':'))
    if model.get('seg'):
        # the congestion overlay: each stop-to-stop stretch's speed (km/h) all day and by hour
        kmh = lambda v: round(v * 3.6) if v and v > 0 else -1
        json.dump({'t': model.get('t'), 's': {k: [kmh(r[1])] + [kmh(v) for v in r[2]] for k, r in model['seg'].items()}},
                  open(os.path.join(a.out, 'congestion.json'), 'w'), separators=(',', ':'))
    print(f'{len(dodges)} dodges, {len(alts)} alternative routes ({sum(1 for x in alts if x["n"] >= 3)} seen 3+ times), {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
