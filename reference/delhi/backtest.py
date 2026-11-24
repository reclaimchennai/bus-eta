#!/usr/bin/env python3
"""How well each method predicts where a bus will be, on a day it did not learn from.

    backtest.py --stops route_stops.json --stop-dist stop_dist.json --archive DIR
                --model model.json --day YYYY-MM-DD [--speeds speeds.json] [--sample 0.2]

For every fix of the test day, each method predicts the bus's position along its
route H seconds later (H = 30, 60, 120, 300); the truth is the bus's next fix
nearest that time. Error = |predicted - true| metres along the route. Methods:

  hold        the bus stays at its last fix (what showing raw GPS does)
  constant    15 km/h and 15 s at every stop (the collector before learning)
  hourly      the live-speeds file's segment speeds incl. stops (the collector now)
  nowcast     travel model: running speed + dwell, live probes, per-bus pace
"""
import argparse
import bisect
import collections
import json
import random
import statistics
import time

import nowcast
import travel_model as tm

H = (30, 60, 120, 300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stops', required=True)
    ap.add_argument('--stop-dist', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--day', required=True)
    ap.add_argument('--speeds', default='')
    ap.add_argument('--sample', type=float, default=0.2)
    a = ap.parse_args()
    random.seed(7)
    routes = tm.Routes(a.stops, a.stop_dist)
    M = nowcast.Model(a.model)
    pace = nowcast.Pace()
    hourly = json.load(open(a.speeds)) if a.speeds else {}

    def hourly_v(key, h):
        row = (hourly.get('s') or {}).get(key)
        v = row[2 + h] if row and row[2 + h] > 0 else ((hourly.get('city') or {}).get('h') or [None] * 24)[h]
        return (v / 3.6) if v and v > 2 else 4.2

    # pass 1: each bus's day as a list of (t, code, s, i), on its route
    tracks = collections.defaultdict(list)
    last = {}
    path = next(p for d, p in tm.day_files(a.archive, 'obs', 1, a.day))
    for r in tm.lines(path):
        prev = last.get(r['reg'])
        near = prev[3] if prev and prev[1] == r['r'] else None
        pr = routes.project(r['r'], r['la'], r['lo'], near)
        if not pr or pr[2] > 150:
            continue
        rec = (r['t'], r['r'], pr[0], pr[1])
        tracks[r['reg']].append(rec)
        last[r['reg']] = rec
    print(f'{len(tracks)} buses on {a.day}', flush=True)

    err = {m: {h: [] for h in H} for m in ('hold', 'constant', 'hourly', 'nowcast')}
    t0 = time.time()
    for reg, tr in tracks.items():
        if random.random() > a.sample:
            continue
        times = [x[0] for x in tr]
        for n, (t, code, s, i) in enumerate(tr):
            ids, ds = routes.ids[code], routes.ds[code]
            hour = time.gmtime(t + 19800).tm_hour
            # live probes and pace learn from the past only (this fix and before)
            if n:
                pt, pc, ps, pi = tr[n - 1]
                if pc == code and 0 < t - pt <= 180 and s - ps >= 12:
                    v, span = (s - ps) / (t - pt), s - ps
                    for j in range(pi, i + 1):
                        if j + 1 < len(ids):
                            share = max(0.0, min(s, ds[j + 1]) - max(ps, ds[j])) / span
                            if share > 0:
                                M.add_probe(f'{ids[j]}|{ids[j + 1]}', t, v, (t - pt) * share)
                    pace.update(reg, M.expected_time(ids, ds, ps, s, hour, t), t - pt)
                elif pc == code and 0 < t - pt <= 180 and abs(s - ps) < 12 and i + 1 < len(ids):
                    M.add_probe(f'{ids[i]}|{ids[i + 1]}', t, 0.0, t - pt)       # standing counts too
            for h in H:
                j = bisect.bisect_left(times, t + h)
                if j >= len(tr) or abs(tr[j][0] - (t + h)) > 15 or tr[j][1] != code:
                    continue
                truth = tr[j][2]
                if truth < s - 20:                        # trip ended / turned: skip
                    continue
                tt = tr[j][0]
                err['hold'][h].append(abs(truth - s))
                # constant 4.2 m/s + 15 s a stop
                here, t_ = s, t
                k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
                while k < len(ds):
                    need = (ds[k] - here) / 4.2
                    if t_ + need >= tt:
                        here += 4.2 * (tt - t_); break
                    t_ += need + 15; here = ds[k]; k += 1
                    if t_ >= tt:
                        break
                err['constant'][h].append(abs(truth - here))
                # hourly incl. stops, no dwell
                here, t_ = s, t
                k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
                while k < len(ds):
                    v = hourly_v(f'{ids[k - 1]}|{ids[k]}', hour)
                    need = (ds[k] - here) / v
                    if t_ + need >= tt:
                        here += v * (tt - t_); break
                    t_ += need + 15; here = ds[k]; k += 1
                err['hourly'][h].append(abs(truth - here))
                err['nowcast'][h].append(abs(truth - M.advance(ids, ds, s, t, tt, pace.get(reg))))
    print(f'evaluated in {time.time() - t0:.0f}s')
    print('median / p80 error in metres along the route, predicting ahead by:')
    print('method      ' + ''.join(f'{h:>16}s' for h in H))
    for m in err:
        row = ''
        for h in H:
            L = sorted(err[m][h])
            row += f'{(str(round(L[len(L) // 2])) + " / " + str(round(L[int(len(L) * .8)]))) if L else "-":>17}'
        print(f'{m:<12}{row}')
    print('samples per horizon:', {h: len(err["nowcast"][h]) for h in H})


if __name__ == '__main__':
    main()
