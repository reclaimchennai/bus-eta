"""v5 research engine: eta_engine.CityStream with extra live state, so every v5 feature is computed by the
same streaming code a service would run (nothing here reads a future fix).

The first 74 columns of a row are v4's features, unchanged and in v4's order, so v4/v4r score the same rows.
The extra columns (V5_EXTRA) come from state v4 does not keep:

  this bus      how long it has been standing, the gap between its last two fixes, its speed over the last
                3 and 5 minutes, its whole-trip time against the history model, where it is inside the segment
  same route    the bus ahead and the bus behind on the same working (gap in metres, age of that fix, its speed),
                and how many buses of the working are active: headway and bunching
  leader time   the last bus that crossed the target stop: its own time from the stop behind us to the target,
                against the history model's time for that span (the classic "previous bus" predictor), and its age
  last passes   the path ahead timed with each segment's last stop-to-stop traversals (up to an hour old,
                dwell included), where v4's corridor estimate only uses probes of the last 15 minutes
  today         a decayed (30 min) ratio of actual to history-model time over every probe of this working, and
                of the whole city: is today running slower than usual here? (online residual, as a feature)
  history       the share of the path ahead that has its own row in model.json (sparse systems)
"""
from __future__ import annotations

import collections
import math
import time

import eta_engine as E

V5_EXTRA = [
    "stopped_s", "fix_gap_s", "veh_v_180s", "veh_v_300s", "trip_vs_hist_ratio", "dist_to_next_stop_m",
    "dist_from_prev_stop_m", "route_today_ratio", "route_today_evidence_s", "city_today_ratio",
    "lead_gap_m", "lead_fix_age_s", "lead_v", "follow_gap_m", "route_active_buses",
    # per target stop
    "leader_tt_ratio", "leader_tt_age_s", "leader_vs_nowcast", "leader2_tt_ratio",
    "base_eta_lastpass_s", "lastpass_vs_nowcast", "lastpass_cov_frac", "lastpass_age_s", "path_hist_cov_frac",
]
V5_FEATURES = E.V4_FEATURES + V5_EXTRA
TAU = 1800.0            # decay of the "today vs history" ratios
LASTPASS_S = 3600.0     # a segment traversal counts for an hour
ROUTE_ACTIVE_S = 1800.0
NO_CITY_V = [-1] * 24


class Decayed:
    """Exponentially decayed sums of actual and expected seconds."""
    __slots__ = ("a", "e", "t")

    def __init__(self):
        self.a = self.e = self.t = 0.0

    def add(self, t, actual, expected):
        if t > self.t:
            f = math.exp(-(t - self.t) / TAU) if self.t else 0.0
            self.a *= f; self.e *= f; self.t = t
        self.a += actual; self.e += expected

    def get(self, t):
        f = math.exp(-max(0.0, t - self.t) / TAU) if self.t else 0.0
        e = self.e * f
        return (min(4.0, max(0.25, self.a / self.e)) if e > 60.0 else -1.0), e


class V5Stream(E.CityStream):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.x = {}                                         # reg -> per-bus extra state
        self.code_bus = collections.defaultdict(dict)       # working -> reg -> (t, s, recent v)
        self.last_cross = collections.defaultdict(dict)     # working -> stop k -> [(t, that bus's crossings), previous]
        self.seg_trav = {}                                  # "stopA|stopB" -> deque of (t, seconds stop to stop)
        self.r_today = collections.defaultdict(Decayed)
        self.c_today = Decayed()

    def vh(self, sk, hour):
        """The history model's speed for a segment at this hour (as eta_engine.features), and whether it has a row."""
        m = self.M.m                                        # read each time: the service reloads model.json
        row = (m.get("seg") or {}).get(sk)
        v = -1.0
        if row:
            v = row[2][hour] if row[2][hour] > 0 else row[1]
        if not v or v <= 0:
            cv = ((m.get("city") or {}).get("v") or NO_CITY_V)[hour]
            v = cv if cv and cv > 0 else 4.2
        return min(22.0, max(1.0, v)), bool(row)

    def ingest(self, r):
        reg, code = r.get("reg"), r.get("r")
        before = self.last.get(reg) if reg else None
        tp0 = self.trip.get(reg) if reg else None
        prev_fix = tp0[5] if tp0 else None
        prev_code = tp0[0] if tp0 else None
        super().ingest(r)
        now = self.last.get(reg) if reg else None
        if now is None or now is before:
            return                                          # the fix was not used
        t, code, s, i, _f = now
        tp = self.trip[reg]
        x = self.x.get(reg)
        new_trip = x is None or tp is not tp0          # eta_engine starts a trip with a new list
        if new_trip:
            if prev_code and prev_code != code:
                self.code_bus[prev_code].pop(reg, None)
            self.x[reg] = {"stop": None, "gap": -1.0, "q": collections.deque([(t, s)]), "act": 0.0, "exp": 0.0, "cross": {}}
            self.code_bus[code][reg] = (t, s, -1.0)
            return
        pt, ps, pi = prev_fix
        ids, ds = self.R.ids[code], self.R.ds[code]
        dt_, d = t - pt, s - ps
        hour = time.gmtime(t + 19800).tm_hour
        x["gap"] = dt_
        if d > 5.0 and dt_ > 0 and d <= 25.0 * dt_:          # stop crossings, as the truth is extracted
            cr = x["cross"]
            for k in range(pi + 1, i + 1):
                if 0 < k < len(ds) - 1 and k not in cr:
                    tc = pt + (ds[k] - ps) / d * dt_
                    cr[k] = tc
                    lc = self.last_cross[code].get(k)
                    self.last_cross[code][k] = [(tc, cr)] + (lc[:1] if lc else [])
                    if k - 1 in cr:
                        q = self.seg_trav.get(f"{ids[k - 1]}|{ids[k]}")
                        if q is None:
                            q = self.seg_trav[f"{ids[k - 1]}|{ids[k]}"] = collections.deque(maxlen=3)
                        q.append((tc, tc - cr[k - 1]))
        if 0 < dt_ <= self.max_gap and 12.0 <= d <= 25.0 * dt_:
            x["stop"] = None
            ex = 0.0
            for j in range(pi, i + 1):
                if j + 1 < len(ids):
                    part = max(0.0, min(s, ds[j + 1]) - max(ps, ds[j]))
                    if part > 0:
                        ex += part / self.vh(f"{ids[j]}|{ids[j + 1]}", hour)[0]
            x["act"] += dt_; x["exp"] += ex
            self.r_today[code].add(t, dt_, ex)
            self.c_today.add(t, dt_, ex)
        elif 0 < dt_ <= self.max_gap and abs(d) < 12.0:
            if x["stop"] is None:
                x["stop"] = pt
            x["act"] += dt_
            self.r_today[code].add(t, dt_, 0.0)
            self.c_today.add(t, dt_, 0.0)
        else:
            x["stop"] = None
        q = x["q"]
        q.append((t, s))
        while len(q) > 2 and q[0][0] < t - 600.0:
            q.popleft()
        self.code_bus[code][reg] = (t, s, tp[4])

    def features(self, reg):
        t, code, rows, ks, bases, etas_nc = super().features(reg)
        if not rows:
            return t, code, rows, ks, bases, etas_nc
        s = self.last[reg][2]
        x = self.x[reg]
        ids, ds = self.R.ids[code], self.R.ds[code]
        hour = time.gmtime(t + 19800).tm_hour
        next_k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
        j0 = max(0, next_k - 1)
        # this bus
        stopped = t - x["stop"] if x["stop"] is not None else 0.0
        v180 = v300 = -1.0
        for tq, sq in x["q"]:
            age = t - tq
            if v300 < 0 and 30.0 <= age <= 300.0:
                v300 = (s - sq) / age
            if v180 < 0 and 30.0 <= age <= 180.0:
                v180 = (s - sq) / age
                break
        trip_ratio = min(4.0, max(0.25, x["act"] / x["exp"])) if x["exp"] > 60.0 else -1.0
        d_next, d_prev = max(0.0, ds[next_k] - s), max(0.0, s - ds[j0])
        r_ratio, r_ev = self.r_today[code].get(t)
        c_ratio, _ = self.c_today.get(t)
        # same working: the bus ahead and the bus behind
        lead = foll = None
        n_act = 0
        cb = self.code_bus[code]
        for o, (to, so, vo) in list(cb.items()):
            if o == reg:
                continue
            if t - to > ROUTE_ACTIVE_S:
                del cb[o]
                continue
            n_act += 1
            g = so - s
            if g > 0 and (lead is None or g < lead[0]):
                lead = (g, t - to, vo)
            elif g <= 0 and (foll is None or -g < foll):
                foll = -g
        bus = [stopped, x["gap"], v180, v300, trip_ratio, d_next, d_prev, r_ratio, r_ev, c_ratio,
               lead[0] if lead else -1.0, lead[1] if lead else -1.0, lead[2] if lead else -1.0,
               foll if foll is not None else -1.0, n_act]
        # per target: history time of whole segments from stop j0, last-pass time of the path, history coverage
        max_k = ks[-1]
        hfull, th, tl, cov_n, lp_n, lp_age, n = {j0: 0.0}, 0.0, 0.0, 0, 0, 0.0, 0
        per_k = {}
        for seg in range(j0, max_k):
            sk = f"{ids[seg]}|{ids[seg + 1]}"
            L = max(1.0, ds[seg + 1] - ds[seg])
            v, has = self.vh(sk, hour)
            th += L / v
            hfull[seg + 1] = th
            rem = max(0.0, ds[seg + 1] - max(s, ds[seg]))
            q = self.seg_trav.get(sk)
            tt = [(tq, u) for tq, u in q if t - tq <= LASTPASS_S] if q else None
            if tt:
                tl += (sum(u for _, u in tt) / len(tt)) * (rem / L)
                lp_n += 1; lp_age += t - tt[-1][0]
            else:
                tl += rem / v
            cov_n += has; n += 1
            per_k[seg + 1] = (tl, lp_n / n, lp_age / lp_n if lp_n else -1.0, cov_n / n)
        lc_all = self.last_cross[code]
        mine = x["cross"]
        for row, k, b in zip(rows, ks, bases):
            lr = la = lv = lr2 = -1.0
            got = 0
            for tc, cr in lc_all.get(k, ()):
                if cr is mine or t - tc > LASTPASS_S or tc > t:
                    continue
                a = j0 if j0 in cr else (next_k if next_k in cr and next_k < k else None)
                if a is None or cr[a] >= tc:
                    continue
                h = hfull[k] - hfull[a]
                if h <= 1.0:
                    continue
                ratio = min(5.0, max(0.2, (tc - cr[a]) / h))
                if got == 0:
                    lr, la = ratio, t - tc
                    lv = ratio * row[5] / b              # leader ratio x history ETA, against the nowcast
                else:
                    lr2 = ratio
                got += 1
            tl_k, lp_cov, lp_a, hcov = per_k[k]
            tl_k = max(5.0, tl_k)
            row.extend(bus)
            row.extend([lr, la, lv, lr2, tl_k, tl_k / b, lp_cov, lp_a, hcov])
        return t, code, rows, ks, bases, etas_nc

    def gc(self, now, keep=3 * 3600):
        super().gc(now, keep)
        for reg in [r for r in self.x if r not in self.last]:
            del self.x[reg]
        for code in list(self.last_cross):
            d = self.last_cross[code]
            for k in [k for k, v in d.items() if v[0][0] < now - LASTPASS_S]:
                del d[k]
            if not d:
                del self.last_cross[code]
        for sk in [k for k, q in self.seg_trav.items() if q[-1][0] < now - LASTPASS_S]:
            del self.seg_trav[sk]


class EnsembleV5:
    """The v5 models behind eta_engine.Ensemble's interface, on the 98 v5 columns: a weighted mean of the P50
    models listed in v5_report.json (log(true/nowcast)), and LightGBM P10/P90 widened by the margin of the
    horizon bucket. eta_engine.score(V5Stream, EnsembleV5, regs) works unchanged."""

    def __init__(self, model_dir, threads=1, use=None):
        import json
        import os

        import lightgbm as lgb
        rep = json.load(open(os.path.join(model_dir, "v5_report.json")))
        self.cqr = [rep["cqr_by_bucket"][str(b)] for b in range(5)]
        self.threads, self.m = threads, []
        for name, w in rep["weights"].items():
            if w <= 0 or (use and name not in use):
                continue
            path = os.path.join(model_dir, rep["files"][name])
            if path.endswith(".txt"):
                bst = lgb.Booster(model_file=path)
                fn = lambda X, bst=bst: bst.predict(X, num_threads=self.threads)
            elif path.endswith((".json", ".ubj")):
                import xgboost as xgb
                bst = xgb.Booster(); bst.load_model(path); bst.set_param({"nthread": threads})
                fn = lambda X, bst=bst, xgb=xgb: bst.predict(xgb.DMatrix(X, nthread=self.threads))
            else:
                from catboost import CatBoost
                bst = CatBoost(); bst.load_model(path)
                fn = lambda X, bst=bst: bst.predict(X, thread_count=self.threads)
            self.m.append((name, w, fn))
        tot = sum(w for _, w, _ in self.m)
        self.m = [(n, w / tot, fn) for n, w, fn in self.m]
        self.q10 = lgb.Booster(model_file=os.path.join(model_dir, rep["files"]["q10"]))
        self.q90 = lgb.Booster(model_file=os.path.join(model_dir, rep["files"]["q90"]))

    def predict(self, X, b, margins=None):
        import numpy as np
        X = np.asarray(X, dtype=np.float32)
        r = np.zeros(len(X))
        for _n, w, fn in self.m:
            r += w * fn(X)
        b = np.asarray(b, dtype=float)
        hb = np.digitize(b, [180.0, 300.0, 600.0, 900.0])
        mg = np.asarray(margins if margins is not None else self.cqr)[hb]
        q10 = self.q10.predict(X, num_threads=self.threads)
        q90 = self.q90.predict(X, num_threads=self.threads)
        lo, hi = np.minimum(q10 - mg, r), np.maximum(q90 + mg, r)
        return b * np.exp(r), b * np.exp(lo), b * np.exp(hi), q10, q90, hb
