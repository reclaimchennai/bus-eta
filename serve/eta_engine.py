"""Streaming v4 ETA engine: the training replay (train_v3_ensemble_agaci.build_v3_day) turned into a
service, so the features a live bus gets are computed by the same steps, in the same order, as the
ones the models were trained on.

Feed it the collector's archive lines (obs-DATE.jsonl, one GPS fix per line, in the order the
collector wrote them) with ingest(); each fix on a known route working updates the fleet-wide live
state (segment probes, corridor speeds, per-bus pace) exactly as in training, and queues the bus.
predict() then scores every queued bus at ITS fix time: arrival times at the next stops, with an
80% range.

Kept stdlib + numpy + the three GBDT libraries. Nothing here reads a future fix.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import time
import zlib

import numpy as np

import nowcast

K = 111320.0
MAX_SPEED_MS = 25.0
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CITIES = ["mtc", "bmtc", "mumbai", "ncr", "delhi"]          # city_idx order of the trained models
CLASS_VOCAB = ["", "u", "a", "n", "k", "dn", "ln", "dc", "lc", "ev", "ac", "e", "o", "s", "d"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_VOCAB)}
HORIZON_MAX = float(os.environ.get("ETA_HMAX", 1e9))   # the model scores every stop ahead (training did too: it kept stops nowcast put past 30 min); 1800 = nowcast-scaled beyond that (worse: 109 vs 93 s on MTC 23 Nov)
N_AHEAD = 30

V2_FEATURES = [
    "city_idx", "bus_class_idx", "dist_ahead_m", "n_stops_ahead", "base_eta_nowcast_s", "base_eta_hist_s",
    "base_eta_corr_s", "live_vs_hist_ratio", "stop_seq_now", "stop_seq_target", "route_num_stops", "progress_now",
    "progress_target", "path_mean_sinuosity", "lat_now", "lon_now", "hour_ist", "min_of_day", "dow", "is_weekend",
    "sin_tod", "cos_tod", "sin_dow", "cos_dow", "nc_pace", "veh_pace_ewma", "veh_recent_v", "trip_elapsed_s",
    "trip_dist_m", "cur_seg_live_v_15m", "cur_corr_live_v_5m", "cur_corr_live_v_15m", "cur_corr_live_v_30m",
    "cur_corr_n_15m", "cur_corr_cv_30m", "cur_corr_stopgo_30m", "up_live_v_15m", "dn_live_v_15m",
    "path_live_corr_v", "path_live_cov_frac", "city_live_v_15m", "lead_headway_s", "corr_lead_headway_s",
    "temp_c", "humidity", "precip_mm", "precip_3h_mm", "precip_6h_mm", "precip_24h_mm", "is_raining",
    "is_heavy_rain", "wind_kph", "gust_kph", "cloud_cover", "vis_km", "pm25", "aqi", "pol_precip_mean",
    "pol_rain_accum_mean",
]
V3_FEATURES = V2_FEATURES + [
    "path_stops_per_km", "path_max_sinuosity", "path_turn_rad_sum", "path_turn_rad_per_km", "path_osm_signals",
    "path_osm_flyovers", "path_osm_rail_crossings", "path_osm_roundabouts", "path_osm_signals_per_km",
    "path_osm_flyovers_per_km", "is_major_holiday", "is_festival_or_procession", "route_bias_prior",
    "route_wknd_bias_prior", "city_hour_wknd_bias_prior", "rain_x_peak", "decayed_pace_effect",
    "live_ratio_x_horizon",
]
DROP = {"route_bias_prior", "route_wknd_bias_prior", "city_hour_wknd_bias_prior"}
KEEP_IDX = [i for i, c in enumerate(V3_FEATURES) if c not in DROP]
V4_FEATURES = [V3_FEATURES[i] for i in KEEP_IDX]
assert len(V4_FEATURES) == 74


def haversine_m(lat1, lon1, lat2, lon2):      # extract_dataset's (flat) version, as trained
    cx = math.cos(math.radians(0.5 * (lat1 + lat2))) * K
    return math.hypot((lon2 - lon1) * cx, (lat2 - lat1) * K)


def quant_link_id(lat1, lon1, lat2, lon2, grid_m=60.0):
    dlat = grid_m / K
    dlon = grid_m / (K * max(0.2, math.cos(math.radians(0.5 * (lat1 + lat2)))))
    return f"{int(round(lat1 / dlat))}_{int(round(lon1 / dlon))}>{int(round(lat2 / dlat))}_{int(round(lon2 / dlon))}"


def bearing_rad(lat1, lon1, lat2, lon2):
    dlo = math.radians(lon2 - lon1)
    la1, la2 = math.radians(lat1), math.radians(lat2)
    return math.atan2(math.sin(dlo) * math.cos(la2), math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo))


class OSMIndex:
    """Same counts as osm_infra_fetch.CityOSMIndex (points within 65 m of a chord's two ends and midpoint),
    on a plain grid instead of scipy's KD-tree."""
    KINDS = ("signals", "flyovers", "rail_crossings", "roundabouts")

    def __init__(self, path, cell=65.0):
        data = json.load(open(path)) if path and os.path.exists(path) else {}
        self.cell, self.cx, self.grid, self.pts = cell, 100000.0, {}, {}
        for kind in ("signals", "rail_crossings", "speed_bumps", "roundabouts", "flyovers"):   # cx: last non-empty kind
            p = data.get(kind) or []
            if p:
                self.cx = math.cos(math.radians(sum(x[0] for x in p) / len(p))) * K
        for kind in self.KINDS:
            xy = [(la * K, lo * self.cx) for la, lo in (data.get(kind) or [])]
            g = collections.defaultdict(list)
            for j, (y, x) in enumerate(xy):
                g[(int(y // cell), int(x // cell))].append(j)
            self.pts[kind], self.grid[kind] = xy, g

    def count(self, la1, lo1, la2, lo2, r=65.0):
        q = [(la1 * K, lo1 * self.cx), (0.5 * (la1 + la2) * K, 0.5 * (lo1 + lo2) * self.cx), (la2 * K, lo2 * self.cx)]
        out = []
        for kind in self.KINDS:
            g, xy, hits = self.grid[kind], self.pts[kind], set()
            for y, x in q:
                cy, cx_ = int(y // self.cell), int(x // self.cell)
                for a in (cy - 1, cy, cy + 1):
                    for b in (cx_ - 1, cx_, cx_ + 1):
                        for j in g.get((a, b), ()):
                            if (xy[j][0] - y) ** 2 + (xy[j][1] - x) ** 2 <= r * r:
                                hits.add(j)
            out.append(len(hits))
        return out


class Routes:
    """route_stops.json + stop_dist.json, as travel_model.Routes (projection) and extract_dataset.Routes (the rest)."""

    def __init__(self, stops_path, dist_path, osm_path=None):
        rs, sd = json.load(open(stops_path)), json.load(open(dist_path))
        self.ll, self.ds, self.ids, self.phys, self.sinu, self.turns, self.osm = {}, {}, {}, {}, {}, {}, {}
        osm = OSMIndex(osm_path)
        for c, r in rs.items():
            d_list = r.get("d") or []
            if not d_list:
                continue
            s = d_list[0].get("s") or []
            if not (c in sd and len(sd[c]) == len(s) and len(s) >= 2):
                continue
            ll = [(float(p[2]), float(p[3])) for p in s]
            ds = [float(x) for x in sd[c]]
            self.ll[c], self.ds[c], self.ids[c] = ll, ds, [str(p[0]) for p in s]
            phys, sinu = [], []
            for i in range(len(ll) - 1):
                (a1, o1), (a2, o2) = ll[i], ll[i + 1]
                phys.append(quant_link_id(a1, o1, a2, o2))
                sinu.append(min(4.0, max(1.0, ds[i + 1] - ds[i]) / max(1.0, haversine_m(a1, o1, a2, o2))))   # as trained
            self.phys[c], self.sinu[c] = phys, sinu
            turns, acc, prev = [0.0] * len(ll), 0.0, None
            for i in range(len(ll) - 1):
                b = bearing_rad(ll[i][0], ll[i][1], ll[i + 1][0], ll[i + 1][1])
                if prev is not None:
                    d = abs(b - prev)
                    acc += 2.0 * math.pi - d if d > math.pi else d
                turns[i + 1], prev = acc, b
            self.turns[c] = turns
            cum = [[0] * len(ll) for _ in range(4)]
            run = [0, 0, 0, 0]
            for i in range(len(ll) - 1):
                n = osm.count(ll[i][0], ll[i][1], ll[i + 1][0], ll[i + 1][1])
                for j in range(4):
                    run[j] += n[j]
                    cum[j][i + 1] = run[j]
            self.osm[c] = cum
        for c, r in rs.items():                   # an alias ({"a": chain code}: APSRTC's service keys) shares its chain
            a = r.get("a")
            if a in self.ll and c not in self.ll:
                for t in (self.ll, self.ds, self.ids, self.phys, self.sinu, self.turns, self.osm):
                    t[c] = t[a]

    def project(self, code, lat, lon, near=None):
        ll, ds = self.ll.get(code), self.ds.get(code)
        if not ll:
            return None
        cx = math.cos(math.radians(lat)) * K

        def scan(lo, hi):
            best = None
            for i in range(max(0, lo), min(len(ll) - 1, hi)):
                (ay, ax), (by, bx) = ll[i], ll[i + 1]
                vx, vy = (bx - ax) * cx, (by - ay) * K
                px, py = (lon - ax) * cx, (lat - ay) * K
                L2 = vx * vx + vy * vy
                t = max(0.0, min(1.0, (px * vx + py * vy) / L2)) if L2 else 0.0
                off = math.hypot(px - t * vx, py - t * vy)
                if best is None or off < best[2]:
                    best = (ds[i] + t * (ds[i + 1] - ds[i]), i, off)
            return best

        if near is not None:
            b = scan(near - 2, near + 7)
            if b and b[2] <= 150:
                return b
        return scan(0, len(ll) - 1)


class Tracker:
    """extract_dataset.LiveCorridorTracker, unchanged."""

    def __init__(self):
        self.seg_q = collections.defaultdict(collections.deque)
        self.corr_q = collections.defaultdict(collections.deque)
        self.city_q = collections.deque()
        self.seg_last_pass, self.corr_last_pass = {}, {}

    @staticmethod
    def _prune(q, cutoff):
        while q and q[0][0] < cutoff:
            q.popleft()

    def query_seg(self, key, now):
        q = self.seg_q.get(key)
        if not q:
            return (-1.0, 0)
        self._prune(q, now - 900.0)
        if not q:
            return (-1.0, 0)
        td, tt = sum(x[1] for x in q), sum(x[2] for x in q)
        return (min(MAX_SPEED_MS, td / tt) if tt > 0 else -1.0, len(q))

    def query_corr(self, pid, now):
        q = self.corr_q.get(pid)
        if not q:
            return (-1.0, -1.0, -1.0, 0, -1.0, -1.0)
        self._prune(q, now - 1800.0)
        if not q:
            return (-1.0, -1.0, -1.0, 0, -1.0, -1.0)
        d5 = t5 = d15 = t15 = d30 = t30 = slow = 0.0
        n15, vs = 0, []
        c5, c15 = now - 300.0, now - 900.0
        for ts, d, u in q:
            d30 += d; t30 += u
            v = d / u if u > 0 else 0.0
            vs.append(v)
            if v < 1.0:
                slow += u
            if ts >= c15:
                d15 += d; t15 += u; n15 += 1
                if ts >= c5:
                    d5 += d; t5 += u
        v5 = min(MAX_SPEED_MS, d5 / t5) if t5 > 0 else -1.0
        v15 = min(MAX_SPEED_MS, d15 / t15) if t15 > 0 else -1.0
        v30 = min(MAX_SPEED_MS, d30 / t30) if t30 > 0 else -1.0
        if len(vs) >= 3:
            mu = sum(vs) / len(vs)
            cv = math.sqrt(sum((x - mu) ** 2 for x in vs) / len(vs)) / max(mu, 0.5)
        else:
            cv = -1.0
        return (v5, v15, v30, n15, cv, slow / t30 if t30 > 0 else -1.0)

    def query_city(self, now):
        self._prune(self.city_q, now - 900.0)
        if len(self.city_q) < 5:
            return -1.0
        td, tt = sum(x[1] for x in self.city_q), sum(x[2] for x in self.city_q)
        return min(MAX_SPEED_MS, td / tt) if tt > 0 else -1.0

    def record(self, sk, pid, now, d, u):
        q = self.seg_q[sk]; q.append((now, d, u))
        if len(q) > 40:
            q.popleft()
        q = self.corr_q[pid]; q.append((now, d, u))
        if len(q) > 80:
            q.popleft()
        self.city_q.append((now, d, u))
        if len(self.city_q) > 2000:
            self.city_q.popleft()

    def gc(self, now):
        for qd in (self.seg_q, self.corr_q):
            for k in [k for k, q in qd.items() if not q or q[-1][0] < now - 1800]:
                del qd[k]


def holiday_flags(day, city, holidays):
    info = holidays.get(day)
    if not info:
        return 0.0, 0.0
    sc = info.get("scope", [])
    if "all" not in sc and city not in sc:
        return 0.0, 0.0
    imp = str(info.get("impact", ""))
    return (1.0 if imp == "major_holiday" else 0.0,
            1.0 if imp in ("festival_eve", "procession_peak", "restricted_holiday") else 0.0)


WX_KEYS = ["temp_c", "humidity", "precip_mm", "precip_3h_mm", "precip_6h_mm", "precip_24h_mm", "is_raining",
           "is_heavy_rain", "wind_kph", "gust_kph", "cloud_cover", "vis_km", "pm25", "aqi", "pol_precip_mean",
           "pol_rain_accum_mean"]
WX_ZERO_DEFAULT = {"precip_mm", "precip_3h_mm", "precip_6h_mm", "precip_24h_mm", "is_raining", "is_heavy_rain"}


def wx_vector(wx):
    out = []
    for k in WX_KEYS:
        v = wx.get(k)
        out.append(float(v) if v is not None else (0.0 if k in WX_ZERO_DEFAULT else np.nan))
    return out


class CityStream:
    """One city's live state. ingest() in archive order; predict() for the buses with new fixes."""

    def __init__(self, city, routes, model_json=None, holidays=None, weather=None, max_gap=None, city_idx=None):
        # a system the models never saw borrows a trained city's index (city_idx, chosen on a validation day)
        self.city = city
        self.ci = city_idx if city_idx is not None else (CITIES.index(city) if city in CITIES else 0)
        self.R = routes
        self.M = nowcast.Model(model_json) if model_json and os.path.exists(model_json) else nowcast.Model(None)
        self.pace = nowcast.Pace()
        self.tr = Tracker()
        self.hol = holidays or {}
        self.weather = weather          # callable(city, hour_ts) -> dict, or None
        self.max_gap = max_gap or (600 if city in ("mtc", "mumbai") else 180)
        self.trip_gap = max(self.max_gap, 300)
        self.last = {}                  # reg -> (t, code, s, i, f) last projected fix
        self.trip = {}                  # reg -> [code, t0, s0, pace_ewma, recent_v, prev_fix(t,s,i)]
        self.dirty = {}                 # reg -> t of the newest fix not yet scored
        self.n_in = self.n_used = 0
        self.cal = None                 # AgACI, set by the service
        self.cal_share = 0.25           # share of buses whose predictions feed the calibration
        self.pending = {}               # reg -> deque of (fix t, code, k, b_nowcast, q10, q90, bucket)

    def cal_pick(self, reg):
        return (zlib.crc32(reg.encode()) & 1023) < self.cal_share * 1024

    def crossed(self, reg, code, k, t_cross):
        """A bus crossed stop k at t_cross: settle the predictions made for it (training's truth)."""
        q = self.pending.get(reg)
        if not q:
            return
        keep = collections.deque(maxlen=q.maxlen)
        for item in q:
            t0, c, kk, b, q10, q90, bk = item
            if c == code and kk == k:
                y = t_cross - t0
                if 60.0 <= y < 1800.0:
                    self.cal.update(bk, min(1.5, max(-1.5, math.log(y / b))), q10, q90)
                continue
            if t_cross - t0 < 1800.0 and not (c == code and kk < k):
                keep.append(item)
        self.pending[reg] = keep

    # ── training: load_day_tracks + extract_ground_truth_stop_times + the replay's probe block ──
    def ingest(self, r):
        self.n_in += 1
        reg, code = r.get("reg"), r.get("r")
        if not reg or not code or code not in self.R.ll:
            return
        try:
            t, la, lo = float(r["t"]), float(r["la"]), float(r["lo"])
        except (KeyError, TypeError, ValueError):
            return
        prev = self.last.get(reg)
        if prev and t <= prev[0]:
            return                       # the collector re-wrote an older fix
        near = prev[3] if prev and prev[1] == code else None
        pr = self.R.project(code, la, lo, near)
        if not pr or pr[2] > 150:
            return
        s, i, f = float(pr[0]), int(pr[1]), str(r.get("f") or "")
        self.last[reg] = (t, code, s, i, f)
        self.n_used += 1
        tp = self.trip.get(reg)
        if tp is None or tp[0] != code or t - tp[5][0] > self.trip_gap or s < tp[5][1] - 30.0:
            self.trip[reg] = [code, t, s, 1.0, -1.0, (t, s, i)]
            self.dirty[reg] = t
            return
        pt, ps, pi = tp[5]
        ids, ds, phys = self.R.ids[code], self.R.ds[code], self.R.phys[code]
        dt_, d = t - pt, s - ps
        hour = time.gmtime(t + 19800).tm_hour
        if self.cal is not None and reg in self.pending and d > 5.0 and dt_ > 0 and d <= 25.0 * dt_:
            for k in range(pi + 1, i + 1):           # eval_eta.extract_ground_truth_stop_times
                if 0 < k < len(ds) - 1:
                    self.crossed(reg, code, k, pt + (ds[k] - ps) / d * dt_)
        if 0 < dt_ <= self.max_gap and 12.0 <= d <= 25.0 * dt_:
            v_pair = d / dt_
            for j in range(pi, i + 1):
                if j + 1 < len(ids):
                    share = max(0.0, min(s, ds[j + 1]) - max(ps, ds[j])) / d
                    if share > 0:
                        sk = f"{ids[j]}|{ids[j + 1]}"
                        self.M.add_probe(sk, t, v_pair, dt_ * share)
                        self.tr.record(sk, phys[j], t, d * share, dt_ * share)
                        self.tr.seg_last_pass[sk] = t
                        self.tr.corr_last_pass[phys[j]] = t
            self.pace.update(reg, self.M.expected_time(ids, ds, ps, s, hour, t), dt_)
            ref_v = self.tr.query_corr(phys[min(i, len(phys) - 1)], t)[1]
            if ref_v <= 0:
                ref_v = 4.2
            tp[3] = tp[3] + 0.25 * (max(0.5, min(2.0, v_pair / ref_v)) - tp[3])
            tp[4] = v_pair
        elif 0 < dt_ <= self.max_gap and abs(d) < 12.0 and i < len(phys):
            sk = f"{ids[i]}|{ids[i + 1]}"
            self.M.add_probe(sk, t, 0.0, dt_)
            self.tr.record(sk, phys[i], t, 0.0, dt_)
            tp[4] = 0.0
        tp[5] = (t, s, i)
        self.dirty[reg] = t

    # ── training: the query block of build_v3_day, for every stop ahead instead of only observed ones ──
    def features(self, reg):
        t, code, s, i, bus_class = self.last[reg]
        tp = self.trip[reg]
        ids, ds, phys, sinu = self.R.ids[code], self.R.ds[code], self.R.phys[code], self.R.sinu[code]
        turns, (osm_s, osm_f, osm_r, osm_rb) = self.R.turns[code], self.R.osm[code]
        hour = time.gmtime(t + 19800).tm_hour
        next_k = next((q for q in range(1, len(ds)) if ds[q] > s), len(ds) - 1)
        max_k = min(len(ds) - 1, next_k + N_AHEAD - 1)
        nc_p = self.pace.get(reg)
        etas_nc = dict(self.M.eta(ids, ds, s, next_k, t, pace=nc_p, n=N_AHEAD))
        cum_hist, cum_corr, cum_sinu, cum_max_sinu, cum_lcov = {}, {}, {}, {}, {}
        th = tc = sinu_sum = 0.0
        max_sinu, lcov_n, tot_n, here = 1.0, 0, 0, s
        seg_m = self.M.m.get("seg") or {}
        city_v = (self.M.m.get("city") or {}).get("v") or [-1] * 24
        for seg_i in range(next_k - 1, max_k):
            if seg_i < 0 or seg_i >= len(phys):
                continue
            seg_d = max(0.0, ds[seg_i + 1] - here)
            here = ds[seg_i + 1]
            row_m = seg_m.get(f"{ids[seg_i]}|{ids[seg_i + 1]}")
            vh = -1.0
            if row_m:
                vh = row_m[2][hour] if row_m[2][hour] > 0 else row_m[1]
            if not vh or vh <= 0:
                cv = city_v[hour]
                vh = cv if cv and cv > 0 else 4.2
            vh = min(22.0, max(1.0, vh))
            th += seg_d / vh
            cv15 = self.tr.query_corr(phys[seg_i], t)[1]
            if cv15 > 0.5:
                vc = min(22.0, max(1.0, cv15)); lcov_n += 1
            else:
                vc = vh
            tc += seg_d / vc
            sinu_sum += sinu[seg_i]
            max_sinu = max(max_sinu, sinu[seg_i])
            tot_n += 1
            k = seg_i + 1
            cum_hist[k], cum_corr[k], cum_sinu[k], cum_max_sinu[k], cum_lcov[k] = th, tc, sinu_sum / tot_n, max_sinu, lcov_n / tot_n
        cur_i = min(max(0, next_k - 1), len(phys) - 1)
        sk_cur, ph_cur = f"{ids[cur_i]}|{ids[cur_i + 1]}", phys[cur_i]
        cur_seg_v15, _ = self.tr.query_seg(sk_cur, t)
        c_v5, c_v15, c_v30, c_n15, c_cv, c_sg = self.tr.query_corr(ph_cur, t)
        up_v15 = self.tr.query_seg(f"{ids[cur_i - 1]}|{ids[cur_i]}", t)[0] if cur_i >= 1 else -1.0
        dn_v15 = self.tr.query_seg(f"{ids[cur_i + 1]}|{ids[cur_i + 2]}", t)[0] if cur_i + 2 < len(ids) else -1.0
        city_live = self.tr.query_city(t)
        lead_hw = t - self.tr.seg_last_pass.get(sk_cur, -1e9)
        lead_hw = lead_hw if 0 < lead_hw <= 7200 else -1.0
        corr_hw = t - self.tr.corr_last_pass.get(ph_cur, -1e9)
        corr_hw = corr_hw if 0 < corr_hw <= 7200 else -1.0
        dto = dt.datetime.fromtimestamp(t, IST)
        hour_ist, mod, dow = dto.hour, dto.hour * 60 + dto.minute, dto.weekday()
        is_wknd, is_peak = int(dow >= 5), int((8 <= hour_ist <= 11) or (17 <= hour_ist <= 20))
        tod, dw = mod * (2.0 * math.pi / 1440.0), dow * (2.0 * math.pi / 7.0)
        lat_now, lon_now = self.R.ll[code][cur_i]
        n_stops = len(ds)
        wx = (self.weather(self.city, (int(t) // 3600) * 3600) if self.weather else None) or {}
        wxv = wx_vector(wx)
        precip_3h = wxv[3]
        h_maj, h_fest = holiday_flags(dto.date().isoformat(), self.city, self.hol)
        st0, st1, st2, st3 = tp[3], tp[4], tp[1], tp[2]
        rows, ks, bases = [], [], []
        for k in range(next_k, max_k + 1):
            if k not in etas_nc or k not in cum_hist:
                continue
            b_nc = max(5.0, etas_nc[k] - t)
            b_h, b_c = max(5.0, cum_hist[k]), max(5.0, cum_corr[k])
            dist_ahead = max(1.0, ds[k] - s)
            dist_km = max(0.05, dist_ahead / 1000.0)
            n_ahead = k - next_k + 1
            kk = min(k, len(turns) - 1)
            turn_sum = max(0.0, turns[kk] - turns[cur_i])
            n_sig, n_fly = max(0, osm_s[kk] - osm_s[cur_i]), max(0, osm_f[kk] - osm_f[cur_i])
            n_rail, n_rb = max(0, osm_r[kk] - osm_r[cur_i]), max(0, osm_rb[kk] - osm_rb[cur_i])
            live_ratio = b_c / b_h
            feat = [
                self.ci, CLASS_MAP.get(bus_class, -1), dist_ahead, n_ahead,
                b_nc, b_h, b_c, live_ratio,
                cur_i, k, n_stops, cur_i / max(n_stops, 1), k / max(n_stops, 1),
                cum_sinu[k], lat_now, lon_now,
                hour_ist, mod, dow, is_wknd, math.sin(tod), math.cos(tod), math.sin(dw), math.cos(dw),
                nc_p, st0, st1, t - st2, max(0.0, s - st3),
                cur_seg_v15, c_v5, c_v15, c_v30, c_n15, c_cv, c_sg,
                up_v15, dn_v15, dist_ahead / b_c, cum_lcov[k], city_live,
                lead_hw, corr_hw,
            ] + wxv + [
                n_ahead / dist_km, cum_max_sinu[k], turn_sum, turn_sum / dist_km,
                n_sig, n_fly, n_rail, n_rb, n_sig / dist_km, n_fly / dist_km,
                h_maj, h_fest,
                precip_3h * is_peak,
                (st0 - 1.0) * math.exp(-b_nc / 300.0),
                (live_ratio - 1.0) * math.log1p(b_nc / 60.0),
            ]
            rows.append(feat); ks.append(k); bases.append(b_nc)
        return t, code, rows, ks, bases, etas_nc

    def take_dirty(self):
        d, self.dirty = self.dirty, {}
        return list(d)

    def gc(self, now, keep=3 * 3600):
        for reg in [r for r, v in self.last.items() if v[0] < now - keep]:
            self.last.pop(reg, None); self.trip.pop(reg, None); self.pending.pop(reg, None)
        self.tr.gc(now)


class Ensemble:
    """0.40 LightGBM Huber + 0.50 XGBoost Pseudo-Huber + 0.10 CatBoost MAE on log(true/nowcast), and LightGBM
    P10/P90 widened by the calibrated margin of their horizon bucket (the 80% range)."""

    def __init__(self, model_dir, threads=1, use=("lgb", "xgb", "cat")):
        import lightgbm as lgb
        rep = json.load(open(os.path.join(model_dir, "v4_report.json")))
        self.w = dict(zip(("lgb", "xgb", "cat"), rep["weights"]))
        self.cqr = [rep["cqr_by_bucket"][str(b)] for b in range(5)]
        self.m, self.threads = {}, threads
        if "lgb" in use:
            self.m["lgb"] = lgb.Booster(model_file=os.path.join(model_dir, "v4_lgb_huber.txt"))
        if "xgb" in use:
            import xgboost as xgb
            b = xgb.Booster(); b.load_model(os.path.join(model_dir, "v4_xgb_huber.json"))
            b.set_param({"nthread": threads})
            self.m["xgb"] = b
        if "cat" in use:
            from catboost import CatBoost
            c = CatBoost(); c.load_model(os.path.join(model_dir, "v4_cat_mae.cbm"))
            self.m["cat"] = c
        self.q10 = lgb.Booster(model_file=os.path.join(model_dir, "v4_lgb_q10.txt"))
        self.q90 = lgb.Booster(model_file=os.path.join(model_dir, "v4_lgb_q90.txt"))
        tot = sum(self.w[k] for k in self.m)
        self.w = {k: self.w[k] / tot for k in self.m}

    def predict(self, X, b, margins=None):
        """P50 and the 80% range in seconds, plus the raw log-ratio quantiles (for online calibration).
        margins: per horizon bucket (default: the calibration of v4_report.json)."""
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != 74:
            X = X[:, KEEP_IDX]
        r = np.zeros(len(X))
        for k, m in self.m.items():
            if k == "lgb":
                p = m.predict(X, num_threads=self.threads)
            elif k == "xgb":
                import xgboost as xgb
                p = m.predict(xgb.DMatrix(X, nthread=self.threads))
            else:
                p = m.predict(X, thread_count=self.threads)
            r += self.w[k] * p
        b = np.asarray(b, dtype=float)
        hb = np.digitize(b, [180.0, 300.0, 600.0, 900.0])
        mg = np.asarray(margins if margins is not None else self.cqr)[hb]
        q10 = self.q10.predict(X, num_threads=self.threads)
        q90 = self.q90.predict(X, num_threads=self.threads)
        lo, hi = np.minimum(q10 - mg, r), np.maximum(q90 + mg, r)
        return b * np.exp(r), b * np.exp(lo), b * np.exp(hi), q10, q90, hb


class AgACI:
    """Online Aggregated Adaptive Conformal Inference (Zaffran et al. 2022), as in training
    (train_v3_ensemble_agaci.agaci_calibrate_online): per horizon bucket, four experts with step sizes
    GAMMAS move the margin added to the P10/P90 band towards 20% misses; the margin used is their median.
    Fed with arrivals the stream observes after the prediction (stops 60 s .. 30 min ahead, as trained)."""
    GAMMAS = (0.002, 0.005, 0.015, 0.05)

    def __init__(self, init):
        self.ex = [[float(init[bk])] * len(self.GAMMAS) for bk in range(5)]
        self.n = [0] * 5
        self.miss = [0] * 5

    def margins(self):
        return [float(np.median(e)) for e in self.ex]

    def update(self, bk, r_true, q10, q90):
        miss_med = not (q10 - float(np.median(self.ex[bk])) <= r_true <= q90 + float(np.median(self.ex[bk])))
        self.n[bk] += 1; self.miss[bk] += miss_med
        for gi, g in enumerate(self.GAMMAS):
            m = self.ex[bk][gi]
            miss = not (q10 - m <= r_true <= q90 + m)
            self.ex[bk][gi] = min(0.45, max(-0.15, m + g * ((1.0 if miss else 0.0) - 0.2)))


def score(stream, ens, regs, capture=None):
    """Arrival predictions for these buses at their newest fixes:
    reg -> (code, fix_t, [[k, t_eta, t_lo, t_hi, t_nowcast], ...], n_scored) with absolute epoch seconds.
    Stops the models were not trained for (more than 30 min ahead by nowcast, or the terminus) carry the
    nowcast time scaled by the ratio at the last scored stop; times never go backwards along the route."""
    X, owner, out = [], [], {}
    meta = {}
    for reg in regs:
        if reg not in stream.last or reg not in stream.trip:
            continue
        t, code, rows, ks, bases, etas_nc = stream.features(reg)
        if not etas_nc:
            continue
        meta[reg] = (t, code, ks, bases, etas_nc, len(stream.R.ds[code]))
        for row, k, b in zip(rows, ks, bases):
            if b <= HORIZON_MAX and k < len(stream.R.ds[code]) - 1:
                X.append(row); owner.append((reg, k, b))
                if capture is not None:                  # training export (rtc_audit --dump): the rows as scored
                    capture.append((reg, k, b, row))
    if X:
        p, lo, hi, q10, q90, hb = ens.predict(np.asarray(X, dtype=np.float32), np.asarray([o[2] for o in owner]),
                                              stream.cal.margins() if stream.cal else None)
    else:
        p = lo = hi = q10 = q90 = hb = []
    by = collections.defaultdict(dict)
    for (reg, k, b), a, l_, h, c10, c90, bk in zip(owner, p, lo, hi, q10, q90, hb):
        by[reg][k] = (float(a), float(l_), float(h))
        # ~1 stop in 10 per scored fix of a sampled bus, so a bus's queue reaches back ~30 min at any horizon
        if stream.cal is not None and stream.cal_pick(reg) and zlib.crc32(f"{reg}{k}{meta[reg][0]}".encode()) % 10 == 0:
            stream.pending.setdefault(reg, collections.deque(maxlen=120)).append(
                (meta[reg][0], meta[reg][1], k, b, float(c10), float(c90), int(bk)))
    for reg, (t, code, ks, bases, etas_nc, n) in meta.items():
        got, res = by.get(reg, {}), []
        ratio, rl, rh, last = 1.0, 0.8, 1.25, t
        for k in sorted(etas_nc):
            b = max(5.0, etas_nc[k] - t)
            if k in got:
                a, l_, h = got[k]
                ratio, rl, rh = a / b, l_ / b, h / b
            else:
                a, l_, h = b * ratio, b * rl, b * rh
            e = max(last + 1.0, t + a)
            res.append([k, e, min(e, t + l_), max(e, t + h), t + b])
            last = e
        out[reg] = (code, t, res, len(got))
    return out
