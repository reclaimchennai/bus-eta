#!/usr/bin/env python3
"""Extract rich, leak-free Indian bus segment-traversal & stop-arrival datasets from raw GPS archives.

For each city (delhi, mtc, bmtc, mumbai, ncr), streams day-by-day obs-YYYY-MM-DD.jsonl(.gz)
in strict chronological order and builds two Parquet datasets:

1. segment_events.parquet  — one row per stop-to-stop (or multi-stop spanned) segment traversal:
   - Identity & spatial: city, route_id, seg_key (stopA|stopB), phys_link_id (shared across ALL
     overlapping routes via 50m quantized endpoint coordinates -> 5-20x live probe density!),
     stop_seq, route_num_stops, seg_dist_m, straight_dist_m, sinuosity (curvature proxy),
     mid_lat, mid_lon
   - Temporal: t_start, t_end, day_str, hour_ist, minute_of_day, dow (0=Mon..6=Sun), is_weekend,
     sin_tod, cos_tod, sin_dow, cos_dow
   - Bus & driver behavior: reg (vehicle_id), bus_class (raw 'f' field), reported_k (raw 'k' field),
     veh_pace_ewma (online driver/vehicle speed ratio vs expected baseline, updated causally),
     veh_trip_elapsed_s, veh_trip_dist_m, veh_recent_speed_ms
   - Live fleet & corridor traffic (STRICTLY CAUSAL — only past observations before t_start!):
     * seg_live_v_15m, seg_live_n_15m      (same stop-to-stop segment, last 15 min)
     * corr_live_v_5m, corr_live_v_15m, corr_live_v_30m, corr_live_n_15m
                                           (ALL routes sharing phys_link_id!)
     * corr_speed_cv_30m                   (std/mean of speeds on corridor -> traffic discipline/friction proxy!)
     * corr_stopgo_ratio_30m               (fraction of standing/crawling time -> congestion turbulence!)
     * upstream_live_v_15m                 (previous segment on route, live 15m speed)
     * downstream_live_v_15m               (next segment on route, live 15m speed)
     * city_live_v_15m                     (city-wide live bus speed index!)
   - Bus bunching / headway:
     * lead_headway_s                      (seconds since previous same-route bus passed stopA)
     * corr_lead_headway_s                 (seconds since ANY bus passed this physical link)
   - Targets:
     * dur_s                               (actual traversal time including any stop dwell on segment)
     * eff_speed_ms                        (seg_dist_m / dur_s)
     * run_speed_ms                        (running speed excluding detected stop dwell)

2. stop_arrivals.parquet — one row per observed bus arrival at a stop:
   - Used for:
     (a) explicit Dwell-Time ML model (p_stop, dwell_s given headway, stop, time, weather, bus_class)
     (b) true multi-horizon Stop-Arrival ETA evaluation (1, 3, 5, 10, 15, 30 min ahead!)
     (c) Charul & Biyani (2019) Mask-CNN 2D Trip x Stop matrix construction!

Weather (pol.reclaimchennai.city hyper-local nearest-station for Chennai + Open-Meteo for all cities)
is joined cleanly onto both tables.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
K = 111320.0
STILL_M = 12.0
NEAR_M = 45.0
MAX_DWELL = 600.0
STAND_MAX = 240.0
MAX_SPAN = 6
MAX_SPEED_MS = 25.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    cx = math.cos(math.radians(0.5 * (lat1 + lat2))) * K
    return math.hypot((lon2 - lon1) * cx, (lat2 - lat1) * K)


def quant_link_id(lat1: float, lon1: float, lat2: float, lon2: float, grid_m: float = 60.0) -> str:
    """Quantize directed physical road link endpoints to ~60m grid so overlapping routes share probes."""
    dlat = grid_m / K
    dlon = grid_m / (K * max(0.2, math.cos(math.radians(0.5 * (lat1 + lat2)))))
    return f"{int(round(lat1 / dlat))}_{int(round(lon1 / dlon))}>{int(round(lat2 / dlat))}_{int(round(lon2 / dlon))}"


class Routes:
    """Stop chords, along-route distances, and shared physical link IDs per route code."""

    def __init__(self, stops_path: Path, dist_path: Path):
        rs = json.loads(stops_path.read_text())
        sd = json.loads(dist_path.read_text())
        self.ll: dict[str, list[tuple[float, float]]] = {}
        self.ds: dict[str, list[float]] = {}
        self.ids: dict[str, list[str]] = {}
        self.phys: dict[str, list[str]] = {}
        self.sinuosity: dict[str, list[float]] = {}

        for c, r in rs.items():
            d_list = r.get("d") or []
            if not d_list:
                continue
            s = d_list[0].get("s") or []
            if c in sd and len(sd[c]) == len(s) and len(s) >= 2:
                ll = [(float(p[2]), float(p[3])) for p in s]
                ds = [float(x) for x in sd[c]]
                ids = [str(p[0]) for p in s]
                self.ll[c] = ll
                self.ds[c] = ds
                self.ids[c] = ids
                phys = []
                sinu = []
                for i in range(len(ll) - 1):
                    (a_la, a_lo), (b_la, b_lo) = ll[i], ll[i + 1]
                    phys.append(quant_link_id(a_la, a_lo, b_la, b_lo))
                    chord = max(1.0, haversine_m(a_la, a_lo, b_la, b_lo))
                    seg_d = max(1.0, ds[i + 1] - ds[i])
                    sinu.append(min(4.0, seg_d / chord))
                self.phys[c] = phys
                self.sinuosity[c] = sinu

    def project(self, code: str, lat: float, lon: float, near: int | None = None):
        ll = self.ll.get(code)
        if not ll:
            return None
        ds = self.ds[code]
        cx = math.cos(math.radians(lat)) * K

        def scan(lo: int, hi: int):
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
            if b and b[2] <= 150.0:
                return b
        return scan(0, len(ll) - 1)


def iter_obs_lines(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        try:
            for ln in fh:
                try:
                    yield json.loads(ln)
                except ValueError:
                    continue
        except (EOFError, OSError, zlib.error) as exc:
            print(f"[warn] truncated {path.name}: {exc}", file=sys.stderr)


class LiveCorridorTracker:
    """Strictly causal rolling window tracker for segment, shared-corridor, and city speeds."""

    def __init__(self):
        self.seg_q: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.corr_q: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.city_q: collections.deque = collections.deque()
        self.seg_last_pass: dict[str, float] = {}
        self.corr_last_pass: dict[str, float] = {}

    def _prune(self, q: collections.deque, cutoff: float):
        while q and q[0][0] < cutoff:
            q.popleft()

    def query_seg(self, seg_key: str, now: float) -> tuple[float, int]:
        q = self.seg_q.get(seg_key)
        if not q:
            return (-1.0, 0)
        self._prune(q, now - 900.0)
        if not q:
            return (-1.0, 0)
        tot_d = sum(x[1] for x in q)
        tot_t = sum(x[2] for x in q)
        return (min(MAX_SPEED_MS, tot_d / tot_t) if tot_t > 0 else -1.0, len(q))

    def query_corr(self, phys_id: str, now: float) -> tuple[float, float, float, int, float, float]:
        """Return (v_5m, v_15m, v_30m, n_15m, speed_cv_30m, stopgo_ratio_30m)."""
        q = self.corr_q.get(phys_id)
        if not q:
            return (-1.0, -1.0, -1.0, 0, -1.0, -1.0)
        self._prune(q, now - 1800.0)
        if not q:
            return (-1.0, -1.0, -1.0, 0, -1.0, -1.0)

        d5 = t5 = d15 = t15 = d30 = t30 = 0.0
        n15 = 0
        vs = []
        slow_t = 0.0
        c5, c15 = now - 300.0, now - 900.0
        for ts, dist, dur in q:
            d30 += dist
            t30 += dur
            v = dist / dur if dur > 0 else 0.0
            vs.append(v)
            if v < 1.0:  # < 3.6 km/h crawling/stopped
                slow_t += dur
            if ts >= c15:
                d15 += dist
                t15 += dur
                n15 += 1
                if ts >= c5:
                    d5 += dist
                    t5 += dur
        v5 = min(MAX_SPEED_MS, d5 / t5) if t5 > 0 else -1.0
        v15 = min(MAX_SPEED_MS, d15 / t15) if t15 > 0 else -1.0
        v30 = min(MAX_SPEED_MS, d30 / t30) if t30 > 0 else -1.0
        if len(vs) >= 3:
            mu = sum(vs) / len(vs)
            var = sum((x - mu) ** 2 for x in vs) / len(vs)
            cv = math.sqrt(var) / max(mu, 0.5)
        else:
            cv = -1.0
        stopgo = slow_t / t30 if t30 > 0 else -1.0
        return (v5, v15, v30, n15, cv, stopgo)

    def query_city(self, now: float) -> float:
        self._prune(self.city_q, now - 900.0)
        if len(self.city_q) < 5:
            return -1.0
        tot_d = sum(x[1] for x in self.city_q)
        tot_t = sum(x[2] for x in self.city_q)
        return min(MAX_SPEED_MS, tot_d / tot_t) if tot_t > 0 else -1.0

    def record(self, seg_key: str, phys_id: str, now: float, dist: float, dur: float):
        sq = self.seg_q[seg_key]
        sq.append((now, dist, dur))
        if len(sq) > 40:
            sq.popleft()
        cq = self.corr_q[phys_id]
        cq.append((now, dist, dur))
        if len(cq) > 80:
            cq.popleft()
        self.city_q.append((now, dist, dur))
        if len(self.city_q) > 2000:
            self.city_q.popleft()


def extract_city(
    city: str,
    raw_dir: Path,
    meta_dir: Path,
    out_dir: Path,
    weather_city_path: Path | None,
    chennai_pol_path: Path | None,
    max_gap: int = 180,
    max_days: int | None = None,
) -> None:
    t0 = time.time()
    routes = Routes(meta_dir / "route_stops.json", meta_dir / "stop_dist.json")
    print(f"[{city}] Loaded {len(routes.ll):,} routes from {meta_dir}")

    # Collect unique day files (strict YYYY-MM-DD; prefer .jsonl over .jsonl.gz if both exist)
    import re
    day_re = re.compile(r"^obs-(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")
    by_day: dict[str, Path] = {}
    for p in sorted(raw_dir.glob("obs-*.jsonl*")):
        m = day_re.match(p.name)
        if not m:
            continue
        day_str = m.group(1)
        if day_str not in by_day or p.suffix == ".jsonl":
            by_day[day_str] = p
    days = sorted(by_day.items())
    if max_days:
        days = days[-max_days:]
    print(f"[{city}] Processing {len(days)} days: {days[0][0]} .. {days[-1][0]}")

    seg_rows: list[tuple] = []
    stop_rows: list[tuple] = []

    tracker = LiveCorridorTracker()
    # Per-bus state: reg -> (t, code, s, i, gap, trip_id, trip_t0, trip_s0, pace_ewma, recent_v)
    last_bus: dict[str, tuple] = {}
    # Standing episode per bus: reg -> dict
    stand_ep: dict[str, dict] = {}
    # Per-route-stop last arrival time for leading headway: (code, stop_id) -> last_t
    last_route_stop_t: dict[tuple[str, str], float] = {}
    # Per-physical-stop last arrival time (any route): stop_id -> last_t
    last_any_stop_t: dict[str, float] = {}
    trip_counter = 0

    for day_idx, (day_str, path) in enumerate(days):
        day_seg_start = len(seg_rows)
        # Clear stale state at day boundary
        last_bus.clear()
        stand_ep.clear()
        n_fixes = 0

        for r in iter_obs_lines(path):
            n_fixes += 1
            reg = r.get("reg")
            code = r.get("r")
            t = r.get("t")
            la = r.get("la")
            lo = r.get("lo")
            if not reg or not code or t is None or la is None or lo is None:
                continue
            t = float(t)
            bus_class = str(r.get("f") or "")
            raw_k = float(r["k"]) if r.get("k") is not None else -1.0

            prev = last_bus.get(reg)
            near = prev[3] if prev and prev[1] == code else None
            pr = routes.project(code, float(la), float(lo), near)
            if not pr or pr[2] > 150.0:
                last_bus.pop(reg, None)
                stand_ep.pop(reg, None)
                continue
            s, i, _off = pr
            ds = routes.ds[code]
            ids = routes.ids[code]
            phys = routes.phys[code]
            sinu = routes.sinuosity[code]

            if prev and prev[1] == code and 0.0 < t - prev[0] <= max_gap:
                pt, _pc, ps, pi, pgap, trip_id, trip_t0, trip_s0, pace_ewma, recent_v = prev
                dt_ = t - pt
                d = s - ps

                if abs(d) < STILL_M:
                    e = stand_ep.get(reg)
                    if e is None:
                        stand_ep[reg] = {"ta": pt, "s": ps, "tb": t, "gin": pgap}
                    else:
                        e["tb"] = t
                    if i < len(phys) and (t - stand_ep[reg]["ta"]) <= STAND_MAX:
                        tracker.record(f"{ids[i]}|{ids[i+1]}", phys[i], t, 0.0, dt_)
                    last_bus[reg] = (t, code, s, i, pgap, trip_id, trip_t0, trip_s0, pace_ewma, 0.0)
                    continue

                if d >= STILL_M and d <= MAX_SPEED_MS * dt_:
                    # Resolve any standing episode right before moving
                    dwell_added = 0.0
                    e = stand_ep.pop(reg, None)
                    if e is not None:
                        j_near = min(range(len(ds)), key=lambda q: abs(ds[q] - e["s"]))
                        if abs(ds[j_near] - e["s"]) <= NEAR_M and 0 < j_near < len(ds) - 1:
                            dur_dw = (e["tb"] - e["ta"]) + 0.25 * ((e["gin"] or 0.0) + (t - e["tb"]))
                            if dur_dw <= MAX_DWELL:
                                dwell_added = dur_dw
                                sid = ids[j_near]
                                lhw = e["ta"] - last_route_stop_t.get((code, sid), -1e9)
                                lhw = lhw if 0 < lhw <= 7200 else -1.0
                                ahw = e["ta"] - last_any_stop_t.get(sid, -1e9)
                                ahw = ahw if 0 < ahw <= 7200 else -1.0
                                last_route_stop_t[(code, sid)] = e["ta"]
                                last_any_stop_t[sid] = e["ta"]
                                stop_rows.append((
                                    city, day_str, trip_id, reg, bus_class, code, sid, j_near,
                                    len(ds), e["ta"], 1, dur_dw, lhw, ahw,
                                    routes.ll[code][j_near][0], routes.ll[code][j_near][1],
                                ))

                    # Record stop arrivals for stops crossed between pi and i
                    v_pair = d / dt_
                    for j_cross in range(pi + 1, i + 1):
                        if 0 < j_cross < len(ds) - 1:
                            frac = (ds[j_cross] - ps) / max(d, 1e-6)
                            t_arr = pt + frac * dt_
                            sid = ids[j_cross]
                            # Only add pass-through if we didn't just log a dwell for this exact stop
                            if e is None or abs(ds[j_cross] - e["s"]) > NEAR_M:
                                lhw = t_arr - last_route_stop_t.get((code, sid), -1e9)
                                lhw = lhw if 0 < lhw <= 7200 else -1.0
                                ahw = t_arr - last_any_stop_t.get(sid, -1e9)
                                ahw = ahw if 0 < ahw <= 7200 else -1.0
                                last_route_stop_t[(code, sid)] = t_arr
                                last_any_stop_t[sid] = t_arr
                                stop_rows.append((
                                    city, day_str, trip_id, reg, bus_class, code, sid, j_cross,
                                    len(ds), t_arr, 0, 0.0, lhw, ahw,
                                    routes.ll[code][j_cross][0], routes.ll[code][j_cross][1],
                                ))

                    # Emit segment traversal rows (crediting spanned segments up to MAX_SPAN)
                    if i - pi <= MAX_SPAN:
                        dt_obj = dt.datetime.fromtimestamp(pt, IST)
                        hour_ist = dt_obj.hour
                        min_of_day = hour_ist * 60 + dt_obj.minute
                        dow = dt_obj.weekday()
                        city_v = tracker.query_city(pt)

                        for seg_i in range(pi, i + 1):
                            if seg_i == 0 or seg_i + 2 >= len(ids):
                                continue
                            a_ = max(ps, ds[seg_i])
                            b_ = min(s, ds[seg_i + 1])
                            seg_d = b_ - a_
                            if seg_d < 15.0:
                                continue
                            share = seg_d / d
                            seg_dur = dt_ * share
                            if seg_dur < 1.0 or seg_dur > 600.0:
                                continue

                            seg_key = f"{ids[seg_i]}|{ids[seg_i + 1]}"
                            phys_id = phys[seg_i]

                            # Strictly causal live queries BEFORE recording this traversal!
                            seg_v15, seg_n15 = tracker.query_seg(seg_key, pt)
                            c_v5, c_v15, c_v30, c_n15, c_cv, c_stopgo = tracker.query_corr(phys_id, pt)
                            up_v15 = tracker.query_seg(f"{ids[seg_i - 1]}|{ids[seg_i]}", pt)[0] if seg_i >= 2 else -1.0
                            dn_v15 = tracker.query_seg(f"{ids[seg_i + 1]}|{ids[seg_i + 2]}", pt)[0] if seg_i + 2 < len(ids) else -1.0

                            lead_hw = pt - tracker.seg_last_pass.get(seg_key, -1e9)
                            lead_hw = lead_hw if 0 < lead_hw <= 7200 else -1.0
                            corr_hw = pt - tracker.corr_last_pass.get(phys_id, -1e9)
                            corr_hw = corr_hw if 0 < corr_hw <= 7200 else -1.0

                            full_seg_len = ds[seg_i + 1] - ds[seg_i]
                            (la1, lo1), (la2, lo2) = routes.ll[code][seg_i], routes.ll[code][seg_i + 1]
                            mid_la, mid_lo = 0.5 * (la1 + la2), 0.5 * (lo1 + lo2)

                            seg_rows.append((
                                city, day_str, trip_id, reg, bus_class, raw_k, code, seg_key, phys_id,
                                seg_i, len(ds), full_seg_len, seg_d, sinu[seg_i], mid_la, mid_lo,
                                pt, t, hour_ist, min_of_day, dow, int(dow >= 5),
                                pace_ewma, pt - trip_t0, max(0.0, ps - trip_s0), recent_v,
                                seg_v15, seg_n15, c_v5, c_v15, c_v30, c_n15, c_cv, c_stopgo,
                                up_v15, dn_v15, city_v, lead_hw, corr_hw,
                                seg_dur, seg_d / seg_dur,
                            ))

                            # Update causal tracker AFTER querying
                            tracker.record(seg_key, phys_id, t, seg_d, seg_dur)
                            tracker.seg_last_pass[seg_key] = t
                            tracker.corr_last_pass[phys_id] = t

                    # Update per-bus online pace EWMA against live corridor/city speed
                    ref_v = tracker.query_corr(phys[min(i, len(phys) - 1)], t)[1]
                    if ref_v <= 0:
                        ref_v = 4.2
                    ratio = max(0.5, min(2.0, v_pair / ref_v))
                    new_pace = pace_ewma + 0.25 * (ratio - pace_ewma)
                    last_bus[reg] = (t, code, s, i, dt_, trip_id, trip_t0, trip_s0, new_pace, v_pair)
                    continue

                # Backwards or teleport jump -> new trip
                stand_ep.pop(reg, None)

            # Start a new trip for this bus
            trip_counter += 1
            last_bus[reg] = (t, code, s, i, 0.0, trip_counter, t, s, 1.0, -1.0)

        print(f"  [{city}] {day_str}: {n_fixes:,} fixes -> +{len(seg_rows) - day_seg_start:,} seg events", flush=True)

    seg_schema = [
        "city", "day", "trip_id", "reg", "bus_class", "raw_k", "route_id", "seg_key", "phys_link_id",
        "stop_seq", "route_num_stops", "full_seg_len_m", "seg_dist_m", "sinuosity", "mid_lat", "mid_lon",
        "t_start", "t_end", "hour_ist", "min_of_day", "dow", "is_weekend",
        "veh_pace_ewma", "trip_elapsed_s", "trip_dist_m", "veh_recent_v",
        "seg_live_v_15m", "seg_live_n_15m", "corr_live_v_5m", "corr_live_v_15m", "corr_live_v_30m",
        "corr_live_n_15m", "corr_speed_cv_30m", "corr_stopgo_30m",
        "up_live_v_15m", "dn_live_v_15m", "city_live_v_15m", "lead_headway_s", "corr_lead_headway_s",
        "dur_s", "eff_speed_ms",
    ]
    stop_schema = [
        "city", "day", "trip_id", "reg", "bus_class", "route_id", "stop_id", "stop_seq",
        "route_num_stops", "t_arr", "stopped", "dwell_s", "lead_headway_s", "any_headway_s",
        "lat", "lon",
    ]

    df_seg = pl.DataFrame(seg_rows, schema=seg_schema, orient="row")
    df_stop = pl.DataFrame(stop_rows, schema=stop_schema, orient="row")

    # Add cyclical time features + hour_ts for weather join
    df_seg = df_seg.with_columns(
        ((pl.col("t_start").cast(pl.Int64) // 3600) * 3600).alias("hour_ts"),
        (pl.col("min_of_day") * (2.0 * math.pi / 1440.0)).sin().alias("sin_tod"),
        (pl.col("min_of_day") * (2.0 * math.pi / 1440.0)).cos().alias("cos_tod"),
        (pl.col("dow") * (2.0 * math.pi / 7.0)).sin().alias("sin_dow"),
        (pl.col("dow") * (2.0 * math.pi / 7.0)).cos().alias("cos_dow"),
    )
    df_stop = df_stop.with_columns(
        ((pl.col("t_arr").cast(pl.Int64) // 3600) * 3600).alias("hour_ts"),
    )

    # Join unified city weather (+ Chennai pol.reclaimchennai.city hyper-local nearest station!)
    if weather_city_path and weather_city_path.exists():
        w_df = pl.read_parquet(weather_city_path).filter(pl.col("city") == city).drop("city")
        df_seg = df_seg.join(w_df, on="hour_ts", how="left")
        df_stop = df_stop.join(w_df, on="hour_ts", how="left")

    # For Chennai, also join nearest pol.reclaimchennai.city rain/AQI station per segment!
    if city == "mtc" and chennai_pol_path and chennai_pol_path.exists():
        pol_df = pl.read_parquet(chennai_pol_path)
        st_coords = (
            pol_df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
            .select(["uid", "lat", "lon"])
            .unique()
        )
        if len(st_coords) > 0 and len(df_seg) > 0:
            st_uids = st_coords["uid"].to_list()
            st_lat = st_coords["lat"].to_numpy()
            st_lon = st_coords["lon"].to_numpy()
            # Map each unique seg_key to its nearest Chennai weather station
            uniq_segs = df_seg.select(["seg_key", "mid_lat", "mid_lon"]).unique()
            s_lat = uniq_segs["mid_lat"].to_numpy()[:, None]
            s_lon = uniq_segs["mid_lon"].to_numpy()[:, None]
            dist2 = (s_lat - st_lat[None, :]) ** 2 + (s_lon - st_lon[None, :]) ** 2
            nearest_idx = np.argmin(dist2, axis=1)
            seg_to_uid = pl.DataFrame({
                "seg_key": uniq_segs["seg_key"],
                "nearest_wx_uid": [st_uids[k] for k in nearest_idx],
            })
            pol_keep = [c for c in ["precip_mm", "rain_accum_mm", "pm25", "aqi", "vis_km"] if c in pol_df.columns]
            pol_ren = pol_df.select(["uid", "hour_ts"] + pol_keep).rename(
                {c: f"local_{c}" for c in pol_keep}
            )
            df_seg = (
                df_seg.join(seg_to_uid, on="seg_key", how="left")
                .join(pol_ren, left_on=["nearest_wx_uid", "hour_ts"], right_on=["uid", "hour_ts"], how="left")
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_out = out_dir / f"{city}_segments.parquet"
    stop_out = out_dir / f"{city}_stops.parquet"
    df_seg.write_parquet(seg_out)
    df_stop.write_parquet(stop_out)
    print(
        f"[{city}] DONE in {time.time() - t0:.1f}s: "
        f"{len(df_seg):,} segment events -> {seg_out.name}, "
        f"{len(df_stop):,} stop arrivals -> {stop_out.name}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["delhi", "mtc", "bmtc", "mumbai", "ncr"])
    ap.add_argument("--data-root", default="~/projects/ETA/data")
    ap.add_argument("--max-gap", type=int, default=0, help="0 = auto per city (180 for dense, 600 for sparse Chalo)")
    ap.add_argument("--max-days", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.data_root)
    # Sparse Chalo feeds (mtc, mumbai) have longer gaps between fixes; dense feeds (delhi, bmtc, ncr) use 180s
    max_gap = a.max_gap or (600 if a.city in ("mtc", "mumbai") else 180)
    extract_city(
        city=a.city,
        raw_dir=root / "raw" / a.city,
        meta_dir=root / "meta" / a.city,
        out_dir=root / "processed",
        weather_city_path=root / "weather" / "weather_city_hourly.parquet",
        chennai_pol_path=root / "weather" / "chennai_pol_hourly.parquet",
        max_gap=max_gap,
        max_days=a.max_days or None,
    )


if __name__ == "__main__":
    main()
