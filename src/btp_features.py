#!/usr/bin/env python3
"""Bengaluru Traffic Police (BTP) congestion-profile features for v4 BMTC rows.

Input: stop_congestion_how.json from twtrx pipeline/btp_congestion.py
  stops = {BMTC stop_id: [168 ints]}  % of BTP 15-min congestion checks that found a jam
  within 200 m of the stop, per hour of week (index 0 = Monday 00:00 IST).

Features (NaN where no BTP-covered stop; profile is Aug-Oct 2026, so no overlap with Nov rows):
  btp_p_next      p at the next stop, at its estimated arrival hour-of-week
  btp_p_max3      max p over the next 3 stops
  btp_path_mean   mean p over covered stops between the bus and the target stop
  btp_path_cov    share of stops on that path that BTP covers (0 outside coverage)

Arrival hour for stop j = now + b_nc * (ds[j]-s)/(ds[k]-s)   (nowcast baseline, causal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BTP_FEATURES = ["btp_p_next", "btp_p_max3", "btp_path_mean", "btp_path_cov"]


def load_profile(path: Path) -> dict[str, np.ndarray]:
    d = json.loads(Path(path).read_text())
    return {k: np.asarray(v, dtype=np.float32) for k, v in d["stops"].items()}


def compute(X: np.ndarray, rids: np.ndarray, b_nc: np.ndarray, feat_names: list[str],
            routes_ids: dict, routes_ds: dict, prof: dict[str, np.ndarray]) -> np.ndarray:
    fi = {n: i for i, n in enumerate(feat_names)}
    cur_i = X[:, fi["stop_seq_now"]].astype(np.int64)
    k_arr = X[:, fi["stop_seq_target"]].astype(np.int64)
    dist = X[:, fi["dist_ahead_m"]].astype(np.float64)
    dow = X[:, fi["dow"]].astype(np.int64)
    mod = X[:, fi["min_of_day"]].astype(np.float64)
    now_min = dow * 1440.0 + mod

    out = np.full((len(X), 4), np.nan, dtype=np.float32)
    out[:, 3] = 0.0
    # per-route lookup: covered mask + profile matrix
    cache: dict[str, tuple] = {}
    for r in range(len(X)):
        code = rids[r].split(":", 1)[1]
        ent = cache.get(code)
        if ent is None:
            ids = routes_ids.get(code)
            if ids is None:
                cache[code] = ent = (None, None, None)
            else:
                P = np.full((len(ids), 168), np.nan, dtype=np.float32)
                for j, sid in enumerate(ids):
                    p = prof.get(sid)
                    if p is not None:
                        P[j] = p
                cov = ~np.isnan(P[:, 0])
                cache[code] = ent = (P if cov.any() else None, np.asarray(routes_ds[code], dtype=np.float64), cov)
        P, ds, cov = ent
        if P is None:
            continue
        nk = int(cur_i[r]) + 1
        k = int(k_arr[r])
        if nk > k or k >= len(ds):
            continue
        s = ds[k] - dist[r]
        js = np.arange(nk, k + 1)
        frac = np.clip((ds[js] - s) / max(dist[r], 1.0), 0.0, 1.0)
        how = ((now_min[r] + b_nc[r] * frac / 60.0) // 60).astype(np.int64) % 168
        pv = P[js, how]
        out[r, 3] = float(np.mean(cov[js]))
        if not np.isnan(pv[0]):
            out[r, 0] = pv[0]
        p3 = pv[:3]
        if not np.all(np.isnan(p3)):
            out[r, 1] = np.nanmax(p3)
        if not np.all(np.isnan(pv)):
            out[r, 2] = np.nanmean(pv)
    return out
