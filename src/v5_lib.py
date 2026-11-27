"""Shared pieces of the v5 experiments: loading the leak-free caches (v5_build.py), the production baselines
scored on the same rows, metrics with bus-clustered bootstrap intervals, and strictly causal online tools
(settled-outcome statistics, AgACI margins)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(1, str(ROOT / "src"))
sys.path.insert(2, str(ROOT / "reference" / "mumbai"))

import eta_engine as E  # noqa: E402
import eta_engine_v5 as E5  # noqa: E402
from v5_build import OUT, PLAN  # noqa: E402

MODELS = Path(os.environ.get("ETA_MODELS", ROOT / "models"))
F4, F5 = E.V4_FEATURES, E5.V5_FEATURES
CAT = ["city_idx", "bus_class_idx"]
BUCKETS = [("1-5m", 60, 300), ("5-10m", 300, 600), ("10-20m", 600, 1200), ("20-30m", 1200, 1800)]
RTC_SYS = ("tricity", "tgsrtc", "apsrtc2")
B_COL = F4.index("base_eta_nowcast_s")


def load(system, day, cap=None, std=False, seed=0):
    """One system-day. std: only the buses of the audit's sample (u <= PLAN std). cap: at most this many rows,
    drawn by whole fixes at random."""
    p = OUT / "cache" / f"{system}_{day}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    keep = np.ones(len(z["y"]), dtype=bool)
    if std:
        keep &= z["u"] <= PLAN[system]["std"]
    if cap and keep.sum() > cap:
        rng = np.random.default_rng(seed + int(day[-2:]))
        idx = np.flatnonzero(keep)
        keep[:] = False
        keep[rng.choice(idx, cap, replace=False)] = True
    d = {k: z[k][keep] for k in ("X", "y", "b", "t", "k", "reg", "code", "u", "g")}
    d["y"], d["b"] = d["y"].astype(np.float64), d["b"].astype(np.float64)
    d["system"], d["day"], d["groups"] = system, day, z["groups"]
    d["sys_idx"] = np.full(len(d["y"]), PLAN[system]["ci"], dtype=np.int8)
    return d


def stack(parts):
    parts = [p for p in parts if p is not None and len(p["y"])]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "b", "t", "sys_idx")}
    # a bus id that is unique across systems and days (for clustered resampling)
    off, regs = 0, []
    for p in parts:
        regs.append(p["reg"].astype(np.int64) + off)
        off += int(p["reg"].max()) + 1 if len(p["reg"]) else 0
    out["reg"] = np.concatenate(regs)
    return out


def ratio(y, b):
    return np.clip(np.log(y / b), -1.5, 1.5).astype(np.float32)


_ens = {}


def baseline(name, d):
    """Production model on these rows: 'v4' (RTC systems borrow city_idx 3, as live) or 'v4r' (own index).
    Returns P50, lo, hi in seconds with the model's own static margins, and its raw log-ratio q10/q90."""
    if name not in _ens:
        _ens[name] = E.Ensemble(str(MODELS / {"v4": "v4_clean_ensemble", "v4r": "v4r_rtc"}[name]), threads=10)
    X = d["X"][:, :74].copy()
    if name == "v4" and d["system"] in RTC_SYS:
        X[:, 0] = 3
    cache = OUT / "pred" / f"{name}_{d['system']}_{d['day']}_{len(X)}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["p"], z["lo"], z["hi"], z["q10"], z["q90"]
    p, lo, hi, q10, q90, _hb = _ens[name].predict(X, d["b"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, p=p, lo=lo, hi=hi, q10=q10, q90=q90)
    return p, lo, hi, q10, q90


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────────────
def cluster_sums(reg, *cols):
    """Per-bus sums of each column (and the row count), for a bootstrap that resamples whole buses."""
    _, inv = np.unique(reg, return_inverse=True)
    n = np.bincount(inv).astype(float)
    return n, [np.bincount(inv, weights=c) for c in cols]


def boot_ci(n, sums, fn, B=1000, seed=1):
    """95% interval of fn(sum_1/N, sum_2/N, ...) when buses are resampled with replacement."""
    rng = np.random.default_rng(seed)
    m = len(n)
    idx = rng.integers(0, m, size=(B, m))
    N = n[idx].sum(1)
    vals = fn(*[s[idx].sum(1) / N for s in sums])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae_table(d, preds, ref=None, B=1000):
    """preds: {name: P50 seconds}. MAE and bias per model with a bus-clustered 95% CI, the paired difference
    to `ref`, and MAE by true-horizon bucket."""
    y, reg = d["y"], d["reg"]
    out = {"n": int(len(y)), "buses": int(len(np.unique(reg)))}
    ae = {k: np.abs(p - y) for k, p in preds.items()}
    for k, p in preds.items():
        n, (s_ae, s_e) = cluster_sums(reg, ae[k], p - y)
        r = {"mae": float(ae[k].mean()), "mae_ci": boot_ci(n, [s_ae], lambda a: a, B),
             "bias_mean": float(np.mean(p - y)), "bias_median": float(np.median(p - y)),
             "med_ae": float(np.median(ae[k])), "p90_ae": float(np.percentile(ae[k], 90))}
        for name, lo, hi in BUCKETS:
            m = (y >= lo) & (y < hi)
            if m.any():
                r[name] = float(ae[k][m].mean())
        if ref and k != ref and ref in ae:
            n, (s_d,) = cluster_sums(reg, ae[k] - ae[ref])
            r[f"d_vs_{ref}"] = float((ae[k] - ae[ref]).mean())
            r[f"d_vs_{ref}_ci"] = boot_ci(n, [s_d], lambda a: a, B)
        out[k] = r
    return out


def band_stats(y, lo, hi):
    return {"cov80": float(100 * np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo)),
            "width_med": float(np.median(hi - lo)), "miss_lo": float(100 * np.mean(y < lo)), "miss_hi": float(100 * np.mean(y > hi))}


# ── strictly causal online tools ─────────────────────────────────────────────────────────────────────────
def settled_mean(t, settle, v, group, tau, prior_n=0.0):
    """For every row: the exponentially decayed mean (time constant tau s) of v over rows of the same group whose
    outcome was known before this row's prediction time (settle_i <= t). prior_n shrinks towards 0 with that many
    pseudo-observations. Returns (mean, effective n). A prediction made at t only sees arrivals observed by t."""
    mean, neff = np.zeros(len(t)), np.zeros(len(t))
    t0 = float(np.min(t))
    order_g = np.argsort(group, kind="stable")
    gs = group[order_g]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    for idx in np.split(order_g, cuts):
        o = idx[np.argsort(settle[idx], kind="stable")]
        w = np.exp(np.clip((settle[o] - t0) / tau, None, 600.0))
        cw, cv = np.cumsum(w), np.cumsum(w * v[o])
        pos = np.searchsorted(settle[o], t[idx], side="right")
        has = pos > 0
        sw = np.where(has, cw[np.maximum(pos, 1) - 1], 0.0)
        sv = np.where(has, cv[np.maximum(pos, 1) - 1], 0.0)
        dec = np.exp(-np.clip((t[idx] - t0) / tau, None, 600.0))
        n_e = sw * dec                                   # decayed count of settled outcomes
        raw = np.where(has, sv / np.maximum(sw, 1e-300), 0.0)
        mean[idx] = raw * n_e / (n_e + prior_n) if prior_n else raw
        neff[idx] = n_e
    return mean, neff


def agaci_causal(t, settle, r, q10, q90, bucket, init, alpha=0.2, gammas=(0.002, 0.005, 0.015, 0.05), share=1.0, seed=3):
    """AgACI margins per horizon bucket, as the live service runs them (eta_engine.AgACI), replayed strictly in
    time: a prediction at t uses the margins after every outcome settled by t. `share` of rows feed the experts
    (live: a sample of buses and stops). Returns the margin used for each row."""
    rng = np.random.default_rng(seed)
    feed = rng.random(len(t)) < share
    ev = np.concatenate([np.stack([t, np.zeros(len(t)), np.arange(len(t))], 1),
                         np.stack([settle[feed], np.ones(int(feed.sum())), np.flatnonzero(feed)], 1)])
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]            # at equal times, predictions before outcomes
    ex = [[float(init[b])] * len(gammas) for b in range(5)]
    used = np.zeros(len(t))
    med = [float(np.median(e)) for e in ex]
    for _tt, kind, i in ev:
        i = int(i)
        b = int(bucket[i])
        if kind == 0:
            used[i] = med[b]
        else:
            e = ex[b]
            for gi, g in enumerate(gammas):
                miss = not (q10[i] - e[gi] <= r[i] <= q90[i] + e[gi])
                e[gi] = min(0.45, max(-0.15, e[gi] + g * ((1.0 if miss else 0.0) - alpha)))
            med[b] = float(np.median(e))
    return used
