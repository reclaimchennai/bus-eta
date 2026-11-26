#!/usr/bin/env python3
"""Independent re-check of the v4 ensemble before it goes live.

What it does, per city:
  1. Re-scores the cached test matrices (reproduces BENCHMARK_RESULTS.md).
  2. Rebuilds the historical speed model (model.json) from the 7 days BEFORE the scored
     day only (--until day-1), exactly as the live server would have had it, then rebuilds
     that day's queries with build_v3_day and scores nowcast and v4 on the same queries.
     (The meta/<city>/model.json used for the published figures was built on 24 Nov from
     the 7 days up to 24 Nov, so it contains the test day itself.)
  3. Splits the scored queries by whether the vehicle also appears in a training day.

Usage: audit_v4.py CITY DAY [--sample 0.35] [--stride 3] [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reference" / "mumbai"))

import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

import eval_eta  # type: ignore  # noqa: E402
import extract_dataset as ed  # type: ignore  # noqa: E402
import train_v2_deepreta_cqr as v2  # type: ignore  # noqa: E402
import train_v3_ensemble_agaci as v3  # type: ignore  # noqa: E402
import train_v4_cached_ensemble as v4  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models" / "v4_clean_ensemble"
CITIES = ["mtc", "bmtc", "mumbai", "ncr", "delhi"]  # city_idx order used in training
CLASS_VOCAB = ["", "u", "a", "n", "k", "dn", "ln", "dc", "lc", "ev", "ac", "e", "o", "s", "d"]
# the production travel_model flags per city (systemd units on raw)
TM_ARGS = {"mtc": ["--max-gap", "1800", "--max-span", "40"], "bmtc": ["--max-gap", "1800", "--max-span", "40"],
           "mumbai": ["--max-gap", "1800", "--max-span", "40"], "delhi": [], "ncr": []}
TRAIN_DAYS = {c: v[0] for c, v in {
    "mtc": (["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],),
    "bmtc": (["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],),
    "mumbai": (["2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"],),
    "ncr": (["2026-11-13", "2026-11-14", "2026-11-15", "2026-11-16"],),
    "delhi": (["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],),
}.items()}


class Ensemble:
    def __init__(self, d: Path = MODELS):
        rep = json.loads((d / "v4_report.json").read_text())
        self.w = rep["weights"]
        self.cqr = {int(k): v for k, v in rep["cqr_by_bucket"].items()}
        self.lgb = lgb.Booster(model_file=str(d / "v4_lgb_huber.txt"))
        self.xgb = xgb.XGBRegressor(); self.xgb.load_model(str(d / "v4_xgb_huber.json"))
        self.cat = CatBoostRegressor(); self.cat.load_model(str(d / "v4_cat_mae.cbm"))
        self.q10 = lgb.Booster(model_file=str(d / "v4_lgb_q10.txt"))
        self.q90 = lgb.Booster(model_file=str(d / "v4_lgb_q90.txt"))

    def predict(self, X: np.ndarray, b: np.ndarray):
        w1, w2, w3 = self.w
        r = w1 * self.lgb.predict(X) + w2 * self.xgb.predict(X) + w3 * self.cat.predict(X)
        hb = v2.horizon_bucket_idx(b)
        m = np.array([self.cqr[int(h)] for h in hb])
        lo = self.q10.predict(X) - m
        hi = self.q90.predict(X) + m
        return b * np.exp(r), b * np.exp(lo), b * np.exp(hi)


def summary(y, p, lo=None, hi=None):
    recs = [(float(a), float(b_), None if lo is None else float(c), None if hi is None else float(d))
            for a, b_, c, d in zip(y, p, lo if lo is not None else [None] * len(y), hi if hi is not None else [None] * len(y))]
    return eval_eta.summarize_errors(recs)


def honest_model_json(city: str, day: str, out_dir: Path) -> Path | None:
    meta = DATA / "meta" / city
    if not (meta / "model.json").exists():
        return None  # the published run also had none (Model(None)) for this city
    until = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    out = out_dir / f"model_{city}_until_{until}.json"
    if not out.exists():
        cmd = [sys.executable, str(ROOT / "reference" / "mumbai" / "travel_model.py"),
               "--stops", str(meta / "route_stops.json"), "--stop-dist", str(meta / "stop_dist.json"),
               "--archive", str(DATA / "raw" / city), "--out", str(out), "--days", "7", "--until", until] + TM_ARGS[city]
        t0 = time.time()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print(f"[{city}] model.json until {until} built in {time.time()-t0:.0f}s", flush=True)
    return out


def train_regs(city: str) -> set[str]:
    regs = set()
    for d in TRAIN_DAYS[city]:
        for _day, p in tm.day_files(str(DATA / "raw" / city), "obs", 1, d):
            for r in tm.lines(p):
                if r.get("reg"):
                    regs.add(r["reg"])
    return regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city"); ap.add_argument("day")
    ap.add_argument("--sample", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-q", type=int, default=1_500_000)
    ap.add_argument("--leaky", action="store_true", help="use meta/<city>/model.json (as published) instead")
    ap.add_argument("--out", default=str(ROOT / "data" / "audit"))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    city, day = a.city, a.day
    t0 = time.time()
    ens = Ensemble()
    res: dict = {"city": city, "day": day, "sample": a.sample, "stride": a.stride}

    # 1. the published cached matrix, if this is the published test day
    for f in sorted((DATA / "v3_cache").glob(f"v3_{city}_{day}_*_3_1500000.npz")):
        z = np.load(f)
        X = z["X"][:, v4.KEEP_IDX]
        p, lo, hi = ens.predict(X, z["b"])
        res["published_cache"] = {"file": f.name, "nowcast": summary(z["y"], z["b"])["ALL(1-30m)"],
                                  "v4": summary(z["y"], p, lo, hi)["ALL(1-30m)"]}
        print(f"[{city}] cached {f.name}: nowcast {res['published_cache']['nowcast']['mae_s']} v4 {res['published_cache']['v4']['mae_s']}", flush=True)

    # 2. rebuild with a causal model.json
    mj = (DATA / "meta" / city / "model.json") if a.leaky else honest_model_json(city, day, out_dir)
    meta = DATA / "meta" / city
    routes_tm = tm.Routes(str(meta / "route_stops.json"), str(meta / "stop_dist.json"))
    routes_ed = ed.Routes(meta / "route_stops.json", meta / "stop_dist.json")
    turns = v3.precompute_route_turns(routes_ed)
    osm = v3.precompute_route_osm(routes_ed, DATA / "osm" / f"{city}_osm_infra.json")
    hol = json.loads((DATA / "holidays_india_2025_2026.json").read_text())
    wx = v2.load_weather_map(DATA, city)
    max_gap = 600 if city in ("mtc", "mumbai") else 180
    X, rids, y, b = v3.build_v3_day(city, CITIES.index(city), day, DATA, routes_tm, routes_ed, turns, osm, hol, mj, wx,
                                    {c: i for i, c in enumerate(CLASS_VOCAB)}, a.sample, max_gap,
                                    fix_stride=a.stride, max_queries=a.max_q, cache_dir=None)
    X = X[:, v4.KEEP_IDX]
    print(f"[{city}] {day}: {len(X):,} queries rebuilt in {time.time()-t0:.0f}s (model.json={mj.name if mj else None})", flush=True)
    p, lo, hi = ens.predict(X, b)
    tag = "leaky" if a.leaky else "causal"
    res[tag] = {"model_json": mj.name if mj else None, "n": int(len(X)),
                "nowcast": summary(y, b), "v4": summary(y, p, lo, hi)}
    np.savez_compressed(out_dir / f"q_{city}_{day}_{tag}.npz", X=X, y=y, b=b, p=p, lo=lo, hi=hi, rids=np.asarray(rids))

    # 3. vehicles seen in training vs not: needs the reg per query, which build_v3_day doesn't keep;
    # approximate by route id (rids) is not the same thing, so report the vehicle overlap rate instead.
    tr = train_regs(city) if city in TRAIN_DAYS else set()
    day_regs = set()
    for _d, pth in tm.day_files(str(DATA / "raw" / city), "obs", 1, day):
        for r in tm.lines(pth):
            if r.get("reg"):
                day_regs.add(r["reg"])
    res["vehicles"] = {"scored_day": len(day_regs), "also_in_train_days": len(day_regs & tr)}

    s_n, s_v = res[tag]["nowcast"]["ALL(1-30m)"], res[tag]["v4"]["ALL(1-30m)"]
    print(f">>> {city} {day} [{tag}] n={s_n['n']:,} nowcast MAE {s_n['mae_s']} -> v4 {s_v['mae_s']} "
          f"({100*(1-s_v['mae_s']/s_n['mae_s']):+.1f}%), cov80 {s_v.get('cov80_pct')}", flush=True)
    (out_dir / f"audit_{city}_{day}_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
