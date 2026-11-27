#!/usr/bin/env python3
"""Online residual correction on top of a trained model, replayed strictly in time.

For every prediction made at t, the model's own log error on arrivals that were OBSERVED before t (same working,
same system, or same working x horizon bucket) is averaged with an exponential decay and a shrinkage prior, and a
fraction lam of it is added to the prediction: p' = p * exp(lam * mean). tau, prior and lam are picked on the
validation days; the test days are reported with those settings.

    v5_online.py EXP [EXP2 ...]      (experiment directories under ETA_V5/exp; several = their log-mean blend first)
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

import v5_lib as L
from v5_build import OUT, PLAN


def sets():
    for s in PLAN:
        for split, days in (("val", [PLAN[s]["val"]]), ("test", PLAN[s]["test"])):
            for day in days:
                d = L.load(s, day, std=True)
                if d is not None and len(d["y"]) > 1000:
                    yield split, s, day, d


def pred(names, s, day):
    ps = [np.load(OUT / "exp" / n / f"p_{s}_{day}.npy").astype(np.float64) for n in names]
    return np.exp(np.mean([np.log(p) for p in ps], 0))


def group_of(kind, d):
    hb = np.digitize(d["b"], [300.0, 600.0, 1200.0])
    if kind == "system":
        return np.zeros(len(d["y"]), dtype=np.int64)
    if kind == "route":
        return d["code"].astype(np.int64)
    if kind == "route_h":
        return d["code"].astype(np.int64) * 4 + hb
    if kind == "sys_h":
        return hb.astype(np.int64)
    raise ValueError(kind)


def main():
    names = sys.argv[1:]
    data = [(split, s, day, d, pred(names, s, day)) for split, s, day, d in sets()]
    base = {(s, day): float(np.mean(np.abs(p - d["y"]))) for _, s, day, d, p in data}
    grid = list(itertools.product(("route", "route_h", "system", "sys_h"), (900.0, 2700.0), (5.0, 30.0)))
    res = {}
    for kind, tau, prior in grid:
        for split, s, day, d, p in data:
            e = np.clip(np.log(d["y"] / p), -1.5, 1.5)
            m, _n = L.settled_mean(d["t"].astype(np.float64), d["t"] + d["y"], e, group_of(kind, d), tau, prior)
            for lam in (0.5, 1.0):
                res[(kind, tau, prior, lam, s, day)] = float(np.mean(np.abs(p * np.exp(lam * m) - d["y"])))
    out = {"models": names, "per_system": {}}
    print(f"online residual correction on {'+'.join(names)}  (MAE s)")
    for s in PLAN:
        vd = PLAN[s]["val"]
        if (s, vd) not in base:
            continue
        cands = [(res[(k, t, pr, lam, s, vd)], k, t, pr, lam) for k, t, pr in grid for lam in (0.5, 1.0)]
        best = min(cands)
        _, k, t, pr, lam = best
        row = {"pick": {"group": k, "tau": t, "prior_n": pr, "lam": lam}, "val": {"day": vd, "model": base[(s, vd)], "corrected": best[0]}, "test": {}}
        line = f"  {s:8s} pick {k:8s} tau {t:5.0f} prior {pr:4.0f} lam {lam:.1f} | val {base[(s, vd)]:6.1f} -> {best[0]:6.1f}"
        for day in PLAN[s]["test"]:
            if (s, day) in base:
                row["test"][day] = {"model": base[(s, day)], "corrected": res[(k, t, pr, lam, s, day)]}
                line += f" | test {day[-2:]} {base[(s, day)]:6.1f} -> {res[(k, t, pr, lam, s, day)]:6.1f}"
        out["per_system"][s] = row
        print(line, flush=True)
    (OUT / "exp" / ("online_" + "+".join(names) + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
