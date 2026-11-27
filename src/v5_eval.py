#!/usr/bin/env python3
"""Held-out report of a v5 model directory: per system on its validation day and test day(s), next to the
nowcast, v4 and v4r on identical rows (the audit's bus sample), with bus-clustered bootstrap intervals, paired
differences, horizon buckets, bias, and the 80% band (static margins and the live AgACI replayed causally).

    v5_eval.py [--models models/v5_ensemble] [--tag v5] [--use lgb,xgb]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eta_engine as E
import eta_engine_v5 as E5
import v5_lib as L
from v5_build import OUT, PLAN


def bands(d, p, q10, q90, cqr):
    y, b = d["y"], d["b"]
    r = np.log(y / b)
    rp = np.log(p / b)
    hb = np.digitize(b, [180.0, 300.0, 600.0, 900.0])
    out = {}
    mg = np.asarray(cqr)[hb]
    out["static"] = L.band_stats(y, b * np.exp(np.minimum(q10 - mg, rp)), b * np.exp(np.maximum(q90 + mg, rp)))
    mg = L.agaci_causal(d["t"].astype(np.float64), d["t"] + y, r, q10, q90, hb, cqr, share=0.1)
    lo, hi = b * np.exp(np.minimum(q10 - mg, rp)), b * np.exp(np.maximum(q90 + mg, rp))
    out["agaci"] = L.band_stats(y, lo, hi)
    for name, a, z in L.BUCKETS:
        m = (y >= a) & (y < z)
        if m.any():
            out["agaci"][name] = L.band_stats(y[m], lo[m], hi[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(L.ROOT / "models" / "v5_ensemble"))
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--use", default="")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    ens = E5.EnsembleV5(a.models, threads=10, use=tuple(a.use.split(",")) if a.use else None)
    cq = {"v4": E.Ensemble(str(L.MODELS / "v4_clean_ensemble"), use=()).cqr, "v4r": E.Ensemble(str(L.MODELS / "v4r_rtc"), use=()).cqr}
    res = {}
    for s in PLAN:
        for split, days in (("val", [PLAN[s]["val"]]), ("test", PLAN[s]["test"])):
            for day in days:
                d = L.load(s, day, std=True)
                if d is None or len(d["y"]) < 1000:
                    continue
                p, _lo, _hi, q10, q90, _hb = ens.predict(d["X"], d["b"])
                v4, v4r = L.baseline("v4", d), L.baseline("v4r", d)
                preds = {"nowcast": d["b"], "v4": v4[0], "v4r": v4r[0], a.tag: p}
                r = L.mae_table(d, preds, ref="v4", B=a.B)
                for ref in ("v4r", "nowcast"):
                    x = L.mae_table(d, {ref: preds[ref], a.tag: p}, ref=ref, B=a.B)[a.tag]
                    r[a.tag][f"d_vs_{ref}"], r[a.tag][f"d_vs_{ref}_ci"] = x[f"d_vs_{ref}"], x[f"d_vs_{ref}_ci"]
                r["band"] = {a.tag: bands(d, p, q10, q90, ens.cqr), "v4": bands(d, v4[0], v4[3], v4[4], cq["v4"]),
                             "v4r": bands(d, v4r[0], v4r[3], v4r[4], cq["v4r"])}
                r["split"] = split
                res[f"{s}|{day}"] = r
                np.save(OUT / "pred" / f"{a.tag}_{s}_{day}.npy", p.astype(np.float32))
                m, prod = r[a.tag], "v4r" if s in L.RTC_SYS else "v4"
                bd = r["band"]
                print(f"{split:4s} {s:8s} {day} n={r['n']:>9,} buses {r['buses']:>5} | nowcast {r['nowcast']['mae']:6.1f} v4 {r['v4']['mae']:6.1f} "
                      f"v4r {r['v4r']['mae']:6.1f} {a.tag} {m['mae']:6.1f} [{m['mae_ci'][0]:.1f},{m['mae_ci'][1]:.1f}] | vs {prod} {m['d_vs_'+prod]:+.1f} "
                      f"[{m['d_vs_'+prod+'_ci'][0]:+.1f},{m['d_vs_'+prod+'_ci'][1]:+.1f}] ({100*m['d_vs_'+prod]/r[prod]['mae']:+.1f}%) | bias {m['bias_mean']:+.0f} (prod {r[prod]['bias_mean']:+.0f}) "
                      f"| band cov/width {a.tag} {bd[a.tag]['agaci']['cov80']:.1f}%/{bd[a.tag]['agaci']['width']:.0f}s {prod} {bd[prod]['agaci']['cov80']:.1f}%/{bd[prod]['agaci']['width']:.0f}s", flush=True)
    (OUT / "eval").mkdir(exist_ok=True)
    Path(OUT / "eval" / f"{a.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
