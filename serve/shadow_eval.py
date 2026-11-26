#!/usr/bin/env python3
"""Live accuracy check from a shadow (or live) run of eta_ml_service.py.

Inputs, all from one city's run directory:
  eta_ml-*.jsonl.gz   the service's sampled log: per scored fix, [k, dt_v4, dt_lo, dt_hi, dt_nowcast]
                      and ("s": 1) what the public map showed for the same bus (vehicles.json row ETAs)
  obs-tee.jsonl       the archive lines the service read (truth: when each bus really crossed each stop)
Truth = eval_eta.extract_ground_truth_stop_times on those fixes (crossing interpolated between two fixes).
Scored: stops 60 s .. 30 min ahead of the fix. "same" = only (bus, fix, stop) triples all three methods cover.

Usage: shadow_eval.py RUN_DIR ROUTE_STOPS STOP_DIST [--since EPOCH] [--max-gap 600]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
import eval_eta  # type: ignore  # noqa: E402
import travel_model as tm  # type: ignore  # noqa: E402


def stats(err):
    e = np.abs(np.asarray(err, dtype=float))
    if not len(e):
        return {"n": 0}
    return {"n": int(len(e)), "mae_s": round(float(e.mean()), 1), "med_s": round(float(np.median(e)), 1),
            "p80_s": round(float(np.percentile(e, 80)), 1), "p90_s": round(float(np.percentile(e, 90)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("stops"); ap.add_argument("dist")
    ap.add_argument("--since", type=float, default=0)
    ap.add_argument("--max-gap", type=int, default=600)
    ap.add_argument("--partial", action="store_true", help="score fixes < 30 min before the end too (only arrivals seen by then: fair between methods, biased to short horizons)")
    a = ap.parse_args()
    run = Path(a.run)
    R = tm.Routes(a.stops, a.dist)
    tracks, last = collections.defaultdict(list), {}
    with open(run / "obs-tee.jsonl") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            reg, code = r.get("reg"), r.get("r")
            if not reg or not code or code not in R.ll:
                continue
            prev = last.get(reg)
            if prev and r["t"] <= prev[0]:
                continue
            pr = R.project(code, r["la"], r["lo"], prev[3] if prev and prev[1] == code else None)
            if not pr or pr[2] > 150:
                continue
            rec = (float(r["t"]), code, float(pr[0]), int(pr[1]), str(r.get("f") or ""))
            tracks[reg].append(rec); last[reg] = rec
    gt = eval_eta.extract_ground_truth_stop_times(tracks, R, max_gap=max(a.max_gap, 300))
    t_end = max((t[-1][0] for t in tracks.values() if t), default=0)

    def truth(reg, code, fix):
        for tr in gt.get(reg, ()):
            if tr["code"] == code and tr["fixes"][0][0] <= fix <= tr["fixes"][-1][0]:
                return tr["stop_times"]
        return None

    v4, shown = {}, {}
    for f in sorted(glob.glob(str(run / "eta_ml-*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            lines = []
            try:                                   # a log still being written ends mid-member
                for ln in fh:
                    lines.append(ln)
            except (EOFError, OSError):
                pass
            for ln in lines:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r["fix"] < a.since or (not a.partial and r["fix"] > t_end - 1800):     # need 30 min of truth after the fix
                    continue
                if r.get("s"):
                    fl = r.get("shown") or []
                    shown[(r["reg"], r["fix"])] = (r["r"], r.get("f", ""), {fl[i]: fl[i + 1] for i in range(0, len(fl) - 1, 2)})
                else:
                    v4[(r["reg"], r["fix"])] = (r["r"], {e[0]: e[1:] for e in r["e"]})
    E = collections.defaultdict(list)
    cov = collections.Counter()
    for (reg, fix), (code, ks) in v4.items():
        st = truth(reg, code, fix)
        if not st:
            continue
        sh = shown.get((reg, fix))
        for k, (dt, lo, hi, nc) in ks.items():
            if k not in st:
                continue
            y = st[k] - fix
            if not 60 <= y < 1800 or st[k] > t_end - 60:
                continue
            E["v4"].append(dt - y); E["nowcast"].append(nc - y)
            cov["in80"] += lo <= y <= hi; cov["n"] += 1; cov["w"] += hi - lo
            if sh and sh[0] == code and k in sh[2]:
                E["same_v4"].append(dt - y); E["same_nowcast"].append(nc - y); E["same_shown"].append(sh[2][k] - y)
                src = "ours" if ("m" in sh[1] or "h" in sh[1]) else "operator"
                E[f"shown_{src}"].append(sh[2][k] - y); E[f"v4_vs_{src}"].append(dt - y)
    out = {k: stats(v) for k, v in E.items()}
    out["cov80_pct"] = round(100 * cov["in80"] / max(1, cov["n"]), 1)
    out["width_s"] = round(cov["w"] / max(1, cov["n"]), 1)
    out["window"] = [min((k[1] for k in v4), default=0), t_end]
    out["fixes_v4"], out["fixes_shown"] = len(v4), len(shown)
    out["bias_s"] = {k: round(float(np.median(v)), 1) for k, v in E.items() if v}
    print(json.dumps(out, indent=1))
    (run / "shadow_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
