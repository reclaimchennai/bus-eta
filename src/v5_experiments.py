#!/usr/bin/env python3
"""v5 ablations: one LightGBM per experiment on the leak-free caches, scored per system on its validation day and
its held-out test day(s) next to nowcast, v4 and v4r on identical rows.

Choices (which ideas to keep) are made on the VALIDATION days only; the test columns are reported, not tuned on.
Predictions are kept in ETA_V5/exp/<name>/ so blends and the online steps can be studied without retraining.

    v5_experiments.py NAME [NAME ...] [--cap 300000] [--fast]
    v5_experiments.py --list
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

import v5_lib as L
from v5_build import OUT, PLAN

import os
NJ = int(os.environ.get("V5_JOBS", "12"))
V5X = L.F5[74:]
GROUPS = {   # the extra columns by idea, for drop-one-group ablations
    "busstate": ["stopped_s", "fix_gap_s", "veh_v_180s", "veh_v_300s", "trip_vs_hist_ratio", "dist_to_next_stop_m", "dist_from_prev_stop_m"],
    "today": ["route_today_ratio", "route_today_evidence_s", "city_today_ratio"],
    "headway": ["lead_gap_m", "lead_fix_age_s", "lead_v", "follow_gap_m", "route_active_buses"],
    "leader": ["leader_tt_ratio", "leader_tt_age_s", "leader_vs_nowcast", "leader2_tt_ratio"],
    "lastpass": ["base_eta_lastpass_s", "lastpass_vs_nowcast", "lastpass_cov_frac", "lastpass_age_s"],
    "histcov": ["path_hist_cov_frac"],
}

EXPS = {
    # name: feats ("v4" | "v5" | "v5-<group>" | "v4+<group>"), objective, weight, base column, label clip
    "a0_v4feat_huber":   dict(feats="v4", obj="huber"),
    "a1_v5feat_huber":   dict(feats="v5", obj="huber"),
    "a2_v5_huber_wb":    dict(feats="v5", obj="huber", weight="b"),
    "a3_v5_l1":          dict(feats="v5", obj="l1"),
    "a4_v5_l1_wb":       dict(feats="v5", obj="l1", weight="b"),
    "a5_v5_l1_wsqrt":    dict(feats="v5", obj="l1", weight="sqrt"),
    "a6_v4feat_l1_wb":   dict(feats="v4", obj="l1", weight="b"),
    "b1_base_hist":      dict(feats="v5", obj="l1", weight="b", base="base_eta_hist_s"),
    "b2_base_lastpass":  dict(feats="v5", obj="l1", weight="b", base="base_eta_lastpass_s"),
    "b3_base_corr":      dict(feats="v5", obj="l1", weight="b", base="base_eta_corr_s"),
    "c1_clip25":         dict(feats="v5", obj="l1", weight="b", clip=2.5),
    **{f"d_drop_{g}": dict(feats=f"v5-{g}", obj="l1", weight="b") for g in GROUPS},
    **{f"e_add_{g}": dict(feats=f"v4+{g}", obj="l1", weight="b") for g in GROUPS},
}


def feat_idx(spec):
    if spec == "v4":
        return list(range(74))
    if spec == "v5":
        return list(range(len(L.F5)))
    if spec.startswith("v5-"):
        drop = set(GROUPS[spec[3:]])
        return [i for i, f in enumerate(L.F5) if f not in drop]
    if spec.startswith("v4+"):
        add = set(GROUPS[spec[3:]])
        return list(range(74)) + [i for i, f in enumerate(L.F5) if f in add]
    raise ValueError(spec)


class Data:
    def __init__(self, cap, val_cap=260_000, systems=None):
        t0 = time.time()
        self.systems = systems or list(PLAN)
        tr = []
        self.val, self.test = {}, {}
        for s in self.systems:
            p = PLAN[s]
            tr += [L.load(s, d, cap=cap) for d in p["train"]]
            v = L.load(s, p["val"], std=True)
            if v is not None:
                self.val[s] = v
            for d in p["test"]:
                te = L.load(s, d, std=True)
                if te is not None and len(te["y"]):
                    self.test[(s, d)] = te
        self.tr = L.stack(tr)
        self.va = L.stack([L.load(s, PLAN[s]["val"], cap=val_cap, std=True) for s in self.systems])   # early stopping
        print(f"[data] train {len(self.tr['y']):,} | early-stop val {len(self.va['y']):,} | "
              f"val systems {len(self.val)} test sets {len(self.test)} | {time.time()-t0:.0f}s", flush=True)
        for s in self.systems:
            print(f"   {s:8s} train rows {int(np.sum(self.tr['sys_idx'] == PLAN[s]['ci'])):>9,}", flush=True)


def base_of(X, b, base):
    if not base:
        return b
    return np.maximum(5.0, X[:, L.F5.index(base)].astype(np.float64))


def weights(kind, b):
    if kind == "b":
        return np.clip(b, 30.0, 3600.0)
    if kind == "sqrt":
        return np.sqrt(np.clip(b, 30.0, 3600.0))
    return None


def train_lgb(D, cfg, fast=False, seed=42, params=None, rounds=None):
    idx = feat_idx(cfg.get("feats", "v5"))
    names = [L.F5[i] for i in idx]
    clip = cfg.get("clip", 1.5)
    b_tr, b_va = base_of(D.tr["X"], D.tr["b"], cfg.get("base")), base_of(D.va["X"], D.va["b"], cfg.get("base"))
    r_tr = np.clip(np.log(D.tr["y"] / b_tr), -clip, clip).astype(np.float32)
    r_va = np.clip(np.log(D.va["y"] / b_va), -clip, clip).astype(np.float32)
    dtr = lgb.Dataset(D.tr["X"][:, idx], label=r_tr, weight=weights(cfg.get("weight"), b_tr), feature_name=names,
                      categorical_feature=L.CAT, free_raw_data=False)
    dva = lgb.Dataset(D.va["X"][:, idx], label=r_va, reference=dtr)
    y_va = D.va["y"]

    def mae_s(pred, _ds):                       # the metric riders feel: seconds, pooled over the validation days
        return "mae_s", float(np.mean(np.abs(b_va * np.exp(pred) - y_va))), False

    p = {"objective": cfg.get("obj", "huber"), "alpha": 0.25, "metric": "None", "learning_rate": 0.1 if fast else 0.04,
         "num_leaves": 127 if fast else 191, "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1,
         "colsample_bytree": 0.8, "n_jobs": NJ, "verbose": -1, "seed": seed}
    if p["objective"] == "l1":
        p.pop("alpha")
    p.update(params or {})
    m = lgb.train(p, dtr, num_boost_round=rounds or (500 if fast else 1500), valid_sets=[dva], feval=mae_s,
                  callbacks=[lgb.early_stopping(30 if fast else 45, verbose=False)])
    return m, idx


def predict(m, idx, d, cfg):
    b = base_of(d["X"], d["b"], cfg.get("base"))
    return b * np.exp(m.predict(d["X"][:, idx], num_iteration=m.best_iteration, num_threads=NJ))


def score_sets(D, pred_fn, name, B=300):
    """Per system: validation and test MAE of this predictor next to nowcast, v4, v4r."""
    res = {"val": {}, "test": {}}
    for split, sets in (("val", {(s, PLAN[s]["val"]): d for s, d in D.val.items()}), ("test", D.test)):
        for (s, day), d in sets.items():
            preds = {"nowcast": d["b"], "v4": L.baseline("v4", d)[0], "v4r": L.baseline("v4r", d)[0], name: pred_fn(d)}
            ref = "v4r" if s in L.RTC_SYS else "v4"
            res[split][f"{s}|{day}"] = L.mae_table(d, preds, ref=ref, B=B)
    return res


def show(name, res):
    print(f"\n=== {name} ===  (MAE s; d = paired difference to production: v4 for cities, v4r for RTC; 95% CI by bus)")
    for split in ("val", "test"):
        for key, r in res[split].items():
            s = key.split("|")[0]
            ref = "v4r" if s in L.RTC_SYS else "v4"
            m = r[name]
            ci = m[f"d_vs_{ref}_ci"]
            print(f"  {split:4s} {key:20s} n={r['n']:>9,} nowcast {r['nowcast']['mae']:6.1f} v4 {r['v4']['mae']:6.1f} v4r {r['v4r']['mae']:6.1f} "
                  f"| {m['mae']:6.1f}  d {m[f'd_vs_{ref}']:+6.1f} [{ci[0]:+.1f},{ci[1]:+.1f}] ({100*m[f'd_vs_{ref}']/r[ref]['mae']:+.1f}%) "
                  f"bias {m['bias_mean']:+.0f}/{m['bias_median']:+.0f}  20-30m {m.get('20-30m', float('nan')):.0f} vs {r[ref].get('20-30m', float('nan')):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.list:
        print("\n".join(EXPS)); return
    D = Data(a.cap)
    out = OUT / "exp"
    out.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        cfg = EXPS[name]
        t0 = time.time()
        m, idx = train_lgb(D, cfg, fast=a.fast)
        tag = name + ("_fast" if a.fast else "") + a.tag
        res = score_sets(D, lambda d: predict(m, idx, d, cfg), tag)
        res["cfg"], res["best_iter"], res["cap"], res["train_s"] = cfg, m.best_iteration, a.cap, round(time.time() - t0)
        show(tag, res)
        (out / tag).mkdir(exist_ok=True)
        (out / tag / "result.json").write_text(json.dumps(res, indent=1))
        m.save_model(str(out / tag / "lgb.txt"), num_iteration=m.best_iteration)
        for split, sets in (("val", {(s, PLAN[s]["val"]): d for s, d in D.val.items()}), ("test", D.test)):
            for (s, day), d in sets.items():
                np.save(out / tag / f"p_{s}_{day}.npy", predict(m, idx, d, cfg).astype(np.float32))
        imp = sorted(zip([L.F5[i] for i in idx], m.feature_importance("gain")), key=lambda x: -x[1])
        tot = sum(g for _, g in imp)
        print("  top gain: " + ", ".join(f"{n} {100*g/tot:.1f}%" for n, g in imp[:14]) + f" | iters {m.best_iteration}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
