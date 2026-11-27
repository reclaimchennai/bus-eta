#!/usr/bin/env python3
"""Train the v5 ensemble on the leak-free caches (v5_build.py), all eight systems in one model.

P50: LightGBM L1 weighted by the nowcast seconds, XGBoost Pseudo-Huber and CatBoost MAE on log(true/nowcast),
blended with weights picked on the validation days (mean over systems of MAE / nowcast MAE). Band: LightGBM
P10/P90 with a per-horizon-bucket conformal margin from the validation days (the live AgACI starts from it).
Nothing here reads a test day; src/v5_eval.py does the held-out report.

    train_v5.py [--out models/v5_ensemble] [--cap 300000] [--feats v5] [--skip cat]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

import v5_experiments as X5
import v5_lib as L
from v5_build import PLAN

NJ = X5.NJ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.ROOT / "models" / "v5_ensemble"))
    ap.add_argument("--cap", type=int, default=300_000)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = X5.Data(a.cap)
    Xtr, Xva = D.tr["X"], D.va["X"]
    r_tr, r_va = L.ratio(D.tr["y"], D.tr["b"]), L.ratio(D.va["y"], D.va["b"])
    b_va, y_va, sys_va = D.va["b"], D.va["y"], D.va["sys_idx"]
    w_b = np.clip(D.tr["b"], 30.0, 3600.0)
    files, p_va, info = {}, {}, {}

    def rel_mae(r_hat):                                    # mean over systems of MAE / nowcast MAE
        ae, ae0 = np.abs(b_va * np.exp(r_hat) - y_va), np.abs(b_va - y_va)
        return float(np.mean([ae[sys_va == c].mean() / ae0[sys_va == c].mean() for c in np.unique(sys_va) if (sys_va == c).sum() > 1000]))

    t0 = time.time()
    m, _ = X5.train_lgb(D, dict(feats="v5", obj="l1", weight="b"), params={"learning_rate": 0.06}, rounds=1000)
    m.save_model(str(out / "v5_lgb_l1.txt"), num_iteration=m.best_iteration)
    files["lgb"], p_va["lgb"] = "v5_lgb_l1.txt", m.predict(Xva, num_iteration=m.best_iteration, num_threads=NJ)
    info["lgb"] = {"iters": m.best_iteration, "rel_mae_val": rel_mae(p_va["lgb"]), "s": round(time.time() - t0)}
    print("[v5] lgb", info["lgb"], flush=True)

    if "xgb" not in a.skip:
        t0 = time.time()
        mx = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=0.25, n_estimators=1200, learning_rate=0.05,
                              max_depth=9, min_child_weight=80, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              n_jobs=NJ, early_stopping_rounds=40, eval_metric="mae")
        mx.fit(Xtr, r_tr, eval_set=[(Xva, r_va)], verbose=200)
        mx.save_model(str(out / "v5_xgb_huber.json"))
        files["xgb"], p_va["xgb"] = "v5_xgb_huber.json", mx.predict(Xva)
        info["xgb"] = {"iters": int(mx.best_iteration), "rel_mae_val": rel_mae(p_va["xgb"]), "s": round(time.time() - t0)}
        print("[v5] xgb", info["xgb"], flush=True)

    if "cat" not in a.skip:
        from catboost import CatBoostRegressor
        t0 = time.time()
        mc = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=0.08, depth=8, thread_count=NJ,
                               early_stopping_rounds=40, verbose=200)
        mc.fit(Xtr, r_tr, sample_weight=w_b, eval_set=(Xva, r_va))
        mc.save_model(str(out / "v5_cat_mae.cbm"))
        files["cat"], p_va["cat"] = "v5_cat_mae.cbm", mc.predict(Xva)
        info["cat"] = {"iters": int(mc.get_best_iteration() or 0), "rel_mae_val": rel_mae(p_va["cat"]), "s": round(time.time() - t0)}
        print("[v5] cat", info["cat"], flush=True)

    names = list(p_va)
    best = None
    for ws in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(names)):
        if abs(sum(ws) - 1.0) > 1e-6:
            continue
        s = rel_mae(sum(w * p_va[n] for w, n in zip(ws, names)))
        if best is None or s < best[0]:
            best = (s, ws)
    weights = {n: round(float(w), 2) for n, w in zip(names, best[1])}
    print(f"[v5] blend {weights} rel MAE {best[0]:.4f}", flush=True)

    q_va = {}
    for alpha in (0.10, 0.90):
        dtr = lgb.Dataset(Xtr, label=r_tr, feature_name=L.F5, categorical_feature=L.CAT, free_raw_data=False)
        dva = lgb.Dataset(Xva, label=r_va, reference=dtr)
        qm = lgb.train({"objective": "quantile", "alpha": alpha, "metric": "quantile", "learning_rate": 0.05, "num_leaves": 127,
                        "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                        "n_jobs": NJ, "verbose": -1, "seed": 7}, dtr, num_boost_round=800, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(40, verbose=False)])
        fn = f"v5_lgb_q{int(alpha * 100)}.txt"
        qm.save_model(str(out / fn), num_iteration=qm.best_iteration)
        files[f"q{int(alpha * 100)}"] = fn
        q_va[alpha] = qm.predict(Xva, num_iteration=qm.best_iteration, num_threads=NJ)
        print(f"[v5] q{int(alpha*100)} iters {qm.best_iteration}", flush=True)
    hb = np.digitize(b_va, [180.0, 300.0, 600.0, 900.0])
    score = np.maximum(q_va[0.10] - r_va, r_va - q_va[0.90])
    cqr = {str(k): float(np.quantile(score[hb == k], 0.80)) for k in range(5)}
    rep = {"features": L.F5, "weights": weights, "files": files, "cqr_by_bucket": cqr, "models": info, "cap": a.cap,
           "plan": {s: {"train": PLAN[s]["train"], "val": PLAN[s]["val"], "test": PLAN[s]["test"], "city_idx": PLAN[s]["ci"]} for s in PLAN},
           "rel_mae_val_blend": best[0]}
    (out / "v5_report.json").write_text(json.dumps(rep, indent=1))
    print("[v5] saved", out, flush=True)


if __name__ == "__main__":
    main()
