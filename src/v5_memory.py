#!/usr/bin/env python3
"""Peak memory of one model set alone: load it, score batches of 2,000 and 50,000 rows with 2 threads (a Delhi
tick is ~48,000 rows), print the process's peak and current RSS. Run once per variant; compare with `none`.

    v5_memory.py none | v4 | v4r | v5 [--use lgb,xgb] [--models DIR]
"""
import argparse
import json
import resource
import sys

import numpy as np

import v5_lib as L
import eta_engine as E  # noqa: E402
import eta_engine_v5 as E5  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("which"); ap.add_argument("--use", default=""); ap.add_argument("--models", default="")
a = ap.parse_args()
import gc
import os
import subprocess


def rss():                                  # "peak/now" MB: MemoryMax is charged for what is held now
    gc.collect()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)
    return f"{peak:.0f}/{int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())]).split()[0]) / 1024:.0f}"
import lightgbm, xgboost, catboost  # noqa: E401,F401  (the service imports all three)
out = {"which": a.which, "use": a.use, "rss_mb_imports": rss()}
ens, nf = None, 74
if a.which in ("v4", "v4r"):
    ens = E.Ensemble(a.models or str(L.MODELS / {"v4": "v4_clean_ensemble", "v4r": "v4r_rtc"}[a.which]), threads=2,
                     use=tuple(a.use.split(",")) if a.use else ("lgb", "xgb", "cat"))
elif a.which == "v5":
    ens, nf = E5.EnsembleV5(a.models or str(L.ROOT / "models" / "v5_ensemble"), threads=2, use=tuple(a.use.split(",")) if a.use else None), 98
out["rss_mb_loaded"] = rss()
if ens:
    rng = np.random.default_rng(0)
    for n in (2000, 50000):
        X = rng.random((n, nf), dtype=np.float32) * 100
        ens.predict(X, np.full(n, 600.0))
        out[f"rss_mb_after_{n}"] = rss()
print(json.dumps(out))
