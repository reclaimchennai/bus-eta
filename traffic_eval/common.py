"""Shared paths and imports for the traffic-incident evaluation (traffic_eval/).

Everything is relative to the repository root (ROOT = the folder holding traffic_eval/). The v4 code is
imported from ROOT/src, ROOT/serve and ROOT/reference/mumbai; while the branches still live in separate
worktrees (ETA-wt-baseline / -features / -models beside the repo), those are used as a fallback.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HERE = ROOT / "traffic_eval"
WORK = HERE / "work"            # replays, causal model.json files, merged archives (large, not committed)
RESULTS = HERE / "results"      # small JSON/CSV/PNG outputs (committed)
SRC_DATA = HERE / "src_data"    # incident snapshots used by the evaluation
MODELS = ROOT / "models" / "v4_clean_ensemble"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

_paths = []
for rel in ("src", "serve", "reference/mumbai"):
    if (ROOT / rel).is_dir():
        _paths.append(ROOT / rel)
for wt, rel in (("ETA-wt-models", "serve"), ("ETA-wt-models", "src"), ("ETA-wt-baseline", "src"),
                ("ETA-wt-features", "src")):
    p = ROOT.parent / wt / rel
    if p.is_dir():
        _paths.append(p)
for p in reversed(_paths):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# which bus system each city's traffic police feed is evaluated against
SYSTEM_OF = {"hyderabad": "tgsrtc", "delhi": "delhi", "bengaluru": "bmtc", "chennai": "mtc"}
CITY_OF = {v: k for k, v in SYSTEM_OF.items()}
# days the v4 models trained on (the evaluation uses other days only)
TRAIN_DAYS = {
    "mtc": ["2026-11-14", "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],
    "bmtc": ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],
    "delhi": ["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21"],
    "tgsrtc": [],                 # never trained on: every day is held out
}


def day_of(t: float) -> str:
    return dt.datetime.fromtimestamp(t, IST).strftime("%Y-%m-%d")


def day_bounds(day: str) -> tuple[float, float]:
    t0 = dt.datetime.fromisoformat(day + "T00:00:00+05:30").timestamp()
    return t0, t0 + 86400
