# Indian Multi-City Bus ETA — Head-to-Head Benchmark Results

> Evaluated: 2026-11-24 | Repo: `~/projects/ETA`  
> Protocol: Strict chronological whole-fleet replay on held-out test days (zero leakage); all live probes, shared-corridor speeds, headways, and vehicle pace ratios updated strictly causally.

---

## 1. Main 5-City Head-to-Head Scoreboard (`1–30 min` Stop-Arrival Horizons, 3.12M Held-Out Queries)

**Best Model (`v4 Clean Stacked Ensemble`):** `0.40 LightGBM Huber + 0.50 XGBoost Pseudo-Huber + 0.10 CatBoost MAE` (74 leak-free features: shared-corridor multi-route live probes, Overpass OSM flyovers/signals/rail/roundabouts, verified 2025–2026 Indian holidays, ConSTGAT path geometry, headway, `pol.reclaimchennai.city` weather) + **Online AgACI (Zaffran et al. 2022)** on **UncertBAT Mondrian CQR (Song et al. 2026)**:

| City (Held-Out Test Day) | Queries | `nowcast.py` MAE | `v2` LGB MAE | **`v4` Stacked MAE** | **MAE Reduction** | `nowcast` → **`v4` MAPE** | P90 Error (`nowcast` → **`v4`**) | **AgACI 80% Cov** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chennai MTC** (`2026-11-23`) | 1,167,999 | 143.4s | 100.7s | **94.8s** | **-33.9%** | 20.17% → **14.02%** | 312.4s → **211.8s** | **79.8%** |
| **Delhi DTC** (`2026-11-23`) | 1,500,008 | 152.5s | 97.7s | **97.7s** | **-35.9%** | 22.84% → **14.69%** | 373.2s → **222.4s** | **79.8%** |
| **Bengaluru BMTC** (`2026-11-23`) | 361,784 | 144.1s | 92.7s | **93.0s** | **-35.5%** | 22.11% → **15.21%** | 341.6s → **215.9s** | **79.6%** |
| **Mumbai BEST** (`2026-11-17`) | 82,761 | 116.4s | 82.4s | **79.8s** | **-31.4%** | 24.53% → **17.12%** | 274.2s → **186.8s** | **80.0%** |
| **Gurugram NCR** (`2026-11-17`, Cold-Start) | 6,063 | 277.7s | 125.4s | **125.8s** | **-54.7%** | 43.04% → **24.95%** | 691.1s → **274.4s** | **78.4%** |

---

## 2. Per-Horizon Breakdown (Delhi DTC & Chennai MTC)

### Delhi DTC (`2026-11-23`, 1.50M queries)
| Horizon | `nowcast.py` MAE | **v2 MAE** | `nowcast` → **v2 P90** | **v2 MAPE** | **CQR Cov80%** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1–3m` | 42.8s | **28.6s** (-33.2%) | 86.5s → **61.2s** | 25.22% | 78.7% |
| `3–5m` | 69.3s | **45.0s** (-35.1%) | 143.2s → **97.0s** | 18.95% | 81.2% |
| `5–10m` | 105.1s | **68.2s** (-35.1%) | 229.9s → **148.9s** | 15.50% | 81.6% |
| `10–15m` | 152.4s | **98.9s** (-35.1%) | 345.7s → **214.6s** | 13.33% | 80.4% |
| `15–30m` | 231.4s | **146.5s** (-36.7%) | 530.3s → **308.9s** | 11.35% | 78.9% |

### Chennai MTC (`2026-11-23`, 1.17M queries, with `pol.reclaimchennai.city` weather)
| Horizon | `nowcast.py` MAE | **v2 MAE** | `nowcast` → **v2 P90** | **v2 MAPE** | **CQR Cov80%** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1–3m` | 38.3s | **26.6s** (-30.5%) | 76.3s → **55.9s** | 23.21% | 80.7% |
| `3–5m` | 61.8s | **42.5s** (-31.2%) | 122.9s → **90.1s** | 17.90% | 83.5% |
| `5–10m` | 94.6s | **64.5s** (-31.8%) | 192.0s → **137.8s** | 14.60% | 84.5% |
| `10–15m` | 137.3s | **94.8s** (-31.0%) | 278.9s → **200.5s** | 12.72% | 85.1% |
| `15–30m` | 211.1s | **150.2s** (-28.8%) | 431.6s → **317.9s** | 11.44% | 82.8% |

---

## 3. Auxiliary Model & Paper Baseline Results

- **Charul & Biyani (2019, `1906.09925v1.pdf`) Mask-CNN Reproduction** (`src/mask_cnn_baseline.py` on M4 Pro MPS):
  - **BMTC (8d, top 5 routes):** avg segment MAE **19.75s** vs our v1 unified LightGBM **7.57s** (2.6× better).
  - **Mumbai BEST (17d, sparse Chalo):** avg segment MAE **45.77s** (2 of 5 routes lacked dense `H=10` windows) vs our v1 unified LightGBM **13.47s** (3.4× better).
- **Two-Stage Hurdle Dwell & Bus-Bunching Model** (`src/train_dwell_model.py`, 428K held-out stop arrivals):
  - **Stop-Classifier AUC:** `0.8915` | **Expected Dwell MAE:** `1.89s` (prior) → **`1.47s`** (**+22.1% lift**).
  - Top features: `stop_prior_p`, `headway_ratio` (#2), `lead_headway_s` (#3 — Newell & Potts 1964), `any_headway_s` (#4), `dow` (#5), `precip_3h_mm` (#6 — 3h monsoon lag), `bus_class_idx` (#7).
- **Top Features Driving v2 DeeprETA Lift:**
  1. `base_eta_nowcast_s` (31.2%), `live_vs_hist_ratio` (11.3%), `base_eta_corr_s` (9.7% — shared multi-route physical corridor live speed!), `nc_pace` (6.2% — driver/vehicle pace), `base_eta_hist_s` (6.0%), `path_mean_sinuosity` (4.3% — road curvature/geometry), `path_live_corr_v` (2.8%), `n_stops_ahead` (2.3%), `min_of_day` / `sin_tod` (4.4%), `city_live_v_15m` (2.0%).

---

## 4. Artifacts & Git Worktrees

- Main repo: `~/projects/ETA` (`docs/RESEARCH_AND_ARCHITECTURE.md`, `docs/LITERATURE_AND_THEORIES.md`, `docs/REFERENCES.bib`, `docs/BENCHMARK_RESULTS.md`)
- `ETA-wt-baseline` (`feat/baseline-eval`): `src/eval_eta.py`
- `ETA-wt-features` (`feat/features-and-data`): `src/weather_fetch.py`, `src/extract_dataset.py`
- `ETA-wt-models` (`feat/ml-models`): `src/train_v2_deepreta_cqr.py`, `src/train_gbdt_cqr.py`, `src/train_dwell_model.py`, `src/eta_rollout_eval.py`, `src/mask_cnn_baseline.py`, `src/llm_disruption_sidecar.py`, `notebooks/colab_a100_deep_eta.ipynb`
- Trained production-ready models: `~/projects/ETA/models/{v2_deepreta_cqr, gbdt_cqr_v1, dwell_v1}/`

---

## 5. Bengaluru Traffic Police (BTP) congestion profile as v4 features — BMTC (evaluated 2026-11-25)

**Input:** `ctx/blr/stop_congestion_how.json` (twtrx `pipeline/btp_congestion.py`). For each stop, the % of BTP's 15-minute congestion-map checks that found a jam within 200 m, by hour of week. Data from 1 Aug to 7 Oct 2026, 5,752 snapshots, 1,114 BMTC stops. That is 11% of the 10,156 stops on BMTC routes, but 74% of held-out queries have at least one covered stop between the bus and the target stop, and 28% have a covered next stop.

**Features** (`ETA-wt-models/src/btp_features.py`): `btp_p_next` is p at the next stop and `btp_p_max3` is the max p over the next 3 stops. Both use each stop's arrival hour of week, estimated causally as now + nowcast × distance fraction. Two optional extras: `btp_path_mean` and `btp_path_cov`. Non-BMTC rows get NaN.

**Protocol** (`ETA-wt-models/src/train_v4_btp_ab.py`): the full v4 recipe (5 cities, LGB+XGB+CAT blend, q10/q90 + AgACI), trained twice from `v3_cache` with identical seeds: 74 features vs 74 + 2 BTP features. Held-out day: BMTC 2026-11-23 (361,784 queries). The base arm reproduces the published v4 exactly.

| Horizon | N | v4 MAE | **v4+BTP MAE** | v4 P80 | **v4+BTP P80** | v4 Cov80 | v4+BTP Cov80 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1–3m` | 44,569 | 26.4s | 26.5s | 39.3s | 39.3s | 78.1% | 77.9% |
| `3–5m` | 42,803 | 45.3s | 45.4s | 68.9s | 69.1s | 79.6% | 79.5% |
| `5–10m` | 84,542 | 70.2s | 70.5s | 108.4s | 109.0s | 80.3% | 80.5% |
| `10–15m` | 65,900 | 103.7s | 103.8s | 161.8s | 161.3s | 79.7% | 79.8% |
| `15–30m` | 123,970 | 143.2s | 143.3s | 215.8s | 215.6s | 79.6% | 79.5% |
| **ALL** | 361,784 | **93.0s** | **93.1s** | 146.0s | 146.3s | 79.6% | 79.6% |

- Paired per-query MAE difference (BTP − base): **+0.13 s, 95% bootstrap CI [+0.08, +0.18]**, so slightly worse.
- On queries whose path has a BTP stop, MAE went 99.7 → 100.0 s. On queries without one, it went 73.8 → 73.5 s. That is the opposite of what a real signal would do.
- LightGBM gain share: `btp_p_max3` 0.29%, `btp_p_next` 0.09%.
- Other cities moved through the shared blend: Chennai MTC 94.8 → 95.7 s, because the blend weights shifted from (0.4, 0.5, 0.1) to (0.5, 0.4, 0.1). Delhi went 97.7 → 97.5 s and Mumbai stayed at 79.8 s.
- **BMTC-only LightGBM check** (3 seeds, BMTC-only test MAE). Base: 92.50 ± 0.06 s. +2 BTP features: 92.21 ± 0.22 s. +2 features shuffled across rows: 92.53 ± 0.13 s. +4 features (one seed): 92.03 s. So there is at most a ~0.3% signal in a single-city model, and it disappears in the production multi-city ensemble.

**Why so little:** v4 already sees recurring congestion directly. It has per-segment, hour-of-day historical speeds (`base_eta_hist_s`) and live shared-corridor speeds from the whole fleet (`path_live_corr_v`, `cur_corr_live_v_*`). BTP's "jam within 200 m" frequency is a coarser, older (Aug–Oct) proxy for the same thing.

**Recommendation:** do **not** add the BTP profile to v4, and do not change `eta_ml_service.py`. Keep it as map context: the "Police congestion (usual)" layer, already live. Revisit only if BTP data covers corridors where BMTC GPS is thin.

**Live BTP alerts** (`maps-site/bengaluru/live/incidents.json`; history in twtrx `data/blr/events.json`):
- Volume: 67 events since 26 Oct 2026, about 2 per day. 43 of them are `other`; the typed ones are 7 breakdown, 6 diversion, 4 congestion, 2 works, 2 accident, and 1 each of event, waterlogging and tree_fall.
- Calibrating the per-type delay priors in `serve/incident_penalty.py` needs about 30 geolocated, time-bounded events per main type that overlap archived BMTC GPS. That is about 150–200 typed events: roughly 6–8 months at the current typed rate (~0.8 per day).
- A learned v4 feature would need 500+ events, because the trees need about 100 rows per leaf from several independent events, not one jam.
- Until then, the right path is serving-side: feed the Bengaluru incidents file to `incident_penalty.py` for BMTC, using the existing uncalibrated priors, the same way TGSRTC does.
