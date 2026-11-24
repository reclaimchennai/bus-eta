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
