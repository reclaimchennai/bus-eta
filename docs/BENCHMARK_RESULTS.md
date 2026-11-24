# Indian Multi-City Bus ETA — Head-to-Head Benchmark Results

> Evaluated: 2026-11-24 | Repo: `~/projects/ETA`  
> Protocol: Strict chronological whole-fleet replay on held-out test days (zero leakage); all live probes, shared-corridor speeds, headways, and vehicle pace ratios updated strictly causally.

---

## 1. Main 5-City Head-to-Head Scoreboard (`1–30 min` Stop-Arrival Horizons)

One single unified multi-city **v2 DeeprETA Residual LightGBM (Huber + P10/P90) + UncertBAT Horizon-Grouped Mondrian CQR + Shared-Corridor Live Probes + Headway + `pol.reclaimchennai.city` Weather** model trained on 4.26M queries (59 features) and evaluated on **3.12M held-out stop-arrival queries** across all 5 Indian cities:

| City (Held-Out Test Day) | Queries | `nowcast.py` MAE | **v2 DeeprETA+CQR MAE** | **MAE Reduction** | `nowcast` → **v2 MAPE** | P90 Error (`nowcast` → **v2**) | **CQR 80% Coverage** | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Delhi DTC** (`2026-11-23`) | 1,500,008 | 152.5s | **97.7s** | **-35.9%** | 22.84% → **14.87%** | 373.2s → **222.6s** | **80.0%** | 10.2 µs/q |
| **Bengaluru BMTC** (`2026-11-23`) | 361,784 | 144.1s | **92.7s** | **-35.7%** | 22.11% → **15.19%** | 341.6s → **214.8s** | **80.2%** | 10.5 µs/q |
| **Chennai MTC** (`2026-11-23`) | 1,167,999 | 143.4s | **100.7s** | **-29.8%** | 20.17% → **14.03%** | 312.4s → **232.7s** | **83.5%** | 10.5 µs/q |
| **Mumbai BEST** (`2026-11-17`) | 82,761 | 116.4s | **82.4s** | **-29.2%** | 24.53% → **17.66%** | 274.2s → **192.5s** | **81.4%** | ~10 µs/q |
| **Gurugram NCR** (`2026-11-17`, Cold-Start) | 6,063 | 277.7s | **125.4s** | **-54.8%** | 43.04% → **24.48%** | 691.1s → **282.4s** | **80.9%** | 12.9 µs/q |

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
