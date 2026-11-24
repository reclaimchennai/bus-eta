# Indian Multi-City Bus ETA Prediction — Research, Data Audit & Architecture

> Created: 2026-11-24  
> Repo: `~/projects/ETA` (read-only audit of `user@map-server` and `~/Desktop/projects`; zero edits to existing repos/servers)

---

## 1. Audit of Existing Assets & Infrastructure

### 1.1 Historical GPS Archives Found on `user@map-server` (`MAP_SERVER_IP`)
| System | Path on `raw` | Date Range | Days | Size | Polling & Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Delhi (DTC + Cluster)** | `~/projects/delhi-live/archive/` | 2026-10-22 → 2026-11-24 | 35 | ~11 GB | Dense (~15–20s OTD fixes), `t, reg, r, la, lo, k, f` |
| **Chennai (MTC)** | `~/projects/mtc-live/archive/` | 2026-10-19 → 2026-11-24 | 38 | ~759 MB | Chalo/MTC feed (sparser route-by-route polling unless rider watches), `f` = bus type (`dn`, `ln`, etc.) |
| **Mumbai (BEST / Chalo)** | `~/projects/mumbai-live/archive/` | 2026-11-03 → 2026-11-18 | 17 | ~72 MB | Sparser Chalo polling (~8–13 min median, ~30s when watched) |
| **Bengaluru (BMTC)** | `~/projects/bmtc-live/archive/` | 2026-11-17 → 2026-11-24 | 8 | ~416 MB | Dense BMTC fixes (`k` = speed/stop, `f` = type) |
| **Gurugram / NCR (GMCBL)** | `~/projects/ncr-live/archive/` | 2026-11-13 → 2026-11-18 | 6 | ~97 MB | Live GMCBL fixes |
| **State RTCs & Others (SVR)** | `SVR RTC_SERVER_LAN` (`raw/live/*.csv.gz`) | varies | — | — | APSRTC, TGSRTC, OSRTC, Kerala KSRTC, CTU Chandigarh, Haryana Roadways, UPSRTC |

### 1.2 Auxiliary Signal Assets Already Available
- **Static GTFS (66 networks, 21 states/UTs):** `https://maps.reclaim.city/gtfs/feeds.json` & `~/Desktop/projects/India_GTFS`
- **Route-dodging / Off-route detours:** `detect_dodges.py` (Mumbai/Delhi/BMTC) — detects shortcutting, flyover bypasses, and distinguishes `usual` GTFS-mismatch routes from driver-specific dodges.
- **Traffic Police Closures & Advisories:** `delhi_bus/server/dtp_closures.py` — scrapes Delhi Traffic Police RSS/alerts + manual X/WhatsApp advisories (`added.jsonl`), geoparses against OSM gazetteer.
- **Weather, Rainfall & AQI:** `raw:~/projects/aqi`, `raw:~/projects/monsoon`, `https://pol.reclaimchennai.city/api/wx`, `~/Desktop/chennai_roads.geojson`, `~/Desktop/Traffic_police.geojson`.

### 1.3 Existing Baseline ETA (`travel_model.py` + `nowcast.py` + `backtest.py`)
The current production baseline is already clean and strong:
- Projects GPS fixes onto stop-to-stop chords with local neighborhood search (`near - 2 .. near + 7`) to avoid loop/parallel-leg jumps.
- Pools effective speed (distance / total time including standing up to `STAND_MAX=240s`) per `stopA|stopB` × hour (0–23).
- At inference (`nowcast.py`): uses **15-min live fleet probes** (`LIVE_S=900s`, if ≥2 probes) → segment hourly speed → segment all-day → city hourly → `4.2 m/s`, multiplied by a per-bus EWMA `Pace` factor (`alpha=0.3`, clamped `[0.6, 1.6]`, decaying with `PACE_T=170s`).

**Key Gaps in Current Baseline:**
1. **No day-of-week / weekend / holiday split** (pools Sunday morning with Monday rush hour).
2. **No cross-segment spatial contagion** (jam on segment `i-1` doesn't inform `i` or `i+1` until ≥2 buses finish `i`).
3. **No bus-bunching / leading-headway dwell model** (headway is the #1 predictor of Indian bus dwell time).
4. **No explicit vehicle-class / historical driver-behavior profile** (AC/EV vs ordinary diesel, chronic stop-skippers/dodgers).
5. **No weather / monsoon / waterlogging / road-closure / OSM infrastructure features**.
6. **No uncertainty intervals** (point estimate only; no P10/P50/P90 or conformal calibration).
7. **Evaluation metric gap:** `backtest.py` measures **distance error (m)** at 30/60/120/300s horizons rather than **stop-arrival time error (seconds)** at 1/3/5/10/15/30 min horizons.

### 1.4 Compute Constraints
- **Local Mac (Training):** Apple M4 Pro, **48 GB unified RAM**, 142 GB free SSD (fast LightGBM/CatBoost + PyTorch MPS).
- **`raw` Server (Live Inference):** 8-core AMD EPYC, **15 GB RAM (~3.7 GB available)**, **NO GPU**, 19 GB free disk.  
  → **Live inference MUST be CPU-friendly (<500 MB RAM, <10 ms per fleet tick)** unless served from a separate GPU host.

---

## 2. Review of Reference Paper (`1906.09925v1.pdf`)
- **Paper:** *To each route its own ETA: A generative modeling framework for ETA prediction* (Charul & Pravesh Biyani, IIIT-Delhi, 2019).
- **Idea:** Formulates each day of a bus route as a 2D `H × K` matrix (trips × consecutive stop travel times) and trains a causal **Masked PixelCNN** to output a discrete softmax distribution over quantized travel times (`C=512` classes, max 1024s), updating sequentially as the bus passes each stop.
- **Strengths:** Captures both running + dwell time jointly, models full conditional probability distributions, handles missing/noisy data, requires zero external traffic APIs.
- **Limitations for our multi-city system (2026):**
  - Requires **1 model per route** and ~2 months of data per route → breaks on low-frequency routes, new cities (BMTC 8 days, NCR 6 days), and cross-route shared corridors.
  - Ignores **other routes sharing the same physical road** (in Delhi/Chennai/Bengaluru, 10–30 routes share major arterials!).
  - Ignores **headway/bunching, day-of-week, weather, bus type, closures, and OSM road geometry**.
  - Softmax quantization over fixed grids doesn't scale as cleanly as **Quantile GBDT + Online Conformal Prediction (CQR)** for calibrated uncertainty intervals.

---

## 3. Deep Research Synthesis: Contemporary ML for Indian Bus ETA (2022–2026)

### 3.1 Architecture Comparison (2–6 Weeks Multi-City Data)
| Architecture | Data Efficiency (2–6 wks) | CPU Inference Latency | Uncertainty Intervals | Cold-Start City Transfer | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM / CatBoost (Residual) + Online CQR** | **Excellent** | **<1 ms** | **Native Quantile + CQR** | **Strong** (normalized target) | **Primary v1 Engine** |
| **ST-GNN (Supersegment / HetETA / ConSTGAT)** | Poor–Fair (overfits) | 10–50 ms (GPU) | Needs MDN/Quantile head | Weak without meta-learning | **Phase 2** (corridor embeddings at 3–6+ mo) |
| **Residual Neural Net (Uber DeeprETA style)** | Fair–Good | 2–10 ms | Quantile / MDN | Moderate | **Phase 2** (hybrid residual corrector) |
| **TSFMs (Chronos-2 / TimesFM-3 / Moirai 2.0)** | **Excellent (Zero-Shot)** | 20–200 ms | Multi-quantile | **Exceptional** | **Async offline prior** for brand-new cities (<1 wk data) |
| **General LLMs (Direct numerical ETA)** | Poor | 200 ms–2 s+ | Poorly calibrated | N/A | **Anti-pattern** for direct ETA |
| **Small Local LLM (Disruption Extraction)** | **High ROI** | Async (every 5–15 min) | Structured severity score | **Works across all cities** | **Context Sidecar** (Traffic Police X/RSS, monsoon/bandh/festival alerts → corridor risk score) |

### 3.2 Recommended Canonical Decomposition
Decompose ETA into **Running Time + Dwell Time + Online Kalman/Residual State + Conformal Interval**, predicting **log-ratio residuals** over a hierarchical Empirical-Bayes prior (`segment×slot → corridor×slot → city×road_class×slot → global`):

$$\widehat{\text{ETA}}(k) = t_{\text{now}} + \hat{T}_{\text{rem\_seg}} + \sum_{i=1}^{k} \hat{T}_{\text{run}, i} + \sum_{j=1}^{k-1} \hat{T}_{\text{dwell}, j} + \Delta_{\text{online}}$$

### 3.3 High-Leverage Indian Bus Features
1. **Shared-Corridor Multi-Route Live Probes:** Map stop-to-stop segments to shared physical road links/OSM ways so *every* bus on Anna Salai / Silk Board / Ring Road acts as a traffic probe for *every other route* (5–20× probe density).
2. **Bus Bunching & Headway:** `leading_headway_sec` (time since previous same-route bus passed upcoming stop) and `trailing_headway_sec` — primary driver of crowd-induced dwell time.
3. **Traffic Discipline & Friction Proxies:** Rolling 15–30 min speed CV ($\sigma_v/\mu_v$), stop-go ratio outside stops, and off-peak (11 PM–5 AM) residual speed to detect chronic potholes / speed breakers / encroachments.
4. **Driver / Vehicle Behavior:** Online EWMA pace ratio relative to expected time (adapts to current driver on shift without driver IDs), bus class (`f` field: EV/AC/ordinary/express), and historical route-dodging propensity (`detect_dodges.py`).
5. **Monsoon / Weather / Closures:** Current + 3-hour cumulative rainfall (waterlogging lag), rain × segment sensitivity, and LLM-extracted traffic police disruption risk ($R_{\text{LLM}}$).
6. **Uncertainty Calibration:** Quantile GBDT (`p10, p50, p90`) + **Online Adaptive Conformal Prediction (CQR)** so passengers see honest intervals (e.g., `6–9 min` vs `12–22 min during heavy rain`).
