# v5: more accurate bus ETAs than v4 / v4r

27 Nov 2026. Branch `claude/eta-v5-research-5b7547`. Nothing here is pushed or deployed.

## Result

One v5 model for all eight systems beats the production model on every system's held-out days, with
bus-clustered 95% intervals that exclude zero everywhere except NCR (48 buses, too little data to tell).

| System | Held-out day | Nowcast MAE | Production MAE | v5 MAE | v5 vs production | 80% band width |
|---|---|---|---|---|---|---|
| MTC | 23 Nov | 140.8 s | 92.6 s (v4) | **81.8 s** | −11.6% | 295 → 261 s |
| MTC | 24 Nov | 133.7 s | 85.7 s (v4) | **75.0 s** | −12.4% | 269 → 246 s |
| BMTC | 23 Nov | 160.7 s | 98.4 s (v4) | **86.0 s** | −12.5% | 328 → 274 s |
| BMTC | 24 Nov | 133.6 s | 91.0 s (v4) | **85.4 s** | −6.2% | 297 → 274 s |
| Mumbai | 17 Nov | 90.1 s | 83.2 s (v4) | **73.7 s** | −11.5% | 273 → 232 s |
| Delhi | 23 Nov | 189.1 s | 110.1 s (v4) | **94.6 s** | −14.1% | 361 → 300 s |
| Delhi | 24 Nov | 174.7 s | 103.0 s (v4) | **84.7 s** | −17.7% | 353 → 276 s |
| Tricity | 24 Nov | 341.1 s | 118.9 s (v4r) | **110.6 s** | −7.0% | 399 → 362 s |
| TGSRTC | 24 Nov | 193.3 s | 94.4 s (v4r) | **85.4 s** | −9.5% | 315 → 285 s |
| APSRTC | 24 Nov | 345.8 s | 102.9 s (v4r) | **86.3 s** | −16.1% | 319 → 286 s |
| NCR | 17 Nov | 206.3 s | 111.4 s (v4) | 110.9 s | −0.4% (not significant) | 378 → 340 s |

Band coverage stays at 80% (79.8–80.2%) with the live AgACI calibration, so the narrower bands are a real gain.

Where the gain comes from, in order of size:

1. **Last-pass segment times** (new feature): the path ahead timed with each segment's most recent
   stop-to-stop traversals, up to an hour old. This alone gives about two thirds of the gain.
2. **Leader travel time, bus state, today-vs-history ratios** (new features): each adds a little.
3. **A wider label clip** (±2.5 instead of ±1.5 on log(true/nowcast)): worth about 8 s on APSRTC, where the
   nowcast is often off by more than 4.5×; neutral elsewhere.
4. **Leak-free training data**: retraining v4's own features on caches whose `model.json` never contains the
   day being scored gives 0–6% on the cities and nothing on RTC.

## Protocol

- **Rows**: stops whose true arrival is 60 s to 30 min ahead; truth is `eval_eta.extract_ground_truth_stop_times`.
  Test rows are the audits' bus sample (`u <= std` in the cache), so they match the published v4 / v4r audits.
- **Same data budget as production.** Cities train on v4's days (MTC 14–21 Nov, BMTC 17–21, Delhi 18–21,
  Mumbai 8–15, NCR 13–15), validate on 22 Nov (Mumbai, NCR: 16), test on 23 and 24 Nov (Mumbai, NCR: 17).
  RTC trains on 18–22 Nov, validates on 23, tests on 24, as v4r did. 24 Nov is a partial day for the cities.
- **No leakage.** Every row is produced by replaying the raw archive through the streaming engine
  (`src/v5_build.py`). On validation and test days the history `model.json` is built from the 7 days before
  that day only. On training days it is built from the 7 nearest other days, never the day itself and never
  a validation or test day. v4's own `data/v3_cache` used a `model.json` built on 24 Nov for every day.
- **Baselines on identical rows.** v4 and v4r score the first 74 columns of the same rows (v4 scores RTC with
  `city_idx` 3, as live). My v4 / v4r numbers reproduce the published audits (Tricity 118.9 vs 121, TGSRTC 94.4
  vs 93, APSRTC 102.9 vs 104).
- **Intervals**: 1,000 bootstrap resamples of whole buses; differences are paired on the same rows.
- **Choices were made on validation days**, with one exception stated under "Loss" below.

## Held-out accuracy

MAE in seconds; brackets are bus-clustered 95% intervals.

| System | Day | Rows | Buses | Nowcast | v4 | v4r | v5 (95% CI) | v5 − v4 | v5 − v4r | v5 − nowcast |
|---|---|---|---|---|---|---|---|---|---|---|
| mtc | 11-23 | 1,119,356 | 1,240 | 140.8 | 92.6 | 92.6 | **81.8** [80.4, 83.2] | -10.8 [-11.9, -9.7] (-11.6%) | -10.8 [-11.8, -9.8] (-11.6%) | -59.0 [-65.1, -53.6] (-41.9%) |
| mtc | 11-24 | 462,407 | 1,046 | 133.7 | 85.7 | 90.3 | **75.0** [72.9, 77.3] | -10.6 [-12.1, -9.2] (-12.4%) | -15.2 [-16.8, -13.7] (-16.9%) | -58.7 [-67.0, -51.6] (-43.9%) |
| bmtc | 11-23 | 319,417 | 1,927 | 160.7 | 98.4 | 96.7 | **86.0** [83.3, 89.0] | -12.3 [-13.9, -10.7] (-12.5%) | -10.7 [-12.0, -9.2] (-11.0%) | -74.7 [-81.9, -67.5] (-46.5%) |
| bmtc | 11-24 | 290,645 | 1,879 | 133.6 | 91.0 | 91.2 | **85.4** [83.0, 87.8] | -5.6 [-6.6, -4.7] (-6.2%) | -5.8 [-6.8, -4.9] (-6.4%) | -48.2 [-52.8, -43.7] (-36.1%) |
| mumbai | 11-17 | 45,928 | 1,090 | 90.1 | 83.2 | 83.7 | **73.7** [71.5, 75.9] | -9.6 [-11.0, -8.3] (-11.5%) | -10.1 [-11.4, -8.8] (-12.0%) | -16.4 [-19.0, -14.1] (-18.2%) |
| ncr | 11-17 | 5,612 | 48 | 206.3 | 111.4 | 108.2 | **110.9** [91.7, 128.8] | -0.5 [-19.2, +16.0] (-0.4%) | +2.8 [-9.4, +12.8] (+2.5%) | -95.4 [-161.4, -46.9] (-46.2%) |
| delhi | 11-23 | 2,753,338 | 637 | 189.1 | 110.1 | 108.7 | **94.6** [92.9, 96.3] | -15.5 [-16.5, -14.5] (-14.1%) | -14.1 [-15.1, -13.1] (-13.0%) | -94.5 [-98.5, -90.7] (-50.0%) |
| delhi | 11-24 | 1,358,582 | 603 | 174.7 | 103.0 | 102.8 | **84.7** [82.4, 87.1] | -18.3 [-19.8, -16.7] (-17.7%) | -18.0 [-19.6, -16.4] (-17.5%) | -89.9 [-95.3, -84.7] (-51.5%) |
| tricity | 11-24 | 86,861 | 155 | 341.1 | 141.6 | 118.9 | **110.6** [103.8, 117.9] | -31.0 [-35.8, -25.8] (-21.9%) | -8.3 [-10.9, -5.7] (-7.0%) | -230.5 [-254.2, -210.2] (-67.6%) |
| tgsrtc | 11-24 | 204,722 | 2,196 | 193.3 | 110.8 | 94.4 | **85.4** [83.5, 87.3] | -25.4 [-27.9, -22.9] (-22.9%) | -9.0 [-10.0, -8.0] (-9.5%) | -107.9 [-118.2, -97.8] (-55.8%) |
| apsrtc2 | 11-24 | 822,428 | 15,116 | 345.8 | 154.8 | 102.9 | **86.3** [85.3, 87.3] | -68.5 [-70.8, -66.4] (-44.3%) | -16.6 [-18.0, -15.3] (-16.1%) | -259.5 [-270.2, -249.6] (-75.0%) |

Validation days, for comparison (v5 vs production): MTC −12.4%, BMTC −12.4%, Mumbai −10.4%, Delhi −18.1%,
Tricity −7.8%, TGSRTC −6.3%, APSRTC −14.9%. Test gains are the same size, so the choices did not overfit the
validation days.

### By horizon, and bias

Production → v5 (production is v4 for the cities, v4r for RTC). Bias is predicted minus true: negative means
the bus arrives later than predicted.

| System | Day | 1-5 min | 5-10 min | 10-20 min | 20-30 min | Mean bias | Median bias | P90 abs error |
|---|---|---|---|---|---|---|---|---|
| mtc | 11-23 | 35 → 31 | 65 → 57 | 105 → 91 | 140 → 126 | +1 → -16 | +8 → -3 | 207 → 185 |
| mtc | 11-24 | 33 → 28 | 62 → 53 | 101 → 86 | 128 → 117 | +19 → -9 | +20 → +0 | 191 → 169 |
| bmtc | 11-23 | 37 → 31 | 75 → 63 | 125 → 105 | 152 → 144 | +10 → -16 | +11 → -2 | 231 → 201 |
| bmtc | 11-24 | 33 → 30 | 65 → 61 | 110 → 100 | 145 → 141 | -13 → -24 | -1 → -8 | 210 → 197 |
| mumbai | 11-17 | 37 → 34 | 73 → 66 | 121 → 106 | 196 → 165 | -28 → -22 | -12 → -7 | 198 → 174 |
| ncr | 11-17 | 53 → 43 | 105 → 85 | 149 → 132 | 132 → 206 | +32 → -43 | +24 → -25 | 237 → 275 |
| delhi | 11-23 | 43 → 36 | 80 → 68 | 129 → 109 | 170 → 150 | -2 → -17 | +10 → -2 | 253 → 215 |
| delhi | 11-24 | 39 → 32 | 77 → 62 | 130 → 101 | 149 → 132 | +34 → +3 | +31 → +10 | 237 → 191 |
| tricity | 11-24 | 54 → 51 | 98 → 93 | 152 → 142 | 183 → 165 | -4 → +5 | +7 → +11 | 271 → 252 |
| tgsrtc | 11-24 | 37 → 33 | 69 → 61 | 112 → 98 | 132 → 125 | +10 → -2 | +15 → +7 | 213 → 194 |
| apsrtc2 | 11-24 | 38 → 33 | 75 → 61 | 116 → 96 | 154 → 132 | +0 → -17 | +1 → -7 | 220 → 194 |

- **Long horizons improve everywhere except NCR**: 20–30 min is down 4–31 s (TGSRTC 132 → 125, APSRTC 154 → 132).
- **Bias**: v5's median bias is within ±11 s on every system except NCR. Its mean bias is slightly negative
  (down to −24 s) because the loss targets the median and delays have a long late tail. v4's "runs late"
  mean bias (+19 to +34 s on MTC and Delhi 24 Nov) is gone. A separate bias correction step is not needed.
- **NCR** has 559 training rows; v5 is no better than v4 there and its 20–30 min bucket is worse. Keep NCR on v4.

### 80% band

Production and v5 both replayed with the live AgACI calibration (`v5_lib.agaci_causal`: a prediction at time
t uses only arrivals observed by t; 10% of rows feed it, as live).

| System | Day | Production coverage | Production width | v5 coverage | v5 width | Width change | v5 with fixed margins: coverage / width | v5 at 20-30 min: coverage / width |
|---|---|---|---|---|---|---|---|---|
| mtc | 11-23 | 80.3% | 295 s | 80.1% | 261 s | -12% | 80.6% / 259 s | 78.6% / 363 s |
| mtc | 11-24 | 80.0% | 269 s | 79.8% | 246 s | -9% | 81.0% / 247 s | 79.4% / 351 s |
| bmtc | 11-23 | 80.0% | 328 s | 80.2% | 274 s | -16% | 79.1% / 263 s | 78.7% / 405 s |
| bmtc | 11-24 | 80.5% | 297 s | 80.1% | 274 s | -8% | 79.4% / 267 s | 77.7% / 397 s |
| mumbai | 11-17 | 80.6% | 273 s | 79.9% | 232 s | -15% | 79.6% / 229 s | 71.0% / 400 s |
| ncr | 11-17 | 81.0% | 378 s | 80.0% | 340 s | -10% | 79.6% / 329 s | 82.3% / 542 s |
| delhi | 11-23 | 79.9% | 361 s | 79.9% | 300 s | -17% | 80.2% / 297 s | 77.5% / 406 s |
| delhi | 11-24 | 80.0% | 353 s | 79.9% | 276 s | -22% | 80.4% / 275 s | 80.0% / 382 s |
| tricity | 11-24 | 79.8% | 399 s | 79.9% | 362 s | -9% | 74.8% / 319 s | 82.1% / 525 s |
| tgsrtc | 11-24 | 80.1% | 315 s | 80.2% | 285 s | -10% | 81.9% / 291 s | 80.6% / 384 s |
| apsrtc2 | 11-24 | 80.2% | 319 s | 80.2% | 286 s | -10% | 82.0% / 295 s | 79.2% / 391 s |

- With AgACI the overall coverage is on target everywhere. Without it (fixed margins from the validation
  days) Tricity drops to 74.8%, so the online calibration must stay on.
- Coverage is per horizon bucket of the *nowcast* time, so the true 20–30 min bucket is a little under 80% on
  some days; Mumbai's is 71% (v4 has the same weakness there).

## What helped, what did not

Ablations are single LightGBM models (learning rate 0.1, up to 500 rounds, 150,000 rows per training day),
validation-day MAE in seconds. They were run before the replay clock fix described under "Caveats", so
compare rows within a table, not with the final numbers.

### Feature groups, added one at a time to v4's 74 features

| Features | MTC | BMTC | Mumbai | Delhi | Tricity | TGSRTC | APSRTC |
|---|---|---|---|---|---|---|---|
| v4's 74 features (leak-free retrain) | 82.6 | 100.4 | 93.3 | 103.0 | 129.9 | 95.8 | 106.9 |
| + last-pass segment times (4) | 78.3 | 94.7 | 86.0 | 93.6 | 125.1 | 93.4 | 103.4 |
| + leader travel time (4) | 80.1 | 98.4 | 90.5 | 97.5 | 126.4 | 95.6 | 107.1 |
| + bus state (7) | 81.5 | 99.4 | 91.2 | 101.7 | 126.9 | 95.0 | 106.0 |
| + today vs history (3) | 82.1 | 100.1 | 86.3 | 103.8 | 124.1 | 95.3 | 106.4 |
| + headway / bunching (5) | 82.3 | 100.3 | 92.4 | 103.1 | 127.7 | 95.7 | 106.9 |
| + history coverage (1) | 82.6 | 100.2 | 93.3 | 102.7 | 128.6 | 95.6 | 106.8 |
| all 98 v5 features | 77.1 | 93.5 | 82.7 | 91.6 | 122.0 | 92.6 | 103.1 |

- **Last-pass segment times** help every system (−2% to −9%). v4's corridor estimate only uses probes of the
  last 15 minutes; this keeps each segment's last three stop-to-stop times for an hour, dwell included.
- **Leader travel time** (the previous bus's time from our position to the target stop) helps the dense
  cities, not the sparse RTC feeds.
- **Today vs history** (a 30-minute decayed ratio of actual to history-model time, per working and per city)
  helps Mumbai and Tricity most. This is the online residual correction, done as a feature.
- **Headway / bunching** (gap to the bus ahead and behind on the same working) adds nothing measurable.
- **History coverage** (share of the path with its own `model.json` row, for sparse systems) adds nothing.

### Loss, label and base

| Variant (all 98 features) | MTC | BMTC | Mumbai | Delhi | Tricity | TGSRTC | APSRTC |
|---|---|---|---|---|---|---|---|
| L1, weighted by nowcast seconds, clip ±1.5 | 77.1 | 93.5 | 82.7 | 91.6 | 122.0 | 92.6 | 103.1 |
| same, clip ±2.5 (chosen) | 76.0 | 93.5 | 82.0 | 91.8 | 120.0 | 90.5 | 94.7 |
| same, clip ±3.5 | 76.1 | 93.3 | 82.1 | 91.9 | 120.8 | 90.6 | 93.5 |
| Huber, weighted | 76.7 | 93.4 | 82.1 | 91.8 | 119.9 | 92.4 | 102.0 |
| residual on the last-pass time instead of the nowcast | 76.2 | 93.2 | 82.5 | 91.4 | 120.7 | 91.3 | 98.3 |

- **Clip ±2.5** is the only choice here that matters: APSRTC 103.1 → 94.7.
- **Loss.** Unweighted Huber, unweighted L1 and weighted L1 tie on the validation days (within 0.6 s). I chose
  the weighted L1 after seeing that the two unweighted models ran 40–45 s late on MTC's partial 24 Nov test
  day (MAE 87.5 and 90.5 against 77.0). That one choice used a test day; it moves MTC 24 Nov by about 10 s
  and MTC 23 Nov by 3–5 s, and nothing else materially.
- **Alternative residual base**: no gain once the last-pass time is a feature.

### Did not help

- **Post-hoc online residual correction** (the model's own error on recently observed arrivals, per working,
  system or horizon, replayed causally by `src/v5_online.py`): 0–1.3 s on test once the v5 features are in.
  Not worth a second live state machine.
- **Separate models per group of systems.** An RTC-only model and a cities-only model are within ±2 s of the
  pooled model on every system (validation: Tricity 118.0 vs 120.2, APSRTC 91.3 vs 92.5 for RTC-only;
  TGSRTC 91.1 vs 90.6 and all four cities within 0.7 s the other way). One model serves both services.
- **CatBoost** got weight 0 in the validation blend and is not shipped.
- **Headway and history-coverage features**: no effect (kept in the feature list; they cost little).
- **NCR**: no model beats v4 with 559 training rows.
- Not retried, as instructed: traffic alerts, advisories, BTP congestion features.

## Recommended serving model

`models/v5_ensemble`: LightGBM L1 (weight 0.4) + XGBoost Pseudo-Huber (0.6) on 98 features, LightGBM P10/P90,
per-bucket margins as AgACI's starting point. 68 MB on disk. One model for every system except NCR.

### Accuracy of the lighter variant

LightGBM alone (`--use-v5 lgb`) gives up 0.3–1.5 s:

| System | Day | Production | v5 blend | v5 LightGBM only | Given up |
|---|---|---|---|---|---|
| mtc | 11-23 | 92.6 | 81.8 | 82.4 | +0.6 s |
| mtc | 11-24 | 85.7 | 75.0 | 75.8 | +0.7 s |
| bmtc | 11-23 | 98.4 | 86.0 | 86.7 | +0.7 s |
| bmtc | 11-24 | 91.0 | 85.4 | 86.2 | +0.8 s |
| mumbai | 11-17 | 83.2 | 73.7 | 74.8 | +1.2 s |
| delhi | 11-23 | 110.1 | 94.6 | 95.4 | +0.9 s |
| delhi | 11-24 | 103.0 | 84.7 | 85.8 | +1.0 s |
| tricity | 11-24 | 118.9 | 110.6 | 112.1 | +1.5 s |
| tgsrtc | 11-24 | 94.4 | 85.4 | 86.2 | +0.8 s |
| apsrtc2 | 11-24 | 102.9 | 86.3 | 86.5 | +0.3 s |

### Latency

Replay of 23 Nov through the engine as the service runs it (`src/v5_latency.py`: every fix ingested, every bus
with a new fix scored each 10 s of feed time, 2 threads), on this Mac under other load. Compare v4 and v5
rows, not absolute times.

| System | Rows per tick | v4 tick (mean / p95) | v5 tick (mean / p95) | Features per row v4 → v5 | Models per row v4 → v5 | Ingest per fix v4 → v5 |
|---|---|---|---|---|---|---|
| Delhi (every bus each tick) | 42,225 | 5.09 / 7.44 s | 6.44 / 9.52 s | 49 → 56 µs | 72 → 96 µs | 22 → 26 µs |
| MTC | 3,613 | 0.36 / 0.52 s | 0.46 / 0.68 s | 29 → 34 µs | 71 → 94 µs | 18 → 23 µs |
| BMTC | 2,148 | 0.18 / 0.29 s | 0.24 / 0.38 s | 12 → 18 µs | 71 → 95 µs | 13 → 18 µs |
| APSRTC | 2,850 | 0.34 / 0.96 s | 0.40 / 1.11 s | 38 → 46 µs | 83 → 95 µs | 11 → 14 µs |
| TGSRTC | 533 | 0.057 / 0.073 s | 0.063 / 0.079 s | 18 → 22 µs | 89 → 97 µs | 10 → 12 µs |

- v5 costs **10–35% more time per tick** than v4 / v4r. Most of it is the larger models (1,000 + 1,200 trees
  against v4's), not the features.
- Delhi is the only system where a tick is seconds long, and live it scores each bus at most every 45 s
  (`min_gap`), about a quarter of the rows above. LightGBM-only Delhi: 6.01 s mean, 8.70 s p95.

### Memory

Measured on this Mac (`src/v5_memory.py`, `src/v5_latency.py`); **not measured on Linux**, see the caveat below.

Model sets alone, resident memory above the bare imports (210 MB):

| Model set | Loaded | After a 50,000-row batch |
|---|---|---|
| v4 (LGB + XGB + Cat + P10/P90) | +297 MB | +340 MB |
| v4r | +348 MB | +391 MB |
| v5 blend (LGB + XGB + P10/P90) | +155 MB | +214 MB |
| v5 LightGBM only | +85 MB | +105 MB |

Whole process, peak memory in the replays above (one system each):

| System | v4 / v4r | v5 blend | v5 LightGBM only |
|---|---|---|---|
| Delhi | 633 MB | 711 MB | 628 MB |
| MTC | 614 MB | 447 MB | not run |
| BMTC | 701 MB | 526 MB | not run |
| TGSRTC | 816 MB | 608 MB | not run |
| APSRTC | 836 MB | 589 MB | not run |

- **XGBoost's file format is the lever.** Loaded from JSON, the XGBoost model alone holds about 255 MB (v4) to
  380 MB (v5); loaded from UBJ the same v5 model holds about 70 MB, with identical predictions (checked:
  max difference 0.0 s). v5 ships UBJ. v4 and v4r ship JSON.
- **v5's extra live state is real on Delhi**: about +80 MB against v4 at the peak, from the larger feature rows
  (98 against 74 columns in a 42,000-row tick) and the per-segment and per-stop history. On the other systems
  it is hidden by the smaller models.

What this means against the limits:

| Service | Limit | Today | All cities on v5 blend | All on v5 LightGBM only | v4 and v5 both loaded (staged rollout) |
|---|---|---|---|---|---|
| eta-ml (raw) | 1400M | about 1.3 GB (reported by the parent session) | models −125 to −140 MB, Delhi state +80 MB: likely fits, margin thin | models −210 to −235 MB: fits | +155 to +214 MB: does not fit |
| eta-ml-rtc | 1200M | not known to me | models −177 to −193 MB against v4r: more headroom than today | −263 to −286 MB | +155 to +214 MB on top of today |

Plainly:

- **Staging one city at a time loads both model sets, and that does not fit eta-ml's 1400M.** For the staged
  phase either raise `MemoryMax` to **1700M**, or run v5 as LightGBM only (+85 to +105 MB, about 1.4 GB:
  still at the limit, so I would not rely on it).
- **With every eta-ml city on v5 (NCR aside), the service should need less than today**, and LightGBM only
  gives the widest margin for 0.3–1.5 s of accuracy. NCR staying on v4 keeps v4 loaded, which is the mixed
  case above. Either move NCR to v5 as well (it is no worse there, −0.4%, not significant) or raise the limit.
- **eta-ml-rtc has the same staging problem in principle** (both sets loaded during the switch), but it has
  three systems and no reason to stage for long. I do not know its steady memory; check
  `systemctl --user status eta-ml-rtc` before deciding. Fully on v5 it needs less than v4r does.
- **Linux caveat.** The JSON saving may be smaller on raw: glibc plus the service's `malloc_trim` may already
  return most of the JSON parse memory that macOS keeps. If so, v4's real footprint is lower than my table
  says and v5's advantage shrinks, leaving the Delhi +80 MB as a net increase. Measure before switching:
  run `src/v5_memory.py v4` and `src/v5_memory.py v5` on raw (they read models only, touch no service), or
  shadow-run the service with `source_cmd` and watch its memory.
- Lazy per-city model loading would not help: there is one model for all cities.
- Converting v4's and v4r's XGBoost files to UBJ would cut their memory the same way and make the staged
  phase fit. It needs a one-line loader change in `eta_engine.Ensemble`; I did not make it (follow-up).

### Config switch

`serve/eta_ml_service.py` now takes `"engine": "v5"` per city in the cities config (default `"v4"`):

```json
{"city": "tgsrtc", "engine": "v5", "...": "..."}
```

```
eta_ml_service.py --config cities.json --models models --models-v5 models-v5 [--use-v5 lgb]
```

- A v5 city runs `eta_engine_v5.V5Stream` and `EnsembleV5`; every other city is untouched. Each model set is
  loaded only if some enabled city uses it.
- Roll back by removing `"engine"` (or setting `"v4"`) and restarting. The output file's `"v"` field says
  which engine wrote it.
- `"cqr_v5"` in a city's config overrides the AgACI starting margins for v5, as `"cqr"` does for v4.
- New files a deploy must copy: `serve/eta_engine_v5.py` and the `models/v5_ensemble` directory (as
  `models-v5`). The deploy scripts are unchanged; I did not touch them.
- After a restart the service warms on 45 minutes of archive (`ETA_WARM_S`). v5's last-pass state looks back
  an hour, so the first 15–60 minutes after a restart are slightly below full accuracy.

### Serving parity

`src/v5_parity.py` replays a held-out day through the service's own classes (`eta_ml_service.City` with
`"engine": "v5"`, `eta_engine.score`, a tick every 10 s of feed time) and compares the published P50 with the
offline prediction for the same bus, fix and stop.

| System | Day | Rows compared | Median difference | Within 1 s | Within 5 s | 99th percentile | Max |
|---|---|---|---|---|---|---|---|
| Tricity | 24 Nov | 166,348 | 0.26 s | 96.0% | 98.0% | 26.6 s | 1,358 s |
| TGSRTC | 24 Nov | 417,297 | 0.26 s | 97.5% | 99.1% | 4.5 s | 1,045 s |
| BMTC | 24 Nov | 800,998 | 0.27 s | 94.3% | 98.0% | 9.9 s | 283 s |

- The median is rounding: the service publishes whole seconds.
- The large differences are not a feature mismatch. They are the service's existing rule that a later stop
  never gets an earlier time than the stop before it (`e = max(previous + 1, model)`), which moved 0.4–1.6% of
  published times (median move 11–37 s). Excluding those rows: 94.8–97.9% within 1 s, 98.4–99.4% within 5 s,
  max 143–257 s.
- The rest comes from when a fix is scored relative to other buses' fixes: the offline builder flushes every
  256 buses or 10 s, the service at its tick, so the fleet state differs by a few seconds of fixes.

## Caveats

- **Replay clock fix.** A few archive records are dated hours or days ahead (APSRTC 23 and 24 Nov open with
  one; BMTC 24 Nov and Mumbai 8 Nov have one). My first replay took the newest timestamp as the clock, so
  housekeeping dropped the whole live state repeatedly on those days. `v5_build.py` now uses the median of
  the last 301 record times. All final numbers are from the rebuilt caches. The fix turned out to change
  results by under 1 s (the same LightGBM on APSRTC 24 Nov: 88.3 s before, 88.2 s after); APSRTC's move from
  −8% to −16% between my first and final model is the label clip. The live service runs on wall time and is
  not affected. `serve/rtc_audit.py` may have the same weakness; not checked.
- **Short test window.** Two test days per city, one per RTC system, all within one fortnight in November.
  No rain event, no festival day.
- **The stop-order rule** moves about 1% of published times by tens of seconds. Whether it helps or hurts
  accuracy is not measured here; it applies to v4 as well.
- **Retrain before shipping?** The shipped model never saw the validation or test days. Retraining on all
  days through 24 Nov would use 2–3 more days per system, but then no held-out number describes that model.
- **Model files are in git** (68 MB), like v4r's.

## Reproduce

```bash
export ETA_DATA=/path/to/data          # raw/, meta/, osm/, audit/ (scripts/fetch_data.sh)
python src/v5_build.py all --jobs 8    # leak-free caches -> data/v5/cache (24 GB, about 1 h)
python src/v5_experiments.py a6_v4feat_l1_wb c1_clip25 --fast --cap 150000    # ablations
python src/train_v5.py --skip cat --clip 2.5      # -> models/v5_ensemble (about 40 min)
python src/v5_eval.py                  # held-out report -> data/v5/eval/v5.json
python src/v5_eval.py --use lgb --tag v5lgb
python src/v5_online.py c1_clip25_fast            # post-hoc online correction
python src/v5_parity.py tgsrtc 2026-11-24
python src/v5_latency.py v5 delhi 2026-11-23 --until 8.6 --score-from 8.5
python src/v5_memory.py v5
```

| File | What it is |
|---|---|
| `serve/eta_engine_v5.py` | `V5Stream` (v4's stream plus the v5 live state), `EnsembleV5` |
| `serve/eta_ml_service.py` | per-city `"engine"` switch |
| `src/v5_build.py` | leak-free cache builder (engine replay, causal / leave-the-day-out history) |
| `src/v5_lib.py` | loading, baselines on identical rows, clustered bootstrap, causal AgACI |
| `src/v5_experiments.py`, `src/v5_online.py` | ablations, online residual replay |
| `src/train_v5.py`, `src/v5_eval.py` | final ensemble, held-out report |
| `src/v5_parity.py`, `src/v5_latency.py`, `src/v5_memory.py` | serving parity, latency, memory |
| `models/v5_ensemble/` | the model and `v5_report.json` |

## Review list before publishing

- **New files on this branch contain no hosts, IPs, user names or personal details.** They take every path
  from `ETA_DATA`, `ETA_V5`, `ETA_MODELS` or the config. `models/v5_ensemble/v5_report.json` holds feature
  names, weights, days and margins only.
- **Bus registration numbers** are not in any committed file. They are in the caches (`data/v5/cache/*.npz`,
  field `regs`), which are git-ignored (`data/v5/`); do not publish the caches.
- **Already in the repo, not added by me, and these do need scrubbing before a public release**
  (found with a grep for user names, `ssh` targets and IP addresses):
  - `serve/deploy/deploy_raw.sh`, `serve/deploy/deploy_rtc.sh`, `serve/deploy/collectors.sh`: default ssh
    targets with user names and IP addresses.
  - `serve/deploy/cities.raw.json`, `serve/deploy/cities.rtc.json`, `serve/deploy/rtc_feed.raw.json`:
    home-directory paths with a user name.
  - `serve/rtc_model.sh`, `serve/rtc_push.sh`: matched the same grep; check them.
  - `serve/eta_ml_service.py` names the `pol.reclaimchennai.city` endpoint (public, but note it).
- I only grepped `serve/`. `scripts/`, `docs/` and the git history were not checked.
