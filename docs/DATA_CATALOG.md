# ETA Project — Organized Data Catalog & Storage Map

> Maintained in `~/projects/ETA` | Mirrored to Google Drive (`the owner's Google Drive/My Drive/ETA/`)  
> Read-only policy: zero original files were modified on `raw`, `svr`, `nia`, or any local `~/Desktop/projects/*` repo.

---

## Directory Layout (`~/projects/ETA/`)

| Folder | Contents | Format & Scale | Google Drive Mirror |
| :--- | :--- | :--- | :---: |
| `data/raw/{mtc,bmtc,mumbai,ncr,delhi}/` | Historical bus GPS fix archives (`obs-YYYY-MM-DD.jsonl.gz`) copied read-only from `user@map-server` | MTC 38d (573 MB), BMTC 8d (378 MB), Mumbai 17d (61 MB), NCR 6d (7.7 MB), Delhi 14d (3.5 GB) | Local only (reproducible via rsync) |
| `data/meta/{city}/` | `route_stops.json`, `stop_dist.json`, `bus_stops.geojson`, `mtc_routes.geojson`, production `model.json` | 5 cities (~45 MB total) | Local |
| `data/weather/` | `chennai_pol_hourly.parquet` (**272 stations, 130,765 station-hours** from `pol.reclaimchennai.city`) + `weather_city_hourly.parquet` (all 5 cities + 3h/6h/24h waterlogging lag) | Parquet (~1.2 MB) | `My Drive/ETA/weather/` |
| `data/osm/` | Fresh Overpass API road infrastructure per city (`{city}_osm_infra.json`): traffic signals, **new flyovers/bridges**, railway crossings, roundabouts, speed bumps | JSON (5 cities: 5,101 flyovers, 5,368 signals, 1,666 rail crossings) | `My Drive/ETA/osm/` |
| `data/uturns/` | Empirically discovered mid-route bus U-turn hotspots from GPS logs (`{city}_uturns.geojson`, `all_cities_uturns.geojson`) with median & P90 delay | GeoJSON FeatureCollections | `My Drive/ETA/uturns/` |
| `maps/` | `indian_bus_uturns_map.html` — interactive Leaflet dark-mode map of all discovered Indian bus U-turn hotspots with city switcher & delay popups | Standalone HTML | `My Drive/ETA/maps/` |
| `data/holidays_india_2025_2026.json` | Web-verified 2025–2026 Indian National & State Public/Festival/Rain Holiday Calendar (TN, Delhi, Haryana, Karnataka, Maharashtra — incl. Oct–Nov 2026 Cyclone Murasu & Chhath) | JSON (54 dates) | `My Drive/ETA/aux/` |
| `data/closures_dodges/` | `delhi_dtp_items.json` — historical Delhi Traffic Police closures/breakdowns/diversions copied from `raw:~/projects/delhi-live/dtp/items.json` | JSON (46 KB) | `My Drive/ETA/aux/` |
| `data/processed/` | Extracted causal trajectory Parquet tables (`{city}_segments.parquet`, `{city}_stops.parquet`) + `disruptions_live.json` | 21.7M segment events, 12.3M stop arrivals (~2.4 GB) | `My Drive/ETA/processed/` |
| `data/v3_cache/` | Per-day `.npz` cached v3 feature arrays (69 features incl. OSM, weather, holidays, ConSTGAT geometry) for instant re-training | `.npz` | `My Drive/ETA/v3_queries/` (`v3_train_1.5m.npz`, `v3_val.npz`) |
| `models/` | Trained models & reports: `v2_deepreta_cqr/`, `v3_ensemble_agaci/`, `gbdt_cqr_v1/`, `dwell_v1/` | LightGBM / XGBoost / CatBoost / JSON | `My Drive/ETA/models/` |
| `docs/` | `RESEARCH_AND_ARCHITECTURE.md`, `LITERATURE_AND_THEORIES.md`, `REFERENCES.bib`, `BENCHMARK_RESULTS.md`, `DATA_CATALOG.md` | Markdown + BibTeX (git-tracked) | Git (`main`) |

---

## Git Worktrees
- `ETA` (`main`): Documentation, BibTeX, verified holidays, benchmark reports, data catalog
- `ETA-wt-baseline` (`feat/baseline-eval`): `src/eval_eta.py` (1–30m stop-arrival replay evaluator)
- `ETA-wt-features` (`feat/features-and-data`): `src/weather_fetch.py`, `src/osm_infra_fetch.py`, `src/detect_uturns.py`, `src/extract_dataset.py`
- `ETA-wt-models` (`feat/ml-models`): `src/train_v2_deepreta_cqr.py`, `src/train_v3_ensemble_agaci.py`, `src/train_dwell_model.py`, `src/nowcast_ml.py`, `src/mask_cnn_baseline.py`, `src/llm_disruption_sidecar.py`, `src/colab_a100_train_v3.py`, `notebooks/colab_a100_deep_eta.ipynb`
