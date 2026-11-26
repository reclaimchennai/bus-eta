# traffic_eval: do police traffic alerts improve v4 ETAs?

The write-up is [`docs/TRAFFIC_INTEGRATION.md`](../docs/TRAFFIC_INTEGRATION.md). This folder holds the code that
produced it and its small outputs.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `events.py` | Normalises the four police feeds (Hyderabad, Delhi, Bengaluru, Chennai) into one event list: type, severity, start/end, point or road line | `work/events.json` |
| 2 | `replay_queries.py SYSTEM DAY` (`run_all.sh` for all) | Replays a held-out day through the **live** v4 engine (`serve/eta_engine.py`), with a causal `model.json` (7 days before the day), and keeps every scored query with its bus position | `work/q_SYSTEM_DAY.npz` |
| 3 | `match_incidents.py [--window alert\|pre\|post]` | Finds the active incidents on each query's remaining path with `serve/incident_penalty.py`'s own geometry, and the delay it would add | `work/m_*.npz` |
| 4 | `gps_sees_it.py SYSTEM` | Event study: bus speed through each incident spot vs the same spot and clock time on other days (with a placebo) | `results/gps_SYSTEM_passes.csv` |
| 5 | `analyse.py` | Difference-in-differences against matched controls, the three integration routes, bootstrap CIs over incidents, charts | `results/summary.json`, `results/per_incident_excess.csv`, `results/*.png` |

Inputs: the GPS archives in `data/raw/<system>/obs-<day>.jsonl[.gz]`, the networks in `data/meta/<system>/`, the v4
models in `models/v4_clean_ensemble/`, and the incident snapshots in `src_data/` (copied from the live pipelines on
26 Nov 2026: twtrx's `out/eta/incidents.jsonl` and `out/{bengaluru,chennai}/incidents_all.geojson`, and the Delhi map's
`route-dodging/data/closures.json`).

```bash
python traffic_eval/events.py
PAR=4 traffic_eval/run_all.sh                 # ~1 h on a 14-core laptop; Delhi days take longest
python traffic_eval/match_incidents.py
python traffic_eval/match_incidents.py --window pre
python traffic_eval/match_incidents.py --window post
for s in tgsrtc delhi bmtc mtc; do python traffic_eval/gps_sees_it.py $s; done
python traffic_eval/analyse.py
```

`work/` (replays, ~2 GB) is not committed; everything in it is rebuilt by the steps above.
