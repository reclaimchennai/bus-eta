# eta-ml: the v4 ensemble as a live service

`eta_engine.py` is `train_v3_ensemble_agaci.build_v3_day` turned into a stream: the same projection, trip
cuts, probes, corridor tracker, pace and 74 features, computed in the same order. `eta_ml_service.py` runs
it for every city in one process (models loaded once), tailing each collector's own archive
(`archive/obs-<IST day>.jsonl`), and writes `out/<city>.json` every 10 s:

    {"t": now, "v": "v4", "b": {reg: [code, fix, [k, dt, dlo, dhi, ...]]}}   (seconds after the fix)

The collector merges that into `vehicles.json` just before writing it (`eta_ml_merge.py`, a copy beside each
collector; `--eta-ml` / env `ETA_ML`, mode `ETA_ML_MODE` = all | ours | off). Rows keep their shape, so the
cards, the delta feed and the GTFS-RT `trip_updates` (nia's gtfs_rt.py reads the same rows) all carry v4
times with no change of their own. 80% ranges go to `live/eta_range.json` beside it, which the cards fetch
only while open (maps-site branch `claude/eta-v4`). A bus keeps the collector's own times when eta-ml is
down (file older than 2 min), has no answer for its fix, or the bus is waiting/in depot/stray/off-progress.

The 80% band is the LightGBM P10/P90 widened by an AgACI margin per horizon bucket, adapted online from the
arrivals the stream observes (training's calibration, now fed live).

## Checks
- `src/audit_v4.py CITY DAY` rebuilds the historical speed model from the 7 days before DAY only and
  re-scores nowcast and v4 on the same queries.
- `replay_eval.py CITY DAY --model-json M [--order file|time] [--agaci]` runs this engine over an archived
  day and scores it (parity with training; `file` = the order the collector wrote the lines).
- `shadow_eval.py RUN_DIR ROUTE_STOPS STOP_DIST` scores a live run's log against arrivals seen afterwards,
  with what the public map showed at the same moment (operator's or nowcast times).

## Deploy (raw)
    serve/deploy/deploy_raw.sh --start          # code, models, venv, eta-ml.service (MemoryMax 1400M)
    serve/deploy/collectors.sh mtc ours         # one city at a time; remove = back to the collector's own
