# v4 on the RTC maps — audit, 24 Nov 2026

The v4 ensemble was trained on five city systems (MTC, BMTC, Mumbai, Gurugram, Delhi). These checks ask, for each
other live map, whether it beats what that map shows today. Every system borrows city_idx 3; on Tricity's validation
day (22 Nov) indexes 0–4 gave 141.3–142.1 s, so the choice doesn't matter.

Two checks, MAE in seconds on stops 1–30 min ahead:
- **Held-out day.** `rtc_audit.py`: model.json built only from the 7 days before the scored day
  (`travel_model --until day-1`). The live engine is replayed in the collector's write order. Nowcast and v4 are
  scored on identical queries. Truth is the stop crossing interpolated between GPS fixes.
- **Shown vs v4.** `rtc_shadow_eval.py`: the public map's etas.json was recorded on 24 Nov, evening, from 19:20 IST.
  For the same bus, fix and stop, it compares what the map showed with what v4 would have shown.

## Results

| system | held-out day: nowcast → v4 (n, 80% band cover) | what the map shows vs v4 | verdict |
|---|---|---|---|
| **Tricity** (CTU) | 23 Nov 342 → 153 (100k, 73%); 24 Nov 344 → 141 | CTU ETA + collector nowcast **299 vs 122** (n=8,527) | **on** |
| **APSRTC** | 23 Nov 421 → 182 (69k, 75%); 24 Nov 403 → 169. City 371 → 172, mofussil 394 → 171, intercity 556 → 224 | timetable: median **975 vs 98** (mean 4,963 vs 179) | **on** (the map had no ETAs) |
| **APSRTC**, cumulative network (25 Nov) | 24 Nov 344 → 155 (236k at a 15% bus sample, 78%). City 301 → 135, pallevelugu 357 → 164, intercity 492 → 191; 20–30 min 593 → 307 | — | **on** (replaces the row above) |
| **TGSRTC** | 23 Nov 184 → 122 (184k, 82%). City 141 → 108, mofussil 221 → 139, intercity 392 → 171 | tsrtc-eta **135 vs 93** (n=631). 1–5 min 119 vs 31; 20–30 min 208 vs **285** (n=74) | **on, ≤ 20 min ahead** (ETA_ML_HMAX=1200) |
| AMTS | 24 Nov 423 → 153 (55k), but band covers only 55% and bias is +82 s; model.json has 1 day of history | the map shows an ETA for 15 of 513 buses | off: the feed collects; re-audit after 7 days |
| UPSRTC | no history: the archive has no route workings | 191 vs 193 (n=6,048), v4 with no model.json | off: the feed collects; re-audit after 7 days |
| OSRTC | 23 Nov 174 → 166 (5.9k, +4.6%); 20–30 min worse (297 → 391); band 65% | no ETAs; vehicles.json has no rid | off |
| Kerala | 23 Nov 127 → 109 but n=647: a fix every ~40 min per bus | — | not viable |
| Haryana /hr | n=0: stops ~60 km apart, fixes ~10 min apart | — | not viable |
| RSRTC, GSRTC/MPRTC (katch) | origin/destination-only services; no rid or ETAs in the feed | — | not viable |
| Kolkata | no live buses | — | — |

- Retraining with these systems is not needed for the gains above. The 80% band covers 73–82% before the
  online AgACI adapts it live.
- The engine's own nowcast is weak on these feeds: heavy tails from long dwells, and sparse fixes.
- The shown checks cover one evening only. TGSRTC's 20–30 min bucket is small, so re-check it after a full day:
  - `rtc_feed.py` with `--shown` on the Mac, then
  - `rtc_shadow_eval.py tgsrtc DIR DAY`.
  - If v4 holds, drop `ETA_ML_HMAX`.

## APSRTC network, 25 Nov

- **The problem.** The first network came from `live/trips.json`. That file lists only the services running at that
  moment, and the 3-hourly rebuild replaced the network each time. Only 1,472 of the day's ~31k service-days matched,
  so the map had v4 times for about 500–1,000 buses.
- **The new network** (`rtc_adapt.py ap-net`) is cumulative:
  - Every service ever met is kept in `net/apsrtc/ap_chains.json.gz`.
  - It was seeded from SVR's discover.py templates and trips.py's fetched routes (read-only): 73,522 services.
  - It grows from `trips.json` and from the feed's `new_workings.jsonl`. The feed reads `trips.json` every 5 min.
- **Chains and aliases.**
  - Those 73,522 services share 14,672 stop chains. Each chain is one entry; each service key is an alias `{"a": chain}`.
  - The engine's `Routes` resolves aliases, and `route_names.json` writes them out in full, so the collectors' merge is unchanged.
  - travel_model reads full entries only, so it gets `rtc_adapt.py canon`: the chains alone, plus an archive with codes mapped to chains.
- **Stop ids are places** (`name@lat,lon`, 4 dp), so services on the same road share segment speeds and dwells.
  - The old per-service ids (`KEY:i`) never pooled.
  - Because the ids changed, the seed model.json was rebuilt (`model_apsrtc2_until_2026-11-24.json`).
- **Coverage.** 98% of the collector's fixes now fall on a known service, up from ~20%.
- **Memory.** Engine Routes 38 → 111 MB. Collector merge tables 8 → 23 MB; apsrtc-live runs at ~110 MB of 512.
  travel_model peaks at 352 MB and ap-net at 495 MB, under eta-rtc-model's 800M cap.
- **Deploy:** `deploy_rtc.sh apnet`.

## v4r: retrained with the RTC systems, 26 Nov

- **Data.** `rtc_audit.py --dump` writes the rows the live engine scored, with their truth, for each day. Each day's
  model.json is causal (built from the 7 days before it only).
  - Days 18–24 Nov for Tricity (all buses), TGSRTC (40% of buses) and APSRTC (15% of buses).
  - Rows: Tricity 1.1M, TGSRTC 2.5M, APSRTC 1.4M (`ETA/data/rtc_cache/`).
- **Training.** `src/train_v4_rtc.py` is v4's recipe unchanged, on the five city caches plus these rows.
  - Each RTC system gets its own city_idx: Tricity 5, TGSRTC 6, APSRTC 7.
  - The RTC days are split: 18–22 Nov train, 23 Nov validation, 24 Nov held out.
- **Held-out day.** 24 Nov, live-engine replay (`rtc_audit.py --models v4r_rtc --city-idx N`), MAE in seconds:

| system | nowcast | v4 (ci 3) | v4r | 20–30 min v4 → v4r | 80% band | bias v4 → v4r |
|---|---|---|---|---|---|---|
| Tricity | 340 | 143 | **121** | 269 → 187 | 73 → 74% | −21 → +3 |
| TGSRTC | 193 | 109 | **93** | 195 → 129 | 83 → 81% | −16 → +15 |
| APSRTC | 344 | 155 | **104** | 308 → 158 | 78 → 81% | −46 → −1 |

- **The five city systems**, on their own test days with identical rows: MTC 94.8 → 95.0, BMTC 93.0 → 92.6,
  Mumbai 79.8 → 81.2, NCR 125.8 → 119.9, Delhi 97.7 → 96.9. Mumbai is 1.4 s worse.
  - So v4r is served only to the RTC maps: `models-rtc/`, read by eta-ml-rtc. The city maps' eta-ml keeps `models/`.
- **Deploy:** `deploy_rtc.sh v4r`.
  - TGSRTC keeps ETA_ML_HMAX=1200 until a fresh shown-vs-v4 check. v4r's 20–30 min error, 129 s, is now below
    what tsrtc-eta showed in that bucket on 24 Nov (208 s, n=74).

## How it runs

```
SVR collectors (unchanged) --vehicles.json--> raw eta-rtc-feed --obs-*.jsonl--> raw eta-ml-rtc (2nd instance, cap 1200M)
     ^                                          (RTC_SERVER_LAN:8091,               |
     |  ETA_ML=/data/rtc/eta-ml/out/<sys>.json   read-only)                      v
     +------ eta-ml-push (rsync -z every 15 s, one kept-open ssh) <-- out/rtc/<sys>.json + rtc/net/<sys>/route_names.json
```

- **eta-rtc-model.timer**, every 3 h:
  - rebuilds each enabled system's network from what its map serves (TGSRTC: adds the new workings the feed met);
  - rebuilds model.json from 7 days once 8 days of feed archive exist. Until then it keeps the laptop-built seed.
- **Collector hooks**, inert unless ETA_ML is set:
  - chandigarh_bus `server/live.py`
  - tsrtc `server/eta.py`
  - apsrtc `live.py` (writes a new live/etas.json)
- **Cards**: maps-site branch `claude/eta-v4-rtc`.
  - The 80% range shows on bus cards and stop arrivals for /tricity, /tgsrtc and /apsrtc.
  - /apsrtc shows the model's predicted times on the stops ahead.
  - rtc.js renders `tt.rng` / `a.rng`.
- **Memory**:
  - eta-ml-rtc measured ~525 MB on the Mac with all three systems.
  - TGSRTC's network costs ~28 MB in tsrtc-eta, which already runs at 466 MB of its 512 MB MemoryHigh (route_names.json is indexed for this).
  - The main eta-ml (MTC/BMTC/Delhi) is untouched.
