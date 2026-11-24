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
