# traffic/: the live traffic-alert scrapers

This folder has the code that collects traffic police alerts and advisories for Hyderabad, Cyberabad, Delhi, Chennai
and Bengaluru, and turns them into geocoded, typed, time-bounded events. Those events feed the bus maps' incident layers
and the ETA experiments in `../traffic_eval/`. The copies here come from the working repos (`twtrx`, the Nitter setup,
`delhi_bus/server/dtp_closures.py`). Hosts, users and keys were replaced with environment variables; see
`config.env.example` and `twtrx/sorsa.env.example`. The data these scrapers produced is in `../data/traffic/`, limited
to official accounts and with phone numbers redacted. How the history was obtained, and what each source can and
cannot give, is in [`../docs/TRAFFIC_SOURCES.md`](../docs/TRAFFIC_SOURCES.md).

```
 sources                         code here                                     output
 ───────                         ─────────                                     ──────
 X history (Sorsa API)     ──►  twtrx/scrape.py, scrape_x.py, probe.py    ──►  raw pages (not published) → posts
 X live (self-hosted Nitter)──► nitter/ (podman Nitter + poll.py, cron)   ──►  ~/nitter/out/latest.json
                                twtrx/pipeline/x_live.py (every minute)
 WhatsApp channel (Mac)    ──►  twtrx/pipeline/whatsapp_db.py, ingest.py
 HTP e-challan site        ──►  twtrx/pipeline/hydtp_web.py
 BTP Power BI dashboard    ──►  twtrx/pipeline/btp_powerbi.py, btp_congestion.py
 Delhi Traffic Police site ──►  delhi_dtp/dtp_closures.py
                                twtrx/pipeline: relevance → classify → extract → gazetteer → route → events
                                                                                ──►  events.jsonl, eta/incidents.jsonl,
                                                                                     live incident layers, priors
```

## twtrx/: Hyderabad + Cyberabad pipeline, and the Chennai / Bengaluru / Delhi forwarders

| File | What it does |
|---|---|
| `scrape.py` | @CYBTRAFFIC history through Sorsa `/v3/search-tweets`: one keyword `OR` query, every page cached, resumable, budgeted. |
| `probe.py` | Date-windowed version of the same query (`since:` / `until:`) to fill stretches that long pagination skipped. |
| `scrape_x.py` | Generic "zero-waste" scraper for any account (used for @HYDTP): `back` / `new` / `window` modes, re-anchors on `until:` instead of paging deep, stops at 429 or when 3 requests remain. `XTAG=HYDTPinfo` switches to the short hashtag query that reaches further back. |
| `clean.py` | First relevance pass on the raw pages (road event vs awareness/campaign posts). |
| `pipeline/relevance.py` | Keeps posts about road movement, drops awareness drives, fines, inspections. |
| `pipeline/classify.py` | Type (accident, breakdown, spillage, waterlogging, tree fall, works, event, closure/diversion, congestion, U-turn change), severity, speed multiplier, start/end, "cleared" updates. |
| `pipeline/extract.py` | Place phrases, `from A towards B via C`, the HTP `📍 A – B – C` lines, stated time windows. |
| `pipeline/gazetteer.py` | Resolves places: aliases (`data/*aliases.csv`, human-curated), bus stops, OSM names, Nominatim; region-aware (HTP vs CTP jurisdiction). |
| `pipeline/route.py` | Road-following lines between places through OSRM (never straight joins). |
| `pipeline/events.py`, `build.py`, `ingest.py`, `manual.py` | Event store, rebuilds, paste/export ingest, curated corrections (`data/manual_events.json`). |
| `pipeline/x_live.py` | Reads Nitter's `latest.json` each minute, ingests new Hyderabad/Cyberabad posts, forwards Delhi (`dtptraffic`), Chennai (`ChennaiTraffic`) and Bengaluru posts. |
| `pipeline/whatsapp_db.py` | Copies WhatsApp Desktop's `ChatStorage.sqlite` read-only to a temp dir and reads **only** the named channel's rows (Mac). |
| `pipeline/hydtp_web.py` | Hyderabad Traffic Police e-challan site: live alert JSON (start = first seen, end = when it changes), "Latest News" advisories with ETag. |
| `pipeline/chennai.py`, `bengaluru.py` | Chennai and Bengaluru road updates into events with their own gazetteers and MTC/BMTC stop ids. |
| `pipeline/btp_powerbi.py` | Bengaluru Traffic Police's public Power BI congestion report, dumped through the same calls the embedded page makes. Needs `BTP_POWERBI_KEY` (the public embed key on the page). |
| `pipeline/btp_congestion.py` | Hour-of-week congestion profile per BTP grid cell, per BMTC stop (200 m), per corridor. |
| `pipeline/collect.py`, `publish.py`, `refdata.py`, `make_map.py` | The live loop (cron every minute), atomic publishing of the live layer, daily reference data, the history map. |
| `scripts/blr_accounts.py`, `nitter_*.py` | Finding and probing the Bengaluru police accounts, paging a Nitter search for history. |
| `scripts/mss.crontab`, `mss_push_code.sh`, `publish_live.sh` | How it was deployed (cron on a small Linux box next to Nitter). |
| `data/` | Hand-made reference only: aliases, curated events, the 67 Bengaluru police accounts, BTP junctions/station areas, Power BI lookups. OSM cuts and caches are rebuilt (`ogr2ogr -spat … -where "name IS NOT NULL"` from the India PBF). |

Setup: `python3 -m venv .venv && .venv/bin/pip install rapidfuzz shapely pyarrow`, then copy `config.env.example` to
`config.env` and export it. `TWTRX_NO_PUSH=1` runs everything without pushing anywhere.
Routing uses `OSRM_URL` (public OSRM by default). `pipeline/gazetteer.py` looks for the TGSRTC, MTC and BMTC map
data in `ref/` or in sibling checkouts; point it at your copies.

## nitter/: self-hosted Nitter for live X timelines

`setup.sh` installs Redis and Nitter as rootless podman containers listening on 127.0.0.1 only, and adds cron lines:
`watchdog.sh` every 5 min, `poll.py` every 2 min. `poll.py` writes the latest page (about 20 posts) of each account
to `~/nitter/out/latest.json`, plus rotating searches over groups of accounts (`groups.json`: Bengaluru's police
stations, 7 accounts per query because Nitter refuses queries over about 200 characters). `poll_basic.py` is the
first version (4 accounts, no groups). Nitter needs `sessions.jsonl` with the session of a spare X account, made
with Nitter's own `tools/create_session_*.py`. It is never in git and not included here.

## delhi_dtp/: Delhi Traffic Police site and @dtptraffic

`dtp_closures.py` polls traffic.delhipolice.gov.in (home ticker + RSS of the last 10 posts) hourly with retries,
because the site often returns 503. It keeps every post ever seen, parses stated periods, places posts with a
gazetteer of OSM roads, landmarks and DTC stops, and checks route-dodging detours against the closures. `--x` merges
the @dtptraffic posts forwarded by `x_live.py`.

## What it yields (see `../data/traffic/`)

* `posts/x_official_posts.jsonl`: 4,272 posts from 41 official police accounts. @CYBTRAFFIC runs from 2020 to Nov 2026 and
  @HYDTP from Sep 2024 to Nov 2026; Delhi, Chennai and Bengaluru come from Nitter (Oct–Nov 2026, Chennai from Jul 2026).
* `events/`: geocoded events (Hyderabad/Cyberabad 2,973; Chennai, Bengaluru and Delhi site items), the ETA incident
  feed, and BTP's typical congestion by hour of week.
* `btp_powerbi/`: 299,932 BTP congestion alerts (1 Aug – 7 Oct 2026) and 163,358 corridor travel times.
* `priors/hotspots.csv`: counts per place × type × hour.

None of these sources had a free, complete API. The scrapers stay within what each site serves publicly; see the
limits section of `docs/TRAFFIC_SOURCES.md` before running them.
