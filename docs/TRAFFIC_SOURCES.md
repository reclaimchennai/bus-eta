# Traffic alert sources and how the archives were built

Traffic police in Indian cities publish road alerts (breakdowns, accidents, spillages, tree falls, waterlogging,
processions and VIP movement, diversions, works) on X, WhatsApp channels and their own websites. None of these
has a free archive API. This page lists every source the project used, how its history and live feed were obtained,
what came out, and the limits. The code is in [`../traffic/`](../traffic/README.md) and the data in
[`../data/traffic/`](../data/traffic/).

Everything published is from **official accounts and sites**. Posts by other X users (replies, quoted posts, the
authors of reposts) are not included, and phone numbers in the post text are replaced by `[phone]`.

## Summary

| City | Source | How | Period held | Count |
|---|---|---|---|---|
| Cyberabad | @CYBTRAFFIC on X | Sorsa API history; Nitter live | 18 Jan 2020 – 25 Nov 2026 | 1,894 posts |
| Cyberabad | "Cyberabad Traffic Pulse" WhatsApp channel | WhatsApp Desktop's local DB, channel rows only (Mac) | 23–25 Nov 2026 | 10 events (same alerts as X) |
| Hyderabad | @HYDTP on X | Sorsa API history; Nitter live | 17 Sep 2024 – 25 Nov 2026 | 1,417 posts |
| Hyderabad | HTP e-challan site (live alert + advisories) | polled every 5 min | 25 Nov 2026 on | 3 alerts so far |
| Delhi | @dtptraffic on X | Nitter live + Nitter search backfill | 26 Oct – 25 Nov 2026 | 144 posts |
| Delhi | traffic.delhipolice.gov.in (ticker + RSS) | polled hourly with retries | Oct – Nov 2026 | 75 items |
| Chennai | @ChennaiTraffic on X | Nitter live + Nitter search backfill | 18 Jul – 25 Nov 2026 | 43 posts |
| Bengaluru | @blrcitytraffic + 66 DCP/ACP/station accounts | Nitter live (rotating group searches) + search backfill | 14 Sep – 25 Nov 2026 | 70 + 704 posts (36 more accounts posted) |
| Bengaluru | BTP congestion dashboard (Power BI, published to web) | the report's own query calls | alerts 1 Aug – 7 Oct 2026; corridors Sep 2023 – Jul 2025 | 299,932 alerts, 163,358 travel times |

Geocoded events: 2,973 for Hyderabad/Cyberabad (`events/hyd_cyb_events.jsonl`; 2,201 of them in the ETA incident
format, `events/eta_incidents_hyd.jsonl`), plus the Chennai, Bengaluru and Delhi event files.

## X history: the Sorsa API

[Sorsa](https://api.sorsa.io) resells X search (`POST /v3/search-tweets`, 20 posts per request, `next_cursor`
pagination, `X-Requests-Available` header). Its free tier is a one-time allowance of 100 requests, so the scrapers
are built to make every request return 20 relevant posts and never fetch a page twice.

**@CYBTRAFFIC** (`traffic/twtrx/scrape.py`, 78 pages + `probe.py`, 21 pages):
* one server-side keyword query, so every page is road posts:
  `from:CYBTRAFFIC ("traffic movement" OR "traffic advisory" OR "traffic alert" OR diversion OR diversions OR diverted OR
  accident OR crash OR collision OR spillage OR breakdown OR congestion OR waterlogging OR "road closure" OR overturned
  OR uprooted OR "traffic update" OR "road closed" OR "fire incident") -filter:replies`, `order: latest`;
* every response saved to disk and the cursor kept in a state file, so a run can stop at any point and resume;
* long cursor chains on X search **skip posts**, so the gaps that showed up in the monthly counts were re-queried with
  `since:`/`until:` windows (2023-01-01..2024-04-18, 2025-02-01..2025-08-01, 2026-04-15..2026-10-01).

**@HYDTP** (`traffic/twtrx/scrape_x.py`, 96 requests, 3 left on the key at the end):
* Keyword `OR` queries longer than about 330 characters only reached about **90 days** back. Older posts came from
  the short hashtag query `XTAG=HYDTPinfo` → `from:HYDTP #HYDTPinfo -filter:replies` (HTP tags its alerts
  `#HYDTPinfo`), and the relevance filter then dropped the non-alerts.
* `back` always re-anchors on the oldest post actually held (`until:<that day>`) instead of paging deep on one cursor;
  `new` stops at the first id already held; each run has a hard budget and stops on 429 or at 3 requests left.

**Gaps:** X search returns thin stretches. Months held per account:

```
CYBTRAFFIC  2020-01 … 2023-03: 2 posts        2023-04 … 2024-05: 1-12 a month      2024-06 … 2024-11: 38-62
            2024-12 … 2025-06: 4-35           2025-07 … 2025-11: 28-256             2025-12 … 2026-01: 13-18
            2026-02 … 2026-03: 125-202        2026-04 … 2026-07: 1-35               2026-08 … 2026-11: 45-131
HYDTP       2024-09 … 2024-11: 9-49           2024-12 … 2025-06: 0-5                2025-07 … 2025-09: 42-146
            2025-10 … 2026-07: 0-2 (gap)      2026-08 … 2026-11: 220-292
```

HYDTP's **Sep 2025 – Aug 2026** stretch is nearly empty because X search did not return those posts to the queries we
could afford, not because HTP stopped posting. `scrape_x.py HYDTP window SINCE UNTIL N` fills it with a paid key; at
30-minute polling, a paid key would cost about $2 a month.

## X live: a self-hosted Nitter

X's public embed timeline (`syndication.twitter.com/srv/timeline-profile/...`) answered 429 to about 95% of hourly
requests from the server, so the live feed comes from [Nitter](https://github.com/zedeus/nitter) running on a small
Linux box (`traffic/nitter/`):
* rootless podman, Nitter and Redis on 127.0.0.1 only, one session of a spare X account (`sessions.jsonl`, never in git);
* `poll.py` every 2 min: the RSS of each account's timeline (about 20 posts), RSS cache 1 min;
* @dtptraffic's timeline is mostly reposts of district units (20 posts = less than a day), so its own posts are also
  read with a search: `from:dtptraffic -filter:replies -filter:nativeretweets`;
* Bengaluru: 56 station/ACP/DCP accounts in 8 group searches of 7 (`(from:a OR from:b …) -filter:replies
  -filter:nativeretweets`; Nitter refuses queries over about 200 characters, and without parentheses the filters bind
  to the last `from:` only), two groups per run, so each group every 8 min;
* `pipeline/x_live.py` reads `latest.json` each minute, so a post reaches the maps in about 1.5 min (3 min at worst).
  `"ok": false` is a Nitter or session failure and is never read as "no alerts".

**Backfill through Nitter search** (`traffic/twtrx/scripts/nitter_search_pages.py`): the same searches paged 4 s apart
(Delhi) or 60 s apart (Bengaluru; at 8 s per page every search got 429 for 5 min). X search gave back to
**26 Oct 2026** for @dtptraffic (120 posts, 5 pages) and the Bengaluru groups, and 20 posts back to Jul 2026 for
@ChennaiTraffic. That is the search horizon for one session, so Nitter cannot fill older gaps.

## WhatsApp channel (Cyberabad)

The "Cyberabad Traffic Pulse" channel carries the same alerts as @CYBTRAFFIC. `pipeline/whatsapp_db.py` reads it from
WhatsApp Desktop's local `ChatStorage.sqlite` on a Mac: the DB is copied read-only to a temp folder, **only** rows of
the chat whose name is the channel are read, and the copy is deleted. It needs WhatsApp Desktop linked to a phone.
Posts can also be pasted or exported (`pipeline/ingest.py`). Only 10 channel posts were ingested before the live
pipeline moved to the Linux box, where X via Nitter covers the same alerts.

## Hyderabad Traffic Police e-challan site

`pipeline/hydtp_web.py` polls the site's live alert JSON every 5 min with an honest User-Agent. The alert's start is
when it is first seen; when it changes or disappears, the previous one gets `t_end` = that moment, which gives
observed durations that X posts lack. The "Latest News" list is fetched every 30 min with ETag/If-Modified-Since, and
each new advisory PDF or image is downloaded once. Multi-diversion PDFs are curated by hand, and one is a scan.

## Delhi Traffic Police site

`traffic/delhi_dtp/dtp_closures.py` reads traffic.delhipolice.gov.in's home-page ticker and its RSS of the last ten
posts hourly. There is no archive, so every post ever seen is kept. The site often answers 503 (no healthy backend),
so each fetch is retried up to six times with growing pauses and logged.

## Bengaluru Traffic Police congestion dashboard

The congestion map on btp.karnataka.gov.in/188/congestion-map is a Power BI report published to the web. Its public
embed key lets `pipeline/btp_powerbi.py` make the same calls the embedded page makes (`modelsAndExploration`,
`conceptualschema`, `querydata` in windows of 30,000 rows). The model is a static import, last refreshed 7 Oct 2026
02:58 UTC with refresh turned off:
* 299,932 congestion alerts (ASTraM's 15-minute city-wide checks, 1 Aug – 7 Oct 2026): position, queue length,
  severity, junction, police station, corridor, grid cell;
* 163,358 corridor travel times (23 corridors, Sep 2023 – Jul 2025).
`pipeline/btp_congestion.py` turns them into an hour-of-week profile per grid cell, per BMTC stop (200 m) and per
corridor. That profile was tested as an ETA feature; see [TRAFFIC_INTEGRATION.md](TRAFFIC_INTEGRATION.md).

## Finding the accounts

Bengaluru's 67 accounts (`traffic/twtrx/data/blr/btp_accounts.md`) were found from the accounts @blrcitytraffic
reposts and mentions, likely spellings of station handles (one Nitter profile request each) and Nitter's account
search. They are checked against BTP's own list of 52 stations. Only accounts of police units are kept; one
station account that could not be tied to the police (few followers, unverified) is left out of the published posts.

## Limits and terms

* **Coverage, not completeness.** X search, both through Sorsa and through Nitter, returns a sample that thins out
  with age. Counts per month say more about the search than about how often the police posted.
* **Times.** A post's time is when it was posted, not when the incident started. Ends are rarely stated: the
  pipeline uses stated windows, "cleared" follow-ups, the e-challan alert's disappearance, or a per-type default
  (accident 60 min, breakdown 45, spillage 60, fire 90, tree fall 90, waterlogging 180, works 480, event 240,
  closure 240 or as stated, congestion 45).
* **Terms of use.** Sorsa is a commercial reseller of X data and was used within its free allowance. Nitter reads X
  with a logged-in session, which X's terms do not provide for. The Power BI and police websites are public pages
  read at a low rate with an identifying User-Agent. Anyone re-running these scrapers should check each service's
  current terms. The published posts are official public-service announcements, kept with their X links so they can be
  checked against the source.
* **Personal data.** Only official police accounts and channels are kept. Replies, quoted posts and reposted
  authors are dropped, and phone numbers in the text are redacted (most are official helplines). Officers named in
  official posts are left as posted.
