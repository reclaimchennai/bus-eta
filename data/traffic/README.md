# data/traffic: traffic police alerts (official accounts only)

Built by `traffic/build_traffic_data.py` (`TWTRX_DIR=<scrapers' working folder> python3 traffic/build_traffic_data.py`)
from the scrapers' local store. Sources and methods: [docs/TRAFFIC_SOURCES.md](../../docs/TRAFFIC_SOURCES.md).

| File | Rows | Content |
|---|---|---|
| `posts/x_official_posts.jsonl` | 4,272 | `id, account, agency, t` (UTC epoch), `time_utc, text, url, via` (sorsa / nitter_rss / nitter_search). Official police accounts only; retweets, replies and quoted posts dropped; phone numbers → `[phone]`. |
| `posts/htp_echallan_alerts.jsonl` | 3 | Hyderabad Traffic Police e-challan live alerts: `first_seen, message, by`. |
| `events/hyd_cyb_events.jsonl` | 2,973 | Geocoded, typed events (Hyderabad + Cyberabad): type, severity, speed multiplier, start/end, place, road-following line, TGSRTC stop ids and routes. |
| `events/eta_incidents_hyd.jsonl` | 2,201 | The same events in the ETA incident contract (point + line, unsnapped, UTC). |
| `events/chennai_events.json`, `bengaluru_events.json` | | Chennai (MTC stop ids) and Bengaluru (BMTC stop ids) road events. |
| `events/delhi_dtp_site_items.json` | 75 | Delhi Traffic Police website alerts/advisories: title, text, kind, pub, first/last seen, link. |
| `events/bengaluru_congestion_typical.json` | 558 cells | BTP congestion by hour of week per grid cell. |
| `btp_powerbi/` | 299,932 + 163,358 | BTP Power BI dump: congestion alerts (IST wall time), corridor travel times, corridors, police stations. |
| `priors/hotspots.csv` | | Counts per place × type × hour of day (IST). |
| `SUMMARY.json` | | Per-account counts and date ranges. |
