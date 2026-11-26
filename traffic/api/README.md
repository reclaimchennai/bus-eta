# traffic/api: the traffic API (police alerts, bus speeds and congestion, city by city)

Public, read-only API at **https://api.reclaim.city/traffic/v1/** (docs: https://api.reclaim.city/docs/#traffic and [`../../docs/TRAFFIC_API.md`](../../docs/TRAFFIC_API.md),
machine-readable: `/traffic/v1/openapi.json`). Set up 26 Nov 2026.

## What it serves

| City | Police alerts (`/alerts`, `/alerts/history`) | Bus speeds (`/speeds`, `/speeds/city`) | Official congestion places (`/hotspots`) |
|---|---|---|---|
| hyderabad | `tgsrtc/live/incidents.json` (`../twtrx`: HTP + Cyberabad); history `traffic/history/hyderabad.json` | `tgsrtc/rt/speeds.json` + `tgsrtc/data/segs.json` | `tgsrtc/traffic/htp_congestion.json` (HTP's list) |
| bengaluru | `bengaluru/live/incidents.json` (`../twtrx/pipeline/bengaluru.py`); history `traffic/history/bengaluru.json` | `bengaluru/live/speeds.json` + `data/seg_geo.json` | `bengaluru/live/btp_congestion.json` (`../twtrx/pipeline/btp_congestion.py` from BTP's Power BI) |
| chennai | `mapsv2/live/incidents.json` (`../twtrx/pipeline/chennai.py`); history `traffic/history/chennai.json` | `mapsv2/live/speeds.json` + `data/seg_geo.json` | — |
| delhi | `delhi/route-dodging/data/closures.json` (`../delhi_dtp/dtp_closures.py`) | `delhi/live/speeds.json` + `data/seg_geo.json` | — |
| mumbai | — | `mumbai/live/speeds.json` + `data/seg_geo.json` | — |
| andhra-pradesh, chandigarh, ahmedabad | — | `apsrtc/`, `tricity/`, `amts/` `rt/speeds.json` + `data/segs.json` | — |

All paths are under https://maps.reclaim.city/. The speeds files are the maps' own "observed traffic" data
(`live_speeds.py` for the city maps, `rtc_speeds.py` for the RTC maps): per stop-to-stop road `[now, buses, h0..h23]`
in km/h, -1 where unknown. History files are pushed hourly by `../twtrx/pipeline/publish_history.py` (only when changed).

Not included (yet): Odisha (OSRTC has speeds but no road shapes), Gurugram and Haryana (no speeds file), Chennai's
schedule-based corridor speeds (a prototype from MTC timetables, never deployed).

## How it works

`api.py`, stdlib Python, one process. Each source file is read from disk when the maps' web root is mounted and has it
(mounted read-only at `/maps-site`) and otherwise fetched from its public URL (the RTC networks' files live on another server).
Live files are cached 60 s, road shapes and stop lists 6 h, history 15 min; a failed refresh keeps the last good copy
and the answer says `freshness.ok: false`. Stretch names come from the stop ids (RTC maps) or the nearest bus stop
to each end within 80 m (city maps, whose keys are hashes). Rows are built once per refresh of the speeds file.

* No access log, no stored state. The rate limit (120 requests / minute / IP) is an in-memory counter.
* Memory: about 430 MB with every city loaded, a little more with the Hyderabad history; the container has 1 GB.
* Responses are gzip-compressed when asked; `Cache-Control: public, max-age=30` (300 for hotspots, history, cities).

## Running

On the map server: `MAPS_SITE_DIR=~/projects/maps-site ./run.sh` (container `traffic-api` on the `web` network, port
5150, read-only, no capabilities). Caddy: `api.reclaim.city { handle /traffic/* { reverse_proxy traffic-api:5150 } }`, metered with
`import apimeter /geo/* /traffic/*` (the usage meter records API `traffic`, network = city, and the endpoint; counts only, no addresses).

Locally: `python3 api.py --port 8790` (fetches everything over HTTPS), then
`curl localhost:8790/traffic/v1/cities`.

## Adding a city

Add an entry to `CITIES` in `api.py`: `alerts` (`fmt` "incidents" for the TGSRTC incident format or "closures" for
Delhi's), `speeds` (`geo_fmt` "geo" for a `seg_geo.json`, "segs" for an RTC `segs.json`; `stops_fmt` "geojson" or
"rtc"), `hotspots` (`fmt` "htp" or "btp", or add a parser in `hotspots()`). Then add the row to the docs table.
