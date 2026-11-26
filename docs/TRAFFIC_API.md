<!-- Mirrors the Traffic section of https://api.reclaim.city/docs/#traffic. Machine-readable: ../traffic/api/openapi.json. Code: ../traffic/api/. -->

# Traffic: police alerts, bus speeds, congestion

One API for road conditions in Indian cities, city by city. It brings together three kinds of data, and cities have
different amounts of each:

- **Police alerts.** Breakdowns, crashes, diversions, closures, waterlogging, events and processions, taken from each traffic police force's own posts and site. Each alert is placed on the road and lists the bus routes and stops it touches.
- **Observed bus speeds.** Our own measurement from the live GPS of the city's buses. It gives the speed on every stop-to-stop road in the last hour, and the typical speed at each hour over the last 7 days, with stops and signals included: what a bus rider gets.
- **Official congestion places.** The spots a police force itself names as congested. Where its data allows, it also gives how often each spot is jammed at each hour of the week.

| City (`{city}`) | Police alerts | Bus speeds | Official congestion places |
|---|---|---|---|
| Hyderabad `hyderabad` | Hyderabad and Cyberabad Traffic Police: X, the e-challan site; history since 2020 | TGSRTC buses (Telangana-wide) | Hyderabad Traffic Police's list of major congested areas |
| Bengaluru `bengaluru` | Bengaluru Traffic Police: @blrcitytraffic and 56 station, ACP and DCP accounts | BMTC buses | BTP's congestion dashboard: 558 places, with the share of checks jammed at every hour of the week |
| Chennai `chennai` | Greater Chennai Traffic Police (@ChennaiTraffic) | MTC buses | — |
| Delhi `delhi` | Delhi Traffic Police: advisories site and @dtptraffic | DTC and cluster buses | — |
| Mumbai region `mumbai` | — | BEST, TMT, KDMT, VVMT buses | — |
| Andhra Pradesh `andhra-pradesh` | — | APSRTC buses (state-wide) | — |
| Chandigarh tricity `chandigarh` | — | CTU buses | — |
| Ahmedabad `ahmedabad` | — | AMTS and Janmarg BRTS buses | — |

Everything is GeoJSON or JSON with `[longitude, latitude]` coordinates. Times are Unix seconds; hours of the day
are IST. Every response has a `freshness` block: `source_updated` is when the underlying data last
changed, and `ok: false` means the last refresh failed and you are getting the last good copy. The machine-readable
description is at [/traffic/v1/openapi.json](https://api.reclaim.city/traffic/v1/openapi.json).

## Start here

**`GET https://api.reclaim.city/traffic/v1/cities`**

Every city and which of the three datasets it has.

**`GET https://api.reclaim.city/traffic/v1/{city}`**

One city: active alert count, the city-wide median bus speed now and by hour, how many roads are measured, how many
official places there are, and how fresh each source is.

**`GET https://api.reclaim.city/traffic/v1/near`**

Everything around a point in one call: active alerts, official congestion places and measured bus roads within the
radius, nearest first. The city is found from the point.

| Param | Type | Notes |
|---|---|---|
| `lat`, `lon` | float | Required |
| `radius` | m | 10–5000, default 500 |
| `city` | string | Optional: force a city |

```
curl "https://api.reclaim.city/traffic/v1/near?lat=12.9172&lon=77.6229&radius=400"
```

```
{
  "city": "bengaluru", "lat": 12.9172, "lon": 77.6229, "radius_m": 400,
  "alerts": [],
  "hotspots": [ { "type": "Feature", "geometry": { "type": "Point", "coordinates": [77.6234, 12.91717] },
      "properties": { "name": "Silk Board Junction", "police_station": "Madiwala", "corridor": "ORR East 1",
                      "jam_share_pct": 14, "peak_jam_share_pct": 97, "peak_hour_of_week": 33,
                      "typical_queue_m": 454, "distance_m": 54, ... } } ],
  "speeds": [ { "type": "Feature", "geometry": { "type": "LineString", "coordinates": [...] },
      "properties": { "from_stop": "Roopena Agrahara", "to_stop": "Central Silk Board", "length_m": 849,
                      "speed_kmph_now": 12.2, "buses_last_hour": 10, "speed_kmph_typical": 11.8,
                      "free_flow_kmph": 18.4, "congestion": 0.34, "distance_m": 11, ... } } ],
  "city_median_kmph_now": 16.5
}
```

## Police alerts

**`GET https://api.reclaim.city/traffic/v1/{city}/alerts`**

Alerts in force now, newest first. Each is one GeoJSON feature: a point, or a `GeometryCollection` of the
point and the road stretch it covers (a diversion, closure or procession route).

| Param | Type | Notes |
|---|---|---|
| `state` | string | `active` (default), `upcoming` (announced, not started) or `all` |
| `type` | string | Comma-separated: `congestion`, `breakdown`, `accident`, `diversion`, `closure`, `works`, `waterlogging`, `tree_fall`, `event`, `u_turn_change`, `other`, … |
| `route` | string | Only alerts on this bus route number (the city map's numbers) |
| `bbox` | string | `min_lon,min_lat,max_lon,max_lat` |
| `lat`, `lon`, `radius` | float, m | Within radius (default 1000 m), nearest first, with `distance_m` |
| `limit` | int | 1–1000, default 500 |

```
curl "https://api.reclaim.city/traffic/v1/hyderabad/alerts?type=diversion,closure"
```

```
{ "type": "FeatureCollection", "city": "hyderabad", "state": "active", "count": 6,
  "freshness": { "source_updated": 1795676039, "fetched": 1795676039, "ok": true },
  "features": [ { "type": "Feature", "geometry": { "type": "Point", "coordinates": [...] },
    "properties": {
      "id": "hydtp_x:2125841827635101998", "city": "hyderabad", "agency": "Hyderabad Traffic Police",
      "type": "diversion", "kind": "unplanned", "severity": 0.8, "status": "active",
      "place": "Kavadiguda", "direction": "",
      "text": "TRAFFIC ALERT Due to the Samvidhan Raksha Rally at Old Ambedkar Statue, vehicular traffic is being diverted ...",
      "start": 1795675162, "end": 1795689562, "source_url": "https://x.com/HYDTP/status/2125841827635101998",
      "route_ids": ["16A/20P", "16R/20P"], "stop_ids": [...], "tags": [], "vehicles": [] } }, ... ] }
```

`kind` is `unplanned` (a breakdown, a crash), `scheduled` (announced works, events),
`network_change` (a new U-turn or one-way) or `standing` (a long-running arrangement, in force until
ended). `end` is the time the police gave, or an expected clearing time by type when they gave none.
`severity` runs 0–1. `vehicles` lists a bus registration when the alert is about a bus.

**`GET https://api.reclaim.city/traffic/v1/{city}/alerts/history`**

Every past alert that could be placed, newest first, in pages. Hyderabad has Cyberabad and Hyderabad posts since 2020,
over 2,200 placed alerts. Bengaluru starts on 26 Oct 2026, Chennai in Jul 2026, and Delhi has the advisories still on its
closures list.

| Param | Type | Notes |
|---|---|---|
| `since`, `until` | Unix time or `YYYY-MM-DD` | By start time, IST; a date in `until` includes that day |
| `type`, `route`, `bbox`, `lat`/`lon`/`radius` |  | As above |
| `agency` | string | Part of the name, e.g. `cyberabad` |
| `offset`, `limit` | int | Paging; limit up to 5000 (default 500). The answer gives `total` and `next_offset` |

```
curl "https://api.reclaim.city/traffic/v1/hyderabad/alerts/history?since=2026-11-01&type=waterlogging&limit=50"
```

## Observed bus speeds

**`GET https://api.reclaim.city/traffic/v1/{city}/speeds`**

One GeoJSON line per stop-to-stop road. Each road has its speed in the last hour and its typical speed at the chosen
hour, both medians over the buses that drove it. It also has a free-flow speed: the 90th percentile of its hourly medians,
given when at least 6 hours are known. `congestion` is `1 − speed / free_flow`, so 0 is moving freely
and 0.9 is crawling at a tenth of its usual best.

| Param | Type | Notes |
|---|---|---|
| `when` | string | `now` (default; roads a bus crossed in the last hour) or `typical` |
| `hour` | int | 0–23 IST, for `typical` (default: the current hour) |
| `sort` | string | `slowest` or `congested` |
| `min_kmph`, `max_kmph` | float | Speed range |
| `min_buses` | int | Fewest buses in the last hour; use 3 or more for a firmer number |
| `profile` | `1` | Adds `typical_kmph_by_hour` (24 values, null where no bus ran) |
| `bbox`, `lat`/`lon`/`radius` |  | Radius up to 5000 m, default 500 |
| `limit` | int | Default 5000. A whole city is 4,500–18,500 roads, several MB, so filter with `bbox` or `sort` |

```
# the 20 most congested roads in Bengaluru right now, with at least 3 buses on them
curl "https://api.reclaim.city/traffic/v1/bengaluru/speeds?sort=congested&min_buses=3&limit=20"
```

```
{ "type": "FeatureCollection", "city": "bengaluru", "when": "now", "hour": 12, "window_days": 7, "now_window_min": 60,
  "features": [ { "type": "Feature", "geometry": { "type": "LineString", "coordinates": [...] },
    "properties": { "id": "bengaluru:9JeyjvSD|YpZkmBN8", "from_stop": "Sony World 80Ft Road koramangala",
      "to_stop": "Military Bridge", "length_m": 1764, "speed_kmph_now": 4.3, "buses_last_hour": 3,
      "speed_kmph_typical": 15.4, "typical_hour": 12, "free_flow_kmph": 34.5, "congestion": 0.88,
      "congestion_basis": "now" } }, ... ] }
```

**`GET https://api.reclaim.city/traffic/v1/{city}/speeds/city`**

The city-wide median bus speed now and for each hour of the day, with the slowest and fastest hours.

> Bus speeds describe the roads buses use, at the times they run. A road with one bus in the hour is one observation. Speeds include time at stops, so they are below car speeds on the same road. The `id` is stable for a road between map updates, so it can be used to join requests over time.

## Official congestion places

**`GET https://api.reclaim.city/traffic/v1/{city}/hotspots`**

**Hyderabad:** the 21 areas on Hyderabad Traffic Police's
[list of major congested areas](https://echallan.tspolice.gov.in/htp/tcongestion.html), as points and road
lines (`kind: "listed"`, no hourly data).
**Bengaluru:** 558 places from Bengaluru Traffic Police's
[congestion dashboard](https://btp.karnataka.gov.in/188/congestion-map/kn). That dashboard records every
congested stretch in the city every 15 minutes; here it is pooled per place and hour of the week, 1 Aug – 7 Oct 2026
(`kind: "measured"`). `jam_share_pct` is the share of that hour's checks that found a jam there.

| Param | Type | Notes |
|---|---|---|
| `day` | string | `mon`…`sun` or 0–6 (Bengaluru; default today, IST) |
| `hour` | int | 0–23 IST (Bengaluru; default now) |
| `min_pct` | float | Only places jammed in at least this share of checks |
| `profile` | `1` | Adds `jam_share_pct_by_hour_of_week` (168 values, 0 = Monday 00:00 IST) |
| `bbox` | string | `min_lon,min_lat,max_lon,max_lat` |

```
curl "https://api.reclaim.city/traffic/v1/bengaluru/hotspots?day=tue&hour=18&min_pct=80"
```

```
{ "type": "FeatureCollection", "city": "bengaluru", "hour_of_week": 42, "count": 19,
  "features": [ { "type": "Feature", "geometry": { "type": "Point", "coordinates": [77.61906, 12.92691] },
    "properties": { "id": "btp:2", "name": "Ayyappa Temple Junction", "agency": "Bengaluru Traffic Police",
      "police_station": "Madiwala", "corridor": "Hosur Road", "kind": "measured", "jam_share_pct": 90,
      "peak_jam_share_pct": 100, "peak_hour_of_week": 166, "typical_queue_m": 338, "severe_or_high_pct": 19,
      "recurring_pct": 97, "alerts": 4382 } } ] }
```

For cities without an official list, `/{city}/speeds?sort=congested` gives the roads where buses are
slowest against their own usual speed.

> The alerts are read from what each police force publishes and placed on the map by our own matching, so a place can be wrong when a post is vague, and alerts with no clear place are left out. Bus speeds are our own measurement from the operators' public tracking. Each source keeps its owner's terms. Alerts refresh within a few minutes of the police posting; bus speeds every few minutes; official places when the police update their lists. Answers are cached for 30 s (5 min for places and history).
