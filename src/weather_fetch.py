#!/usr/bin/env python3
"""Fetch and unify historical weather + rainfall + AQI data for all target Indian cities.

Sources:
1. Hyper-local Chennai station network: https://pol.reclaimchennai.city/api/{stations,history}
   (GCC rainfall, KWS Chennai 5-min rain_accum_mm, AQI.in temp/humidity/wind/vis_km/pm25/aqi)
2. Open-Meteo Historical Weather API (free, no key) for Delhi, Chennai, Bengaluru, Mumbai, Gurugram/NCR
   to guarantee continuous gap-free hourly weather across all 5 cities.

Outputs (in ETA/data/weather/):
- chennai_stations.json          : station metadata (uid, lat, lon, source)
- chennai_pol_hourly.parquet     : per-station hourly readings from pol.reclaimchennai.city
- weather_city_hourly.parquet    : unified city-level hourly weather + 3h/6h/24h rolling rainfall (waterlogging lag)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

POL_BASE = "https://pol.reclaimchennai.city/api"
POL_METRICS = [
    "precip_mm",
    "rain_accum_mm",
    "temp_c",
    "humidity",
    "wind_kph",
    "vis_km",
    "pm25",
    "aqi",
]

CITIES = {
    "mtc": ("Chennai", 13.0827, 80.2707),
    "delhi": ("Delhi", 28.6139, 77.2090),
    "bmtc": ("Bengaluru", 12.9716, 77.5946),
    "mumbai": ("Mumbai", 19.0760, 72.8777),
    "ncr": ("Gurugram", 28.4595, 77.0266),
}

OPEN_METEO_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
]


def fetch_pol_chennai(out_dir: Path, hours: int = 960) -> pl.DataFrame | None:
    """Fetch stations + historical hourly readings from pol.reclaimchennai.city."""
    sess = requests.Session()
    sess.headers["User-Agent"] = "ReclaimCity-ETA-Research/1.0"

    print(f"[pol] Fetching stations from {POL_BASE}/stations ...")
    r = sess.get(f"{POL_BASE}/stations", timeout=30)
    r.raise_for_status()
    stations_json = r.json()
    stations = stations_json.get("stations", [])
    (out_dir / "chennai_stations.json").write_text(json.dumps(stations, indent=2))
    print(f"[pol] Saved {len(stations)} stations.")

    station_meta = {
        s["uid"]: (s.get("latitude"), s.get("longitude"), s.get("source", ""))
        for s in stations
        if s.get("uid")
    }

    frames = []
    for metric in POL_METRICS:
        url = f"{POL_BASE}/history?uid=all&metric={metric}&hours={hours}"
        print(f"[pol] Fetching {metric} ({hours}h) ...", end=" ", flush=True)
        try:
            resp = sess.get(url, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"WARN: failed ({exc})")
            continue

        rows = []
        for s in payload.get("series", []):
            uid = s.get("uid")
            lat, lon, src = station_meta.get(uid, (None, None, s.get("source", "")))
            for ts_str, val in s.get("points", []):
                if val is None:
                    continue
                try:
                    t_epoch = int(dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
                    hour_epoch = (t_epoch // 3600) * 3600
                    rows.append((uid, lat, lon, src, hour_epoch, float(val)))
                except Exception:
                    continue

        if not rows:
            print("0 rows")
            continue

        df_m = pl.DataFrame(
            rows,
            schema={
                "uid": pl.String,
                "lat": pl.Float64,
                "lon": pl.Float64,
                "source": pl.String,
                "hour_ts": pl.Int64,
                metric: pl.Float64,
            },
            orient="row",
        )
        # Aggregate multiple readings within the same station-hour
        agg_expr = pl.col(metric).max() if metric in ("precip_mm", "rain_accum_mm") else pl.col(metric).mean()
        df_m = df_m.group_by(["uid", "lat", "lon", "source", "hour_ts"]).agg(agg_expr)
        print(f"{len(df_m):,} station-hours")
        frames.append((metric, df_m))
        time.sleep(0.2)

    if not frames:
        return None

    # Outer-join all metrics on (uid, hour_ts)
    base = frames[0][1]
    for metric, df_m in frames[1:]:
        base = base.join(
            df_m.select(["uid", "hour_ts", metric]),
            on=["uid", "hour_ts"],
            how="full",
            coalesce=True,
        )
        # Fill lat/lon/source from station_meta if introduced by right side
    meta_df = pl.DataFrame(
        [(u, float(v[0]) if v[0] is not None else None, float(v[1]) if v[1] is not None else None, str(v[2] or "")) for u, v in station_meta.items()],
        schema={"uid": pl.String, "lat_m": pl.Float64, "lon_m": pl.Float64, "src_m": pl.String},
        orient="row",
    )
    base = (
        base.join(meta_df, on="uid", how="left")
        .with_columns(
            pl.coalesce(["lat", "lat_m"]).alias("lat"),
            pl.coalesce(["lon", "lon_m"]).alias("lon"),
            pl.coalesce(["source", "src_m"]).alias("source"),
        )
        .drop(["lat_m", "lon_m", "src_m"])
        .sort(["uid", "hour_ts"])
    )
    out_path = out_dir / "chennai_pol_hourly.parquet"
    base.write_parquet(out_path)
    print(f"[pol] Wrote {len(base):,} rows to {out_path}")
    return base


def fetch_open_meteo_city(city_key: str, lat: float, lon: float, start_date: str, end_date: str) -> pl.DataFrame:
    """Fetch hourly weather from Open-Meteo Archive + Forecast (for recent days)."""
    frames = []
    # Archive API covers up to ~5 days ago; forecast API with past_days=10 covers recent days
    urls = [
        (
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(OPEN_METEO_VARS),
                "timezone": "UTC",
            },
        ),
        (
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "past_days": 14,
                "forecast_days": 1,
                "hourly": ",".join(OPEN_METEO_VARS),
                "timezone": "UTC",
            },
        ),
    ]
    for url, params in urls:
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                continue
            h = r.json().get("hourly", {})
            times = h.get("time", [])
            if not times:
                continue
            epochs = [int(dt.datetime.fromisoformat(t + "+00:00").timestamp()) for t in times]
            data = {"city": [city_key] * len(epochs), "hour_ts": epochs}
            for var in OPEN_METEO_VARS:
                vals = h.get(var, [None] * len(epochs))
                data[var] = [float(v) if v is not None else None for v in vals]
            frames.append(pl.DataFrame(data))
        except Exception as exc:
            print(f"[open-meteo:{city_key}] WARN {url}: {exc}")

    if not frames:
        return pl.DataFrame()

    df = pl.concat(frames, how="vertical_relaxed")
    # Deduplicate by hour_ts, preferring non-null values (latest forecast/archive)
    df = (
        df.group_by(["city", "hour_ts"])
        .agg([pl.col(v).drop_nulls().last().alias(v) for v in OPEN_METEO_VARS])
        .sort("hour_ts")
    )
    return df


def build_unified_weather(out_dir: Path, start_date: str, end_date: str, pol_df: pl.DataFrame | None) -> None:
    """Combine Open-Meteo (all 5 cities) + pol.reclaimchennai.city (Chennai hyper-local aggregate)."""
    city_frames = []
    for city_key, (name, lat, lon) in CITIES.items():
        print(f"[open-meteo] Fetching {name} ({city_key}) {start_date}..{end_date} ...", end=" ", flush=True)
        df_c = fetch_open_meteo_city(city_key, lat, lon, start_date, end_date)
        print(f"{len(df_c):,} hours")
        if len(df_c) > 0:
            city_frames.append(df_c)

    df = pl.concat(city_frames, how="vertical_relaxed")
    df = df.rename(
        {
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "humidity",
            "precipitation": "precip_mm",
            "rain": "rain_mm",
            "wind_speed_10m": "wind_kph",
            "wind_gusts_10m": "gust_kph",
        }
    )

    # Merge Chennai pol.reclaimchennai.city city-wide aggregates (mean + p90 precip across Chennai stations!)
    if pol_df is not None and len(pol_df) > 0:
        agg_exprs = []
        if "precip_mm" in pol_df.columns:
            agg_exprs.extend([
                pl.col("precip_mm").mean().alias("pol_precip_mean"),
                pl.col("precip_mm").max().alias("pol_precip_max"),
            ])
        if "rain_accum_mm" in pol_df.columns:
            agg_exprs.append(pl.col("rain_accum_mm").mean().alias("pol_rain_accum_mean"))
        if "vis_km" in pol_df.columns:
            agg_exprs.append(pl.col("vis_km").mean().alias("vis_km"))
        if "pm25" in pol_df.columns:
            agg_exprs.append(pl.col("pm25").mean().alias("pm25"))
        if "aqi" in pol_df.columns:
            agg_exprs.append(pl.col("aqi").mean().alias("aqi"))
        if "temp_c" in pol_df.columns:
            agg_exprs.append(pl.col("temp_c").mean().alias("pol_temp_c"))
        if "humidity" in pol_df.columns:
            agg_exprs.append(pl.col("humidity").mean().alias("pol_humidity"))

        pol_city = (
            pol_df.group_by("hour_ts")
            .agg(agg_exprs)
            .with_columns(pl.lit("mtc").alias("city"))
        )
        df = df.join(pol_city, on=["city", "hour_ts"], how="left")
        # For Chennai, prefer local GCC/KWS rainfall when available and higher resolution
        if "pol_precip_mean" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("pol_precip_mean").is_not_null())
                .then(pl.max_horizontal("precip_mm", "pol_precip_mean"))
                .otherwise(pl.col("precip_mm"))
                .alias("precip_mm"),
                pl.coalesce(["pol_temp_c", "temp_c"]).alias("temp_c"),
                pl.coalesce(["pol_humidity", "humidity"]).alias("humidity"),
            )

    # Add rolling rainfall accumulation (3h, 6h, 24h) per city — critical for Indian monsoon waterlogging lag!
    df = df.sort(["city", "hour_ts"]).with_columns(
        pl.col("precip_mm").fill_null(0.0).alias("precip_mm"),
        pl.col("rain_mm").fill_null(0.0).alias("rain_mm"),
    )
    df = df.with_columns(
        pl.col("precip_mm").rolling_sum(window_size=3, min_samples=1).over("city").alias("precip_3h_mm"),
        pl.col("precip_mm").rolling_sum(window_size=6, min_samples=1).over("city").alias("precip_6h_mm"),
        pl.col("precip_mm").rolling_sum(window_size=24, min_samples=1).over("city").alias("precip_24h_mm"),
        (pl.col("precip_mm") > 0.2).cast(pl.Int8).alias("is_raining"),
        (pl.col("precip_mm") >= 5.0).cast(pl.Int8).alias("is_heavy_rain"),
    )

    out_path = out_dir / "weather_city_hourly.parquet"
    df.write_parquet(out_path)
    print(f"[weather] Wrote unified {len(df):,} city-hours to {out_path}")
    print(df.group_by("city").agg([
        pl.len().alias("hours"),
        pl.col("precip_mm").max().alias("max_precip_mm"),
        pl.col("is_raining").sum().alias("rain_hours"),
    ]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="~/projects/ETA/data/weather")
    ap.add_argument("--start", default="2026-10-18")
    ap.add_argument("--end", default="2026-11-24")
    ap.add_argument("--pol-hours", type=int, default=960)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pol_cache = out_dir / "chennai_pol_hourly.parquet"
    if pol_cache.exists():
        print(f"[pol] Reusing cached {pol_cache}")
        pol_df = pl.read_parquet(pol_cache)
    else:
        pol_df = fetch_pol_chennai(out_dir, hours=a.pol_hours)
    build_unified_weather(out_dir, a.start, a.end, pol_df)


if __name__ == "__main__":
    main()
