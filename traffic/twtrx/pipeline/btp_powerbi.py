"""Bengaluru Traffic Police congestion dashboard (Power BI, published to web) -> data/blr/powerbi/.

  python -m pipeline.btp_powerbi              # dump every table (skips tables already dumped for this model refresh)
  python -m pipeline.btp_powerbi --force      # dump again

The report is embedded on btp.karnataka.gov.in/188/congestion-map ("Congestion Maps": congestion alerts with place,
severity, length, police station, zone and corridor; corridor travel time and speed by hour). Its public key gives
the model through the same calls the embedded page makes (modelsAndExploration, conceptualschema, querydata); rows
come in windows of WINDOW with restart tokens. The model is an import, not a live feed: lastRefreshTime is kept in
meta.json, and a dump is only redone when it changes.

Output (data/blr/powerbi/):
  meta.json                     report ids, model refresh time, row counts, when dumped
  congestion_alerts.csv.gz      one row per alert (timestamp is IST wall time as the model stores it)
  travel_time.csv.gz            corridor x direction x time: distance, duration, speed
  corridors.csv                 corridor lookup (name, code, length, start/end point)
  police_stations.csv           traffic police station -> sub-division, zone
Plain requests, no browser headers: the server sometimes refuses a connection, so each call retries with backoff.
"""
import os
import argparse, csv, gzip, io, json, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/blr/powerbi"
KEY = os.environ.get("BTP_POWERBI_KEY", "")   # the publish-to-web resource key in the iframe on btp.karnataka.gov.in/188/congestion-map (the "k" in the embed URL)
CLUSTER = "https://wabi-india-central-a-primary-api.analysis.windows.net"
WINDOW = 30000

TABLES = {   # file: (entity, columns, order-by column)
    "congestion_alerts": ("Congestion Alerts",
        ["id", "timestamp", "latitude", "longitude", "extent", "severity", "recurring", "incidents", "road", "area",
         "junction_name", "address", "traffic_police_station", "ps_sub_division", "ps_zone", "corridor", "ward",
         "event_area", "grid_id", "osm_id", "link"], "timestamp"),
    "travel_time": ("Travel Time",
        ["Corridor Name", "direction", "Date Time", "distance_km", "duration_hr", "Speed_Kmph"], "Date Time"),
    "corridors": ("Corridor LU",
        ["SI.No", "City", "Corridor Name", "Corridor Code", "Corridor Length", "Column2", "Start Point", "End Point"], None),
    "police_stations": ("Final_MasterPS_LU",
        ["Id", "Traffic PS Names", "Traffic PS Names2", "Sub-Division Case Corrected", "Zone Case Corrected"], None),
}


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def call(path, body=None, tries=6):
    url = f"{CLUSTER}/public/reports/{path}"
    data = json.dumps(body).encode() if body is not None else None
    for i in range(tries):
        try:
            r = urllib.request.Request(url, data=data, headers={"X-PowerBI-ResourceKey": KEY, "Content-Type": "application/json",
                                                                "Accept-Encoding": "gzip"})
            b = urllib.request.urlopen(r, timeout=300).read()
            return json.loads(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                raise
            wait = 5 * 2 ** i
            log(f"{path.split('?')[0]}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{path}: gave up after {tries} tries")


def model():
    j = call(f"{KEY}/modelsAndExploration?preferReadOnlySession=true")
    m = j["models"][0]
    return {"model_id": m["id"], "dataset": m["dbName"], "report": j["exploration"]["report"]["objectId"],
            "name": m.get("displayName"), "last_refresh_utc": m.get("LastRefreshTime")}


def query(meta, entity, cols, order=None, restart=None):
    sel = [{"Column": {"Expression": {"SourceRef": {"Source": "t"}}, "Property": c}, "Name": f"t.{c}"} for c in cols]
    q = {"Version": 2, "From": [{"Name": "t", "Entity": entity, "Type": 0}], "Select": sel}
    if order:
        q["OrderBy"] = [{"Direction": 1, "Expression": {"Column": {"Expression": {"SourceRef": {"Source": "t"}}, "Property": order}}}]
    win = {"Count": WINDOW, **({"RestartTokens": restart} if restart else {})}
    cmd = {"SemanticQueryDataShapeCommand": {"Query": q, "Binding": {
        "Primary": {"Groupings": [{"Projections": list(range(len(cols)))}]},
        "DataReduction": {"DataVolume": 6, "Primary": {"Window": win}}, "Version": 1}, "ExecutionMetricsKind": 1}}
    body = {"version": "1.0.0", "modelId": meta["model_id"], "cancelQueries": [],
            "queries": [{"Query": {"Commands": [cmd]}, "QueryId": "",
                         "ApplicationContext": {"DatasetId": meta["dataset"], "Sources": [{"ReportId": meta["report"]}]}}]}
    res = call("querydata?synchronous=true", body)["results"][0]["result"]
    if "dsr" not in res.get("data", {}):
        raise RuntimeError(json.dumps(res)[:500])
    return decode(res["data"]["dsr"], len(cols))


def decode(dsr, n):
    """Power BI DSR -> (rows, restart tokens). Row 'C' holds the values not repeated (bit set in 'R': same as the
    row before) and not null (bit set in 'Ø'); a column with 'DN' stores indexes into ValueDicts[DN]."""
    ds = dsr["DS"][0]
    if "PH" not in ds:
        raise RuntimeError(json.dumps(dsr)[:500])
    dicts = ds.get("ValueDicts", {})
    rows, prev, schema = [], [None] * n, None
    for ph in ds["PH"]:
        for r in ph.get("DM0", []):
            if "S" in r:
                schema = r["S"]
            rep, nul = r.get("R", 0), r.get("Ø", 0)
            vals = iter(r.get("C", []))
            row = []
            for i in range(n):
                if rep >> i & 1:
                    v = prev[i]
                elif nul >> i & 1:
                    v = None
                else:
                    v = next(vals)
                    dn = schema[i].get("DN") if schema and i < len(schema) else None
                    if dn is not None and isinstance(v, int):
                        v = dicts[dn][v]
                row.append(v)
            rows.append(row); prev = row
    return rows, ds.get("RT")


def fmt(v, t):
    if v is None:
        return ""
    if t == 7 and isinstance(v, (int, float)):      # datetime: ms, model wall time
        return datetime.fromtimestamp(v / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return v


def dump(meta, name, entity, cols, order):
    rows, rt, page = [], None, 0
    while True:
        got, rt2 = query(meta, entity, cols, order, rt)
        rows += got; page += 1
        log(f"{name}: page {page}, +{len(got)} = {len(rows)}")
        if not got or not rt2 or rt2 == rt or len(got) < WINDOW // 2:
            break
        rt = rt2
        time.sleep(2)
    # the window boundary row can repeat: drop exact duplicates, keep order
    seen, uniq = set(), []
    for r in rows:
        k = json.dumps(r)
        if k not in seen:
            seen.add(k); uniq.append(r)
    dt = {c: (7 if c in ("timestamp", "Date Time") else 0) for c in cols}
    path = OUT / (f"{name}.csv.gz" if len(uniq) > 2000 else f"{name}.csv")
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
    for r in uniq:
        w.writerow([fmt(v, dt[c]) for v, c in zip(r, cols)])
    data = buf.getvalue().encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data); tmp.replace(path)
    return path.name, len(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_p = OUT / "meta.json"
    old = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta = model()
    log(f"model {meta['model_id']} '{meta['name']}', refreshed {meta['last_refresh_utc']} UTC")
    if not a.force and old.get("last_refresh_utc") == meta["last_refresh_utc"] and old.get("tables") and not a.only:
        log("unchanged since the last dump"); return 0
    tables = dict(old.get("tables", {})) if a.only else {}
    for name, (entity, cols, order) in TABLES.items():
        if a.only and name not in a.only:
            continue
        f, n = dump(meta, name, entity, cols, order)
        tables[name] = {"file": f, "rows": n, "entity": entity}
    meta.update(source="https://btp.karnataka.gov.in/188/congestion-map/kn", key=KEY, cluster=CLUSTER,
                dumped=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), tables=tables)
    meta_p.write_text(json.dumps(meta, indent=1))
    log(json.dumps(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
