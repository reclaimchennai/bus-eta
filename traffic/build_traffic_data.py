"""Build the publishable traffic dataset ETA/data/traffic/ from twtrx's local store (official accounts only)."""
import csv, glob, gzip, json, os, re, shutil, time, collections
from datetime import datetime, timezone
from pathlib import Path
S = Path(os.environ.get("TWTRX_DIR", "../twtrx")).resolve()          # the scrapers' working folder (raw pages, store)
E = Path(__file__).resolve().parent.parent                            # this repo
O = E / "data/traffic"
for d in ("posts", "events", "btp_powerbi", "priors"): shutil.rmtree(O / d, ignore_errors=True); (O / d).mkdir(parents=True)
OFFICIAL = {"HYDTP": "Hyderabad Traffic Police", "CYBTRAFFIC": "Cyberabad Traffic Police",
            "dtptraffic": "Delhi Traffic Police", "ChennaiTraffic": "Greater Chennai Traffic Police"}
for r in csv.DictReader(open(S / "data/blr/btp_accounts.csv")):
    if "unverified" in (r.get("note") or ""): continue     # keep only accounts we could tie to a police unit
    OFFICIAL[r["handle"]] = "Bengaluru Traffic Police: " + (r["unit"] or r["role"])
OFF_L = {k.lower(): k for k in OFFICIAL}
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
def clean(t): return PHONE.sub("[phone]", t or "")
def iso(t): return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
posts, dropped = {}, collections.Counter()
def add(acct, pid, t, text, via):
    a = OFF_L.get((acct or "").lower())
    if not a: dropped[acct] += 1; return
    if (text or "").startswith("RT @"): dropped["retweet"] += 1; return
    k = (a, str(pid))
    if k in posts: posts[k]["via"] = sorted(set(posts[k]["via"]) | {via}); return
    posts[k] = {"id": str(pid), "account": a, "agency": OFFICIAL[a], "t": int(t), "time_utc": iso(int(t)),
                "text": clean(text), "url": f"https://x.com/{a}/status/{pid}", "via": [via]}
# Sorsa pages (CYBTRAFFIC at raw/*.json, HYDTP at raw/hydtp/*.json)
for f in sorted(glob.glob(str(S / "raw/*.json")) + glob.glob(str(S / "raw/hydtp/*.json"))):
    try: j = json.load(open(f))
    except Exception: continue
    if isinstance(j, dict) and "response" in j: j = j["response"] or {}
    for tw in (j.get("tweets") or []) if isinstance(j, dict) else []:
        if tw.get("retweeted_status"): dropped["retweet"] += 1; continue
        t = int(datetime.strptime(tw["created_at"], "%a %b %d %H:%M:%S %z %Y").timestamp())
        add((tw.get("user") or {}).get("username"), tw["id"], t, tw.get("full_text"), "sorsa")
for f, via in [("raw/nitter/posts.jsonl", "nitter_rss"), ("raw/nitter/blr_backfill.jsonl", "nitter_search"),
               ("raw/nitter/dtptraffic_backfill.jsonl", "nitter_search"), ("raw/nitter/chennaitraffic_backfill.jsonl", "nitter_search"),
               ("out/bengaluru/btp.jsonl", "nitter_rss"), ("out/delhi/dtptraffic.jsonl", "nitter_rss"), ("out/chennai/chennaitraffic.jsonl", "nitter_rss")]:
    default = {"dtptraffic": "dtptraffic", "chennaitraffic": "ChennaiTraffic"}
    for l in open(S / f):
        p = json.loads(l)
        acct = p.get("user") or p.get("acct") or next((v for k, v in default.items() if k in f), None)
        if p.get("url"): acct = re.search(r"x\.com/(\w+)/status", p["url"]).group(1)
        add(acct, p["id"], p["t"], p.get("text"), via)
P = sorted(posts.values(), key=lambda p: (p["account"], p["t"]))
with open(O / "posts/x_official_posts.jsonl", "w") as w:
    for p in P: w.write(json.dumps(p, ensure_ascii=False) + "\n")
# HTP e-challan live alert history
with open(O / "posts/htp_echallan_alerts.jsonl", "w") as w:
    for l in open(S / "raw/hydtp_echallan/alerts.jsonl"):
        a = json.loads(l); a["message"] = clean(a.get("message")); w.write(json.dumps(a, ensure_ascii=False) + "\n")
# derived events (text phone-redacted)
def evcopy(src, dst):
    with open(dst, "w") as w:
        for l in open(src):
            e = json.loads(l)
            if "text" in e: e["text"] = clean(e["text"])
            w.write(json.dumps(e, ensure_ascii=False) + "\n")
evcopy(S / "out/events.jsonl", O / "events/hyd_cyb_events.jsonl")
evcopy(S / "out/eta/incidents.jsonl", O / "events/eta_incidents_hyd.jsonl")
for src, dst in [("data/chn/events.json", "chennai_events.json"), ("data/blr/events.json", "bengaluru_events.json")]:
    t = (S / src).read_text(); (O / "events" / dst).write_text(PHONE.sub("[phone]", t))
d = json.load(open(E / "data/closures_dodges/delhi_dtp_items.json"))
for v in d.values():
    for k in ("title", "text"): v[k] = clean(v.get(k))
json.dump(d, open(O / "events/delhi_dtp_site_items.json", "w"), ensure_ascii=False, indent=0)
shutil.copy(S / "out/bengaluru/congestion_typical.json", O / "events/bengaluru_congestion_typical.json")
for f in ("congestion_alerts.csv.gz", "travel_time.csv.gz", "corridors.csv", "police_stations.csv"):
    shutil.copy(S / "data/blr/powerbi" / f, O / "btp_powerbi" / f)
m = json.load(open(S / "data/blr/powerbi/meta.json")); m.pop("key", None); json.dump(m, open(O / "btp_powerbi/meta.json", "w"), indent=1)
shutil.copy(S / "out/priors/hotspots.csv", O / "priors/hotspots.csv")
# summary
acc = collections.defaultdict(list)
for p in P: acc[p["account"]].append(p["t"])
summ = {"built_utc": iso(int(time.time())), "posts": len(P),
        "by_account": {a: {"n": len(ts), "first": iso(min(ts))[:10], "last": iso(max(ts))[:10]} for a, ts in sorted(acc.items(), key=lambda x: -len(x[1]))},
        "dropped_non_official_or_retweet": dict(dropped)}
json.dump(summ, open(O / "SUMMARY.json", "w"), indent=1)
print(json.dumps(summ, indent=1)[:3000])
