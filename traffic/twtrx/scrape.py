"""Scrape @CYBTRAFFIC road-traffic posts via Sorsa /search-tweets.

Server-side keyword filtering means every billed request returns 20 relevant
posts. Every raw page is saved to raw/ and progress (cursor, pages used) to
state.json, so the run is resumable and no request is ever spent twice.

Usage: python3 scrape.py <max_requests>
"""
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw"; RAW.mkdir(exist_ok=True)
STATE = HERE / "state.json"

def load_key():
    for line in (HERE / "sorsa.env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key in sorsa.env")

TERMS = ['"traffic movement"', '"traffic advisory"', '"traffic alert"',
         'diversion', 'diversions', 'diverted', 'accident', 'crash', 'collision',
         'spillage', 'breakdown', 'congestion', 'waterlogging', '"road closure"',
         'overturned', 'uprooted', '"traffic update"', '"road closed"', '"fire incident"']
QUERY = "from:CYBTRAFFIC (" + " OR ".join(TERMS) + ") -filter:replies"

def post(key, body):
    req = urllib.request.Request("https://api.sorsa.io/v3/search-tweets",
        data=json.dumps(body).encode(), method="POST",
        headers={"ApiKey": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), r.headers.get("X-Requests-Available")

def main():
    budget = int(sys.argv[1])
    key = load_key()
    st = json.loads(STATE.read_text()) if STATE.exists() else {"pages": 0, "cursor": None, "done": False, "query": QUERY}
    used = 0
    while used < budget and not st["done"]:
        body = {"query": st["query"], "order": "latest"}
        if st["cursor"]:
            body["next_cursor"] = st["cursor"]
        try:
            data, remaining = post(key, body)
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, e.read()[:300].decode(errors="replace")); break
        used += 1; st["pages"] += 1
        (RAW / f"page_{st['pages']:04d}.json").write_text(json.dumps(data, ensure_ascii=False))
        tw = data.get("tweets") or []
        cur = data.get("next_cursor")
        oldest = tw[-1]["created_at"] if tw else "-"
        print(f"page {st['pages']}: {len(tw)} tweets, oldest {oldest}, remaining {remaining}")
        if not tw or not cur or cur == "0" or str(cur).startswith("0|") or cur == st["cursor"]:
            st["done"] = True
        st["cursor"] = cur
        STATE.write_text(json.dumps(st, indent=1))
        if remaining is not None and remaining.isdigit() and int(remaining) <= 0:
            break
        time.sleep(0.3)
    print("used this run:", used, "| total pages:", st["pages"], "| done:", st["done"])

if __name__ == "__main__":
    main()
