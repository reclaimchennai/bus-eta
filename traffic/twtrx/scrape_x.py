"""Zero-waste Sorsa scraper for a traffic police X account (default @HYDTP).

  python3 scrape_x.py HYDTP back  <max_requests>          # walk backwards from the oldest post held
  python3 scrape_x.py HYDTP new   <max_requests>          # only posts newer than the newest held
  python3 scrape_x.py HYDTP window SINCE UNTIL <max_req>  # fill a date gap (YYYY-MM-DD, until exclusive)

Rules that keep every billed request useful:
  * server-side keyword filter (only road-traffic posts come back, 20 per request)
  * every raw response is cached in raw/<acct>/; parsing never re-fetches
  * 'new' stops on the first page that reaches an id already held (incremental, no re-reads)
  * 'back' resumes from state; when a cursor stops yielding older posts it restarts with until:<oldest date>
    instead of paging deeper (deep pagination on X search skips posts)
  * hard budget per run; stops when X-Requests-Available <= RESERVE, on 429 (honours Retry-After once), on 401/402/403
Key: SORSA_API_KEY in sorsa.env (never printed).
"""
import json, os, sys, time, urllib.request, urllib.error
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
RESERVE = 3     # keep a few requests for live checks

TERMS = ['"traffic advisory"', '"traffic alert"', '"traffic update"', 'diversion', 'diversions', 'diverted',
         'accident', 'crash', 'collision', 'breakdown', '"broke down"', 'congestion', '"slow moving"',
         '"traffic is slow"', '"slow traffic"', '"heavy traffic"', '"heavy vehicular"', '"heavy flow"',
         'waterlogging', '"water logging"', '"road closed"', '"road closure"', 'spillage', 'overturned',
         'uprooted', '"tree fall"', '"fire incident"', 'procession', 'rally']

def load_key():
    for line in (HERE / "sorsa.env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no key in sorsa.env")

TAG = os.environ.get("XTAG")   # e.g. XTAG=HYDTPinfo: short hashtag query (long OR queries only reach ~90 days back)

def query(acct, extra=""):
    if TAG:
        return f"from:{acct} #{TAG} -filter:replies" + (" " + extra if extra else "")
    return f"from:{acct} (" + " OR ".join(TERMS) + ") -filter:replies" + (" " + extra if extra else "")

class Out(Exception): pass

def post(key, body, st):
    for attempt in (1, 2):
        req = urllib.request.Request("https://api.sorsa.io/v3/search-tweets", data=json.dumps(body).encode(),
                                     method="POST", headers={"ApiKey": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                avail = r.headers.get("X-Requests-Available")
                return json.load(r), int(avail) if avail and avail.lstrip("-").isdigit() else None
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode(errors="replace")
            if e.code == 429 and attempt == 1:
                wait = min(int(e.headers.get("Retry-After") or 30), 120)
                print(f"429 rate limited, waiting {wait}s once"); time.sleep(wait); continue
            raise Out(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 1:
                print("network error, retrying once in 15s:", e); time.sleep(15); continue
            raise Out(f"network: {e}")
    raise Out("gave up after retry")

def created(t):
    return datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S %z %Y")

def main():
    acct, mode = sys.argv[1], sys.argv[2]
    raw = HERE / "raw" / acct.lower(); raw.mkdir(parents=True, exist_ok=True)
    sp = raw / "state.json"
    st = json.loads(sp.read_text()) if sp.exists() else {"pages": 0, "ids": [], "back_cursor": None,
                                                          "back_until": None, "back_done": False}
    held = set(st["ids"])
    if TAG:                                   # own back-walk position for the hashtag query
        bk = st.setdefault("tag_back", {"back_cursor": None, "back_until": os.environ.get("XUNTIL"), "back_done": False})
        main_bk = {k: st[k] for k in ("back_cursor", "back_until", "back_done")}
        st.update(bk)
    if mode == "window":
        since, until, budget = sys.argv[3], sys.argv[4], int(sys.argv[5])
    else:
        budget = int(sys.argv[3])
    key = load_key(); used = 0; cursor = None
    created_by = {}                       # id -> datetime, from the cached raw pages (no requests)
    for f in raw.glob("*_*.json"):
        for t in json.load(open(f))["response"].get("tweets") or []:
            created_by[t["id"]] = created(t)

    def save():
        st["ids"] = sorted(held, key=int)
        out = dict(st)
        if TAG:
            out["tag_back"] = {k: st[k] for k in ("back_cursor", "back_until", "back_done")}
            out.update(main_bk)
        sp.write_text(json.dumps(out))

    try:
        while used < budget:
            if mode == "back":
                if st["back_done"]: print("history complete"); break
                extra = f"until:{st['back_until']}" if st["back_until"] else ""
                body = {"query": query(acct, extra), "order": "latest"}
                if st["back_cursor"]: body["next_cursor"] = st["back_cursor"]
            elif mode == "new":
                body = {"query": query(acct), "order": "latest"}
                if cursor: body["next_cursor"] = cursor
            else:
                body = {"query": query(acct, f"since:{since} until:{until}"), "order": "latest"}
                if cursor: body["next_cursor"] = cursor
            data, avail = post(key, body, st)
            used += 1; st["pages"] += 1
            (raw / f"{mode}_{st['pages']:04d}.json").write_text(json.dumps({"request": body, "response": data}, ensure_ascii=False))
            tw = [t for t in (data.get("tweets") or []) if (t.get("user") or {}).get("username", "").lower() == acct.lower()]
            new = [t for t in tw if t["id"] not in held]
            held.update(t["id"] for t in tw)
            for t in tw: created_by[t["id"]] = created(t)
            nxt = data.get("next_cursor")
            end = not nxt or nxt == "0" or str(nxt).startswith("0|")
            span = f"{created(tw[-1]):%Y-%m-%d} .. {created(tw[0]):%Y-%m-%d}" if tw else "-"
            print(f"[{mode}] page {st['pages']}: {len(tw)} posts ({len(new)} new) {span} | requests left {avail}")
            if mode == "new":
                if len(new) < len(tw) or end: break          # reached what we already hold
                cursor = nxt
            elif mode == "window":
                if end or not new: break
                cursor = nxt
            else:   # back: always anchor on the oldest post actually held, never on a stale date
                oldest_held = min((created_by[i] for i in held if i in created_by), default=None)
                if tw and new and not end:
                    st["back_cursor"] = nxt
                    if st["pages"] % 25 == 0 and oldest_held:      # re-anchor: deep paging skips posts
                        st["back_cursor"] = None; st["back_until"] = f"{oldest_held:%Y-%m-%d}"
                else:
                    anchor = f"{oldest_held:%Y-%m-%d}" if oldest_held else None
                    if not tw and not st["back_cursor"] or anchor == st["back_until"] and not new:
                        st["back_done"] = True                  # an anchored query found nothing older
                    st["back_cursor"] = None; st["back_until"] = anchor
            save()
            if avail is not None and avail <= RESERVE:
                print(f"stopping: only {avail} requests left (reserve {RESERVE})"); break
            time.sleep(0.4)
    except Out as e:
        print("stopped:", e)
    save()
    print(f"used {used} this run | pages total {st['pages']} | posts held {len(held)}")

if __name__ == "__main__":
    main()
