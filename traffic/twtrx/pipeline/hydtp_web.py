"""Hyderabad Traffic Police e-challan site: live alert + "Latest News (Traffic Updates)".

  python -m pipeline.hydtp_web            # one polite poll (run by the 5-min launchd job)
  python -m pipeline.hydtp_web --check    # show what it would see, ingest nothing

Sources (both public, read by the page itself in any browser):
  * alert JSON  https://echallan.tspolice.gov.in/TSeChallanRS/IDDetails/data  -> {"message", "By"}
    The page polls it every 60 s; we poll every 5 min. It sends no ETag/Last-Modified, so the body is
    hashed: an unchanged message costs no processing, a new one is ingested (time = first seen; the
    source gives none), and when it changes or disappears the previous alert gets t_end = that moment.
  * index      https://echallan.tspolice.gov.in/htp/index.html  (~300 KB)
    Fetched at most every 30 min with If-None-Match / If-Modified-Since (304 = nothing new).
    News items link to advisory PDFs / route-map images: each new one is downloaded once into
    raw/hydtp_echallan/docs/ and listed for curation (multi-diversion advisories are not auto-geocoded).
Honest User-Agent, no retries on refusal; anything gated is reported, not worked around.
"""
import argparse, hashlib, html, json, re, subprocess, time, urllib.error, urllib.request
from datetime import datetime
from pathlib import Path
from .classify import IST

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw/hydtp_echallan"; DOCS = RAW / "docs"
STATE = ROOT / "data/hydtp_web_state.json"
BASE = "https://echallan.tspolice.gov.in"
ALERT_URL = BASE + "/TSeChallanRS/IDDetails/data"
INDEX_URL = BASE + "/htp/index.html"
UA = "twtrx-traffic-research/1.0 (personal, non-commercial; polls every 5 min)"
INDEX_EVERY = 30 * 60
NO_ALERT = re.compile(r"no active|^\s*$", re.I)

def get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b""

def parse_news(page):
    items = []
    for m in re.finditer(r'<div class="news-item">\s*<a href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        href, inner = m.group(1), m.group(2)
        title = re.search(r"<strong>(.*?)</strong>", inner, re.S)
        t = html.unescape(re.sub(r"<[^>]+>", " ", title.group(1))).strip() if title else ""
        body = html.unescape(re.sub(r"<[^>]+>", " ", inner[title.end():] if title else inner))
        items.append({"href": href, "title": re.sub(r"\s+", " ", t).strip(" ➤"),
                      "summary": re.sub(r"\s+", " ", body).strip()})
    return items

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    now = datetime.now(IST); ts = int(now.timestamp())
    st = json.loads(STATE.read_text()) if STATE.exists() else {"alert": None, "news": {}, "index": {}}
    print(now.strftime("[%Y-%m-%d %H:%M]"), end=" ")

    # ---- live alert
    code, hdr, body = get(ALERT_URL)
    msgs = []; closed = None
    if code != 200:
        print(f"alert endpoint HTTP {code} (not retried)", end="; ")
    else:
        try: j = json.loads(body)
        except ValueError: j = {}
        msg = (j.get("message") or "").strip()
        h = hashlib.sha1(msg.encode()).hexdigest()[:12] if msg and not NO_ALERT.search(msg) else None
        cur = st.get("alert")
        if a.check:
            print("alert:", msg[:160] or "(none)"); return 0
        if (cur or {}).get("hash") != h:
            if cur and cur.get("event_id"):
                closed = (cur["event_id"], ts)                 # observed end of the previous alert
            if h:
                RAW.mkdir(parents=True, exist_ok=True)
                with open(RAW / "alerts.jsonl", "a") as f:
                    f.write(json.dumps({"first_seen": ts, "message": msg, "by": j.get("By")}, ensure_ascii=False) + "\n")
                msgs.append({"time": now, "text": msg})
            st["alert"] = {"hash": h, "first_seen": ts, "event_id": None} if h else None
        print("alert:", "new" if msgs else ("ended" if closed and not h else "unchanged" if h else "none"), end="; ")

    # ---- news list (conditional GET, at most every 30 min)
    ix = st.setdefault("index", {})
    if ts - ix.get("checked", 0) >= INDEX_EVERY:
        hd = {}
        if ix.get("etag"): hd["If-None-Match"] = ix["etag"]
        if ix.get("last_modified"): hd["If-Modified-Since"] = ix["last_modified"]
        code, hdr, body = get(INDEX_URL, hd)
        ix["checked"] = ts
        if code == 304:
            print("news: not modified", end="; ")
        elif code == 200:
            ix["etag"] = hdr.get("ETag"); ix["last_modified"] = hdr.get("Last-Modified")
            items = parse_news(body.decode("utf-8", "replace"))
            fresh = [i for i in items if i["href"] not in st["news"]]
            for i in fresh:
                i["first_seen"] = ts
                url = i["href"] if i["href"].startswith("http") else f"{BASE}/htp/{i['href'].lstrip('/')}"
                DOCS.mkdir(parents=True, exist_ok=True)
                fn = DOCS / Path(i["href"]).name
                if not fn.exists():
                    c2, _, doc = get(url, timeout=60)
                    if c2 == 200: fn.write_bytes(doc)
                if fn.suffix.lower() == ".pdf" and fn.exists():
                    try:
                        txt = subprocess.run(["pdftotext", "-layout", str(fn), "-"], capture_output=True, text=True, timeout=60).stdout
                        i["doc_text_chars"] = len(txt.strip())
                        (fn.with_suffix(".txt")).write_text(txt)
                    except (OSError, subprocess.SubprocessError):
                        pass
                st["news"][i["href"]] = i
                print(f"\n  NEW ADVISORY (curate into data/manual_events.json): {i['title']} | {i['summary'][:140]} | {fn.name}", end="")
            print(f"news: {len(items)} items, {len(fresh)} new", end="; ")
        else:
            print(f"news page HTTP {code} (not retried)", end="; ")

    # ---- ingest / close
    if msgs or closed:
        from . import build
        if closed:
            ev = build.load_store(); e = ev.get(closed[0])
            if e and (e["t_end"] is None or e["t_end"] > closed[1]):
                e["t_end"] = closed[1]; e["t_expires"] = closed[1]; build.save_store(ev)
        if msgs:
            from .ingest import ingest
            ids = ingest(msgs, a.publish, source="hydtp_web")
            if ids and st.get("alert"): st["alert"]["event_id"] = ids[0]
        elif closed:
            build.write_outputs(build.load_store(), a.publish)
    print()
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
