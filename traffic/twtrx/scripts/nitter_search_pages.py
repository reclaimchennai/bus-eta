#!/usr/bin/env python3
"""Page a search on the local Nitter (run on mss over ssh stdin): prints one JSON line per post
{id, user, url, t, text}. Stops at --until (epoch), at --max-pages, or when the cursor ends. PAUSE s (default 4)
between pages: python3 - QUERY UNTIL_EPOCH MAX_PAGES [PAUSE]. Over ssh, quote QUERY for the remote shell (shlex.quote)."""
import html, json, re, sys, time, urllib.parse, urllib.request, email.utils
from datetime import datetime, timezone
q, until, maxp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PAUSE = float(sys.argv[4]) if len(sys.argv) > 4 else 4
BASE = 'http://127.0.0.1:8095/search'
cur = None; seen = set()
for page in range(maxp):
    u = BASE + '?' + urllib.parse.urlencode({'f': 'tweets', 'q': q, **({'cursor': cur} if cur else {})})
    for attempt in (1, 2):
        try:
            s = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'twtrx-local/1'}), timeout=60).read().decode('utf-8', 'replace'); break
        except Exception as e:
            print(f'# page {page} error {e}', file=sys.stderr); time.sleep(30); s = None
    if s is None: break
    oldest = None; n = 0
    for blk in s.split('<div class="timeline-item')[1:]:
        m = re.search(r'class="tweet-date"><a href="/(\w+)/status/(\d+)[^"]*" title="([^"]+)"', blk)
        c = re.search(r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>', blk, re.S)
        if not m or not c or m.group(2) in seen: continue
        seen.add(m.group(2))
        t = int(datetime.strptime(m.group(3).replace('·', '').replace('  ', ' '), '%b %d, %Y %I:%M %p UTC').replace(tzinfo=timezone.utc).timestamp())
        txt = html.unescape(re.sub(r'<[^>]+>', '', re.sub(r'<br\s*/?>', '\n', c.group(1)))).strip()
        print(json.dumps({'id': m.group(2), 'user': m.group(1), 'url': f'https://x.com/{m.group(1)}/status/{m.group(2)}', 't': t, 'text': txt}, ensure_ascii=False), flush=True)
        oldest = t if oldest is None else min(oldest, t); n += 1
    nx = re.findall(r'show-more"><a href="\?([^"]+)"', s)
    print(f'# page {page}: {n} posts, oldest {datetime.fromtimestamp(oldest or 0, timezone.utc):%Y-%m-%d}', file=sys.stderr, flush=True)
    if not n or not nx or (oldest and oldest < until): break
    cur = urllib.parse.parse_qs(html.unescape(nx[-1])).get('cursor', [None])[0]
    if not cur: break
    time.sleep(PAUSE)
