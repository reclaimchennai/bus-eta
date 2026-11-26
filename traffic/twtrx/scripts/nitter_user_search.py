#!/usr/bin/env python3
"""Account search on the local Nitter (run on mss over ssh stdin): prints one JSON line per account found
{q, handle, name, bio}. Pages each query up to MAXP pages, 3 s between pages.
  ssh mss python3 - MAXP 'query one' 'query two' ... < scripts/nitter_user_search.py"""
import html, json, re, sys, time, urllib.parse, urllib.request

BASE = 'http://127.0.0.1:8095/search'
maxp, qs = int(sys.argv[1]), sys.argv[2:]
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', '', s or '')).strip()
for q in qs:
    cur, seen = None, set()
    for page in range(maxp):
        u = BASE + '?' + urllib.parse.urlencode({'f': 'users', 'q': q, **({'cursor': cur} if cur else {})})
        try:
            s = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'twtrx-local/1'}), timeout=60).read().decode('utf-8', 'replace')
        except Exception as e:
            print(f'# {q!r} page {page}: {e}', file=sys.stderr); break
        n = 0
        for blk in s.split('<div class="timeline-item')[1:]:
            h = re.search(r'class="username"[^>]*title="@(\w+)"', blk)
            if not h or h.group(1).lower() in seen:
                continue
            seen.add(h.group(1).lower()); n += 1
            nm = re.search(r'class="fullname"[^>]*title="([^"]*)"', blk)
            bio = re.search(r'class="tweet-content[^"]*"[^>]*>(.*?)</div>', blk, re.S)
            print(json.dumps({'q': q, 'handle': h.group(1), 'name': clean(nm.group(1) if nm else ''),
                              'bio': clean(bio.group(1) if bio else '')[:200]}, ensure_ascii=False), flush=True)
        nx = re.findall(r'show-more"><a href="\?([^"]+)"', s)
        print(f'# {q!r} page {page}: {n}', file=sys.stderr, flush=True)
        if not n or not nx:
            break
        cur = urllib.parse.parse_qs(html.unescape(nx[-1])).get('cursor', [None])[0]
        if not cur:
            break
        time.sleep(3)
    time.sleep(3)
