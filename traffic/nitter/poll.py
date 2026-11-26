#!/usr/bin/env python3
"""Newest posts of the traffic police accounts (Hyderabad, Cyberabad, Delhi, Chennai, Bengaluru), from the local Nitter's RSS (127.0.0.1:8095), into ~/nitter/out/latest.json.

    {"t": epoch, "accounts": {"HYDTP": {"ok": true, "t": epoch, "posts": [{"id", "url", "t", "text"}, ...]}, ...}}

Only the latest page (Nitter's RSS: about 20 posts), newest first; no history is kept here (twtrx keeps its own).
url points at x.com, under the post's own author (a repost or a group search can be someone else's); "user" is that
author. A failed account keeps its last good posts with "ok": false. Stdlib only; run by cron every 2 min.

Groups (groups.json beside this file, written by twtrx scripts/blr_accounts.py): name -> an X search over several
accounts' own posts, e.g. Bengaluru's traffic police stations ("BLR1".."BLR8"). Nitter refuses a query over about
200 characters, so each group holds about 7 accounts. GROUPS_PER_RUN groups are read per run, in turn; the others
keep their last posts (with their own "t"), so latest.json always carries every group.
"""
import email.utils
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ACCOUNTS = ['HYDTP', 'CYBTRAFFIC', 'dtptraffic', 'ChennaiTraffic', 'blrcitytraffic']   # Hyderabad, Cyberabad, Delhi, Chennai, Bengaluru traffic police
BASE = 'http://127.0.0.1:8095'
# Accounts whose timeline is mostly reposts: also read their own posts via search, merged in (same schema).
SEARCH = {'dtptraffic': 'from:dtptraffic -filter:replies -filter:nativeretweets'}
OUT = os.path.expanduser('~/nitter/out/latest.json')
GROUPS = os.path.expanduser('~/nitter/groups.json')
GROUPS_PER_RUN = 2


def text_of(desc):
    s = re.sub(r'<br\s*/?>', '\n', desc or '', flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).strip()


def fetch(acct, path=None):
    with urllib.request.urlopen(urllib.request.Request(f'{BASE}/{path or acct + "/rss"}', headers={'User-Agent': 'twtrx-local/1'}), timeout=30) as r:
        root = ET.fromstring(r.read())
    posts = []
    for it in root.iter('item'):
        link = (it.findtext('link') or '').strip()
        m = re.search(r'/status/(\d+)', link)
        if not m:
            continue
        when = it.findtext('pubDate')
        t = int(email.utils.parsedate_to_datetime(when).timestamp()) if when else 0
        user = re.search(r'/(\w+)/status/\d+', link).group(1)
        posts.append({'id': m.group(1), 'user': user, 'url': f'https://x.com/{user}/status/{m.group(1)}', 't': t,
                      'text': text_of(it.findtext('description')) or (it.findtext('title') or '').strip()})
    posts.sort(key=lambda p: p['t'], reverse=True)
    return posts


def main():
    try:
        old = json.load(open(OUT))
    except (OSError, ValueError):
        old = {'accounts': {}}
    out = {'t': int(time.time()), 'accounts': {}}
    for a in ACCOUNTS:
        try:
            posts = fetch(a)
            if a in SEARCH:
                ids = {x['id'] for x in posts}
                posts += [x for x in fetch(a, 'search/rss?f=tweets&q=' + urllib.parse.quote(SEARCH[a])) if x['id'] not in ids]
                posts.sort(key=lambda x: x['t'], reverse=True)
            out['accounts'][a] = {'ok': True, 't': out['t'], 'posts': posts}
        except Exception as e:
            prev = old.get('accounts', {}).get(a, {})
            out['accounts'][a] = {'ok': False, 't': prev.get('t', 0), 'error': type(e).__name__, 'posts': prev.get('posts', [])}
    try:
        groups = json.load(open(GROUPS))
    except (OSError, ValueError):
        groups = {}
    names = sorted(groups)
    turn = int(time.time() // 120)
    now = {names[(turn * GROUPS_PER_RUN + i) % len(names)] for i in range(GROUPS_PER_RUN)} if names else set()
    for g in names:
        prev = old.get('accounts', {}).get(g)
        if g not in now:                      # not its turn: keep what it had (a new group waits for its turn)
            if prev:
                out['accounts'][g] = prev
            continue
        try:
            out['accounts'][g] = {'ok': True, 't': out['t'], 'group': True,
                                  'posts': fetch(g, 'search/rss?f=tweets&q=' + urllib.parse.quote(groups[g]))}
        except Exception as e:
            prev = prev or {}
            out['accounts'][g] = {'ok': False, 't': prev.get('t', 0), 'group': True, 'error': type(e).__name__, 'posts': prev.get('posts', [])}
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, OUT)
    print(time.strftime('%F %T'), ' '.join(f"{a}:{'ok' if v['ok'] else v.get('error')}:{len(v['posts'])}" for a, v in out['accounts'].items()
                                           if not v.get('group') or a in now), flush=True)


if __name__ == '__main__':
    main()
