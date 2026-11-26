#!/usr/bin/env python3
"""Newest posts of the traffic police accounts (Hyderabad, Cyberabad, Delhi, Chennai), from the local Nitter's RSS (127.0.0.1:8095), into ~/nitter/out/latest.json.

    {"t": epoch, "accounts": {"HYDTP": {"ok": true, "t": epoch, "posts": [{"id", "url", "t", "text"}, ...]}, ...}}

Only the latest page (Nitter's RSS: about 20 posts), newest first; no history is kept here (twtrx keeps its own).
url points at x.com. A failed account keeps its last good posts with "ok": false. Stdlib only; run by cron every 2 min.
"""
import email.utils
import html
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

ACCOUNTS = ['HYDTP', 'CYBTRAFFIC', 'dtptraffic', 'ChennaiTraffic']   # Hyderabad, Cyberabad, Delhi, Chennai traffic police
BASE = 'http://127.0.0.1:8095'
OUT = os.path.expanduser('~/nitter/out/latest.json')


def text_of(desc):
    s = re.sub(r'<br\s*/?>', '\n', desc or '', flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s).strip()


def fetch(acct):
    with urllib.request.urlopen(urllib.request.Request(f'{BASE}/{acct}/rss', headers={'User-Agent': 'twtrx-local/1'}), timeout=30) as r:
        root = ET.fromstring(r.read())
    posts = []
    for it in root.iter('item'):
        link = (it.findtext('link') or '').strip()
        m = re.search(r'/status/(\d+)', link)
        if not m:
            continue
        when = it.findtext('pubDate')
        t = int(email.utils.parsedate_to_datetime(when).timestamp()) if when else 0
        posts.append({'id': m.group(1), 'url': f'https://x.com/{acct}/status/{m.group(1)}', 't': t,
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
            out['accounts'][a] = {'ok': True, 't': out['t'], 'posts': fetch(a)}
        except Exception as e:
            prev = old.get('accounts', {}).get(a, {})
            out['accounts'][a] = {'ok': False, 't': prev.get('t', 0), 'error': type(e).__name__, 'posts': prev.get('posts', [])}
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, OUT)
    print(time.strftime('%F %T'), ' '.join(f"{a}:{'ok' if v['ok'] else v.get('error')}:{len(v['posts'])}" for a, v in out['accounts'].items()), flush=True)


if __name__ == '__main__':
    main()
