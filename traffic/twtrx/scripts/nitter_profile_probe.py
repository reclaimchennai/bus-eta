#!/usr/bin/env python3
"""Probe X accounts on the local Nitter (run on mss over ssh stdin): one JSON line per account with its profile
(name, bio, posts, followers, joined) and its latest timeline page (own posts vs reposts, whose posts it reposts,
newest and oldest date, a few own-post texts). 3 s between accounts.
  ssh mss python3 - user1 user2 ... < scripts/nitter_profile_probe.py"""
import html, json, re, sys, time, urllib.request
from collections import Counter
from datetime import datetime, timezone

BASE = 'http://127.0.0.1:8095/'

def clean(s):
    return html.unescape(re.sub(r'<[^>]+>', '', re.sub(r'<br\s*/?>', '\n', s or ''))).strip()

def num(s):
    return int(re.sub(r'\D', '', s)) if s and re.search(r'\d', s) else None

def probe(u):
    try:
        s = urllib.request.urlopen(urllib.request.Request(BASE + u, headers={'User-Agent': 'twtrx-local/1'}), timeout=60).read().decode('utf-8', 'replace')
    except Exception as e:
        return {'user': u, 'error': str(e)[:120]}
    if 'error-panel' in s and 'profile-card' not in s:
        return {'user': u, 'error': clean(re.search(r'error-panel">(.*?)</div>', s, re.S).group(1))[:120]}
    g = lambda p: (m.group(1) if (m := re.search(p, s, re.S)) else None)
    out = {'user': u,
           'handle': g(r'class="profile-card-username"[^>]*title="@?(\w+)"'),
           'name': clean(g(r'class="profile-card-fullname"[^>]*title="([^"]*)"')),
           'bio': clean(g(r'<div class="profile-bio">(.*?)</div>'))[:200],
           'joined': clean(g(r'class="profile-joindate"><span title="([^"]+)"')),
           'posts': num(g(r'class="posts">.*?profile-stat-num">([^<]+)')),
           'followers': num(g(r'class="followers">.*?profile-stat-num">([^<]+)')),
           'protected': 'icon-lock' in (g(r'(class="profile-card-fullname".*?</div>)') or '')}
    own, rts, ts, from_, ment = 0, 0, [], Counter(), Counter()
    texts = []
    for blk in s.split('<div class="timeline-item')[1:]:
        if 'class="pinned"' in blk[:600]:
            continue
        m = re.search(r'class="tweet-date"><a href="/(\w+)/status/(\d+)[^"]*" title="([^"]+)"', blk)
        if not m:
            continue
        t = int(datetime.strptime(m.group(3).replace('·', '').replace('  ', ' '), '%b %d, %Y %I:%M %p UTC').replace(tzinfo=timezone.utc).timestamp())
        if 'retweet-header' in blk[:800]:
            rts += 1; from_[m.group(1)] += 1
        else:
            own += 1; ts.append(t)
            for h in re.findall(r'href="/(\w+)"[^>]*>@', blk):
                ment[h] += 1
            if len(texts) < 3:
                c = re.search(r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>', blk, re.S)
                texts.append(clean(c.group(1) if c else '')[:160])
    out.update(own=own, reposts=rts, repost_of=dict(from_.most_common()),
               mentions=dict(ment.most_common(15)), newest_own=max(ts) if ts else None, oldest_own=min(ts) if ts else None, samples=texts)
    return out

for u in sys.argv[1:]:
    print(json.dumps(probe(u), ensure_ascii=False), flush=True)
    time.sleep(3)
