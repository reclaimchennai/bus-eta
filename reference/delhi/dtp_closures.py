"""Delhi Traffic Police alerts and advisories -> closures on the map.

  dtp_closures.py --gaz delhi_gazetteer.json.gz --stops bus_stops.geojson \
      --state DIR --alts alternatives.json --out closures.json

traffic.delhipolice.gov.in publishes alerts (breakdowns, blockages, works) and
advisories (rallies, VIP movement, trial diversions) as a ticker on its home
page and an RSS feed of the last ten posts. There is no archive, so this runs
hourly and keeps every post it has ever seen in STATE/items.json.

The site answers "503 Service Unavailable - No server is available to handle
this request" a good part of the time (its load balancer has no healthy
backend). Each source is tried up to six times with a growing pause, every
attempt is logged to STATE/fetch-log.jsonl, and a run where nothing could be
fetched keeps the posts already stored.

Reading a post:
  period  - dates written in the post ("05.10.2026 - 11.10.2026", "05.10-30.10.2026",
            "till 30.10.2026", "for 1 month", "Today (3 PM - 5 PM)"); otherwise an
            alert lasts 6 h from posting and an advisory the day it was posted.
  places  - every gazetteer name (OSM named roads and landmarks, DTC stop
            names) found word for word in the title and text. A road plus
            landmarks gives the stretch of that road near / between them;
            landmarks alone give points; a road alone is used only if short.
            Where a name exists in several places, the copy nearest the other
            names in the post wins.
  conf    - high: a road with landmarks on it; medium: landmarks only; low: a
            guess among same-named places. Posts that name nothing we know are
            listed without a shape.
STATE/manual.json may override any post by id: {"id": {"from":..,"to":..,"geom":..}}.
STATE/added.jsonl holds posts typed in by hand from the police's X / Facebook /
WhatsApp channels (most posts never reach the website, and those channels have
no free API): one JSON per line {title, text, kind, pub, [from, to], [link]};
with "id" set to a stored post's id it only fills in that post's text.

Detours on route-dodging are then checked against the closures: an
alternative whose planned line passes within 150 m of a closure, on a day the
closure was in force, is listed under that closure ("alts").
"""
import argparse
import datetime as dt
import gzip
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

BASE = 'https://traffic.delhipolice.gov.in'
SOURCES = ['/en/rss.xml', '/en']          # the category pages carry the same posts in other markup
UA = 'reclaimchennai-maps/1.0 (+https://maps.reclaimchennai.city/delhi/route-dodging/)'
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
K = 111320.0
GENERIC = {'delhi', 'new delhi', 'road', 'main road', 'market', 'metro station', 'bus stand', 'bus stop', 'police station', 'hospital',
           'school', 'temple', 'park', 'flyover', 'chowk', 'gate', 'colony', 'nagar', 'village', 'sector', 'block', 'phase', 'extension',
           'today', 'traffic', 'police', 'public', 'stretch', 'loop', 'red light', 'crossing', 'junction', 'service road', 'ring road',
           'outer ring road', 'india', 'national highway', 'expressway', 'metro', 'office', 'court', 'club', 'station', 'bridge', 'underpass'}
# big roads the police name constantly: kept, but only as a road (a stretch near the landmarks)
ROAD_ONLY = {'ring road', 'outer ring road'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def norm(s):
    s = html.unescape(s).lower()
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
    s = re.sub(r'\b(rd)\b', 'road', s)
    s = re.sub(r'\bmetro stn\b|\bmetro\s+station\b', 'metro station', s)
    return ' ' + re.sub(r'\s+', ' ', s).strip() + ' '


# ── fetching ─────────────────────────────────────────────────────────────────
def get(url, logf):
    wait = 10
    for attempt in range(6):
        code, body = 0, b''
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept-Language': 'en'})
            with urllib.request.urlopen(req, timeout=40) as r:
                code, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            code = e.code
        except Exception:
            code = 0
        logf.write(json.dumps({'t': int(time.time()), 'url': url, 'try': attempt + 1, 'code': code}) + '\n')
        logf.flush()
        if code == 200 and b'No server is available' not in body[:400]:
            return body.decode('utf-8', 'replace')
        if code in (404, 403, 429):                 # 429: rate limited - wait for the next hourly run, never hammer
            return None
        time.sleep(wait)
        wait = min(wait * 2, 90)
    return None


def parse_rss(s):
    out = []
    root = ET.fromstring(s.lstrip().encode())
    for it in root.iter('item'):
        d = html.unescape(it.findtext('description') or '')
        cat = re.findall(r'news-alertadvisory/(\w+)', d)
        if not cat:
            continue                                   # press releases, officer profiles
        text = re.findall(r'field-name-field-news-alert-advisory.*?field-item even">(.*?)</div>', d, re.S)
        pub = dt.datetime.strptime(it.findtext('pubDate').strip(), '%a, %d %b %Y %H:%M:%S %z')
        out.append({'title': it.findtext('title').strip(), 'text': re.sub('<[^>]+>', ' ', text[0]).strip() if text else '',
                    'kind': 'advisory' if cat[0].startswith('advis') else 'alert', 'pub': int(pub.timestamp()),
                    'link': (it.findtext('link') or '').strip()})
    return out


def parse_ticker(s, page):
    """Home page cards and the two category pages: title, text, date, Facebook link."""
    out = []
    blocks = re.split(r'(?=<article[^>]+id="news-)', s) if page == '/en' else [s]
    for b in blocks:
        kind = 'advisory' if ('news-advisory' in b[:200] or page.endswith('advisory')) else 'alert'
        if page == '/en' and 'news-' not in b[:200]:
            continue
        for a in re.finditer(r'<a href="\s*([^"]+)"[^>]*>\s*<div class="news-main-title">(.*?)</div>\s*<div class="subline-content">(.*?)</div>\s*<div class="news-date">(.*?)</div>', b, re.S):
            link, title, text, date = (html.unescape(re.sub('<[^>]+>', '', x)).strip() for x in a.groups())
            pub = None
            for fmt in ('%d-%m-%Y', '%d-%b-%Y'):
                try:
                    pub = int(dt.datetime.strptime(date, fmt).replace(hour=9, tzinfo=IST).timestamp()); break
                except ValueError:
                    pass
            if pub:
                out.append({'title': title, 'text': text, 'kind': kind, 'pub': pub, 'link': link})
    return out


HEADER = re.compile(r'^(traffic (alert|advisory|update)|alert|advisory|update)$', re.I)
STOP = set('the a an of to on at in near from and or for is are be will with by due this that its traffic road delhi'.split())


def words(t):
    return {w for w in norm(t).split() if len(w) > 2 and w not in STOP}


def shape(raw_text):
    """title, text, kind of a post. "📍 Location: ... 🔧 Reason: ..." posts get those
    two as the title; header lines ("TRAFFIC ALERT") and lines with phone numbers
    (drivers' contacts) are dropped."""
    raw = [re.sub(r'[*_~]', '', l).strip() for l in raw_text.splitlines()]
    lines = [l for l in raw if l and not re.search(r'\d{5}\s?\d{5}|contact\s*:', l, re.I)]
    clean = [l for l in (re.sub(r'^\W+', '', l).strip() for l in lines) if l]
    if not clean:
        return None
    field = lambda k: next((re.sub(rf'^{k}\s*:\s*', '', l, flags=re.I) for l in clean if re.match(rf'{k}\s*:', l, re.I)), '')
    loc, why = field('location'), field('reason')
    body = [l for l in clean if not HEADER.match(l)]
    first = re.sub(r'^(traffic (advisory|alert)\s*[:\-]\s*)', '', body[0] if body else clean[0], flags=re.I)
    title = (f'{loc} - {re.split(r"[.;]", why)[0]}' if loc and why else loc or re.split(r'(?<=[.!])\s', first)[0])[:140]
    kind = 'alert' if re.search(r'traffic alert|broken down|breakdown|blocked|accident|waterlog', raw_text, re.I) and not re.search(r'traffic advisory', raw_text, re.I) else 'advisory'
    rest = body[1:] if body and not (loc and why) and title.startswith(first[:40]) and len(first) <= 140 else [l for l in body if l != title]
    return title, ' '.join(rest), kind


def wa_posts(path):
    """Posts of the police's WhatsApp channel (wa/wa_bridge.mjs) as items.
    "📍 Location: ..." / "🔧 Reason: ..." posts get those two as the title;
    lines with phone numbers (drivers' contacts) are dropped."""
    out = []
    for ln in open(path):
        if not ln.strip():
            continue
        p = json.loads(ln)
        sh = shape(p.get('text', ''))
        if not sh:
            continue
        title, text, kind = sh
        out.append({'title': title, 'text': text, 'kind': kind, 'pub': int(p['t']), 'link': '', 'wa': p['id']})
    return out


X_TL = 'https://syndication.twitter.com/srv/timeline-profile/screen-name/dtptraffic'
X_KEEP = re.compile(r'advisory|traffic alert|diver(t|sion)|restriction|clos(ed|ure)|blocked|blockage|broken down|breakdown|waterlog|alternate route|carriageway|location\s*:', re.I)
X_DROP = re.compile(r'prompt action|arrest|felicitat|recognis|cleanliness|swachh|challan|crackdown|awareness|commemoration|traced|patrolling|recovered|seized|nabbed|campaign|felicitation', re.I)


def x_posts(s):
    """@dtptraffic's latest ~20 posts (and the traffic DCPs' posts it reposts) from
    X's public embed timeline, the one websites use: no login, no API key. Each post
    is cut at 280 characters there (long posts and posters keep the rest in the
    image), so the image is kept too."""
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', s, re.S)
    out = []
    for e in json.loads(m.group(1))['props']['pageProps']['timeline']['entries'] if m else []:
        t = (e.get('content') or {}).get('tweet')
        if not t:
            continue
        t = t.get('retweeted_status') or t
        txt = html.unescape(re.sub(r'https://t\.co/\S+', '', t.get('full_text', ''))).strip()
        if not X_KEEP.search(txt) or X_DROP.search(txt):
            continue
        sh = shape(txt)
        if not sh:
            continue
        title, text, kind = sh
        med = ((t.get('extended_entities') or t.get('entities') or {}).get('media') or [{}])[0].get('media_url_https')
        who = (t.get('user') or {}).get('screen_name', 'dtptraffic')
        pub = int(dt.datetime.strptime(t['created_at'], '%a %b %d %H:%M:%S %z %Y').timestamp())
        out.append({'title': title, 'text': text, 'kind': kind, 'pub': pub,
                    'link': f'https://x.com/{who}/status/{t["id_str"]}', 'x': t['id_str'], 'image': med})
    return out


def merge(store, it, src):
    """One post on several channels is one item: similar words within 3 days. The
    website's title wins; the fuller text and any image are kept."""
    if any(v.get(src) == it[src] for v in store.values()):
        return
    w = words(it['title'] + ' ' + it['text'])
    for v in store.values():
        vw = words(v['title'] + ' ' + v.get('text', ''))
        if abs(v['pub'] - it['pub']) < 3 * 86400 and len(w & vw) / max(1, len(w | vw)) >= 0.3:
            v[src] = it[src]
            if len(it['text']) > len(v.get('text', '')):
                v['text'] = it['text']
            for k in ('image', 'link'):
                if it.get(k) and not v.get(k):
                    v[k] = it[k]
            return
    k = item_id(it)
    store.setdefault(k, dict(it, id=k, first_seen=int(time.time()), last_seen=int(time.time())))


def item_id(it):
    return hashlib.sha1(norm(it['title']).strip().encode()).hexdigest()[:12]


# ── period ───────────────────────────────────────────────────────────────────
def day(d, m, y):
    return dt.datetime(y, m, d, tzinfo=IST)


def period(it):
    s = it['title'] + ' ' + it['text']
    pub = dt.datetime.fromtimestamp(it['pub'], IST)
    y0 = pub.year
    D = r'(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?'
    yr = lambda y: (int(y) + 2000 if len(y) == 2 else int(y)) if y else y0
    m = re.search(D + r'\s*(?:–|-|to|till|until)\s*' + D, s)
    if m:
        d1, m1, y1, d2, m2, y2 = m.groups()
        y2 = yr(y2); y1 = int(yr(y1)) if y1 else y2
        try:
            return int(day(int(d1), int(m1), y1).timestamp()), int((day(int(d2), int(m2), y2) + dt.timedelta(days=1)).timestamp()), 'dates'
        except ValueError:
            pass
    m = re.search(r'(?:till|until|upto|up to)\s+' + D, s, re.I)
    if m:
        d2, m2, y2 = m.groups()
        try:
            return it['pub'], int((day(int(d2), int(m2), yr(y2)) + dt.timedelta(days=1)).timestamp()), 'dates'
        except ValueError:
            pass
    m = re.search(r'for\s+(\d+|one|two|a)\s+(month|week|day)s?', s, re.I)
    if m:
        n = {'one': 1, 'a': 1, 'two': 2}.get(m.group(1).lower()) or int(m.group(1))
        days = n * {'month': 30, 'week': 7, 'day': 1}[m.group(2).lower()]
        return it['pub'], it['pub'] + days * 86400, 'duration'
    t = re.search(r'(\d{1,2})(?::(\d\d))?\s*([AP]M)\s*(?:–|-|to)\s*(\d{1,2})(?::(\d\d))?\s*([AP]M)', s, re.I)
    if t:
        h = lambda hh, mm, ap: (int(hh) % 12 + (12 if ap.upper() == 'PM' else 0), int(mm or 0))
        (h1, n1), (h2, n2) = h(*t.group(1, 2, 3)), h(*t.group(4, 5, 6))
        a = pub.replace(hour=h1, minute=n1, second=0); b = pub.replace(hour=h2, minute=n2, second=0)
        if b <= a:
            b += dt.timedelta(days=1)
        return int(a.timestamp()), int(b.timestamp()), 'hours'
    if it['kind'] == 'alert':
        return it['pub'], it['pub'] + 6 * 3600, 'default'
    d0 = pub.replace(hour=0, minute=0, second=0)
    return int(d0.timestamp()), int((d0 + dt.timedelta(days=1)).timestamp()), 'default'


# ── places ───────────────────────────────────────────────────────────────────
class Gaz:
    def __init__(self, gaz, stops):
        g = json.load(gzip.open(gaz, 'rt'))
        self.roads = {}
        for nm, lines in g['roads'].items():
            self.roads.setdefault(norm(nm), []).extend(lines)
        self.places = {}
        for nm, x, y, kind in g['places']:
            self.places.setdefault(norm(nm), []).append((x, y, kind))
        for f in json.load(open(stops))['features']:
            nm = f['properties'].get('Stop Name') or f['properties'].get('name') or ''
            if nm:
                x, y = f['geometry']['coordinates'][:2]
                self.places.setdefault(norm(nm), []).append((x, y, 'bus_stop'))
        # a stop name like "Karala Village" also answers to "Karala"
        for key in list(self.places):
            for suf in (' village ', ' bus stand ', ' terminal ', ' depot ', ' main ', ' more '):
                if key.endswith(suf) and len(key) - len(suf) > 6:
                    self.places.setdefault(key[:-len(suf) + 1], []).extend(self.places[key])
        # Ring Road is Mahatma Gandhi Marg in OSM; the police use the common name
        for alias, real in ((' ring road ', ' mahatma gandhi marg '), (' ring road ', ' mahatma gandhi road ')):
            self.roads.setdefault(alias, []).extend(self.roads.get(real, []))
        # "Padmashree Guru Hanuman Marg" is written "Guru Hanuman Marg"
        for key in list(self.roads):
            w = key.split()
            for n in (3, 2):
                if len(w) > n and len(' '.join(w[-n:])) >= 12 and w[-1] in ('marg', 'road', 'flyover', 'path', 'bridge', 'chowk'):
                    self.roads.setdefault(' ' + ' '.join(w[-n:]) + ' ', []).extend(self.roads[key])
        ok = lambda k: len(k.strip()) >= 5 and k.strip() not in GENERIC and not k.strip().isdigit()
        self.roads = {k: v for k, v in self.roads.items() if ok(k) or k.strip() in ROAD_ONLY}
        AREA = ('bus_stop', 'place=', 'railway=station', 'station=', 'public_transport=station')
        self.places = {k: [c for c in v if ' ' in k.strip() or c[2].startswith(AREA)] for k, v in self.places.items() if ok(k)}
        self.places = {k: v for k, v in self.places.items() if v}
        log(f'gazetteer: {len(self.roads)} roads, {len(self.places)} place names')

    def find(self, text):
        """Longest names first; a name inside a longer match does not count again."""
        t = norm(text)
        taken, hits = [], []
        for key in sorted(set(self.roads) | set(self.places), key=len, reverse=True):
            i = t.find(key)
            while i >= 0:
                j = i + len(key)
                if not any(a < j and i < b for a, b in taken):
                    taken.append((i, j)); hits.append((i, key)); break
                i = t.find(key, i + 1)
        return [k for _, k in sorted(hits)]


def mxy(x, y, lat0):
    return x * K * math.cos(math.radians(lat0)), y * K


def seg_d(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    L = ax * ax + ay * ay
    u = 0 if L == 0 else max(0, min(1, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L))
    return math.hypot(p[0] - a[0] - u * ax, p[1] - a[1] - u * ay)


def line_d(p, lines, lat0):
    P = mxy(*p, lat0)
    best = 1e12
    for ln in lines:
        q = [mxy(x, y, lat0) for x, y in ln]
        for a, b in zip(q, q[1:]):
            best = min(best, seg_d(P, a, b))
    return best


DIVERT = re.compile(r'\b(?:divert\w*|via|use|take|prefer|alternate)\b[^.;!]*', re.I)


def locate(gz, it):
    # "divert via Rani Jhansi Flyover" names the way round, not the closure
    names = gz.find(DIVERT.sub(' ', it['title']) + ' . ' + DIVERT.sub(' ', it['text']))
    roads = [n for n in names if n in gz.roads]
    marks = [n for n in names if n in gz.places and n not in gz.roads]
    lat0 = 28.6
    conf = 'medium'
    # landmarks: for each name pick the copy nearest the roads named, else nearest the other landmarks
    pts = {}
    for n in marks:
        cands = gz.places[n]
        if roads:
            scored = sorted(cands, key=lambda c: min(line_d(c[:2], gz.roads[r], lat0) for r in roads))
            if min(line_d(scored[0][:2], gz.roads[r], lat0) for r in roads) > 1500:
                continue
        elif len(marks) > 1:
            others = [c for m in marks if m != n for c in gz.places[m]]
            scored = sorted(cands, key=lambda c: min(math.hypot(*(a - b for a, b in zip(mxy(*c[:2], lat0), mxy(*o[:2], lat0)))) for o in others))
        else:
            scored = sorted(cands, key=lambda c: 0 if c[2].startswith(('railway', 'station', 'public_transport')) else 1)
        far = max(math.hypot(*(a - b for a, b in zip(mxy(*scored[0][:2], lat0), mxy(*c[:2], lat0)))) for c in cands)
        if far > 2000 and not roads and len(marks) == 1:
            conf = 'low'
        pts[n] = scored[0][:2]
    geom = None
    if roads and pts:
        P = [mxy(*p, lat0) for p in pts.values()]
        keep = []
        for r in roads:
            for ln in gz.roads[r]:
                part = []
                for x, y in ln:
                    q = mxy(x, y, lat0)
                    # between landmarks: a corridor that widens with the stretch, as roads bend away from the straight line
                    near = (min(seg_d(q, a, b) - max(0, 0.2 * math.hypot(b[0] - a[0], b[1] - a[1]) - 250) for a, b in zip(P, P[1:])) < 250) if len(P) > 1 \
                        else (math.hypot(q[0] - P[0][0], q[1] - P[0][1]) < 600)
                    if near:
                        part.append([x, y])
                    elif len(part) > 1:
                        keep.append(part); part = []
                    else:
                        part = []
                if len(part) > 1:
                    keep.append(part)
        if keep:
            geom = {'type': 'MultiLineString', 'coordinates': keep}
            conf = 'high' if conf != 'low' else 'medium'
    if geom is None and pts:
        if len(pts) > 1 and re.search(r'\bbetween\b|\bfrom\b', it['title'] + ' ' + it['text'], re.I):
            geom = {'type': 'MultiLineString', 'coordinates': [list(pts.values())]}
        else:
            geom = {'type': 'MultiPoint', 'coordinates': list(pts.values())}
    if geom is None and roads:
        lines = [ln for r in roads for ln in gz.roads[r]]
        length = sum(math.hypot(*(a - b for a, b in zip(mxy(*p, lat0), mxy(*q, lat0)))) for ln in lines for p, q in zip(ln, ln[1:]))
        if length < 4000:
            geom = {'type': 'MultiLineString', 'coordinates': lines}
    return geom, (conf if geom else None), [n.strip() for n in roads + marks]


# ── detours near closures ───────────────────────────────────────────────────
def link_alts(items, alts_path, dodges_path):
    """Detours made while a closure was in force (from an hour before) whose
    planned line passes within 150 m of it: [alternative key, detours then]."""
    if not alts_path or not os.path.exists(alts_path) or not os.path.exists(dodges_path or ''):
        return
    A = {a['id']: a for a in json.load(open(alts_path))['alts']}
    D = json.load(open(dodges_path))
    c = {k: i for i, k in enumerate(D['cols'])}
    lat0 = 28.6
    for it in items:
        it['alts'] = []
        g = it.get('geom')
        if not g:
            continue
        lines = g['coordinates'] if g['type'] == 'MultiLineString' else [[p, p] for p in g['coordinates']]
        xs = [p[0] for ln in lines for p in ln]; ys = [p[1] for ln in lines for p in ln]
        box = (min(xs) - .003, min(ys) - .003, max(xs) + .003, max(ys) + .003)
        when = {}
        for d in D['d']:
            if it['from'] - 3600 <= d[c['t0']] <= it['to']:
                when[d[c['alt']]] = when.get(d[c['alt']], 0) + 1
        near = {}
        for aid, n in when.items():
            a = A.get(aid)
            pl = (a or {}).get('planned') or []
            if not any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pl):
                continue
            if min(line_d(p, lines, lat0) for p in pl) < 150:
                near[a['key']] = near.get(a['key'], 0) + n
        it['alts'] = sorted(([k, n] for k, n in near.items()), key=lambda kn: -kn[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gaz', required=True)
    ap.add_argument('--stops', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--alts', default='')
    ap.add_argument('--dodges', default='', help='dodges.json, for its list of days')
    ap.add_argument('--out', required=True)
    ap.add_argument('--wa', default='', help='posts.jsonl of the WhatsApp channel bridge')
    ap.add_argument('--no-fetch', action='store_true', help='re-read the stored posts only')
    a = ap.parse_args()
    os.makedirs(a.state, exist_ok=True)
    store_p = os.path.join(a.state, 'items.json')
    store = json.load(open(store_p)) if os.path.exists(store_p) else {}
    reached = 0
    with open(os.path.join(a.state, 'fetch-log.jsonl'), 'a') as logf:
        for src in ([] if a.no_fetch else SOURCES):
            s = get(BASE + src, logf)
            if s is None:
                log('unreachable', src); continue
            reached += 1
            try:
                got = parse_rss(s) if src.endswith('.xml') else parse_ticker(s, src)
            except Exception as e:
                log('parse failed', src, e); continue
            now = int(time.time())
            for it in got:
                k = item_id(it)
                old = store.get(k)
                if old:
                    old['last_seen'] = now
                    if src.endswith('.xml'):            # the RSS has the exact posting time and the site's own link
                        old.update(pub=it['pub'], link=it['link'] or old['link'])
                    if not old.get('text') and it['text']:
                        old['text'] = it['text']
                else:
                    store[k] = dict(it, id=k, first_seen=now, last_seen=now)
            log(src, len(got), 'posts')
    # X (@dtptraffic): most posts never reach the website
    if not a.no_fetch:
        with open(os.path.join(a.state, 'fetch-log.jsonl'), 'a') as logf:
            s = get(X_TL, logf)
        if s:
            try:
                got = x_posts(s)
                for it in got:
                    merge(store, it, 'x')
                log('x', len(got), 'traffic posts')
            except Exception as e:
                log('x parse failed', e)
    if a.wa and os.path.exists(a.wa):
        for it in wa_posts(a.wa):
            merge(store, it, 'wa')
    add_p = os.path.join(a.state, 'added.jsonl')
    for ln in (open(add_p) if os.path.exists(add_p) else []):
        if not ln.strip():
            continue
        it = json.loads(ln)
        if it.get('id') in store:
            store[it['id']].update({k: v for k, v in it.items() if k in ('text', 'from', 'to')})
            continue
        k = it.get('id') or item_id(it)
        if k not in store:
            store[k] = dict(it, id=k, link=it.get('link', ''), by_hand=True, first_seen=int(time.time()), last_seen=int(time.time()))
    json.dump(store, open(store_p + '.tmp', 'w'), ensure_ascii=False, indent=0)
    os.replace(store_p + '.tmp', store_p)
    gz = Gaz(a.gaz, a.stops)
    man_p = os.path.join(a.state, 'manual.json')
    manual = json.load(open(man_p)) if os.path.exists(man_p) else {}
    items = []
    for k, it in sorted(store.items(), key=lambda kv: -kv[1]['pub']):
        if it.get('hide'):
            continue
        f, t, how = (it['from'], it['to'], 'given') if it.get('from') else period(it)
        geom, conf, names = locate(gz, it)
        rec = {'id': k, 'kind': it['kind'], 'title': re.sub(r'^\W+', '', it['title']).strip(), 'text': it['text'], 'link': it['link'],
               'pub': it['pub'], 'from': f, 'to': t, 'when': how, 'image': it.get('image'), 'names': names, 'conf': conf, 'geom': geom}
        rec.update(manual.get(k, {}))
        items.append(rec)
    link_alts(items, a.alts, a.dodges)
    ok = [json.loads(l) for l in open(os.path.join(a.state, 'fetch-log.jsonl'))][-2000:]
    week = [r for r in ok if r['t'] > time.time() - 7 * 86400 and 'delhipolice' in r['url']]
    health = {'tries_7d': len(week), 'ok_7d': sum(r['code'] == 200 for r in week), 'e503_7d': sum(r['code'] == 503 for r in week),
              'last_ok': max([r['t'] for r in ok if r['code'] == 200 and 'delhipolice' in r['url']] or [0])}
    out = {'t': int(time.time()), 'reached': reached, 'sources': len(SOURCES), 'health': health, 'items': items}
    json.dump(out, open(a.out + '.tmp', 'w'), ensure_ascii=False, separators=(',', ':'))
    os.replace(a.out + '.tmp', a.out)
    log(f'{len(items)} posts, {sum(1 for i in items if i["geom"])} located, {sum(n for i in items for _, n in i.get("alts", []))} detour links -> {a.out}')


if __name__ == '__main__':
    main()
