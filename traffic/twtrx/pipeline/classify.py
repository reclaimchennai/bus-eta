"""Classify a traffic post: event type, severity, expected duration, time window.

Types line up with ETA-wt-models/src/llm_disruption_sidecar.py (`sidecar_type`),
so the same record can feed that sidecar's corridor-risk table.
"""
import re
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# (type, regex, sidecar_type, severity, default_minutes, speed_multiplier)
# Ordered by priority: the first match is the primary type; all matches are tags.
TYPES = [
    ("u_turn_change", r"u[- ]?turn",                                                            "other",              0.3, None, 0.85),
    ("closure",      r"road (is )?closed|road closure|closed for (all )?(vehicular )?traffic|(will|shall) not be allowed|no entry|blocked", "road_closure", 0.9, 240, 0.30),
    ("diversion",    r"divert|diversion|alternate route|alternative route",                    "road_closure",       0.8, 240, 0.45),
    ("accident",     r"accident|collision|crash|overturn|skid|rammed|hit by|fatal|injur",       "breakdown_accident", 0.65, 60, 0.50),
    ("fire",         r"fire incident|fire accident|caught fire|\bfire\b",                       "other",              0.65, 90, 0.50),
    ("waterlogging", r"water[\s-]?logg|inundat|flood|submerg|heavy rain",                            "waterlogging",       0.75, 180, 0.45),
    ("tree_fall",    r"tree (fell|fall|fallen|uproot)|uprooted|fallen tree|branch",              "other",              0.55, 90, 0.55),
    ("spillage",     r"spill",                                                                  "other",              0.55, 60, 0.55),
    ("breakdown",    r"break[\s-]?down|broke(n)?[\s-]?down|stalled|tyre burst|puncture",                 "breakdown_accident", 0.5, 45, 0.60),
    ("event",        r"rally|procession|vip|v\.i\.p|dignitar|convoy|concert|match|festival|ganesh|immersion|nimajjanam|marathon|\b(5|10|21)k\b|\brun\b|bandobast|celebration|sabha|public meeting|bonalu|bathukamma|new year", "rally_procession", 0.6, 240, 0.55),
    ("works",        r"works?\b|construction|repair|pipeline|widening|metro|road laying|digging|excavat|maintenance", "construction", 0.5, 480, 0.65),
    ("congestion",   r"traffic movement is slow|movement is slow|traffic is slow|slow[- ]moving|congestion|heavy (vehicular|traffic)|traffic jam|high traffic volume", "other", 0.4, 45, 0.60),
]
_TYPES = [(t, re.compile(rx, re.I), s, sev, d, sm) for t, rx, s, sev, d, sm in TYPES]

CLEARED = re.compile(r"\b(cleared|restored|normali[sz]ed|resumed|back to normal|now normal|removed|moving smoothly|smooth(ly)? flow)", re.I)
SEV_UP = re.compile(r"both (sides|directions)|heavy|complete(ly)?|fatal|severe|long queue|standstill|huge", re.I)
SEV_DOWN = re.compile(r"no traffic (impact|disruption|congestion)|minor|partial", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")

def _dt(d, mth, y, hh=0, mm=0):
    y = int(y); y = y + 2000 if y < 100 else y
    return datetime(y, int(mth), int(d), int(hh), int(mm), tzinfo=IST)

DATE = r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})"
TIME = r"(\d{1,2})[:.](\d{2})\s*(?:hrs|hours|h)?\s*(am|pm)?"

def _hm(h, m, ap):
    h, m = int(h), int(m)
    if ap and ap.lower() == "pm" and h < 12: h += 12
    if ap and ap.lower() == "am" and h == 12: h = 0
    return h, m

def parse_window(text, posted):
    """Return (start, end, stated) in IST. stated=True when the post gives an end."""
    t = text
    # "Date: 22-11-2026, Time: 13:00 hrs" -> observation time
    m = re.search(r"Date\s*:?\s*" + DATE + r".{0,15}?Time\s*:?\s*" + TIME, t, re.I | re.S)
    if m:
        d, mo, y, h, mi, ap = m.groups(); h, mi = _hm(h, mi, ap)
        try: return _dt(d, mo, y, h, mi), None, False
        except ValueError: pass
    dates = [ _safe(*g) for g in re.findall(DATE, t) ]
    dates = [d for d in dates if d and abs((d.date() - posted.date()).days) < 60]
    tm = re.search(r"from\s+" + TIME + r"\s*(?:to|till|-|–)\s*" + TIME, t, re.I)
    if dates or tm:
        day0 = dates[0] if dates else posted.replace(hour=0, minute=0, second=0, microsecond=0)
        day1 = dates[-1] if len(dates) > 1 else day0
        if tm:
            h1, m1 = _hm(*tm.groups()[0:3]); h2, m2 = _hm(*tm.groups()[3:6])
            s = day0.replace(hour=h1, minute=m1); e = day1.replace(hour=h2, minute=m2)
            if e <= s: e += timedelta(days=1)
            return s, e, True
        if len(dates) > 1 and day1 > day0:
            return day0, day1 + timedelta(days=1), True
        if day0.date() > posted.date():          # advisory for a future day, no times
            return day0.replace(hour=6), day0.replace(hour=23, minute=59), True
    fd = re.search(r"\bfor\s+(?:about\s+|next\s+)?(\d{1,3})\s*(min|mins|minutes|hrs?|hours?)\b", t, re.I)
    if fd:                                  # "for 30Min" -> explicit duration
        n = int(fd.group(1)); mins = n if fd.group(2).lower().startswith("m") else n * 60
        return posted, posted + timedelta(minutes=mins), True
    return posted, None, False

def _safe(d, mo, y):
    try: return _dt(d, mo, y)
    except ValueError: return None

def classify(text, posted):
    """posted: aware datetime. Returns dict of classification fields."""
    tags = [(t, s, sev, d, sm) for t, rx, s, sev, d, sm in _TYPES if rx.search(text)]
    if not tags:
        tags = [("other", "other", 0.3, 45, 0.8)]
    typ, side, sev, dur, sm = tags[0]
    if SEV_UP.search(text): sev = min(1.0, sev + 0.1)
    if SEV_DOWN.search(text): sev = max(0.1, sev - 0.2)
    status = "cleared" if CLEARED.search(text) and typ not in ("closure", "diversion", "u_turn_change") else "active"
    start, end, stated = parse_window(text, posted)
    if typ == "u_turn_change" and not stated:
        kind = "network_change"           # a lasting change to the road network
    elif stated:
        kind = "scheduled"
    else:
        kind = "unplanned"
    if end is None and dur is not None:
        exp = start + timedelta(minutes=dur)
    else:
        exp = end
    return {
        "type": typ, "tags": [t[0] for t in tags], "sidecar_type": side,
        "severity": round(sev, 2), "speed_multiplier": sm,
        "default_duration_min": dur, "status": status, "kind": kind,
        "start": start, "end": end, "expires": exp, "end_stated": stated,
    }

def scrub(text):
    """Remove phone numbers (no personal data downstream)."""
    return PHONE.sub("[phone]", text)
