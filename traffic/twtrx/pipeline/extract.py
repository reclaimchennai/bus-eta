"""Pull location mentions (place, from->to, direction) out of a traffic post."""
import re, html
STOP = r"(?=\s+(?:on the occasion|along with|coupled with|involving|as well as|due|because|owing|as|and\s+(?:our|traffic|commuters)|commuters|motorists|our|please|kindly|traffic|vehicles?|where|which|is|are|has|have|till|until|for|during|while|so|will|shall|would|was|were|has been|had|be|being|in view|from \d|via)\b|[.,;:!\n()|]|$)"
PLACE = r"([A-Z0-9][\w’'&\-/. ]{1,60}?)"
PATS = [
 ("at_from_to", re.compile(r"\b(?:at|on|near)\s+(?:the\s+)?"+PLACE+r"\s+from\s+"+PLACE+r"\s+(?:to|towards|twds?)\s+"+PLACE+STOP, re.I)),
 ("from_to", re.compile(r"\bfrom\s+"+PLACE+r"\s+(?:to|towards|till|up ?to)\s+"+PLACE+STOP, re.I)),
 ("from_to", re.compile(PLACE+r"\s*(?:→|➡️|➡|->|=>)\s*"+PLACE+STOP, re.I)),
 ("at_towards", re.compile(r"\b(?:at|near|on|in front of|opposite|beside)\s+"+PLACE+r"\s+(?:(?:towards|twds?|to)\s+|\s*(?:→|➡️|➡|->)\s*)"+PLACE+STOP, re.I)),
 ("at_from", re.compile(r"\b(?:at|near|on)\s+"+PLACE+r"\s+from\s+"+PLACE+STOP, re.I)),
 ("at", re.compile(r"\b(?:at|near|on|in front of|opposite|beside|under|below)\s+(?:the\s+)?"+PLACE+STOP, re.I)),
 ("pin", re.compile(r"(?:📍|Location\s*:)\s*"+PLACE+STOP, re.I)),
]
JUNK = re.compile(r"^(pillar no.*|road no\.?|need basis|\d+\s*[ap]\.?|around \d.*|side|sides|all routes.*|both sides.*|both directions.*|vice versa|h-citi.*|occasion.*|route|routes|bridge|leadership.*|the|a|an|this|that|road|roads|time|present|present time|around|about|\d+[\d:.\s]*(hrs|am|pm)?|hrs|night|morning|evening|duty|ground|spot|site|place|location|both sides|all|least|once|risk|your|our|x|twitter|instagram|facebook|heavy|earliest|congestion|heavy congestion|vehicle breakdown|vehicle break down|traffic|water logging|waterlogging|this stretch|heavy vehicular flow|opp|signal|junction|ground.*|proceed.*|duty|roads?|stretch|route|way|road accident|accident|time|hrs|spot|the spot|priority|caution|alert|immediate.*|regular.*|weekends?|weekdays?|night.*|morning.*|evening.*|peak.*|all times|any time|the earliest|least|risk|ease|smooth.*|control|speed)$", re.I)
TRAIL = re.compile(r"\s+(due|opp|towards|twd|from|to|and|the|side|both|area|road side|in|of|on|at|near)$", re.I)
def clean(p):
    p = re.sub(r"\s+", " ", p).strip(" -–.'’/&:")
    while TRAIL.search(p): p = TRAIL.sub("", p)
    p = re.sub(r"^(the|a)\s+", "", p, flags=re.I)
    p = re.sub(r"^(?:the\s+)?(?:starting|ending|end) point of (?:the\s+)?", "", p, flags=re.I)
    p = re.sub(r"\s+(?:starting|ending) point$", "", p, flags=re.I)
    return p
NOISE = re.compile(r"\b(clear|cleared|affected|ensure|ease|several|following|left turn|right turn|wet|waterlogged|necessary|damage|field|spot|measures|steps|congestion|movement|vehicular|stretch(es)?|locations|landing|starting|both ?sides|commuters|officers|alternate|caution|delays?|situation|traffic)\b", re.I)
PINLINE = re.compile(r"^[ \t]*📍[ \t]*(.+?)[ \t]*$", re.M)
CHAIN_SEP = re.compile(r"\s+(?:–|—|-|→|➡️|➡|->)\s+|\s+(?:towards|twds?|to)\s+", re.I)

def pin_paths(t):
    """'📍 Omega Hospital – Taj Mahal Hotel – BVB towards Road No. 45' -> path mention (waypoints in order)."""
    out, spans = [], []
    for m in PINLINE.finditer(t):
        line = m.group(1)
        if line.rstrip().endswith(":"):            # "📍 Route:" header -> the stops are on the next line
            nxt = re.match(r"\s*\n\s*([^\n]+)", t[m.end():])
            if not nxt: continue
            line = nxt.group(1).rstrip(" .")
            spans.append((m.end(), m.end() + nxt.end()))
        line = re.sub(r"\s*\((?:both sides|both directions)\)|\s+on both sides$", "", line, flags=re.I)
        parts = [clean(x) for x in CHAIN_SEP.split(line)]
        parts = [x for x in parts if x and len(x) >= 3 and not JUNK.match(x)]
        if len(parts) >= 2:
            out.append({"kind": "path", "places": parts}); spans.append(m.span())
        elif parts:
            out.append({"kind": "at", "places": parts}); spans.append(m.span())
    return out, spans

ARROW = re.compile(r"([A-Z][\w’'.& ]{2,40}?)[ \t]*(?:→|➡️?|-+>)[ \t]*([A-Z][\w’'.& ]{2,40}?)(?=[ \t]*(?:[^\w\s’'.&]|\n|$))")

def arrow_paths(t, spans):
    """'Parry’s Corner → Broadway', 'Butt Road → Kathipara Bridge' (Chennai): a stretch, in the direction of travel."""
    out = []
    for m in ARROW.finditer(t):
        if any(a <= m.start() < b for a, b in spans): continue
        parts = [clean(x) for x in m.groups()]
        if all(_ok(x) for x in parts):
            out.append({"kind": "path", "places": parts}); spans.append(m.span())
    return out

END = r"(?=\s*(?:[.;:\n!]|,?\s+(?:due|because|as well as|coupled|along|resulting|causing|commuters|motorists|and\s+(?:vice|from|on|in)|on both|in both|is|are|has|have|will|for|while|where|which))|$)"
LIST_FROM = re.compile(r"\bfrom\s+((?:(?!\bfrom\b)[^.;:\n!]){3,220}?)\s+(?:towards|twds?\.?|to|up ?to|till)\s+((?:the\s+)?[^.;:\n!,]{3,80}?)" + END, re.I)
END2 = r"(?=\s*[.,;:\n!]|\s+(?:due|because|resulting|causing|commuters|is|are|has|have|will|for|while)\b|$)"
BETWEEN = re.compile(r"\bbetween\s+([^.;:\n!,]{3,60}?)\s+(?:and|&)\s+([^.;:\n!,]{3,60}?)" + END2, re.I)
THROUGH = re.compile(r"\bpassing\s+(?:through\s+)?([^.;:\n!,]{3,60}?)\s+(?:towards|twds?|to)\s+([^.;:\n!,]{3,60}?)" + END2, re.I)
SPLIT_LIST = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*", re.I)

def _ok(p):
    return p and len(p) >= 3 and not JUNK.match(p) and not (NOISE.search(p) and not re.search(r"flyover|junction|rotary|x ?roads?", p, re.I))

def list_paths(t, spans):
    """'from A, B and C towards D' (route through all), 'between X and Y', 'passing through X towards Y'."""
    out = []
    for kind, rx in (("between", BETWEEN), ("through", THROUGH), ("list", LIST_FROM)):
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in spans): continue
            if kind == "list":
                src = [clean(x) for x in SPLIT_LIST.split(m.group(1))]
                if len(src) < 2: continue                     # the plain from->to patterns handle one origin
                parts = [x for x in src if _ok(x)] + [clean(m.group(2))]
            else:
                parts = [clean(m.group(1)), clean(m.group(2))]
            parts = [x for x in parts if _ok(x)]
            if len(parts) >= 2:
                out.append({"kind": "path", "places": parts}); spans.append(m.span())
    return out

AROUND = [("at", re.compile(r"\b(?:around|along)\s+(?:the\s+)?" + PLACE + STOP, re.I))]

def extract(text, arrows=False):
    """arrows: also read 'A → B' as a stretch, and 'around / along X' as a place (Chennai and Bengaluru posts;
    Hyderabad uses arrows for diversion routes)."""
    t = html.unescape(text); t = re.sub(r"https?://\S+|[*#]", " ", t)
    t = re.sub(r"\s*\((?!\s*both)[^()\n]{2,40}\)", "", t)          # "Lepakshi (Upper Tank Bund)" -> "Lepakshi"
    t = re.sub(r"\b(?:opp\.?|opposite)\s+", "", t, flags=re.I)
    t = re.sub(r"\b(No|Nos|Jn|Rd|St|Opp)\.\s*", r"\1 ", t)                # "6 No. Amberpet", "Road No. 45"
    out, spans = pin_paths(t)
    out += list_paths(t, spans)
    if arrows: out += arrow_paths(t, spans)
    for kind, rx in PATS + (AROUND if arrows else []):
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in spans): continue
            parts = [clean(g) for g in m.groups()]
            parts = [re.sub(r"\s+(on )?both ?sides$|\s+(starting|landing)( point)?$", "", p, flags=re.I) for p in parts]
            if any(not p or JUNK.match(p) or len(p) < 3 or (NOISE.search(p) and not re.search(r"flyover|junction|rotary|x ?roads?", p, re.I)) for p in parts): continue
            spans.append(m.span()); mm = {"kind": kind, "places": parts}
            v = re.match(r"\s+via\s+" + PLACE + STOP, t[m.end():], re.I)
            if v and not JUNK.match(clean(v.group(1))): mm["via"] = clean(v.group(1))
            out.append(mm)
    ps = re.search(r"([A-Za-z &/]+?)\s*Traffic\s*(?:PS|Police Station)", t, re.I)
    return {"mentions": out, "traffic_ps": clean(ps.group(1)) if ps else ""}
