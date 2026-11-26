"""Is a post about something affecting road movement? (same rules used on the X scrape)"""
import re
INCIDENT = re.compile(r"trafficalert|(vehicle|vehicular) movements? (is|are) (currently |presently |slightly |very )?slow|movements? of (the )?vehicles (is|are) (currently |presently |slightly )?slow|slow (vehicular|vehicle) movement|road damage|pothole|(procession|shobha ?yatra|yatra|rally|nimajjanam)\b.{0,60}\b(passing|entered|reached)|traffic (movement|alert|advisory|update|diversion)|diver(sion|ted)|road (closed|closure)|congestion (is|on|at|near|expected)|"
 r"broke(n)?[\s-]?down|tree fell|congestion alert|traffic is slow|congestion due to|(traffic police|ps) update|heavy (traffic )?congestion|spillage|water[\s-]?logg|overturn|uprooted|fire incident|"
 r"(accident|breakdown|broke down|collision|crash)\b.{0,60}\b(at|near|on|towards|from)\b|u[- ]?turn (is |will be )?(closed|opened|shifted|blocked)|"
 r"Date: ?\d|slow (at|near|on)|traffic (is|will be) (slow|diverted|restricted|stopped)|not (be )?allowed|alternate route", re.I)
AWARE = re.compile(r"#?roadsafety|never |always |stay safe|remember|awareness|cyberabad moves|helmet|drunk|drink|seat ?belt|"
 r"wrong[- ]side|challan|tips|campaign|do you know|be a responsible|habit|subscription|opt[- ]?out|subscribe", re.I)
STRONG = re.compile(r"trafficalert|(vehicle|vehicular) movements? (is|are) (currently |presently |slightly |very )?slow|movements? of (the )?vehicles (is|are) (currently |presently |slightly )?slow|slow (vehicular|vehicle) movement|road damage|pothole|(procession|shobha ?yatra|yatra|rally|nimajjanam)\b.{0,60}\b(passing|entered|reached)|Date: ?\d|traffic (alert|advisory|update|diversion|movement)|divert|broke(n)? down (at|near|on|under)|tree fell|congestion\s+at|traffic ps:|spillage|water[\s-]?logg|"
 r"congestion alert|traffic is slow|congestion due to|(traffic police|ps) update|heavy (traffic )?congestion|u[- ]?turn", re.I)
CLEAR_UPDATE = re.compile(r"\b(cleared|restored|normali[sz]ed|resumed|back to normal|moving smoothly)\b.*|.*\b(cleared|restored|normali[sz]ed|resumed|back to normal|moving smoothly)\b", re.I | re.S)
INCIDENT_WORD = re.compile(r"break[\s-]?down|broke(n)?[\s-]?down|accident|spill|water[\s-]?logg|tree|congestion|traffic|diversion|closure|u[- ]?turn", re.I)
# officials inspecting / reviewing / meeting / campaigning: not a live condition on the road
OFFICIAL = re.compile(r"\binspect|review meeting|field visit|\bvisited\b|stakeholder|meeting (with|regarding|on)|conducted a (meeting|review)|"
 r"\battended\b|traffic monitoring|open discussion|felicitat|launched|route guide|peako|press meet|preparedness|coordination meeting|"
 r"special drive|enforcement|awareness programme|campaign", re.I)
ALERT_HEAD = re.compile(r"traffic (update|movement is)|ps update|traffic ps\s*[:-]|vehicle breakdown|movement is slow|traffic is slow|traffic alert|trafficalert|traffic advisory|diver(sion|ted)|road closed|breakdown alert|water[\s-]?logg", re.I)

def is_road_event(text):
    if re.search(r"subscription|opt[- ]?out", text, re.I): return False
    if OFFICIAL.search(text[:300]) and not ALERT_HEAD.search(text[:200]): return False
    if CLEAR_UPDATE.search(text) and INCIDENT_WORD.search(text) and not AWARE.search(text):
        return True                      # "... has been cleared" updates close open events
    return bool(INCIDENT.search(text)) and (not AWARE.search(text) or bool(STRONG.search(text)))
