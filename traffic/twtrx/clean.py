import json,glob,re,csv,html
from datetime import datetime,timezone,timedelta
IST=timezone(timedelta(hours=5,minutes=30))
seen={}
for f in glob.glob('raw/*.json'):
    for t in json.load(open(f)).get('tweets') or []:
        if t['user']['username'].lower()=='cybtraffic' and not t.get('retweeted_status'): seen[t['id']]=t
INCIDENT=re.compile(r"traffic (movement|alert|advisory|update|diversion)|diver(sion|ted)|road (closed|closure)|congestion (is|on|at|near|expected)|"
 r"broke(n)? down|tree fell|congestion alert|traffic is slow|congestion due to|broke(n)? down (under|at|near|on)|(traffic police|ps) update|heavy (traffic )?congestion|spillage|water ?logg|overturn|uprooted|fire incident|(accident|breakdown|broke down|collision|crash)\b.{0,60}\b(at|near|on|towards|from)\b|"
 r"Date: ?\d|slow (at|near|on)|traffic (is|will be) (slow|diverted|restricted|stopped)|not (be )?allowed|alternate route", re.I)
AWARE=re.compile(r"#?roadsafety|never |always |stay safe|remember|awareness|cyberabad moves|helmet|drunk|drink|seat ?belt|"
 r"wrong[- ]side|challan|tips|campaign|do you know|be a responsible|habit", re.I)
CATS=[("Accident/Crash",r"accident|crash|collision|overturn|hit by|rammed|skid"),("Breakdown",r"break ?down|broke down"),
 ("Spillage",r"spill"),("Waterlogging/Rain",r"water ?logg|rain|flood|inundat"),("Fire",r"fire"),("Tree fall",r"tree|uproot"),
 ("Diversion/Closure",r"divert|diversion|road clos|not (be )?allowed|restrict"),("Works",r"works|construction|repair|pipeline|metro"),
 ("Event/VIP",r"rally|procession|vip|festival|match|event|ganesh|bandobast|meeting"),("Slow movement",r"traffic movement|slow|congestion")]
keep,drop=[],[]
for t in seen.values():
    x=html.unescape(t['full_text'])
    inc=bool(INCIDENT.search(x)); aw=bool(AWARE.search(x))
    # awareness posts need a strong, location-specific incident marker to stay
    ok = inc and (not aw or re.search(r"Date: ?\d|traffic (alert|advisory|update|diversion|movement)|divert|broke(n)? down (at|near|on)|tree fell|congestion\s+at|traffic ps:|spillage|water ?logg|congestion alert|traffic is slow|congestion due to|broke(n)? down (under|at|near|on)|(traffic police|ps) update|heavy (traffic )?congestion",x,re.I))
    (keep if ok else drop).append((t,x))
def row(t,x):
    dt=datetime.strptime(t['created_at'],'%a %b %d %H:%M:%S %z %Y').astimezone(IST)
    m=re.search(r"([A-Za-z &/]+?) ?Traffic PS",x)
    return {"id":t['id'],"datetime_ist":dt.strftime('%Y-%m-%d %H:%M'),
     "category":"; ".join(c for c,p in CATS if re.search(p,x,re.I)) or "Other",
     "traffic_ps":m.group(1).strip() if m else "","text":x,"views":t.get('view_count'),"likes":t.get('likes_count'),
     "reposts":t.get('retweet_count'),"replies":t.get('reply_count'),
     "media":" ".join(e.get('preview') or e.get('link') or '' for e in t.get('entities') or []),
     "url":f"https://x.com/CYBTRAFFIC/status/{t['id']}"}
rows=sorted((row(*k) for k in keep),key=lambda r:r['datetime_ist'],reverse=True)
with open('cybtraffic_road_incidents.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
json.dump(rows,open('cybtraffic_road_incidents.json','w'),ensure_ascii=False,indent=1)
json.dump(sorted((row(*d) for d in drop),key=lambda r:r['datetime_ist']),open('excluded_non_traffic.json','w'),ensure_ascii=False,indent=1)
print('unique',len(seen),'kept',len(keep),'dropped',len(drop),'range',rows[-1]['datetime_ist'],'->',rows[0]['datetime_ist'])
