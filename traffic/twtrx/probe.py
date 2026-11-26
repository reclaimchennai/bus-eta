"""Date-windowed search to fill gaps. Usage: python3 probe.py SINCE UNTIL MAXPAGES"""
import json,glob,sys,time
from pathlib import Path
import scrape
since,until,maxp=sys.argv[1],sys.argv[2],int(sys.argv[3])
seen={t['id'] for f in glob.glob('raw/*.json') for t in (json.load(open(f)).get('tweets') or [])}
key=scrape.load_key(); q=scrape.QUERY+f" since:{since} until:{until}"
prev=sorted(glob.glob(f"raw/win_{since}_{until}_*.json"),key=lambda f:int(f.rsplit('_',1)[1][:-5]))
start=len(prev); cur=json.load(open(prev[-1])).get('next_cursor') if prev else None
if prev and not cur: sys.exit('window exhausted')
for i in range(start,start+maxp):
    body={"query":q,"order":"latest"}
    if cur: body["next_cursor"]=cur
    d,rem=scrape.post(key,body)
    Path(f"raw/win_{since}_{until}_{i+1}.json").write_text(json.dumps(d,ensure_ascii=False))
    tw=d.get('tweets') or []; new=[t for t in tw if t['id'] not in seen]; seen|={t['id'] for t in tw}
    print(f"{since}..{until} p{i+1}: {len(tw)} tweets, {len(new)} new, oldest {tw[-1]['created_at'] if tw else '-'}, remaining {rem}")
    nc=d.get('next_cursor')
    if not tw or not nc or nc==cur or str(nc).startswith('0'): break
    cur=nc; time.sleep(0.3)
