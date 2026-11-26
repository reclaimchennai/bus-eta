"""Write out/map/index.html: an interactive map of every geocoded incident (data embedded).

  python -m pipeline.make_map
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/map"

COLORS = {
    "accident": "#d62728", "breakdown": "#ff7f0e", "spillage": "#8c564b", "waterlogging": "#1f77b4",
    "fire": "#e377c2", "tree_fall": "#2ca02c", "closure": "#111111", "diversion": "#9467bd",
    "u_turn_change": "#17becf", "works": "#bcbd22", "event": "#7f7f7f", "congestion": "#f2b701", "other": "#aaaaaa",
}
LABELS = {
    "accident": "Crash", "breakdown": "Breakdown", "spillage": "Spillage", "waterlogging": "Waterlogging",
    "fire": "Fire", "tree_fall": "Tree fall", "closure": "Road closure", "diversion": "Diversion",
    "u_turn_change": "U-turn change", "works": "Road works", "event": "Event / VIP / rally",
    "congestion": "Slow movement", "other": "Other",
}

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hyderabad Traffic Incidents</title>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--panel:#ffffffee;--line:#e3e3e3;--accent:#2b6cb0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121212;--fg:#eee;--muted:#aaa;--panel:#1c1c1cee;--line:#333;--accent:#7fb3ff}}
:root[data-theme="dark"]{--bg:#121212;--fg:#eee;--muted:#aaa;--panel:#1c1c1cee;--line:#333;--accent:#7fb3ff}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
#map{position:absolute;inset:0}
#panel{position:absolute;top:10px;left:10px;width:300px;max-height:calc(100% - 20px);overflow:auto;background:var(--panel);
 border:1px solid var(--line);border-radius:10px;padding:12px 14px;box-sizing:border-box;backdrop-filter:blur(6px)}
h1{font-size:15px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin-bottom:8px}
.row{display:flex;align-items:center;gap:6px;margin:2px 0;cursor:pointer}.sw{width:11px;height:11px;border-radius:50%;flex:none}
.n{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
label.t{display:block;margin:10px 0 3px;font-weight:600;font-size:12px}
input[type=range]{width:100%}input[type=search]{width:100%;box-sizing:border-box;padding:5px 7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:4px 8px;cursor:pointer;font-size:12px}
button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#hot{margin:4px 0 0;padding-left:18px}#hot li{cursor:pointer}
.maplibregl-popup-content{background:var(--bg);color:var(--fg);max-width:320px;font-size:12px}
.pt{font-weight:600}.pm{color:var(--muted);font-size:11px;margin:2px 0 6px}.ptext{max-height:150px;overflow:auto;white-space:pre-wrap}
a{color:var(--accent)}
@media (max-width:640px){#panel{width:auto;right:10px;max-height:45%}}
</style></head><body>
<div id="map"></div>
<div id="panel">
 <h1>Hyderabad &amp; Cyberabad traffic incidents</h1>
 <div class="sub" id="sub"></div>
 <input type="search" id="q" placeholder="Search place, road, text…">
 <label class="t">Period: <span id="pl"></span></label>
 <input type="range" id="from" min="0" max="0" value="0"><input type="range" id="to" min="0" max="0" value="0">
 <label class="t">Hour of day: <span id="hl"></span></label>
 <input type="range" id="h0" min="0" max="23" value="0"><input type="range" id="h1" min="0" max="23" value="23">
 <label class="t">Type <span style="font-weight:400;color:var(--muted)">(click to toggle)</span></label>
 <div id="types"></div>
 <div class="btns"><button id="heat">Heatmap</button><button id="lines" class="on">Road segments</button><button id="all">All types</button></div>
 <label class="t">Hotspots (current filter)</label><ol id="hot"></ol>
 <div class="sub" style="margin-top:8px">Sources: @CYBTRAFFIC and @HYDTP posts, Cyberabad Traffic Pulse WhatsApp, HTP e-challan alerts · Basemap © OpenStreetMap, OpenFreeMap</div>
</div>
<script>
const DATA=__DATA__, COLORS=__COLORS__, LABELS=__LABELS__;
const pts=DATA.features.filter(f=>f.geometry.type==="Point"), lns=DATA.features.filter(f=>f.geometry.type!=="Point");
const months=[...new Set(pts.map(f=>f.properties.t_start_ist.slice(0,7)))].sort();
const st={types:new Set(Object.keys(COLORS)),from:0,to:months.length-1,h0:0,h1:23,q:"",heat:false,lines:true};
const $=id=>document.getElementById(id);
["from","to"].forEach(k=>{$(k).max=months.length-1});$("to").value=months.length-1;
// default to the last 12 months that have data
st.from=Math.max(0,months.length-12);$("from").value=st.from;
function keep(f){const p=f.properties,m=p.t_start_ist.slice(0,7),h=+p.t_start_ist.slice(11,13);
 if(!st.types.has(p.type))return false;if(m<months[st.from]||m>months[st.to])return false;
 if(h<st.h0||h>st.h1)return false;
 if(st.q){const s=(p.text_en+" "+(p.place||"")+" "+(p.police_station||"")+" "+(p.direction||"")).toLowerCase();if(!s.includes(st.q))return false}
 return true}
const map=new maplibregl.Map({container:"map",style:"https://tiles.openfreemap.org/styles/positron",center:[78.38,17.45],zoom:11.2,attributionControl:{compact:true}});
map.addControl(new maplibregl.NavigationControl(),"top-right");
const colorExpr=["match",["get","type"],...Object.entries(COLORS).flat(),"#888"];
function fc(a){return{type:"FeatureCollection",features:a}}
function refresh(){
 const P=pts.filter(keep),ids=new Set(P.map(f=>f.properties.id));
 const L=lns.filter(f=>ids.has(f.properties.id.replace(/:line$/,"")));
 if(map.getSource("p")){map.getSource("p").setData(fc(P));map.getSource("l").setData(fc(L))}
 $("sub").textContent=`${P.length.toLocaleString()} of ${pts.length.toLocaleString()} incidents · ${L.length} road segments`;
 $("pl").textContent=`${months[st.from]} → ${months[st.to]}`;$("hl").textContent=`${st.h0}:00–${st.h1}:59 IST`;
 const cnt={};P.forEach(f=>cnt[f.properties.type]=(cnt[f.properties.type]||0)+1);
 $("types").innerHTML=Object.keys(COLORS).filter(t=>pts.some(f=>f.properties.type===t)).map(t=>`<div class="row" data-t="${t}" style="opacity:${st.types.has(t)?1:.35}"><span class="sw" style="background:${COLORS[t]}"></span>${LABELS[t]}<span class="n">${cnt[t]||0}</span></div>`).join("");
 document.querySelectorAll("#types .row").forEach(r=>r.onclick=()=>{const t=r.dataset.t;st.types.has(t)?st.types.delete(t):st.types.add(t);refresh()});
 const hp={};P.forEach(f=>{const k=f.properties.place;if(!k)return;(hp[k]=hp[k]||{n:0,c:f.geometry.coordinates}).n++});
 $("hot").innerHTML=Object.entries(hp).sort((a,b)=>b[1].n-a[1].n).slice(0,10).map(([k,v])=>`<li data-c="${v.c}">${k} <span class="n">${v.n}</span></li>`).join("");
 document.querySelectorAll("#hot li").forEach(li=>li.onclick=()=>map.flyTo({center:li.dataset.c.split(",").map(Number),zoom:15}));
 if(map.getLayer("pc")){map.setLayoutProperty("heat","visibility",st.heat?"visible":"none");map.setLayoutProperty("pc","visibility",st.heat?"none":"visible");
  map.setLayoutProperty("lc","visibility",st.lines?"visible":"none")}
 $("heat").classList.toggle("on",st.heat);$("lines").classList.toggle("on",st.lines);
}
map.on("load",()=>{
 map.addSource("l",{type:"geojson",data:fc([])});map.addSource("p",{type:"geojson",data:fc([])});
 map.addLayer({id:"lc",type:"line",source:"l",paint:{"line-color":colorExpr,"line-width":["interpolate",["linear"],["zoom"],10,1.5,15,4],"line-opacity":.55},layout:{"line-cap":"round"}});
 map.addLayer({id:"heat",type:"heatmap",source:"p",layout:{visibility:"none"},paint:{"heatmap-radius":["interpolate",["linear"],["zoom"],9,8,15,30],"heatmap-intensity":["interpolate",["linear"],["zoom"],9,.6,15,1.5],"heatmap-opacity":.8}});
 map.addLayer({id:"pc",type:"circle",source:"p",paint:{"circle-color":colorExpr,"circle-radius":["interpolate",["linear"],["zoom"],9,["+",2,["*",3,["get","severity"]]],15,["+",5,["*",7,["get","severity"]]]],"circle-stroke-color":"#fff","circle-stroke-width":.7,"circle-opacity":.85}});
 map.on("click","pc",e=>{const fs=e.features.slice(0,6);new maplibregl.Popup({maxWidth:"340px"}).setLngLat(e.lngLat).setHTML(fs.map(f=>{const p=f.properties,r=JSON.parse(p.route_ids||"[]");
  return `<div class="pt" style="color:${COLORS[p.type]}">${LABELS[p.type]}${p.place?" · "+p.place:""}</div><div class="pm">${p.t_start_ist} IST · ${p.police_station||""} ${p.direction?"· "+p.direction:""} · severity ${p.severity}${r.length?"<br>TGSRTC routes: "+r.slice(0,15).join(", "):""}</div><div class="ptext">${p.text_en.replace(/</g,"&lt;")}</div>${p.src?`<a href="${p.src}" target="_blank" rel="noopener">source</a>`:""}`}).join("<hr>")).addTo(map)});
 map.on("mouseenter","pc",()=>map.getCanvas().style.cursor="pointer");map.on("mouseleave","pc",()=>map.getCanvas().style.cursor="");
 refresh()});
["from","to","h0","h1"].forEach(k=>$(k).oninput=()=>{st[k]=+$(k).value;if(st.from>st.to)[st.from,st.to]=[st.to,st.from];if(st.h0>st.h1)[st.h0,st.h1]=[st.h1,st.h0];refresh()});
$("q").oninput=()=>{st.q=$("q").value.trim().toLowerCase();refresh()};
$("heat").onclick=()=>{st.heat=!st.heat;refresh()};$("lines").onclick=()=>{st.lines=!st.lines;refresh()};
$("all").onclick=()=>{st.types=new Set(Object.keys(COLORS));refresh()};
</script></body></html>"""

def main():
    data = json.loads((OUT / "incidents_all.geojson").read_text())
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))) \
               .replace("__COLORS__", json.dumps(COLORS)).replace("__LABELS__", json.dumps(LABELS))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("wrote", OUT / "index.html", f"{len(html)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
