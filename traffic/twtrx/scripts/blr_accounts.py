#!/usr/bin/env python3
"""Bengaluru Traffic Police accounts on X -> data/blr/btp_accounts.csv (+ .md table for the README).

Role and station are hand-assigned below (station names as in BTP's own list, data/blr/powerbi/police_stations.csv,
which also gives sub-division and zone). Activity comes from the Nitter profile probes in raw/bengaluru/probe_*.jsonl
(scripts/nitter_profile_probe.py: the latest timeline page, about 20 items, of 25 Nov 2026):
  status  active    own post in the last 30 days
          quiet     own post in the last 12 months, none in 30 days
          reposter  no own post on the latest page, only reposts (reposts are already collected from their source)
          dormant   last own post over 12 months ago
  poll    yes when the account's own posts are collected (active or quiet); reposters and dormant ones are not:
          their reposts repeat posts already collected from the source account.
  python3 scripts/blr_accounts.py
"""
import csv, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOW = time.mktime((2026, 11, 25, 12, 0, 0, 0, 0, -1))

# handle: (role, BTP station name or unit, note)
ACC = {
    "blrcitytraffic": ("city", "Bengaluru Traffic Police", "main account; advisories, diversions"),
    "Jointcptraffic": ("joint_cp", "Joint/Special CP Traffic", ""),
    "DCPTrEastBCP": ("dcp", "Traffic East", ""), "DCPTrWestBCP": ("dcp", "Traffic West", ""),
    "DCPTrNorthBCP": ("dcp", "Traffic North", ""), "DCPSouthTrBCP": ("dcp", "Traffic South", ""),
    "DCPTrCentralBCP": ("dcp", "Traffic Central", "new account (Nov 2026)"),
    "acpeasttraffic": ("acp", "East Sub Division", ""), "ACP_TrafficEast": ("acp", "East Sub Division", "old account"),
    "acpcentraltrf": ("acp", "Central Sub Division", ""), "acpsouthtrf": ("acp", "South Sub Division", ""),
    "acpsetraffic": ("acp", "South East Sub Division", ""), "acpwesttrf": ("acp", "West Sub Division", ""),
    "acpwfieldtrf": ("acp", "White Field Sub Division", ""), "acphsrtrps": ("acp", "Hsr Layout Sub Division", ""),
    "acpnortheasttr": ("acp", "North East Sub Division", ""), "acpnorthtraffic": ("acp", "North Sub Division", ""),
    "Acpvnagartrfbcp": ("acp", "Vijayanagar Sub Divi", ""),
    "adugoditrfps": ("station", "Adugodi", ""), "ashoknagartrfps": ("station", "Ashok Nagar", ""),
    "bsktrfps": ("station", "Banashankari", ""), "bwaditrafficps": ("station", "Banaswadi", ""),
    "basvanguditrfps": ("station", "Basavanagudi", ""), "bellandurutrfps": ("station", "Bellandur", ""),
    "chickpetetrfps": ("station", "Chickpet", "display name ChamarajapeteTrafficPs"),
    "chikkajalatrfps": ("station", "Chikkajala", ""), "citymarkettrps": ("station", "City Market", ""),
    "cparktrafficps": ("station", "Cubbon Park", ""), "ecitytrfps": ("station", "Electronic City", ""),
    "halairporttrfps": ("station", "HAL Old Airport", ""), "hsrltrafficps": ("station", "HSR Layout", ""),
    "halasoortrfps": ("station", "Halasur", ""), "hgatetrafficps": ("station", "Halasuru Gate", ""),
    "Hgatetrps": ("station", "Halasuru Gate", "second account (few followers)"),
    "hebbaltrafficps": ("station", "Hebbala", ""), "highgroundtrfps": ("station", "High ground", ""),
    "hulimavutrfps": ("station", "Hulimavu", ""), "jpnagartrfps": ("station", "J.P. Nagar", ""),
    "jalahallitrfps": ("station", "Jalahalli", "older account"), "JalahalliP65260": ("station", "Jalahalli", "newer account"),
    "JnagarTr": ("station", "Jayanagara", ""), "jbnagartrfps": ("station", "Jeevanbheemanagar", ""),
    "JbharathiTrPs24": ("station", "Jnanabharathi", ""), "kghallitrfps": ("station", "K.G. Halli", ""),
    "KRPURATRAFFIC": ("station", "K.R. Pura", ""), "kslayoutrfps": ("station", "K.S. Layout", ""),
    "kmpalyatrfps": ("station", "Kamakshipalya", ""), "kengeritrfps": ("station", "Kengeri", ""),
    "madivalatrfps": ("station", "Madiwala", ""), "magadiroadtrfps": ("station", "Magadi Road", ""),
    "mahadevapuratrf": ("station", "Mahadevapura", ""), "Mvmtraffic": ("station", "Malleshwaram", ""),
    "micolyttrfps": ("station", "Mico Layout", ""), "peenyatrfps": ("station", "Peenya", ""),
    "ftowntrfps": ("station", "Pulikeshinagar(F.Town)", ""), "rtnagartraffic": ("station", "R.T. Nagar", ""),
    "rajajinagartrps": ("station", "Rajajinagar", ""), "ssnagartrfps": ("station", "Sadashivanagar", ""),
    "sanjayanagara78": ("station", "Sanjayanagar", "unverified: few followers, display name 'sanjayanagaratr'"),
    "upparpetetrfps": ("station", "Upparpet", ""), "vvpuramtrfps": ("station", "V.V.Puram (C.Pet)", ""),
    "vijayanagartrps": ("station", "Vijayanagara", ""), "wftrps": ("station", "Whitefield", ""),
    "wgardentrfps": ("station", "Wilson Garden", ""), "yelahankatrfps": ("station", "Yelahanka", "newer of two; both dormant"),
    "yelahankatrps": ("station", "Yelahanka", "older account"), "yprtrps": ("station", "Yeshwanthpura", ""),
}
NO_HANDLE = ["Byatarayanapura", "Chikkabanavara", "Devanahalli Airport", "Hennuru", "Shivajinagar", "Thalagattapura"]


def main():
    probes = {}
    for f in sorted((ROOT / "raw/bengaluru").glob("probe_*.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if "error" not in r:
                probes[r["user"].lower()] = r
    ps = {r["Traffic PS Names"]: r for r in csv.DictReader(open(ROOT / "data/blr/powerbi/police_stations.csv"))}
    rows = []
    for h, (role, unit, note) in ACC.items():
        p = probes.get(h.lower(), {})
        new = p.get("newest_own")
        age = (NOW - new) / 86400 if new else None
        status = ("reposter" if not new and p.get("reposts") else "unknown" if not p else
                  "active" if age <= 30 else "quiet" if age <= 365 else "dormant")
        own, rt = p.get("own", 0), p.get("reposts", 0)
        st = ps.get(unit, {})
        rows.append({"handle": h, "role": role, "unit": unit,
                     "sub_division": st.get("Sub-Division Case Corrected", ""), "zone": st.get("Zone Case Corrected", ""),
                     "name_on_x": p.get("name", ""), "followers": p.get("followers", ""), "posts": p.get("posts", ""),
                     "last_own_post": time.strftime("%Y-%m-%d", time.gmtime(new)) if new else "",
                     "repost_share": f"{rt / (own + rt):.0%}" if own + rt else "", "status": status,
                     "poll": "yes" if status in ("active", "quiet") else "no",
                     "url": f"https://x.com/{h}", "note": note})
    order = {"city": 0, "joint_cp": 1, "dcp": 2, "acp": 3, "station": 4}
    rows.sort(key=lambda r: (order[r["role"]], r["unit"], r["handle"].lower()))
    out = ROOT / "data/blr/btp_accounts.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    covered = {r["unit"] for r in rows if r["role"] == "station"}
    missing = [s for s in ps if s not in covered]
    md = ["| Account | Role | Station / unit | Zone | Last own post | Reposts | Status | Collected |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| [@{r['handle']}]({r['url']}) | {r['role']} | {r['unit']} | {r['zone']} | {r['last_own_post'] or '-'} | "
                  f"{r['repost_share'] or '-'} | {r['status']} | {r['poll']} |" + (f" {r['note']}" if r['note'] else ""))
    md.append("")
    md.append("Stations with no account found: " + ", ".join(missing) + ".")
    (ROOT / "data/blr/btp_accounts.md").write_text("\n".join(md) + "\n")
    # Nitter poller search groups (~/nitter/groups.json on mss): own posts of the polled accounts, a few searches of
    # at most GROUP_CHARS characters (Nitter answers 400 above about 200); @blrcitytraffic is read from its own timeline every run instead.
    GROUP_CHARS, FILTERS = 195, " -filter:replies -filter:nativeretweets"
    hs = [r["handle"] for r in rows if r["poll"] == "yes" and r["handle"] != "blrcitytraffic"]
    groups, cur = [], []
    for h in hs:
        if cur and len(" OR ".join(f"from:{x}" for x in cur + [h])) + len(FILTERS) > GROUP_CHARS:
            groups.append(cur); cur = []
        cur.append(h)
    groups.append(cur)
    gj = {f"BLR{i + 1}": "(" + " OR ".join(f"from:{x}" for x in g) + ")" + FILTERS for i, g in enumerate(groups)}
    (ROOT / "data/blr/nitter_groups.json").write_text(json.dumps(gj, indent=1) + "\n")
    print(len(gj), "search groups,", [len(g) for g in groups], "handles, longest", max(len(q) for q in gj.values()), "chars")
    from collections import Counter
    print(len(rows), "accounts;", dict(Counter(r["status"] for r in rows)), "; polled:", sum(r["poll"] == "yes" for r in rows))
    print("stations covered", len(covered), "of", len(ps), "; missing:", missing)


if __name__ == "__main__":
    main()
