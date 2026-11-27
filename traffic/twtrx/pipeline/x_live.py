"""Live @HYDTP and @CYBTRAFFIC posts from the self-hosted Nitter on mss (no Sorsa requests).

  python -m pipeline.x_live          # run by pipeline.collect as soon as Nitter rewrites latest.json

mss cron rewrites ~/nitter/out/latest.json every 2 min:
  {"t", "accounts": {"HYDTP"|"CYBTRAFFIC": {"ok", "t", "posts": [{"id", "url", "t", "text"}] newest first}}}
On mss (where the collector runs) the file is read directly; elsewhere over ssh via the nia jump host (direct ssh
to mss drops). Only the latest ~20 posts per account are there; gaps can be back-filled with scrape_x.py.
Dedupe is by post id (same ids as the Sorsa history: twitter:<id>, hydtp_x:<id>), then by text within 12 h
(the same alert via WhatsApp). "ok": false = Nitter/session failure: logged, never read as "no alerts".
"""
import json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path
from .classify import IST

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw/nitter"
LOCAL = Path(os.environ.get("NITTER_LATEST", Path.home() / "nitter/out/latest.json"))
NO_PUSH = os.environ.get("TWTRX_NO_PUSH") == "1"     # dry run: build everything, push nothing to raw/SVR
SSH = ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5",
       *os.environ.get("NITTER_SSH", "user@nitter-host").split(), "cat nitter/out/latest.json"]   # e.g. NITTER_SSH="-J user@jump -p 22 user@nitter-host"
SOURCE = {"HYDTP": "hydtp_x", "CYBTRAFFIC": "twitter"}
# other cities: archived as JSON lines and forwarded, not geocoded here
FORWARD = {"dtptraffic": ROOT / "out/delhi/dtptraffic.jsonl",          # -> raw:projects/delhi-live/x/posts.jsonl
           "ChennaiTraffic": ROOT / "out/chennai/chennaitraffic.jsonl",
           "trafficchd": ROOT / "out/chandigarh/trafficchd.jsonl"}      # read twice a day by the poller (SCHEDULED)
# Chandigarh: advisories carry #TrafficAdvisory / #TrafficAlert; most other posts are awareness drives
CHD_UPDATE = re.compile(r"#?traffic ?(advisory|alerts?)\b|divert|diversion|restrict|road (closed|closure|blocked)|closed for|"
    r"procession|yatra|nagar kirtan|shobha yatra|marathon|cyclothon|rally|vip movement|waterlogg|breakdown|accident|"
    r"tree (fall|fell|uproot)|signal (out of order|not working)|congestion is expected|avoid the (route|road|stretch)", re.I)
CHD_NOT = re.compile(r"awareness|road[- ]safety (week|month|stall|team|drive|class|lecture)|helmet|seat ?belt|drunk|challan|"
    r"special drive|does traffic congestion bother you|#drivebetter|#drivesafe|lost (and|&) found|felicitat|congratulat|wishes", re.I)
def chd_update(text): return bool(CHD_UPDATE.search(text)) and not CHD_NOT.search(text)
CHENNAI_NOT = re.compile(r"action taken|awareness|helmet|seat ?belt|drunk|wrong side|triple riding|stop line|sentenced|mobile phone|tail ?lights?|zebra|number plate|hsrp|overspeed|மது அருந்தி|தலைக்கவசம்", re.I)
DELHI_HOST, DELHI_DIR = os.environ.get("MAP_SERVER", "user@map-server"), "projects/delhi-live/x"
# Chennai posts that are about the roads today (the account is mostly road-safety awareness)
CHENNAI_UPDATE = re.compile(r"#TrafficUpdate|#TrafficAdvisory|traffic (advisory|update|alert|diversion)|divert|diversion|"
    r"road (closed|closure|blocked)|blocked|closed for|congestion|slow[- ]moving|waterlogg|breakdown|vip movement|pothole|pit causing|accident|zero traffic|traffic (changes|restrictions|arrangements)|போக்குவரத்து (மாற்றம்|நெரிசல்)|மாற்றுப் பாதை", re.I)
# Bengaluru: @blrcitytraffic's timeline and the poller's station groups (BLR1..), all into one file; reposts were
# dropped by the group searches, and a post seen in two places is kept once (ids are global)
BLR_FILE = ROOT / "out/bengaluru/btp.jsonl"
def is_blr(acct): return acct == "blrcitytraffic" or re.fullmatch(r"BLR\d+", acct) is not None
BLR_NOT = re.compile(r"awareness|helmet|seat ?belt|drunk|drink and drive|wrong side|triple riding|cases? (booked|registered)|"
    r"\bfines?\b|penalt|special drive|lost (and|&) found|returned to|handed over|felicitat|congratulat|wishes|greetings|"
    r"no parking drive|defective number|tinted|hsrp|school children|road safety (week|month)|#roadsafety|safety tips|"
    r"footpath (clearance )?drive|encroachment|clearance drive|conducted (a|an)\b|#trafficawareness|"
    r"visited|inspected the|site inspection|beyond (their |regular )?duty|ಭೇಟಿ ನೀಡಿ|ಪರಿಶೀಲನೆ|"
    r"ಹೆಲ್ಮೆಟ್|ಪ್ರಕರಣ ದಾಖಲ|ದಂಡ|ಜಾಗೃತಿ|ಅರಿವು|ಶುಭಾಶಯ", re.I)
BLR_UPDATE = re.compile(r"#?traffic ?(update|advisory|alert|diversion)|divert|diversion|alternat(e|ive) route|slow[- ]?moving|"
    r"congest|heavy (traffic|vehicle)|traffic (jam|movement is|is slow|density)|break ?down|broke down|breakdown|"
    r"tree (fall|fell|uproot)|fallen tree|branch fell|water ?logg|stagnat|flood|pothole|road (closed|closure|blocked|work)|"
    r"closed for|blocked|accident|overturn|skid|procession|rally|protest|vip movement|metro work|white ?topping|"
    r"one ?way|no entry|restricted|cleared|clearance|moving smoothly|normal(cy)? (restored|now)|kindly avoid|plan your (travel|journey)|"
    r"ಸಂಚಾರ (ದಟ್ಟಣೆ|ಸಲಹೆ|ಸೂಚನೆ|ಬದಲಾವಣೆ|ನಿಧಾನ|ಸುಗಮ|ಮಾರ್ಪಾಡು)|ಮಾರ್ಗ ಬದಲಾವಣೆ|ಪರ್ಯಾಯ ಮಾರ್ಗ|ಕೆಟ್ಟು ನಿಂತ|ಮರ (ಬಿದ್ದ|ಉರುಳಿ)|ನೀರು ನಿಂತ|ಅಪಘಾತ|"
    r"ನಿಧಾನಗತಿ|ಸ್ಥಗಿತ", re.I)
def blr_update(text): return bool(BLR_UPDATE.search(text)) and not BLR_NOT.search(text)
SEEN = ROOT / "data/x_live_seen.json"        # every post id already processed (incl. non-road posts)

def main():
    print(datetime.now(IST).strftime("[%Y-%m-%d %H:%M]"), end=" ")
    r = None
    try:
        if LOCAL.exists():
            d = json.loads(LOCAL.read_text())
        else:
            r = subprocess.run(SSH, capture_output=True, timeout=60)
            d = json.loads(r.stdout) if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        d = None
    if not d:
        err = r.stderr.decode(errors="replace").strip()[:200] if r is not None else "timeout / bad json"
        print("x_live: could not read latest.json:", err); return 0
    from . import build
    ev = build.load_store()
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    msgs, notes = [], []
    RAW.mkdir(parents=True, exist_ok=True)
    pushed = False
    blr_new, blr_bad = 0, []
    for acct, v in (d.get("accounts") or {}).items():
        if is_blr(acct):
            BLR_FILE.parent.mkdir(parents=True, exist_ok=True)
            new = [p for p in v.get("posts") or [] if str(p["id"]) not in seen]
            seen.update(str(p["id"]) for p in v.get("posts") or [])
            with open(BLR_FILE, "a") as f:
                for p in reversed(new):
                    f.write(json.dumps({"id": p["id"], "user": p.get("user") or acct, "url": p.get("url"), "t": p["t"],
                                        "text": p["text"], "update": blr_update(p["text"])}, ensure_ascii=False) + "\n")
            blr_new += len(new)
            if not v.get("ok"): blr_bad.append(acct)
            continue
        if acct in FORWARD:
            fp = FORWARD[acct]; fp.parent.mkdir(parents=True, exist_ok=True)
            new = [p for p in v.get("posts") or [] if str(p["id"]) not in seen]
            seen.update(str(p["id"]) for p in v.get("posts") or [])
            with open(fp, "a") as f:
                for p in reversed(new):
                    rec = {"id": p["id"], "url": p.get("url"), "t": p["t"], "text": p["text"]}
                    if acct == "ChennaiTraffic":
                        rec["update"] = bool(CHENNAI_UPDATE.search(p["text"])) and not CHENNAI_NOT.search(p["text"])
                    elif acct == "trafficchd":
                        rec["update"] = chd_update(p["text"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            notes.append(f"{acct} {len(new)} new" + ("" if v.get("ok") else " (NOT OK)"))
            if acct == "dtptraffic" and new: pushed = True
            continue
        src = SOURCE.get(acct)
        if not src: continue
        if not v.get("ok"):
            notes.append(f"{acct} NOT OK (Nitter/session failure)")
        new = [p for p in v.get("posts") or [] if f"{src}:{p['id']}" not in ev and str(p["id"]) not in seen]
        seen.update(str(p["id"]) for p in v.get("posts") or [])
        with open(RAW / "posts.jsonl", "a") as f:
            for p in new: f.write(json.dumps({"acct": acct, **p}, ensure_ascii=False) + "\n")
        for p in reversed(new):              # oldest first, so a 'cleared' update closes an earlier alert
            msgs.append({"time": datetime.fromtimestamp(p["t"], IST), "text": p["text"], "source": src,
                         "src": p.get("url") or f"https://x.com/{acct}/status/{p['id']}", "pid": p["id"]})
        notes.append(f"{acct} {len(new)} new")
    if blr_new or blr_bad:
        notes.append(f"Bengaluru {blr_new} new" + (f" (NOT OK: {' '.join(blr_bad)})" if blr_bad else ""))
    print("x_live:", ", ".join(notes))
    SEEN.write_text(json.dumps(sorted(seen)[-5000:]))
    if not NO_PUSH and (pushed or not (ROOT / "out/delhi/.pushed").exists()):
        # the whole file (small, append-only) to delhi-live; dtp_closures.py --x reads it hourly
        # streamed over ssh stdin (launchd children may not read ~/Desktop), tmp file then mv
        cmd = f"mkdir -p {DELHI_DIR} && cat > {DELHI_DIR}/.posts.jsonl.tmp && mv {DELHI_DIR}/.posts.jsonl.tmp {DELHI_DIR}/posts.jsonl"
        r2 = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", DELHI_HOST, cmd],
                            input=FORWARD["dtptraffic"].read_bytes(), capture_output=True, timeout=60)
        if r2.returncode == 0: (ROOT / "out/delhi/.pushed").touch()
        else: print("  delhi push failed:", r2.stderr.decode(errors="replace").strip()[:200])
    if msgs:
        from .ingest import ingest
        ingest(msgs)
    return 0

if __name__ == "__main__":
    sys.exit(main())
