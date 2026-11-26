"""Push out/map/incidents_live.json to the TGSRTC map's live/ dir on SVR, atomically.

  python -m pipeline.publish          # run by pipeline.collect --publish (mss cron) after each step

Served at maps.reclaim.city/tgsrtc/live/incidents.json, also read by tsrtc-eta on SVR. On the Mac it ran under the
venv Python from launchd (launchd's /bin/bash may not read ~/Desktop). Python reads the file and streams it over ssh stdin into a temp file, then mv: no partial file is
ever served, and ssh/scp never need Desktop access. Skips when the file has not changed since the last push.
One run at a time; ssh has short timeouts and BatchMode, so a dropped link cannot stack runs.
Exactly one machine may run this: two stores publishing in turn would make the map flip between them.
"""
import fcntl, os, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out/map/incidents_live.json"
STAMP = ROOT / "out/.published_mtime"
HOST = os.environ.get("PUBLISH_HOST", "user@rtc-map-server")
DIR = "/data/rtc/web/tgsrtc/live"
OPTS = ["-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", "-o", "BatchMode=yes"]

def log(msg):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)

def main():
    lock = open("/tmp/twtrx_publish.pylock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("locked"); return 0
    if os.environ.get("TWTRX_NO_PUSH") == "1":
        return 0
    if not SRC.exists() or SRC.stat().st_size == 0:
        log(f"no {SRC.name}"); return 0
    mt = str(SRC.stat().st_mtime_ns)
    if STAMP.exists() and STAMP.read_text() == mt:
        return 0                                   # unchanged since last push (regenerated every 5 min)
    data = SRC.read_bytes()
    cmd = f"cat > {DIR}/.incidents.json.tmp && mv {DIR}/.incidents.json.tmp {DIR}/incidents.json"
    try:
        r = subprocess.run(["ssh", *OPTS, HOST, cmd], input=data, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        log("ssh timed out"); return 1
    if r.returncode != 0:
        log(f"publish failed ({r.returncode}): {r.stderr.decode(errors='replace').strip()[:300]}"); return 1
    STAMP.write_text(mt)
    log(f"published {len(data)} bytes")
    return 0

if __name__ == "__main__":
    sys.exit(main())
