"""Live collector: X via Nitter (Hyderabad, Cyberabad; Delhi, Chennai and Bengaluru forwarded), the HTP e-challan alert/news,
Chennai incidents for the MTC map, Bengaluru incidents for the BMTC map, Chandigarh incidents for the tricity map, the WhatsApp channel (Mac only), then the TGSRTC live layer to SVR.

  python -m pipeline.collect --watch 55 --publish     # mss cron, every minute (scripts/mss.crontab)
  python -m pipeline.collect                          # one pass, no push to SVR (the old Mac launchd job)

With --watch N it keeps running for N seconds and runs x_live as soon as Nitter rewrites ~/nitter/out/latest.json
(mss cron, every 2 min), so a post is on the map within seconds of Nitter seeing it. The slow sources (e-challan,
Chennai, WhatsApp) and an output refresh (expired events drop, "generated" stays fresh) run every 5 min at most.
--publish pushes out/map/incidents_live.json to SVR after any step that rewrote it (pipeline.publish skips unchanged).
Skips while a full rebuild is writing the store, and never overlaps itself (also flock'd by cron)."""
import argparse, fcntl, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "out/.collect_state.json"
LIVE = ROOT / "out/map/incidents_live.json"
SLOW_EVERY = 290          # seconds: e-challan (its UA promises 5 min), Chennai, Bengaluru, WhatsApp, output refresh
WHATSAPP = sys.platform == "darwin" and os.environ.get("TWTRX_WHATSAPP", "1") != "0"   # needs WhatsApp Desktop

def run(mod, *args):
    subprocess.run([sys.executable, "-m", mod, *args], cwd=ROOT)

def nitter_mtime():
    from .x_live import LOCAL
    try: return LOCAL.stat().st_mtime_ns
    except OSError: return None             # not on mss: x_live reads it over ssh, so run it every slow pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0, help="keep watching Nitter's latest.json for this many seconds")
    ap.add_argument("--publish", action="store_true", help="push the TGSRTC live layer to SVR when it changes")
    a = ap.parse_args()
    t0 = time.time()
    lock = open(ROOT / "out/.collect.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0
    if subprocess.run(["pgrep", "-f", "pipeline.build"], capture_output=True).returncode == 0:
        print(datetime.now().strftime("[%Y-%m-%d %H:%M]"), "rebuild running, skipped", flush=True); return 0
    st = json.loads(STATE.read_text()) if STATE.exists() else {}

    def save(): STATE.write_text(json.dumps(st))
    def publish():
        if a.publish: run("pipeline.publish")
    def x_if_new(force=False):
        m = nitter_mtime()
        if force or m is None or m != st.get("nitter_mtime"):
            st["nitter_mtime"] = m; save()
            run("pipeline.x_live"); return True
        return False

    if time.time() - st.get("slow_at", 0) >= SLOW_EVERY:
        st["slow_at"] = time.time(); save()
        if WHATSAPP: run("pipeline.whatsapp_db")
        run("pipeline.hydtp_web")
        x_if_new(force=nitter_mtime() is None)
        run("pipeline.chennai")
        run("pipeline.bengaluru")
        run("pipeline.chandigarh")
        run("pipeline.publish_history")      # alert history for the traffic API, hourly at most
        try: age = time.time() - LIVE.stat().st_mtime
        except OSError: age = 1e9
        if age > 240:                        # nothing rewrote the layer this pass: refresh expiries and "generated"
            from . import build
            build.write_outputs(build.load_store())
        publish()
    elif x_if_new():
        publish()
    while time.time() - t0 < a.watch - 3:
        time.sleep(2)
        if nitter_mtime() is not None and x_if_new():
            publish()
    return 0

if __name__ == "__main__":
    sys.exit(main())
