"""Ingest WhatsApp channel alerts (or any pasted traffic message) into the event store.

  python -m pipeline.ingest inbox/                 # every .txt in inbox/, then moved to inbox/done/
  python -m pipeline.ingest --text "🚨 TRAFFIC ALERT ..." [--time "2026-11-25 10:05"]
  pbpaste | python -m pipeline.ingest -            # from clipboard / stdin
  add --publish to refresh maps-site/tgsrtc/traffic/incidents.json

Accepted inputs:
  * WhatsApp chat export (.txt): "[25/11/26, 10:05:12 AM] Name: msg" or "25/11/2026, 10:05 am - Name: msg"
    (forward the channel's posts to a chat/yourself and export it). Sender names are dropped.
  * Plain pasted text: one message, or several separated by a line containing only ---
    Time comes from the message ("Date: .. Time: .."), --time, or the file's modified time.
"""
import argparse, re, shutil, sys
from datetime import datetime
from pathlib import Path
from .classify import IST
from .relevance import is_road_event
from .events import Builder, norm_text
from .gazetteer import hav
from . import build

HDR = re.compile(r"^‎?\[?(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?,?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([APap]\.?[Mm]\.?)?\]?\s*(?:-\s*)?([^:\n]{1,60}):\s?(.*)$")

def parse_export(raw):
    msgs, cur = [], None
    for line in raw.splitlines():
        m = HDR.match(line)
        if m:
            if cur: msgs.append(cur)
            d, mo, y, h, mi, s, ap, _sender, body = m.groups()
            now = datetime.now(IST)
            if y is None:            # "[25/11, 08:58]" (copied from WhatsApp): no year
                y = now.year if (int(mo), int(d)) <= (now.month, now.day) else now.year - 1
            y = int(y) + (2000 if int(y) < 100 else 0); h = int(h)
            if ap:
                ap = ap.lower().replace(".", "")
                if ap == "pm" and h < 12: h += 12
                if ap == "am" and h == 12: h = 0
            cur = {"time": datetime(y, int(mo), int(d), h, int(mi), tzinfo=IST), "text": body}
        elif cur:
            cur["text"] += "\n" + line
    if cur: msgs.append(cur)
    return msgs

def parse_any(raw, fallback_time):
    msgs = parse_export(raw)
    if msgs: return msgs
    return [{"time": fallback_time, "text": p.strip()} for p in re.split(r"(?m)^\s*---+\s*$", raw) if p.strip()]

def core_text(t):
    """Wording without hashtags, alert headers, emoji and spacing: the same alert posted on X, WhatsApp and the
    e-challan site compares equal."""
    t = re.sub(r"#\w+|https?://\S+", " ", t)
    t = re.sub(r"\b(?:traffic alert|traffic update|traffic advisory|dear citizen)\b", " ", t, flags=re.I)
    return norm_text(t)[:120]

def ingest(messages, publish=False, source="whatsapp"):
    ev = build.load_store(); b = Builder(web=True)
    added = dup = skipped = closed = 0; new_ids = []
    for m in messages:
        text = m["text"].replace("<Media omitted>", "").strip()
        if not text or not is_road_event(text):
            skipped += 1; continue
        e = b.build(text, m["time"], m.get("source", source), m.get("src"), m.get("pid"))
        nt = core_text(e["text"])
        # duplicate: same id, or the same wording (any source: X, WhatsApp, e-challan) within 12 h
        same = ev.get(e["id"]) or next((x for x in ev.values() if abs((x["post_time"] or 0) - e["post_time"]) < 43200
                                        and core_text(x["text"]) == nt), None)
        if same:
            dup += 1; new_ids.append(same["id"]); continue
        if e["status"] == "cleared" and e["lat"] is not None:
            for x in ev.values():   # close the matching open event nearby
                if x["status"] == "active" and x["lat"] is not None and x["t_end"] is None \
                   and 0 <= e["post_time"] - (x["t_start"] or 0) < 12 * 3600 and hav((x["lat"], x["lon"]), (e["lat"], e["lon"])) < 600:
                    x["t_end"] = e["post_time"]; x["t_expires"] = e["post_time"]; closed += 1
        ev[e["id"]] = e; added += 1; new_ids.append(e["id"])
        print(f"+ {e['t_start_ist']} {e['type']:<13} {e['place'] or '(no location)'} | {e['text'][:70]!r}")
    b.save(); build.save_store(ev); build.write_outputs(ev, publish)
    print(f"added {added}, duplicates {dup}, not road-related {skipped}, closed by 'cleared' updates {closed}")
    unres = [p for p, _ in b.unresolved.items()]
    if unres: print("unresolved places (add to data/aliases.csv):", "; ".join(unres[:20]))
    return new_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="file, folder, or - for stdin")
    ap.add_argument("--text"); ap.add_argument("--time", help="IST 'YYYY-MM-DD HH:MM' for pasted text")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    t0 = datetime.strptime(a.time, "%Y-%m-%d %H:%M").replace(tzinfo=IST) if a.time else datetime.now(IST)
    msgs, done = [], []
    if a.text: msgs += parse_any(a.text, t0)
    elif a.src == "-": msgs += parse_any(sys.stdin.read(), t0)
    elif a.src:
        p = Path(a.src); files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
        for f in files:
            ft = a.time and t0 or datetime.fromtimestamp(f.stat().st_mtime, IST)
            msgs += parse_any(f.read_text(encoding="utf-8", errors="replace"), ft); done.append(f)
    if not msgs: ap.error("nothing to ingest")
    ingest(msgs, a.publish)
    for f in done:
        if f.parent.name == "inbox":
            (f.parent / "done").mkdir(exist_ok=True); shutil.move(str(f), f.parent / "done" / f.name)

if __name__ == "__main__":
    main()
