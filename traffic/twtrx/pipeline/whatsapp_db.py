"""Auto-ingest new posts from the 'Cyberabad Traffic Pulse' WhatsApp channel via WhatsApp Desktop's local DB.

  python -m pipeline.whatsapp_db            # ingest new channel posts since last run, refresh outputs
  python -m pipeline.whatsapp_db --publish  # ...and copy the live layer into maps-site
  python -m pipeline.whatsapp_db --check    # only report whether the channel is found (no message text shown)

Privacy: the DB is copied read-only to a temp dir, and ONLY rows of the chat session whose name
matches CHANNEL are read; the copy is deleted afterwards. Nothing from other chats is read or stored.
Requires WhatsApp Desktop (Mac) to be linked to the phone and to have synced the channel.
"""
import argparse, json, os, shutil, sqlite3, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .classify import IST

CHANNEL = os.environ.get("WA_CHANNEL", "Cyberabad Traffic Pulse")
CANDIDATES = [
    Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite",
    Path.home() / "Library/Group Containers/group.net.whatsapp.WhatsApp.private/ChatStorage.sqlite",
]
STATE = Path(__file__).resolve().parent.parent / "data/whatsapp_state.json"
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

def find_db():
    for p in CANDIDATES:
        if p.exists(): return p
    return None

def _pb_strings(blob, out, depth=0):
    """Collect UTF-8 strings from a protobuf blob (template-message metadata)."""
    i, n = 0, len(blob)
    def varint(i):
        v = sh = 0
        while i < n:
            b = blob[i]; i += 1; v |= (b & 0x7F) << sh; sh += 7
            if b < 0x80: return v, i
        raise ValueError
    try:
        while i < n:
            key, i = varint(i); wt = key & 7
            if wt == 0: _, i = varint(i)
            elif wt == 1: i += 8
            elif wt == 5: i += 4
            elif wt == 2:
                ln, i = varint(i); chunk = blob[i:i + ln]; i += ln
                if len(chunk) != ln: return
                try:
                    t = chunk.decode("utf-8")
                    if t.isprintable() or "\n" in t: out.append(t)
                except UnicodeDecodeError:
                    pass
                if depth < 6: _pb_strings(chunk, out, depth + 1)
            else:
                return
    except (ValueError, IndexError):
        return

def message_text(text, title, meta):
    """Plain text (ZTEXT), image caption (media ZTITLE), or text inside template metadata."""
    if text: return text
    if title: return title
    if meta:
        out = []; _pb_strings(bytes(meta), out)
        out = [t for t in out if len(t) >= 20 and " " in t]
        out = [t for t in out if not any(o != t and o in t and len(o) >= 0.8 * len(t) for o in out)]
        if out: return max(out, key=len)
    return None

def read_channel(db, after_ts):
    tmp = Path(tempfile.mkdtemp(prefix="wa_"))
    try:
        for suf in ("", "-wal", "-shm"):
            src = Path(str(db) + suf)
            if src.exists(): shutil.copy2(src, tmp / (db.name + suf))
        con = sqlite3.connect(f"file:{tmp / db.name}?mode=ro", uri=True)
        ses = con.execute("SELECT Z_PK, ZPARTNERNAME, ZCONTACTJID FROM ZWACHATSESSION WHERE ZPARTNERNAME = ? "
                          "OR (ZPARTNERNAME LIKE ? AND ZCONTACTJID LIKE '%@newsletter')",
                          (CHANNEL, "%" + CHANNEL + "%")).fetchall()
        if not ses: return None, [], after_ts
        sid = ses[0][0]
        # business posts arrive as plain text (0), text templates (19) or image templates with a caption (20);
        # type 10 is a system notice. Z_PK is not chronological, so track time.
        rows = con.execute("SELECT m.Z_PK, m.ZMESSAGEDATE, m.ZMESSAGETYPE, m.ZTEXT, mi.ZTITLE, mi.ZMETADATA FROM ZWAMESSAGE m "
                           "LEFT JOIN ZWAMEDIAITEM mi ON mi.Z_PK = m.ZMEDIAITEM WHERE m.ZCHATSESSION = ? "
                           "AND m.ZMESSAGEDATE > ? AND m.ZMESSAGETYPE != 10 ORDER BY m.ZMESSAGEDATE", (sid, after_ts)).fetchall()
        con.close()
        msgs, notext = [], {}
        for pk, ts, typ, txt, title, meta in rows:
            t = message_text(txt, title, meta)
            if t: msgs.append({"pk": pk, "ts": ts, "time": (APPLE_EPOCH + timedelta(seconds=ts)).astimezone(IST), "text": t})
            else: notext[typ] = notext.get(typ, 0) + 1
        if notext: print("new messages without readable text (type: count):", notext)
        last = rows[-1][1] if rows else after_ts
        return ses[0], msgs, last
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--publish", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    print(datetime.now(IST).strftime("[%Y-%m-%d %H:%M]"), end=" ")
    db = find_db()
    if not db:
        print("WhatsApp Desktop message DB not found: open WhatsApp on this Mac and link it to your phone."); return 0
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.setdefault("last_ts", 0)
    r = read_channel(db, st["last_ts"])
    if r[0] is None:
        print(f"Channel '{CHANNEL}' not in the local DB yet (open it once in WhatsApp Desktop so it syncs)."); return 0
    ses, msgs, last = r
    if a.check:
        print(f"found channel ({ses[2]}); {len(msgs)} new posts with text since {st['last_ts']}"); return 0
    if msgs:
        from .ingest import ingest
        ingest([{"time": m["time"], "text": m["text"]} for m in msgs], a.publish)
    if last > st["last_ts"]:
        st["last_ts"] = last; st["last_run"] = datetime.now(IST).isoformat(timespec="seconds")
        STATE.write_text(json.dumps(st))
    if msgs:
        pass
    else:
        from . import build                 # still refresh: expired events drop, "generated" stays fresh
        build.write_outputs(build.load_store())
        print("no new channel posts")
    return 0

if __name__ == "__main__":
    sys.exit(main())
