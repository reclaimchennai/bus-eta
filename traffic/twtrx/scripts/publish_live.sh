#!/bin/bash
# Push out/map/incidents_live.json to the TGSRTC map's live/ dir on SVR, atomically.
# Served at maps.reclaim.city/tgsrtc/live/incidents.json. Production: enable it yourself (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=out/map/incidents_live.json
DEST_HOST=${PUBLISH_HOST:?user@rtc-map-server}
DEST_DIR=/data/rtc/web/tgsrtc/live
LOCK=/tmp/twtrx_publish.lock
[ -s "$SRC" ] || { echo "$(date '+%F %T') no $SRC"; exit 0; }
# one run at a time (a sleeping Mac or dropped link must not stack runs)
if ! mkdir "$LOCK" 2>/dev/null; then
  [ $(( $(date +%s) - $(stat -f %m "$LOCK") )) -gt 600 ] && rm -rf "$LOCK" && mkdir "$LOCK" || { echo "locked"; exit 0; }
fi
trap 'rm -rf "$LOCK"' EXIT
OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o BatchMode=yes)
scp -q "${OPTS[@]}" "$SRC" "$DEST_HOST:$DEST_DIR/.incidents.json.tmp"
ssh "${OPTS[@]}" "$DEST_HOST" "mv $DEST_DIR/.incidents.json.tmp $DEST_DIR/incidents.json"
echo "$(date '+%F %T') published $(wc -c < "$SRC") bytes"
