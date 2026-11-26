#!/usr/bin/env bash
# After the cutover, mss holds the live store: push only code and hand-curated files from the Mac, never state.
#   bash scripts/mss_push_code.sh            # pipeline/, scripts/, aliases, manual events, Bengaluru reference data
# (data/blr: OSM cut, accounts, search groups, aliases; rsync sends only what changed)
# Takes effect on the next minute's collector run. To re-apply manual events now:
#   ssh ${NITTER_SSH:?set NITTER_SSH, e.g. "-J user@jump -p 22 user@nitter-host"} 'cd twtrx && flock out/.collect.lock .venv/bin/python -m pipeline.manual'
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -azR -e "ssh -o BatchMode=yes -o ConnectTimeout=15 ${NITTER_SSH_OPTS:-}" --exclude __pycache__ \
  pipeline/ scripts/mss.crontab scripts/mss_cutover.sh scripts/mss_push_code.sh \
  data/aliases.csv data/chn/aliases.csv data/manual_events.json README.md \
  data/blr/aliases.csv data/blr/btp_accounts.csv data/blr/btp_accounts.md data/blr/nitter_groups.json \
  data/blr/osm_points.geojsonseq data/blr/osm_lines.geojsonseq data/blr/osm_polys.geojsonseq ${NITTER_HOST:?user@nitter-host}:twtrx/
# not pushed: data/blr/powerbi/, junctions.csv, station_areas.json (mss rebuilds them: weekly btp_powerbi + btp_congestion)
echo "pushed code to mss:~/twtrx"
