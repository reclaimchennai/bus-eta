#!/usr/bin/env bash
# traffic-api container. Re-run to restart with a new api.py.
# MAPS_SITE_DIR: a checkout of the bus maps' web root on this machine (read-only). Files found there are read from disk;
# anything else is fetched from https://maps.reclaim.city. Leave it unset to fetch everything over HTTPS.
# Expects a docker network "web" shared with the reverse proxy, which routes /traffic/* to traffic-api:5150.
set -euo pipefail
cd "$(dirname "$0")"
MOUNT=()
if [ -n "${MAPS_SITE_DIR:-}" ]; then MOUNT=(-v "$MAPS_SITE_DIR:/maps-site:ro"); fi
docker rm -f traffic-api >/dev/null 2>&1 || true
docker run -d --name traffic-api --restart unless-stopped --network web --user "$(id -u):$(id -g)" \
  --memory 1g --cap-drop ALL --security-opt no-new-privileges:true --read-only --tmpfs /tmp:size=16m \
  -e MAPS_SITE=/maps-site/ "${MOUNT[@]}" \
  -v "$PWD/api.py:/app/api.py:ro" \
  -w /app python:3.12-alpine python3 -u api.py --port 5150
