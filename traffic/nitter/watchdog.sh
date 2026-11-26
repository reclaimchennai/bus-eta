#!/usr/bin/env bash
# Keep Redis and Nitter running (rootless podman, host network, 127.0.0.1 only). Safe to run any time.
cd "$HOME/nitter" || exit 1
running() { [ "$(podman inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
if ! running nitter-redis; then
  podman rm -f nitter-redis >/dev/null 2>&1
  podman run -d --name nitter-redis --network host docker.io/library/redis:7-alpine \
    redis-server --bind 127.0.0.1 --port 6390 --save 60 1 --maxmemory 128mb --maxmemory-policy allkeys-lru >/dev/null \
    && echo "$(date '+%F %T') started redis"
fi
if ! running nitter; then
  podman rm -f nitter >/dev/null 2>&1
  SESS=(); [ -f sessions.jsonl ] && SESS=(-v "$HOME/nitter/sessions.jsonl:/src/sessions.jsonl:ro,Z")
  podman run -d --name nitter --network host -v "$HOME/nitter/nitter.conf:/src/nitter.conf:ro,Z" "${SESS[@]}" \
    docker.io/zedeus/nitter:latest >/dev/null && echo "$(date '+%F %T') started nitter"
fi
