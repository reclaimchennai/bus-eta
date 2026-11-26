#!/usr/bin/env bash
# Self-hosted Nitter on mss, only for the traffic police feeds (HYDTP, CYBTRAFFIC): latest posts, no history.
#
#   ./setup.sh            from the Mac: copies this folder to mss:~/nitter and starts Redis + Nitter (rootless podman)
#   ./setup.sh status     is it up, what the two feeds return
#
# Needs on mss (once, by the owner): sudo apt-get install -y podman
# Needs ~/nitter/sessions.jsonl on mss: session tokens of a spare X account, made by the owner with Nitter's
# tools/create_session_browser.py or create_session_curl.py (it asks for that account's password / 2FA). Never in git.
#
# Both containers listen on 127.0.0.1 only (Nitter :8095, Redis :6390): nothing is exposed. cron keeps them up
# (mss has no user systemd): watchdog.sh every 5 min, and poll.sh every 2 min writes ~/nitter/out/latest.json
# (the newest posts of each account) for twtrx to fetch.
set -euo pipefail
cd "$(dirname "$0")"
MSS=(-o ConnectTimeout=20 ${NITTER_SSH:?set NITTER_SSH, e.g. "-J user@jump -p 22 user@nitter-host"})
if [ "${1:-}" = status ]; then
  ssh "${MSS[@]}" 'podman ps --format "{{.Names}} {{.Status}}"; for a in HYDTP CYBTRAFFIC; do printf "%s " $a; curl -s -o /dev/null -w "%{http_code} %{size_download}B\n" http://127.0.0.1:8095/$a/rss; done; ls -la ~/nitter/out/ 2>/dev/null; tail -3 ~/nitter/logs/*.log 2>/dev/null'
  exit 0
fi
COPYFILE_DISABLE=1 tar cf - nitter.conf.in watchdog.sh poll.py | ssh "${MSS[@]}" 'mkdir -p ~/nitter/out ~/nitter/logs && tar xf - -C ~/nitter 2>/dev/null; chmod +x ~/nitter/watchdog.sh
  cd ~/nitter
  command -v podman >/dev/null || { echo "podman missing: run  sudo apt-get install -y podman  on mss first"; exit 1; }
  mkdir -p ~/.config/containers; [ -f ~/.config/containers/containers.conf ] || printf "[engine]\ncgroup_manager = \"cgroupfs\"\nevents_logger = \"file\"\n" > ~/.config/containers/containers.conf   # no user systemd on mss
  [ -f nitter.conf ] || sed "s/@HMAC@/$(head -c 24 /dev/urandom | od -An -tx1 | tr -d " \n")/" nitter.conf.in > nitter.conf
  podman pull -q docker.io/library/redis:7-alpine docker.io/zedeus/nitter:latest
  ./watchdog.sh
  ( crontab -l 2>/dev/null | grep -v "/nitter/" ; echo "*/5 * * * * flock -n \$HOME/nitter/.wd \$HOME/nitter/watchdog.sh >> \$HOME/nitter/logs/watchdog.log 2>&1"; echo "@reboot sleep 60 && \$HOME/nitter/watchdog.sh >> \$HOME/nitter/logs/watchdog.log 2>&1"; echo "*/2 * * * * flock -n \$HOME/nitter/.poll /usr/bin/python3 \$HOME/nitter/poll.py >> \$HOME/nitter/logs/poll.log 2>&1" ) | crontab -
  sleep 8; podman ps --format "{{.Names}} {{.Status}}"
  [ -f sessions.jsonl ] || echo "NOTE: no sessions.jsonl yet: Nitter runs, but X will refuse it until the owner adds one"'
