#!/bin/bash
# v4 arrival times for the RTC maps (collectors on SVR, model on raw). Run from the Mac, one step at a time:
#
#   serve/deploy/deploy_rtc.sh feed             1. raw: eta-rtc-feed (reads the maps' vehicles.json from SVR's feed server,
#                                                  read-only) -> ~/projects/eta-ml/rtc/<system>/obs-*.jsonl. Touches no map.
#   serve/deploy/deploy_rtc.sh model            2. raw: code to serve-rtc/, networks + seed model.json + OSM, cities.rtc.json
#                                                  (all systems off), units eta-ml-rtc, eta-ml-push, eta-rtc-model.timer
#                                                  installed; eta-ml-rtc + push started (scores nothing until step 3)
#   serve/deploy/deploy_rtc.sh apnet            APSRTC's cumulative network (25 Nov: ~73k services on ~15k stop chains, every
#                                                  service ever met, not only those running at the last rebuild) + its seed
#                                                  model.json (stop ids are now places, so the old one no longer applies) + the
#                                                  code that reads it (engine aliases, feed, rtc_model.sh); eta-ml-rtc and the
#                                                  feed restarted, eta-rtc-model run once. The SVR collector needs nothing.
#   serve/deploy/deploy_rtc.sh on SYSTEM        3. one system: on in cities.rtc.json (eta-ml-rtc restarted), then its SVR
#                                                  collector gets the merge (files checked against git first, old copies to
#                                                  /data/rtc/backups/<ts>-eta-ml-<system>/) + a drop-in with ETA_ML, restarted
#   serve/deploy/deploy_rtc.sh off SYSTEM       the collector's drop-in removed and restarted (its own times again),
#                                                  the system off in cities.rtc.json
#   SYSTEM = tricity | tgsrtc | apsrtc          (audited and better than what the map shows; see eta-rtc-audit.md)
#
# The cards (80% range, APSRTC's predicted times) are maps-site branch claude/eta-v4-rtc:
#   assets/busmap/server/deploy_city.sh shared|tricity|tgsrtc|apsrtc   (after step 3 for that system; --dry first)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SERVE=$(dirname "$HERE")
P=~/projects
ETA=$P/ETA
RAWH=${RAWH:-"-J user@jump-host user@map-server"}
SVRH=${SVRH:-"-J user@map-server user@RTC_SERVER_IP"}
PUSH=$P/transit-density/tools/push.sh
STEP=${1:?feed|model|on|off}; SYS=${2:-}; EXTRA=
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

ship() {   # $1 = tar of files under eta-ml/
  RAWH="$RAWH" "$PUSH" "$1" eta-rtc-deploy.tar
  ssh $RAWH 'set -e; mkdir -p ~/projects/eta-ml/{out/rtc,logs,rtc/net} ~/backups/deploy
    tar --warning=no-unknown-keyword -C ~/projects -xf ~/eta-rtc-deploy.tar && rm -f ~/eta-rtc-deploy.tar'
}

case $STEP in
feed)
  mkdir -p "$T/eta-ml/serve-rtc"
  cp "$SERVE"/rtc_feed.py "$SERVE"/rtc_adapt.py "$T/eta-ml/serve-rtc/"
  cp "$HERE"/rtc_feed.raw.json "$T/eta-ml/rtc_feed.json"
  cp "$HERE"/eta-rtc-feed.service "$T/eta-ml/"
  COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/x.tar" eta-ml && ship "$T/x.tar"
  ssh $RAWH 'set -e; cd ~/projects/eta-ml; python3 -m py_compile serve-rtc/rtc_feed.py serve-rtc/rtc_adapt.py
    install -m 644 eta-rtc-feed.service ~/.config/systemd/user/ && systemctl --user daemon-reload
    systemctl --user enable --now eta-rtc-feed.service && sleep 60
    systemctl --user --no-pager status eta-rtc-feed.service | head -5; wc -l rtc/*/obs-*.jsonl'
  ;;
model)
  mkdir -p "$T/eta-ml/serve-rtc" "$T/eta-ml/rtc/net" "$T/eta-ml/osm"
  cp "$SERVE"/eta_engine.py "$SERVE"/nowcast.py "$SERVE"/eta_ml_service.py "$SERVE"/eta_ml_merge.py "$SERVE"/rtc_feed.py \
     "$SERVE"/rtc_adapt.py "$SERVE"/rtc_model.sh "$SERVE"/rtc_push.sh "$T/eta-ml/serve-rtc/"
  for s in tricity tgsrtc apsrtc; do
    mkdir -p "$T/eta-ml/rtc/net/$s"
    cp "$ETA/data/meta/$s/route_stops.json" "$ETA/data/meta/$s/stop_dist.json" "$ETA/data/meta/$s/route_names.json" "$T/eta-ml/rtc/net/$s/"
    cp "$ETA/data/audit/model_${s}_until_2026-11-23.json" "$T/eta-ml/rtc/net/$s/model.json"
    cp "$ETA/data/osm/${s}_osm_infra.json" "$T/eta-ml/osm/"
  done
  cp "$HERE"/cities.rtc.json "$T/eta-ml/cities.rtc.json.new"
  cp "$HERE"/eta-ml-rtc.service "$HERE"/eta-ml-push.service "$HERE"/eta-rtc-model.service "$HERE"/eta-rtc-model.timer "$T/eta-ml/"
  COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/x.tar" eta-ml && ship "$T/x.tar"
  ssh $RAWH 'set -e; cd ~/projects/eta-ml
    [ -f cities.rtc.json ] && rm -f cities.rtc.json.new || mv cities.rtc.json.new cities.rtc.json   # a live config is kept
    .venv/bin/python -m py_compile serve-rtc/*.py
    for u in eta-ml-rtc.service eta-ml-push.service eta-rtc-model.service eta-rtc-model.timer; do install -m 644 $u ~/.config/systemd/user/; done
    systemctl --user daemon-reload
    systemctl --user enable --now eta-rtc-model.timer eta-ml-push.service
    systemctl --user enable --now eta-ml-rtc.service && sleep 30
    systemctl --user --no-pager status eta-ml-rtc.service eta-ml-push.service | grep -E "Active|Memory"; tail -3 logs/eta-ml-rtc.log'
  ;;
apnet)
  M=$ETA/data/meta/apsrtc2
  SEED=$ETA/data/audit/model_apsrtc2_until_2026-11-24.json
  [ -f "$SEED" ] || { echo "no $SEED (build it first: see RTC_AUDIT.md)"; exit 1; }
  for f in eta_engine.py eta_ml_service.py rtc_feed.py rtc_adapt.py rtc_model.sh; do   # never over a hot patch on raw
    known=$(for c in $(git -C "$SERVE" log --format=%H -- "$f"); do git -C "$SERVE" rev-parse "$c:serve/$f" 2>/dev/null; done)
    live=$(ssh $RAWH "cd ~/projects/eta-ml/serve-rtc && if [ -f $f ]; then git hash-object $f; else echo none; fi")
    if [ "$live" != none ] && ! grep -qx "$live" <<<"$known"; then echo "SKIP: raw serve-rtc/$f is not a committed version (hot patch?); nothing sent"; exit 1; fi
  done
  mkdir -p "$T/eta-ml/serve-rtc" "$T/eta-ml/rtc/net/apsrtc.new"
  cp "$SERVE"/eta_engine.py "$SERVE"/eta_ml_service.py "$SERVE"/rtc_feed.py "$SERVE"/rtc_adapt.py "$SERVE"/rtc_model.sh "$T/eta-ml/serve-rtc/"
  cp "$M"/route_stops.json "$M"/stop_dist.json "$M"/route_names.json "$T/eta-ml/rtc/net/apsrtc.new/"
  cp "$M"/ap_chains.json.gz "$T/eta-ml/rtc/net/apsrtc.new/ap_chains.seed.json.gz"
  cp "$SEED" "$T/eta-ml/rtc/net/apsrtc.new/model.json"
  COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/x.tar" eta-ml && ship "$T/x.tar"
  ssh $RAWH 'set -e; cd ~/projects/eta-ml; .venv/bin/python -m py_compile serve-rtc/*.py
    ts=$(date +%Y%m%d-%H%M%S); mkdir -p ~/backups/deploy/$ts-apnet && cp -p rtc/net/apsrtc/* ~/backups/deploy/$ts-apnet/
    for f in route_stops.json stop_dist.json route_names.json model.json; do mv rtc/net/apsrtc.new/$f rtc/net/apsrtc/$f.new; done
    mv rtc/net/apsrtc.new/ap_chains.seed.json.gz rtc/net/apsrtc/ && rmdir rtc/net/apsrtc.new
    for f in model.json stop_dist.json route_stops.json route_names.json; do mv rtc/net/apsrtc/$f.new rtc/net/apsrtc/$f; done
    systemctl --user restart eta-rtc-feed.service eta-ml-rtc.service && systemctl --user start eta-rtc-model.service
    sleep 90; tail -3 logs/eta-ml-rtc.log; tail -4 logs/eta-rtc-model.log
    echo "old network: ~/backups/deploy/$ts-apnet"'
  echo "check: curl -s https://maps.reclaim.city/apsrtc/live/etas.json | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"eta\"))'"
  ;;
on|off)
  case $SYS in
    tricity) REPO=$P/chandigarh_bus; DIR=/data/rtc/tricity; UNIT=tricity-live.service; FILES="server/live.py server/eta_ml_merge.py" ;;
    tgsrtc)  REPO=$P/tsrtc;          DIR=/data/rtc/tsrtc;   UNIT=tsrtc-eta.service;    FILES="server/eta.py server/eta_ml_merge.py server/incident_penalty.py"
             EXTRA=" ETA_ML_HMAX=1200" ;;   # shadow 24 Nov: v4 better to 20 min ahead, tsrtc-eta's timetable blend better beyond (n=74)
    apsrtc)  REPO=$P/apsrtc;         DIR=/data/rtc/apsrtc;  UNIT=apsrtc-live.service;  FILES="live.py eta_ml_merge.py" ;;
    *) echo "SYSTEM: tricity | tgsrtc | apsrtc"; exit 1 ;;
  esac
  flip() { ssh $RAWH "cd ~/projects/eta-ml && python3 -c 'import json; p=\"cities.rtc.json\"; j=json.load(open(p))
for c in j[\"cities\"]:
    c[\"enabled\"] = (c[\"enabled\"] if c[\"city\"] != \"$SYS\" else $1)
json.dump(j, open(p + \".tmp\", \"w\"), indent=1)' && mv cities.rtc.json.tmp cities.rtc.json && systemctl --user restart eta-ml-rtc.service && echo 'eta-ml-rtc: $SYS $1'"; }
  if [ "$STEP" = off ]; then
    ssh $SVRH "rm -f ~/.config/systemd/user/$UNIT.d/60-eta-ml.conf && systemctl --user daemon-reload && systemctl --user restart $UNIT && echo '$UNIT: eta-ml removed, restarted'"
    flip False; exit 0
  fi
  flip True
  # the live copies must be a committed version of the file (or absent, for a file new to the collector): never over a hot patch
  for f in $FILES; do
    known=$(for c in $(git -C "$REPO" log --format=%H -- "$f"); do git -C "$REPO" rev-parse "$c:$f" 2>/dev/null; done)
    live=$(ssh $SVRH "cd $DIR && if [ -f $f ]; then git hash-object $f; else echo none; fi")
    if [ "$live" != none ] && ! grep -qx "$live" <<<"$known"; then echo "SKIP: SVR $DIR/$f is not a committed version (hot patch?); not touching it"; exit 1; fi
  done
  ts=$(date +%Y%m%d-%H%M%S)
  ssh $SVRH "mkdir -p /data/rtc/backups/$ts-eta-ml-$SYS && cd $DIR && for f in $FILES; do [ -f \$f ] && cp -p --parents \$f /data/rtc/backups/$ts-eta-ml-$SYS/ || true; done"
  for f in $FILES; do git -C "$REPO" show "HEAD:$f" | ssh $SVRH "cat > $DIR/$f.new && mv $DIR/$f.new $DIR/$f"; done
  ssh $SVRH "set -e; cd $DIR && python3 -m py_compile $FILES
    until [ -f /data/rtc/eta-ml/net/$SYS/route_names.json ]; do echo 'waiting for eta-ml-push to send the network'; sleep 20; done
    mkdir -p ~/.config/systemd/user/$UNIT.d
    printf '[Service]\nEnvironment=ETA_ML=/data/rtc/eta-ml/out/$SYS.json ETA_ML_STOPS=/data/rtc/eta-ml/net/$SYS/route_names.json ETA_ML_MODE=all${EXTRA:-}\n' > ~/.config/systemd/user/$UNIT.d/60-eta-ml.conf
    systemctl --user daemon-reload && systemctl --user restart $UNIT && sleep 90 && systemctl --user --no-pager status $UNIT | head -4
    echo backups: /data/rtc/backups/$ts-eta-ml-$SYS"
  echo "check: curl -s https://maps.reclaim.city/$SYS/live/etas.json | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"eta\"))'"
  ;;
*) echo "feed | model | apnet | on SYSTEM | off SYSTEM"; exit 1 ;;
esac
