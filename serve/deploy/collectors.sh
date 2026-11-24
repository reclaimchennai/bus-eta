#!/bin/bash
# Point ONE city's collector at eta-ml (one system at a time). Run from the Mac after deploy_raw.sh --start:
#
#   serve/deploy/collectors.sh mtc   [all|ours|off]    (default: all = v4 also replaces the operator's own ETAs; shadow 24 Nov: MTC 72 vs 129 s, BMTC 77 vs 335 s)
#   serve/deploy/collectors.sh bmtc  [all|ours|off]
#   serve/deploy/collectors.sh delhi [all|ours|off]
#   serve/deploy/collectors.sh CITY remove            drop the drop-in: the collector's own times again
#
# 1. copies the collector's three changed files (committed in its own repo) to raw, old copies to
#    ~/backups/deploy/<ts>-eta-ml-<city>/ (aborts if the live copy is not the committed HEAD~ version:
#    a hot patch on raw is never overwritten);
# 2. writes ~/.config/systemd/user/<unit>.d/60-eta-ml.conf with ETA_ML / ETA_ML_MODE;
# 3. restarts the collector. MTC's map empties for ~5 min after a restart (no resume), BMTC's ~2 min.
set -euo pipefail
CITY=${1:?city}; MODE=${2:-}
RAWH=${RAWH:-"-J user@jump-host user@map-server"}
P=~/projects
case $CITY in
  mtc)   REPO=$P/chennai_one; SUB=v36;    DIR=projects/mtc-live;   UNIT=mtc-its.service;   MAIN=mtcits_collector.py; DEF=all ;;
  bmtc)  REPO=$P/bengaluru_bus; SUB=server; DIR=projects/bmtc-live; UNIT=bmtc-live.service; MAIN=bmtc_collector.py;   DEF=all ;;
  delhi) REPO=$P/delhi_bus; SUB=server;   DIR=projects/delhi-live; UNIT=delhi-live.service; MAIN=delhi_collector.py;  DEF=all ;;
  *) echo "unknown city $CITY"; exit 1 ;;
esac
MODE=${MODE:-$DEF}
if [ "$MODE" = remove ]; then
  ssh $RAWH "rm -f ~/.config/systemd/user/$UNIT.d/60-eta-ml.conf && systemctl --user daemon-reload && systemctl --user restart $UNIT && echo '$UNIT: eta-ml removed, restarted'"
  exit 0
fi
FILES="live_collector.py eta_ml_merge.py $MAIN"
# the live copies must be what this repo had before the eta-ml commit (HEAD~1 of each file's history)
for f in $FILES; do
  want=$(git -C "$REPO" log -n 2 --format=%H -- "$SUB/$f" | tail -1)
  prev=$( [ "$f" = eta_ml_merge.py ] && echo none || git -C "$REPO" rev-parse "$want:$SUB/$f")
  head=$(git -C "$REPO" rev-parse "HEAD:$SUB/$f")
  live=$(ssh $RAWH "cd ~/$DIR && [ -f $f ] && git hash-object $f || echo none")
  if [ "$live" != "$prev" ] && [ "$live" != "$head" ]; then echo "SKIP: raw ~/$DIR/$f is neither the committed old version nor HEAD (hot patch?); not touching it"; exit 1; fi
done
ts=$(date +%Y%m%d-%H%M%S)
ssh $RAWH "mkdir -p ~/backups/deploy/$ts-eta-ml-$CITY && cd ~/$DIR && cp -p $(echo $FILES | sed 's/eta_ml_merge.py//') ~/backups/deploy/$ts-eta-ml-$CITY/"
for f in $FILES; do
  git -C "$REPO" show "HEAD:$SUB/$f" | ssh $RAWH "cat > ~/$DIR/$f.new && mv ~/$DIR/$f.new ~/$DIR/$f"
done
ssh $RAWH "cd ~/$DIR && python3 -m py_compile $FILES && mkdir -p ~/.config/systemd/user/$UNIT.d && printf '[Service]\nEnvironment=ETA_ML=%%h/projects/eta-ml/out/$CITY.json ETA_ML_MODE=$MODE\n' > ~/.config/systemd/user/$UNIT.d/60-eta-ml.conf && systemctl --user daemon-reload && systemctl --user restart $UNIT && sleep 60 && systemctl --user --no-pager status $UNIT | head -5; echo backups: ~/backups/deploy/$ts-eta-ml-$CITY"
echo "check: curl -s https://maps.reclaimchennai.city/<app>/live/vehicles.json | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"eta\"))'"
