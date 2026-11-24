#!/bin/bash
# Install / update the eta-ml service on raw (user@map-server:~/projects/eta-ml). Run from the Mac:
#
#   serve/deploy/deploy_raw.sh            ship code + models + config, venv, install the unit (not started)
#   serve/deploy/deploy_raw.sh --start    ... and (re)start it
#   serve/deploy/deploy_raw.sh --no-push --start   files already on raw: only venv, unit, start
#   serve/deploy/deploy_raw.sh --code --start      only serve/*.py (one small piece), then restart
#
# The venv is Python 3.12 made by uv (~/.local/bin/uv): the models were saved by xgboost 3.4.1, which needs
# Python >= 3.12, and raw's system python is 3.11 (newest xgboost there: 3.2.0).
#
# What goes over: serve/*.py, the v4 models (models/v4_clean_ensemble: 6 files, ~55 MB), the holiday
# calendar, OSM infrastructure points per city, cities.raw.json (as cities.json), the systemd user unit.
# One tarball through transit-density/tools/push.sh (the laptop->VPS path-MTU blackhole wedges big scp).
# Nothing else on raw is touched: the collectors only use eta-ml once collectors.sh adds --eta-ml to them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SERVE=$(dirname "$HERE")
ETA=~/projects/ETA
RAWH=${RAWH:-"-J user@jump-host user@map-server"}
PUSH=~/projects/transit-density/tools/push.sh
NOPUSH=0; START=0; CODE=0
for a in "$@"; do case $a in --no-push) NOPUSH=1 ;; --start) START=1 ;; --code) CODE=1 ;; *) echo "unknown option $a"; exit 1 ;; esac; done
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
if [ $NOPUSH = 0 ]; then
mkdir -p "$T/eta-ml/serve" "$T/eta-ml/models" "$T/eta-ml/osm"
cp "$SERVE"/eta_engine.py "$SERVE"/nowcast.py "$SERVE"/eta_ml_service.py "$SERVE"/eta_ml_merge.py "$SERVE"/shadow_eval.py "$T/eta-ml/serve/"
cp "$ETA"/models/v4_clean_ensemble/{v4_lgb_huber.txt,v4_xgb_huber.json,v4_cat_mae.cbm,v4_lgb_q10.txt,v4_lgb_q90.txt,v4_report.json} "$T/eta-ml/models/"
cp "$ETA"/data/holidays_india_2025_2026.json "$T/eta-ml/"
cp "$ETA"/data/osm/*_osm_infra.json "$T/eta-ml/osm/"
cp "$HERE"/cities.raw.json "$T/eta-ml/cities.json"
cp "$HERE"/requirements.txt "$HERE"/eta-ml.service "$T/eta-ml/"
if [ $CODE = 1 ]; then COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/eta-ml.tar" eta-ml/serve
else COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/eta-ml.tar" eta-ml; fi
RAWH="$RAWH" "$PUSH" "$T/eta-ml.tar" eta-ml-deploy.tar
ssh $RAWH 'set -e
  mkdir -p ~/projects/eta-ml/{out,logs} ~/backups/deploy
  if [ -d ~/projects/eta-ml/serve ]; then ts=$(date +%Y%m%d-%H%M%S); tar -C ~/projects -cf ~/backups/deploy/eta-ml-$ts.tar --exclude=eta-ml/.venv --exclude=eta-ml/out --exclude=eta-ml/logs eta-ml; echo "old copy: ~/backups/deploy/eta-ml-$ts.tar"; fi
  tar --warning=no-unknown-keyword -C ~/projects -xf ~/eta-ml-deploy.tar && rm -f ~/eta-ml-deploy.tar'
fi
ssh $RAWH 'set -e
  cd ~/projects/eta-ml
  UV=~/.local/bin/uv
  if ! [ -x .venv/bin/python ] || ! .venv/bin/python -c "import sys; sys.exit(sys.version_info < (3, 12))"; then
    rm -rf .venv && $UV venv -q -p 3.12 .venv
  fi
  $UV pip install -q -p .venv/bin/python "numpy>=1.24,<2.4" lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10
  .venv/bin/python -c "import sys; sys.path.insert(0, \"serve\"); import eta_engine as E; e = E.Ensemble(\"models\"); print(\"models load:\", sorted(e.m))"
  install -m 644 eta-ml.service ~/.config/systemd/user/eta-ml.service
  systemctl --user daemon-reload
  systemctl --user enable eta-ml.service >/dev/null 2>&1
  echo installed'
if [ $START = 1 ]; then
  ssh $RAWH 'systemctl --user restart eta-ml.service; sleep 90; systemctl --user --no-pager status eta-ml.service | head -12; tail -n 8 ~/projects/eta-ml/logs/eta-ml.log; ls -la ~/projects/eta-ml/out'
fi
