#!/bin/bash
# Install / update the eta-ml service on raw (user@map-server:~/projects/eta-ml). Run from the Mac:
#
#   serve/deploy/deploy_raw.sh            ship code + models + config, venv, install the unit (not started)
#   serve/deploy/deploy_raw.sh --start    ... and (re)start it
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
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/eta-ml/serve" "$T/eta-ml/models" "$T/eta-ml/osm"
cp "$SERVE"/eta_engine.py "$SERVE"/nowcast.py "$SERVE"/eta_ml_service.py "$SERVE"/eta_ml_merge.py "$SERVE"/shadow_eval.py "$T/eta-ml/serve/"
cp "$ETA"/models/v4_clean_ensemble/{v4_lgb_huber.txt,v4_xgb_huber.json,v4_cat_mae.cbm,v4_lgb_q10.txt,v4_lgb_q90.txt,v4_report.json} "$T/eta-ml/models/"
cp "$ETA"/data/holidays_india_2025_2026.json "$T/eta-ml/"
cp "$ETA"/data/osm/*_osm_infra.json "$T/eta-ml/osm/"
cp "$HERE"/cities.raw.json "$T/eta-ml/cities.json"
cp "$HERE"/requirements.txt "$HERE"/eta-ml.service "$T/eta-ml/"
COPYFILE_DISABLE=1 tar -C "$T" -cf "$T/eta-ml.tar" eta-ml
RAWH="$RAWH" "$PUSH" "$T/eta-ml.tar" eta-ml-deploy.tar
ssh $RAWH 'set -e
  mkdir -p ~/projects/eta-ml/{out,logs} ~/backups/deploy
  if [ -d ~/projects/eta-ml/serve ]; then ts=$(date +%Y%m%d-%H%M%S); tar -C ~/projects -cf ~/backups/deploy/eta-ml-$ts.tar --exclude=eta-ml/.venv --exclude=eta-ml/out --exclude=eta-ml/logs eta-ml; echo "old copy: ~/backups/deploy/eta-ml-$ts.tar"; fi
  tar -C ~/projects -xf ~/eta-ml-deploy.tar && rm -f ~/eta-ml-deploy.tar
  cd ~/projects/eta-ml
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip >/dev/null
  .venv/bin/pip install -q -r requirements.txt || .venv/bin/pip install -q "numpy>=1.24,<2.4" lightgbm==4.7.0 xgboost==3.4.1 catboost==1.2.10
  .venv/bin/python -c "import sys; sys.path.insert(0, \"serve\"); import eta_engine as E; e = E.Ensemble(\"models\"); print(\"models load:\", sorted(e.m))"
  install -m 644 eta-ml.service ~/.config/systemd/user/eta-ml.service
  systemctl --user daemon-reload
  systemctl --user enable eta-ml.service >/dev/null 2>&1
  echo installed'
if [ "${1:-}" = "--start" ]; then
  ssh $RAWH 'systemctl --user restart eta-ml.service; sleep 90; systemctl --user --no-pager status eta-ml.service | head -12; tail -n 8 ~/projects/eta-ml/logs/eta-ml.log; ls -la ~/projects/eta-ml/out'
fi
