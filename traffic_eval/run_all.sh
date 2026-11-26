#!/bin/bash
# Replays every evaluated (system, day) through the live v4 engine (3 at a time). Output: work/q_*.npz, work/logs/
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
mkdir -p work/logs
jobs_list=(
  "tgsrtc 2026-11-17" "tgsrtc 2026-11-18" "tgsrtc 2026-11-19" "tgsrtc 2026-11-20" "tgsrtc 2026-11-21" "tgsrtc 2026-11-22" "tgsrtc 2026-11-23" "tgsrtc 2026-11-24"
  "bmtc 2026-11-22" "bmtc 2026-11-24"
  "mtc 2026-10-20" "mtc 2026-10-21" "mtc 2026-10-22" "mtc 2026-10-27" "mtc 2026-10-28" "mtc 2026-10-31" "mtc 2026-11-01" "mtc 2026-11-02"
  "delhi 2026-11-10" "delhi 2026-11-11" "delhi 2026-11-12" "delhi 2026-11-13" "delhi 2026-11-14" "delhi 2026-11-15" "delhi 2026-11-16" "delhi 2026-11-17" "delhi 2026-11-22" "delhi 2026-11-23" "delhi 2026-11-24"
)
for j in "${jobs_list[@]}"; do
  set -- $j
  [ -f "work/q_$1_$2.npz" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "${PAR:-3}" ]; do sleep 5; done
  echo "start $1 $2 $(date +%T)"
  extra=""; [ "$1" = delhi ] && extra="--every 6 --other 0.03"   # Delhi: ~10x the fixes of the others
  $PY replay_queries.py "$1" "$2" $extra > "work/logs/$1_$2.log" 2>&1 &
done
wait
echo done $(date +%T)
