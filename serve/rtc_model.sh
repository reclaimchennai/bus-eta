#!/bin/bash
# Every 3 h (eta-rtc-model.timer), for each enabled RTC system:
#  1. its network, as the map has it now: Tricity/AMTS/UP from live/net (services, stops, segs), APSRTC from
#     live/trips.json, TGSRTC + the workings eta-rtc-feed met that the network lacked (new_workings.jsonl);
#  2. model.json (historical segment speeds): the travel_model.py the MTC/BMTC/Mumbai units run (~/projects/mtc-live),
#     7 days, fix gaps up to 30 min read as one movement (RTC fixes are 30 s .. 5 min apart). Until a system has
#     8 days of feed archive it keeps the seed model.json deploy_rtc.sh shipped (built on the laptop from the
#     collectors' own archives, days before deploy only).
# eta-ml-rtc reloads a network whose file changed; eta-ml-push sends route_names.json on to SVR for the collectors.
set -u
cd ~/projects/eta-ml
BASE=http://RTC_SERVER_LAN:8091
TM=~/projects/mtc-live/travel_model.py
A=serve-rtc/rtc_adapt.py
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
for s in $(python3 -c 'import json; print(" ".join(c["city"] for c in json.load(open("cities.rtc.json"))["cities"] if c.get("enabled")))'); do
  N=rtc/net/$s
  case $s in
    apsrtc) curl -sf -m 120 $BASE/apsrtc/live/trips.json -o $T/trips.json && python3 $A ap-net $T/trips.json $N ;;
    tgsrtc) [ -f rtc/tgsrtc/new_workings.jsonl ] && python3 $A tg-extend $N rtc/tgsrtc/new_workings.jsonl ;;
    *) mkdir -p $T/$s && for f in services stops segs; do curl -sf -m 120 $BASE/$s/live/net/$f.json -o $T/$s/$f.json || continue 2; done
       python3 $A net $T/$s $N ;;
  esac
  n=$(ls rtc/$s/obs-*.jsonl* 2>/dev/null | wc -l)
  if [ "$n" -lt 8 ]; then echo "$(date +%F\ %T) $s: $n days of feed archive, keeping the seed model.json"; continue; fi
  python3 $TM --stops $N/route_stops.json --stop-dist $N/stop_dist.json --archive rtc/$s \
    --out $N/model.json.new --days 7 --max-gap 1800 && mv $N/model.json.new $N/model.json \
    && echo "$(date +%F\ %T) $s: model.json rebuilt"
done
