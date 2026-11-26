#!/bin/bash
# Every 3 h (eta-rtc-model.timer), for each enabled RTC system:
#  1. its network, as the map has it now: Tricity/AMTS/UP from live/net (services, stops, segs); TGSRTC + the workings
#     eta-rtc-feed met that the network lacked (new_workings.jsonl); APSRTC cumulative (net/apsrtc/ap_chains.json.gz:
#     every service ever met, from deploy_rtc.sh's seed ap_chains.seed.json.gz once, live/trips.json and the feed's
#     new_workings.jsonl), one entry per stop chain and an alias per service key;
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
mkdir -p tmp; T=$(mktemp -d -p ~/projects/eta-ml/tmp); trap 'rm -rf "$T"' EXIT   # not /tmp: APSRTC's canonical archive is GBs
for s in $(python3 -c 'import json; print(" ".join(c["city"] for c in json.load(open("cities.rtc.json"))["cities"] if c.get("enabled")))'); do
  N=rtc/net/$s
  case $s in
    apsrtc) [ -f $N/ap_chains.seed.json.gz ] && python3 $A ap-net $N $N/ap_chains.seed.json.gz && rm -f $N/ap_chains.seed.json.gz
            curl -sf -m 120 $BASE/apsrtc/live/trips.json -o $T/trips.json || : > $T/trips.json
            python3 $A ap-net $N $T/trips.json rtc/apsrtc/new_workings.jsonl ;;
    tgsrtc) [ -f rtc/tgsrtc/new_workings.jsonl ] && python3 $A tg-extend $N rtc/tgsrtc/new_workings.jsonl ;;
    *) mkdir -p $T/$s && for f in services stops segs; do curl -sf -m 120 $BASE/$s/live/net/$f.json -o $T/$s/$f.json || continue 2; done
       python3 $A net $T/$s $N ;;
  esac
  n=$(ls rtc/$s/obs-*.jsonl* 2>/dev/null | wc -l)
  if [ "$n" -lt 8 ]; then echo "$(date +%F\ %T) $s: $n days of feed archive, keeping the seed model.json"; continue; fi
  S=$N/route_stops.json; D=$N/stop_dist.json; AR=rtc/$s
  if grep -q '"a":' $S; then                  # aliases (APSRTC): travel_model reads the canonical view
    nice python3 $A canon $N $T/canon --archive rtc/$s --days 7 || continue
    S=$T/canon/route_stops.json; D=$T/canon/stop_dist.json; AR=$T/canon/archive
  fi
  nice python3 $TM --stops $S --stop-dist $D --archive $AR \
    --out $N/model.json.new --days 7 --max-gap 1800 && mv $N/model.json.new $N/model.json \
    && echo "$(date +%F\ %T) $s: model.json rebuilt"
  rm -rf $T/canon
done
