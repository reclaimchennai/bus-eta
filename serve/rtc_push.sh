#!/bin/bash
# eta-ml-rtc's output (~/projects/eta-ml/out/rtc/<system>.json) and the networks it scores against
# (rtc/net/<system>/route_stops.json) -> SVR /data/rtc/eta-ml/{out,net}, where the RTC collectors merge them; every 15 s,
# compressed, over one ssh connection kept open. SVR reaches raw only on 80/443, so raw pushes.
SVR=${SVR:-user@RTC_SERVER_IP}
CM="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-eta-push-%r@%h -o ControlPersist=600 -o ServerAliveInterval=30 -o BatchMode=yes"
cd ~/projects/eta-ml
ssh $CM "$SVR" 'mkdir -p /data/rtc/eta-ml/out /data/rtc/eta-ml/net'
i=0
while true; do
  rsync -az --timeout=20 -e "ssh $CM" out/rtc/ "$SVR":/data/rtc/eta-ml/out/ || echo "$(date +%T) push failed"
  if [ $((i % 20)) = 0 ]; then       # networks: every 5 min (they change every few hours at most)
    rsync -az --timeout=60 -e "ssh $CM" --include='*/' --include='route_stops.json' --exclude='*' rtc/net/ "$SVR":/data/rtc/eta-ml/net/ || echo "$(date +%T) net push failed"
  fi
  i=$((i + 1)); sleep 15
done
