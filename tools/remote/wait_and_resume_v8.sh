#!/usr/bin/env bash
# Wait until the 3090 is actually free, then wake v8 where it was stopped.
#
# "Free" means: no compute process on the card other than v8's own suspended context
# (pid 296900), which SIGSTOP cannot release. The connectome control run holds the card
# until it finishes both arms; this only acts once it is gone.
#
# If v8's processes disappear while waiting, stop and say so rather than resuming
# something that is no longer there.
H=10.8.0.14
V8=296900
while true; do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=15 "$H" \
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader; echo ---; ps -o stat= -p $V8 2>/dev/null" 2>/dev/null) || { sleep 180; continue; }
  alive=$(echo "$out" | sed -n '/^---$/,$p' | tail -n +2 | tr -d ' ')
  if [ -z "$alive" ]; then
    echo "STOP: v8 트레이너($V8)가 사라졌다 — 재개하지 않고 멈춘다"
    exit 1
  fi
  others=$(echo "$out" | sed -n '1,/^---$/p' | grep -v '^---$' | tr -d ' ' | grep -v "^$V8$" | grep -c . || true)
  if [ "$others" = "0" ]; then
    echo "카드가 비었다 — v8 을 깨운다 ($(date '+%m-%d %H:%M'))"
    timeout 120 ssh -o BatchMode=yes "$H" /tmp/resume_v8.sh 2>&1 | tail -8
    exit 0
  fi
  sleep 180
done
