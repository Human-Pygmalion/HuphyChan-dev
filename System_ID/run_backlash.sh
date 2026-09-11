#!/bin/bash
# 2026-09-11. 헐거움 측정: 카메라로 원판을, 모터 엔코더로 회전자 쪽을 **동시에** 본다.
# 카메라를 먼저 켜고 2초 뒤 모터를 시작한다 (앞부분에 안 움직이는 구간이 남도록).
set -e
cd "$(dirname "$0")"
M=${1:-4}; MODEL=${2:-RS04}; OUT=/tmp/cam_backlash_${M}
v4l2-ctl -d /dev/video4 -c auto_exposure=1,white_balance_automatic=0 2>/dev/null || true
v4l2-ctl -d /dev/video4 -c white_balance_temperature=4600,exposure_time_absolute=200 2>/dev/null || true
sleep 1.5
rm -rf "$OUT"
python3 -u v4l2_capture.py --out "$OUT" --seconds 30 --width 1280 --height 720 --fps 30 &
CAM=$!
sleep 2
python3 -u backlash_probe.py --motor-id $M --model $MODEL --period-s 12 --cycles 2
wait $CAM
echo BACKLASH_RUN_DONE
