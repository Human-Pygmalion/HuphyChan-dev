#!/bin/bash
# 2026-09-10. RealSense D435i 로 찍으면서 모터를 돌린다.
#
# 앞의 웹캠 촬영과 다른 점은 **시각을 맞출 필요가 없다**는 것입니다. 카메라와 모터가 같은
# 컴퓨터에 붙어 있으므로 둘 다 그 컴퓨터의 같은 시계로 시각을 적습니다. 앞에서는 초당 장수를
# 몰라서(15인 줄 알았는데 8.6이었습니다) 프레임률·지연·각도눈금 셋을 한꺼번에 수색해야
# 했습니다.
#
# 카메라를 **먼저** 켜고 2초 뒤에 모터를 시작합니다. 그래야 촬영 앞부분에 "아직 안 움직이는"
# 구간이 남아서, 눈으로도 시작점을 확인할 수 있습니다.
#
# 쓰기 전에 반드시:  python3 rs_capture.py --probe
#   이 카메라가 **실제로 내주는** 해상도·장수만 출력합니다. 촬영도 안 하고 모터도 안 건드립니다.
set -e
cd "$(dirname "$0")"
M=${1:-4}; MODEL=${2:-RS04}; W=${3:-1280}; H=${4:-720}; FPS=${5:-60}
OUT=${6:-$HOME/System_ID/data/cam_rs_${M}}

SECS=32          # 모터 파형이 약 22초 + 앞뒤 여유
python3 -u rs_capture.py --out "$OUT" --seconds $SECS --width $W --height $H --fps $FPS &
CAM=$!
sleep 2
python3 -u pd_profiles.py videosync --motor-id $M --model $MODEL --kp 20 --kd 1.0 \
        --amp 30 --period-s 3.0 --cycles 3 --repeats 1 --bound-deg 80 --tag rs
wait $CAM
echo "촬영: $OUT"
echo VIDEO_RUN_DONE
