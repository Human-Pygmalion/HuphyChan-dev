#!/bin/bash
# 2026-09-10. 카메라와 나란히 볼 영상을 위해 한 파형만 돌린다.
# 원점 복귀(움직임) → 3초 정지 → 파형.  정지 구간이 있어야 영상과 자료를 맞출 때
# "언제 움직이기 시작했는가"를 잡을 수 있다.
set -e
cd "$(dirname "$0")"
M=${1:-4}; MODEL=${2:-RS04}; MODE=${3:-videosync}
python3 -u pd_profiles.py $MODE --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 30 \
        --period-s 3.0 --cycles 3 --period-s 3.0 --repeats 1 --bound-deg 80 --tag vid
echo VIDEO_RUN_DONE
