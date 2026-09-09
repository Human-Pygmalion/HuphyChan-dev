#!/bin/bash
# 2026-09-09. 다섯 가지 명령을 각 10회씩. 명령마다 민감한 값이 다르다:
#   큰 계단 ±8도       뜨는 속도(관성)와 끝내 못 지운 오차(마찰)를 한 번에
#   작은 계단 ±1.5도   마찰의 죽은구간 안에서만 논다. frictionloss 에 가장 민감
#   처프 0.3~10 Hz     주파수 응답. armature 에 가장 민감
#   느린 삼각파 20도/s 일정 속도. 점성은 무시할 크기라 쿨롱 마찰만 보인다
#   빠른 삼각파 240도/s 일정 속도. 여기서 점성이 드러난다
set -e
cd "$(dirname "$0")"
python3 -u pd_profiles.py step     --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 8   --period-s 1.5 --cycles 4 --repeats 10 --tag b
python3 -u pd_profiles.py step     --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 1.5 --period-s 1.5 --cycles 4 --repeats 10 --tag b
python3 -u pd_profiles.py chirp    --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 8   --f0 0.3 --f1 10 --chirp-s 20 --repeats 10 --tag b
python3 -u pd_profiles.py triangle --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 20  --period-s 2.0 --cycles 4 --repeats 10 --tag b
python3 -u pd_profiles.py triangle --motor-id 4 --model RS04 --kp 20 --kd 1.0 --amp 60  --period-s 0.5 --cycles 8 --repeats 10 --tag bf
echo ALL_DONE
