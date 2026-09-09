#!/bin/bash
# 2026-09-10. 사용자 지시: "실험에 사용된 각도가 너무 작아. 30도로 다시 실험."
# 그리고 "Chirp 시작 주파수가 너무 커. 좀더 느긋하게 시작해."  → 0.3 Hz -> 0.1 Hz, 20초 -> 45초
#
# 각도를 30도로 키우면 계단 직후 추종 오차가 60도까지 벌어진다(명령이 −30 에서 +30 으로
# 뛰므로). 그래서 안전 경계를 25도에서 80도로 넓힌다. 진짜 폭주는 여전히 잡힌다 — 호스트
# PD 의 토크 상한이 8 뉴턴미터라 최대 가속도가 103 라디안/s^2 이고, 80도를 지나가는 데
# 0.165초(표본 660개)가 걸린다.
#
# 작은 계단(±1.5도)은 그대로 둔다. 마찰의 죽은구간 안에서만 노는 유일한 명령이고,
# 정지 마찰이 빠진 자리를 찾아낸 것이 이 명령이다.
set -e
cd "$(dirname "$0")"
M=${1:-4}; MODEL=${2:-RS04}
python3 -u pd_profiles.py step     --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 30  --period-s 1.5  --cycles 4  --repeats 10 --bound-deg 80 --tag c
python3 -u pd_profiles.py step     --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 1.5 --period-s 1.5  --cycles 4  --repeats 10 --bound-deg 25 --tag c
python3 -u pd_profiles.py chirp    --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 30  --f0 0.1 --f1 10 --chirp-s 45 --repeats 10 --bound-deg 80 --tag c
python3 -u pd_profiles.py triangle --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 30  --period-s 3.0  --cycles 3  --repeats 10 --bound-deg 80 --tag c
python3 -u pd_profiles.py triangle --motor-id $M --model $MODEL --kp 20 --kd 1.0 --amp 30  --period-s 0.25 --cycles 12 --repeats 10 --bound-deg 80 --tag cf
echo ALL_DONE
