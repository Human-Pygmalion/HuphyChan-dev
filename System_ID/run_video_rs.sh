#!/bin/bash
# 카메라로 찍으면서 모터를 돌린다. **시각을 맞출 필요가 없다.**
#
# 2026-09-10. 카메라와 모터가 같은 컴퓨터에 붙어 있고, 둘 다 그 컴퓨터의 같은 시계
# (CLOCK_MONOTONIC)로 시각을 적습니다. 카메라 쪽은 커널이 프레임마다 찍어 주고
# (`v4l2_capture.py`), 모터 쪽은 기록마다 쪽지(`.meta.json`)에 시작 시각을 남깁니다.
# 그래서 맞추는 일이 **뺄셈 하나**입니다.
#
# 앞의 웹캠 촬영에서는 이게 없어서 초당 장수·지연·각도 눈금 셋을 한꺼번에 수색해야 했고,
# 초당 15장인 줄 알았는데 실제로는 8.6장이라 영상 시계가 1.75배 빨리 갔습니다.
#
# ★ 이 카메라는 해상도와 초당 장수를 **동시에** 최대로 못 냅니다 (실측):
#       1920x1080 → 초당 30장      848x480 → 초당 59.5장
#   그래서 보여줄 영상은 높은 해상도로, 빠른 파형은 높은 장수로 따로 찍습니다.
#
# 카메라를 **먼저** 켜고 2초 뒤에 모터를 시작합니다. 앞부분에 "아직 안 움직이는" 구간이
# 남아야 눈으로도 시작점을 확인할 수 있습니다.
set -e
cd "$(dirname "$0")"
M=${1:-4}; MODEL=${2:-RS04}; MODE=${3:-videosync}
W=${4:-1920}; H=${5:-1080}; FPS=${6:-30}; SECS=${7:-32}
OUT=${8:-$HOME/System_ID/data/cam_${MODE}_${M}_${W}x${H}}
EXPOSURE=${9:-150}

# ★ 노출과 화이트밸런스를 **고정**한다. 자동으로 두면 촬영 중에 밝기와 색이 움직이는데,
# 막대를 색(RS04)이나 어두움(RS03)으로 찾기 때문에 그러면 각도 읽기가 흔들린다. 실제로
# 자동일 때 RS04 의 "여유"(진짜 막대가 다른 후보보다 얼마나 뚜렷한가)가 22 에서 4.9 로
# 떨어지는 것을 봤다. 고정하면 프레임을 다시 찍어도 같은 값이 나온다.
#
# 알맞은 값이 모터마다 다르다 (실측):
#   RS04 붉은기로 찾음 → 밝을수록 색이 살아남   노출 200 에서 여유 18
#   RS03 어두운 막대로 찾음 → 어두울수록 대비   노출 100 에서 여유 72 (200 에서는 12)
v4l2-ctl -d /dev/video4 -c auto_exposure=1,white_balance_automatic=0 2>/dev/null || true
v4l2-ctl -d /dev/video4 -c white_balance_temperature=4600,exposure_time_absolute=$EXPOSURE 2>/dev/null || true
sleep 1.5

rm -rf "$OUT"
python3 -u v4l2_capture.py --out "$OUT" --seconds $SECS --width $W --height $H --fps $FPS &
CAM=$!
sleep 2
if [ "$MODE" = "chirp" ]; then
  python3 -u pd_profiles.py chirp --motor-id $M --model $MODEL --kp 20 --kd 1.0 \
          --amp 30 --f0 0.1 --f1 10 --chirp-s 45 --repeats 1 --bound-deg 80 --tag rs
else
  python3 -u pd_profiles.py videosync --motor-id $M --model $MODEL --kp 20 --kd 1.0 \
          --amp 30 --period-s 3.0 --cycles 3 --repeats 1 --bound-deg 80 --tag rs
fi
wait $CAM
echo "촬영: $OUT"
echo VIDEO_RUN_DONE
