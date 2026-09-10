#!/bin/bash
# 벤치 영상 한 편을 **처음부터 끝까지** 만든다. 이 컴퓨터에서 돌린다.
#
# 2026-09-10. 다섯 단계가 두 기계에 나뉘어 있어서, 손으로 하면 순서를 틀리기 쉽습니다.
#
#   ① 파이: 카메라를 켜고 모터를 돌린다            (v4l2_capture.py + pd_profiles.py)
#   ② 파이: 원본에서 **막대 각도**를 읽는다         (yuyv_tools.py track)
#   ③ 파이: 영상에 쓸 장만 **모터 주변만** 잘라 낸다 (yuyv_tools.py crop)
#   ④ 여기: 잘라 온 원본을 그림으로 바꾼다          (crops_to_jpg.py)
#   ⑤ 여기: 실물 위에 시뮬레이터를 겹쳐 영상을 만든다 (make_overlay_video.py)
#
# 원본을 통째로 가져오지 않는 이유는 크기 때문입니다 — 1920x1080 으로 30초면 3.7 기가바이트인데
# 연결이 초당 10.6 메가바이트라 6분 걸립니다. 무거운 일은 파이에서 하고 줄인 것만 가져옵니다.
#
#   ./video_pipeline.sh 4    (무릎용 RS04)
#   ./video_pipeline.sh 3    (엉덩이 회전용 RS03)
set -e
cd "$(dirname "$0")"
M=${1:-4}
PY=../mujoco-sim/mjlab/.venv/bin/python3

# 모터마다 다른 값 — 사진에서 실측한 것 (2026-09-10, 1920x1080)
if [ "$M" = 4 ]; then
  MODEL=RS04; HUB="1052,320"; DISC=177; R0=60; R1=165; SCORE=wood
  CROP="647,0,1457,693"; REACH=1.55
else
  MODEL=RS03; HUB="1738,245"; DISC=139; R0=45; R1=100; SCORE=dark
  CROP="1110,0,1920,693"; REACH=1.76
fi
W=1920; H=1080; FPS=30; SECS=32
RAW=/home/maido/System_ID/data/cam_vs_${M}
LOCAL=/tmp/cam_vs_${M}

echo "===== ① 찍으면서 돌린다 ====="
./pi_run.sh "cd ~/System_ID && bash run_video_rs.sh $M $MODEL videosync $W $H $FPS $SECS $RAW"

echo "===== ② 막대 각도를 파이에서 읽는다 ($SCORE) ====="
./pi_run.sh "cd ~/System_ID && python3 -u yuyv_tools.py track --raw $RAW \
             --cx ${HUB%,*} --cy ${HUB#*,} --r0 $R0 --r1 $R1 --score $SCORE"

echo "===== ③ 영상에 쓸 만큼만 잘라 낸다 ====="
./pi_run.sh "cd ~/System_ID && python3 -u yuyv_tools.py crop --raw $RAW --box $CROP --every 1"

echo "===== 가져오기 ====="
rm -rf "$LOCAL"; mkdir -p "$LOCAL"
ssh -o BatchMode=yes syaro@10.8.0.14 \
    "ssh -o BatchMode=yes maido@192.168.0.16 'cd $RAW && tar cf - crops.bin angles.csv meta.json'" \
    | tar xf - -C "$LOCAL"
CSV=data/pd_videosync_${M}_kp20_kd1_a30_r0_rs.csv
ssh -o BatchMode=yes syaro@10.8.0.14 \
    "ssh -o BatchMode=yes maido@192.168.0.16 'cd ~/System_ID && tar cz $CSV ${CSV%.csv}.meta.json'" \
    | tar xz -C .

echo "===== ④ 그림으로 바꾼다 ====="
$PY crops_to_jpg.py --crops "$LOCAL/crops.bin" --out "$LOCAL/jpg"

echo "===== ⑤ 겹쳐서 영상 ====="
MUJOCO_GL=egl $PY make_overlay_video.py \
  --frames-dir "$LOCAL/jpg" --angles "$LOCAL/angles.csv" --csv "$CSV" \
  --family $MODEL --motor $M --hub "$HUB" --disc-r $DISC --ptr-reach $REACH \
  --crop "$CROP" --out-fps 25 \
  --out "../docs/assets/bench_vs_mujoco_${MODEL,,}.mp4"
echo PIPELINE_DONE
