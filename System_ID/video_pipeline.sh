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
# 2026-09-10 실측 (1920x1080 한 프레임에서 눈금 얹어 읽음). 카메라가 조금씩 움직이므로
# 영상을 찍기 전에 다시 재는 것이 좋습니다 — 실제로 한 번 움직여서 다시 쟀습니다.
#   HUB   축 중심 (원본 프레임 좌표)      DISC  사진 속 도는 원판 반지름 [화소]
#   R0,R1 막대를 찾을 고리 (원판 **안쪽**에 있어야 함 — 바깥은 나무 책상이라 색이 비슷)
#   SCORE 막대를 무엇으로 찾을까 (RS04 붉은기 여유 22.0 / RS03 어두움 여유 92.1)
#   REACH 실물 막대가 원판 반지름의 몇 배까지 뻗는가
# 좌표는 **1280x720 에서 잰 값**입니다. 해상도를 바꾸면 이 값도 다시 재야 합니다.
if [ "$M" = 4 ]; then
  MODEL=RS04; HUB="483,159"; DISC=119; R0=40; R1=105; SCORE=wood
  CROP="214,0,754,462"; REACH=1.49; EXPOSURE=200   # 추적 여유 21.0
else
  MODEL=RS03; HUB="908,224"; DISC=88; R0=38; R1=76; SCORE=dark
  CROP="638,0,1178,462"; REACH=1.70; EXPOSURE=100
fi
# ★ 2026-09-10 — 1920x1080 을 초당 30장 원본으로 쓰면 초당 124 메가바이트인데 **SD 카드가
# 못 받습니다.** 처음 13초는 캐시가 받아 주다 차면 카메라가 멈춥니다(실측: 모터 4 는 12.9초
# 뒤 초당 2장으로 무너지고, 모터 3 은 11.7초가 통째로 사라졌습니다). 그래서
#   ① 해상도를 1280x720 으로 낮춰 초당 55 메가바이트로 만들고
#   ② **램 위(tmpfs, /tmp)** 에 씁니다 — 32초에 1.8 기가바이트, 남은 3.3 기가바이트 안.
# 그러면 초당 30장이 끝까지 유지됩니다. 촬영 프로그램도 이제 자리·속도를 미리 따져 거부합니다.
W=1280; H=720; FPS=30; SECS=32
RAW=/tmp/cam_vs_${M}
LOCAL=/tmp/cam_vs_${M}

# 램은 좁다(3.3 기가바이트). 지난 촬영이 남아 있으면 이번 것이 안 들어가므로 먼저 치운다.
./pi_run.sh "rm -rf /tmp/cam_vs_*"

echo "===== ① 찍으면서 돌린다 ====="
./pi_run.sh "cd ~/System_ID && bash run_video_rs.sh $M $MODEL videosync $W $H $FPS $SECS $RAW $EXPOSURE"

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
# 램에 둔 원본은 크다. 필요한 것(각도·자른 그림)은 이미 가져왔으니 비운다.
./pi_run.sh "rm -f $RAW/frames.raw $RAW/crops.bin"
echo PIPELINE_DONE
