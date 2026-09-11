#!/bin/bash
# 벤치 화면을 **한 줄로** 되살린다. 재부팅하면 전부 내려가므로 필요하다.
#
# 2026-09-11 · 재부팅 뒤에 통신선·카메라·관절 화면이 모두 사라진 것을 겪고 만듦.
#
#   1) 통신선 두 가닥을 올리고, 어느 선에 어느 다리가 붙었는지 **확인**한다
#      (고유 일련번호로 열고, 모터 번호를 물어봐 판정 — can_up.py)
#   2) 카메라 중계를 띄운다 (8099)
#   3) 관절 확인 화면을 띄운다 (8098). 뷰어로 텔레메트리도 같이 보낸다
#
# 모터로 나가는 것은 고장값 조회와 세기 0·토크 0 프레임뿐이다. **축이 움직일 수 없다.**
#
#   ./bench_up.sh                       # 그대로 띄움
#   ./bench_up.sh 10.8.0.16:9870        # 뷰어 주소를 바꿔서
#
# 멈추려면:  pkill -f 'cam_stream.py|joint_web.py'
set -e
cd "$(dirname "$0")"
VIEWER=${1:-10.8.0.16:9870}
CAM=/dev/v4l/by-id/usb-4K_USB_Camera_4K_USB_Camera_01.00.00-video-index0

echo "== 1) 통신선"
python3 can_up.py || echo "  (확인할 것이 있습니다 — 위 내용을 볼 것)"

echo
echo "== 2) 카메라"
pkill -f cam_stream.py 2>/dev/null || true
sleep 1
nohup python3 cam_stream.py --device "$CAM" --width 1280 --height 720 --fps 15 \
  --port 8099 > /tmp/cam_stream.log 2>&1 &
sleep 4
tail -1 /tmp/cam_stream.log

echo
echo "== 3) 관절 화면"
pkill -f joint_web.py 2>/dev/null || true
sleep 1
nohup python3 joint_web.py --port 8098 --hz 20 --telemetry "$VIEWER" \
  > /tmp/joint_web.log 2>&1 &
sleep 4
tail -3 /tmp/joint_web.log

echo
echo "== 확인"
pgrep -af "cam_stream.py|joint_web.py" | sed 's/^/  /'
