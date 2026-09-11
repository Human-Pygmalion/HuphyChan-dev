#!/bin/bash
# 관절 화면을 **한 줄로** 띄운다. 12관절 전부, 한계는 시뮬레이터 모델에서 그대로.
#
# 2026-09-12. 한계를 손으로 적다가 빠뜨리면 그 관절이 하드스톱으로 갑니다. 그래서
# 손으로 적지 않고 `mjcf_limits.json` 에서 만들어 넣습니다.
#
#     ./joint_web_up.sh              읽기만 (힘이 절대 안 나감)
#     ./joint_web_up.sh --command    조작판을 띄우되 **보내지는 않음**(시험)
#     ./joint_web_up.sh --live       실제로 힘이 나감. 사람이 옆에 있을 때만.
#
# 세기는 낮게 둡니다. 엉덩이 앞뒤·좌우와 무릎은 다리 무게가 걸려 **처집니다** —
# 그것이 정상이고, 억지로 들려고 세기를 올리지 마십시오.
#
# 명령이 끊겼다고 보는 시간(--deadman-s)을 0.3초에서 **3.0초**로 올렸습니다
# (2026-09-12). 0.3초는 휴대전화와 원격망을 건너오는 명령에는 너무 짧아서,
# 통신이 잠깐 늦을 때마다 힘이 빠졌습니다 — 실측으로 한 번 쓰는 동안 188번
# 끊겼습니다. 사용자 지적: *"모터 지멋대로 계속 꺼지잖아."*
#
# 짧게 잡아 두던 이유는 "멈춤을 누르면 빨리 꺼지게" 였는데, 이제 멈춤은 데드맨이
# 아니라 `/release` 로 **바로** 끊으므로 그 이유가 사라졌습니다. 데드맨이 맡는 일은
# 하나만 남습니다 — **화면이 죽어 아무도 안 보는 경우**. 거기엔 3초가 넉넉합니다.
set -e
cd "$(dirname "$0")"

MAP=$(python3 - <<'PY'
import json
lim = json.load(open("mjcf_limits.json"))
name = {"hip_pitch": "hip_pitch", "hip_roll": "hip_roll", "hip_yaw": "hip_yaw",
        "knee": "knee", "ankle_a": "crank_A", "ankle_b": "crank_B"}
out = []
for leg, side in (("left_leg", "L"), ("right_leg", "R")):
    for key, tag in name.items():
        lo, hi = lim[leg][key]
        out.append(f"{side}_{tag}_joint={lo}:{hi}")
print(" ".join(out))
PY
)

ALLOW="L_hip_pitch_joint R_hip_pitch_joint L_hip_roll_joint R_hip_roll_joint \
L_hip_yaw_joint R_hip_yaw_joint L_knee_joint R_knee_joint \
L_crank_A_joint L_crank_B_joint R_crank_A_joint R_crank_B_joint"

# 관절마다 세기를 따로. 다리 무게가 걸리는 곳일수록 낮게 — 세게 잡으려다 스톱에
# 부딪히는 것보다 처지는 편이 낫습니다.
KP="L_hip_pitch_joint=8 R_hip_pitch_joint=8 L_hip_roll_joint=8 R_hip_roll_joint=8 \
L_knee_joint=12 R_knee_joint=12 L_hip_yaw_joint=10 R_hip_yaw_joint=10 \
L_crank_A_joint=8 L_crank_B_joint=8 R_crank_A_joint=8 R_crank_B_joint=8"

case "${1:-}" in
  --live)    MODE="--command --live" ;;
  --command) MODE="--command" ;;
  *)         MODE="" ;;
esac

exec python3 joint_web.py --port 8098 --telemetry 10.8.0.16:9870 \
  $MODE --allow $ALLOW --limit-map "$MAP" --kp-map "$KP" \
  --span-deg 25 --max-vel-deg-s 15 --deadman-s 3.0 --max-temp-c 50
