#!/bin/bash
# 1 단계: **짧게 훑어서 λ 를 고른다.** 긴 학습은 그 다음이다.
#
# 2026-09-18 · 사용자 질문: *"간단하게 짧은 학습으로 검증하는 방법은 부족한가?"*
#
# 부족하지 않습니다 — **절반은**. 정규화가 신경망을 얼마나 눌렀는지는 0 회부터 보이고,
# 굴릴 필요도 없이 민감도를 직접 재면 됩니다. 반면 **밀어도 안 넘어지는지**는 밀기
# 커리큘럼이 2000 회부터 램프하므로 짧게는 잴 수 없습니다.
#
# 그래서 이 스크립트는 **λ 고르기만** 합니다. 합격 판정은 긴 학습 + stand_robustness.py.
#
# 왜 지금 필요한가: CAPS 의 λ 를 제가 **크기만 맞춰 찍었습니다**. LCP 는 논문에 표가
# 있지만(0.002 가 최적, 0.005 에서 −8%, 0.01 에서 −38%) 그건 그쪽 로봇 이야기이고,
# CAPS·스펙트럼정규화는 우리 로봇 기준이 아예 없습니다.
#
# 기준선: v7b 민감도 3.41 (떨림), v7 민감도 0.81 (얌전하지만 0.5 m/s 에 넘어짐).
# 목표는 그 사이 어딘가 — **v7 만큼 얌전하면서 v7b 의 강건성을 지키는** 자리.

set -u
cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab || exit 1
ITERS=${ITERS:-500}
ENVS=${ENVS:-4096}
OUT=/tmp/v9_stage1
mkdir -p "$OUT"

# (이름, 환경변수) — 방법마다 두 값씩. LCP 는 논문 표가 있는 구간을 씁니다.
ARMS=(
  "caps_02:PYG_CAPS_S=0.2"
  "caps_10:PYG_CAPS_S=1.0"
  "lcp_0002:PYG_LCP=0.002"
  "lcp_0005:PYG_LCP=0.005"
  "sn_02:PYG_SN=0.2"
  "sn_05:PYG_SN=0.5"
)

CKS=(); LABS=()
for a in "${ARMS[@]}"; do
  NAME="${a%%:*}"; VAR="${a#*:}"
  echo "=== $NAME ($VAR) $ITERS 회 ==="
  ( set -a; . /tmp/pyg_env.sh; set +a
    export "$VAR"
    .venv/bin/python3 -u analysis/train_wandb_video.py Mjlab-Velocity-Flat-Pygmalion \
      --video False --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" \
      --agent.run-name "v9s1_$NAME" --agent.logger tensorboard
  ) > "$OUT/$NAME.log" 2>&1
  D=$(ls -dt logs/rsl_rl/pygmalion_velocity/*v9s1_$NAME 2>/dev/null | head -1)
  CK=$(ls -t "$D"/model_*.pt 2>/dev/null | head -1)
  if [ -n "$CK" ]; then CKS+=("$CK"); LABS+=("$NAME"); echo "  -> $CK"; else echo "  ★ 저장본 없음"; fi
done

echo; echo "=== 민감도 비교 (기준: v7b 3.41 / v7 0.81) ==="
.venv/bin/python3 -u ../../tools/robot_model/policy_sensitivity.py \
  --checkpoint "${CKS[@]}" --label "${LABS[@]}" --seconds 8 --tag v9_stage1_sensitivity
echo "STAGE1 DONE"
