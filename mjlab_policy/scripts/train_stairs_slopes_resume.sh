#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${MJLAB_PYTHON:-python}"
CKPT="${CKPT:-checkpoints/terrain10_union_dr_49999/model_49999.pt}"
NUM_ENVS="${NUM_ENVS:-2048}"
MORE_ITERS="${MORE_ITERS:-50000}"
SEED="${SEED:-43}"
RUN_NAME="${RUN_NAME:-generalized_stairs_slopes12_courseprogress_stagnation_resume49999_50k_v1}"
LOGS="logs/rsl_rl/pygmalion_velocity"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 2; }
START="$($PY -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); missing=[k for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter') if k not in c]; print('MISSING:'+','.join(missing) if missing else c['iter'])" "$CKPT")"
case "$START" in
  MISSING:*) echo "full resume needs ${START#MISSING:}" >&2; exit 2 ;;
esac

SRC="resume_src_stairs_slopes_$(basename "${CKPT%.pt}")"
mkdir -p "$LOGS/$SRC"
cp -f "$CKPT" "$LOGS/$SRC/model_$START.pt"

echo "[stairs-slopes] checkpoint=$CKPT iteration=$START"
echo "[stairs-slopes] envs=$NUM_ENVS more_iterations=$MORE_ITERS target=$((START + MORE_ITERS))"
echo "[stairs-slopes] curriculum=12 levels, +1 every 2000 iterations, cumulative uniform"
echo "[stairs-slopes] reward=signed world-course progress; heading=45deg; stagnation=1.0s+0.5s"
echo "[stairs-slopes] run_name=$RUN_NAME logger=wandb"

exec "$PY" scripts/train.py Pygmalion-Velocity-Generalized-StairsSlopes12-UnionDR \
  --agent.seed "$SEED" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MORE_ITERS" \
  --agent.resume True \
  --agent.load-run "$SRC" \
  --agent.load-checkpoint "model_$START.pt" \
  --agent.run-name "$RUN_NAME" \
  "$@"
