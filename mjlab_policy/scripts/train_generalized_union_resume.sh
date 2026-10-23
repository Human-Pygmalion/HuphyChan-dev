#!/bin/bash
# Full-state resume of model_37400 on the Terrain10 Union-DR task.
# This script defines the server command; it is not run automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${MJLAB_PYTHON:-python}"
CKPT="${CKPT:-checkpoints/terrain10_2048_model_37400.pt}"
NUM_ENVS="${NUM_ENVS:-2048}"
TARGET_ITER="${TARGET_ITER:-50000}"
SEED="${SEED:-43}"
RUN_NAME="${RUN_NAME:-generalized_terrain10_uniondr_resume37400}"
LOGS="logs/rsl_rl/pygmalion_velocity"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 2; }
START="$($PY -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); missing=[k for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter') if k not in c]; print('MISSING:'+','.join(missing) if missing else c['iter'])" "$CKPT")"
case "$START" in
  MISSING:*) echo "full resume needs ${START#MISSING:}" >&2; exit 2 ;;
esac

MORE_ITERS="${MORE_ITERS:-$((TARGET_ITER - START))}"
(( MORE_ITERS > 0 )) || { echo "TARGET_ITER must exceed checkpoint iteration $START" >&2; exit 2; }

SRC="resume_src_union_$(basename "${CKPT%.pt}")"
mkdir -p "$LOGS/$SRC"
cp -f "$CKPT" "$LOGS/$SRC/model_$START.pt"

echo "[union-dr] checkpoint=$CKPT iteration=$START"
echo "[union-dr] task=Pygmalion-Velocity-Generalized-Terrain10-UnionDR"
echo "[union-dr] envs=$NUM_ENVS more_iterations=$MORE_ITERS target=$((START + MORE_ITERS))"
echo "[union-dr] run_name=$RUN_NAME logger=wandb"

exec "$PY" scripts/train.py Pygmalion-Velocity-Generalized-Terrain10-UnionDR \
  --agent.seed "$SEED" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MORE_ITERS" \
  --agent.resume True \
  --agent.load-run "$SRC" \
  --agent.load-checkpoint "model_$START.pt" \
  --agent.run-name "$RUN_NAME" \
  "$@"
