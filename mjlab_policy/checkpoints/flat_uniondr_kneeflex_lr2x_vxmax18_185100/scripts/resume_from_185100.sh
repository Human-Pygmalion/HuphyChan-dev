#!/bin/bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PACKAGE="$POLICY_ROOT/checkpoints/flat_uniondr_kneeflex_lr2x_vxmax18_185100"
PY="${MJLAB_PYTHON:-python}"
UPDATES="${1:-30000}"
ENVS="${2:-4096}"
RUN_NAME="${3:-flat_uniondr_kneeflex_lr2x_vxmax18_resume185100}"
shift $(( $# < 3 ? $# : 3 ))

cd "$POLICY_ROOT"
SOURCE_RUN="release_flat_uniondr_kneeflex_lr2x_vxmax18_parent_185100"
mkdir -p "logs/rsl_rl/pygmalion_velocity/$SOURCE_RUN"
cp -f "$PACKAGE/model_185100.pt" \
  "logs/rsl_rl/pygmalion_velocity/$SOURCE_RUN/model_185100.pt"

exec env PYTHONPATH="$POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" scripts/train.py \
  Pygmalion-Velocity-Flat-UnionDR-FullPlanar-Clearance10-KneeFlex-LR2x \
  --agent.seed 43 \
  --env.commands.twist.ranges.lin-vel-x '(-1.5,1.8)' \
  --env.scene.num-envs "$ENVS" \
  --agent.max-iterations "$UPDATES" \
  --agent.save-interval 100 \
  --agent.resume True \
  --agent.load-run "$SOURCE_RUN" \
  --agent.load-checkpoint model_185100.pt \
  --agent.logger wandb \
  --agent.wandb-project pygmalion_velocity \
  --agent.run-name "$RUN_NAME" \
  "$@"
