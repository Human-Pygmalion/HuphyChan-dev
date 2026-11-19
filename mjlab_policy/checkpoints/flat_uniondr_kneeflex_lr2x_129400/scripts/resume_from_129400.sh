#!/bin/bash
set -euo pipefail

POLICY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PACKAGE="$POLICY_ROOT/checkpoints/flat_uniondr_kneeflex_lr2x_129400"
PY="${MJLAB_PYTHON:-python}"
UPDATES="${1:-30000}"
ENVS="${2:-4096}"
RUN_NAME="${3:-flat_uniondr_kneeflex_lr2x_resume129400}"
shift $(( $# < 3 ? $# : 3 ))

cd "$POLICY_ROOT"
SOURCE_RUN="release_flat_uniondr_kneeflex_lr2x_parent_129400"
mkdir -p "logs/rsl_rl/pygmalion_velocity/$SOURCE_RUN"
cp -f "$PACKAGE/model_129400.pt" \
  "logs/rsl_rl/pygmalion_velocity/$SOURCE_RUN/model_129400.pt"

exec "$PY" scripts/train.py \
  Pygmalion-Velocity-Flat-UnionDR-FullPlanar-Clearance10-KneeFlex-LR2x \
  --agent.seed 43 \
  --env.scene.num-envs "$ENVS" \
  --agent.max-iterations "$UPDATES" \
  --agent.resume True \
  --agent.load-run "$SOURCE_RUN" \
  --agent.load-checkpoint model_129400.pt \
  --agent.run-name "$RUN_NAME" \
  "$@"
