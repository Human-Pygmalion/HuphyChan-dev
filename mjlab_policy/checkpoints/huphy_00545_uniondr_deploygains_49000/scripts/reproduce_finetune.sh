#!/usr/bin/env bash
set -euo pipefail
UNITREE_RL_MJLAB=${UNITREE_RL_MJLAB:?Set UNITREE_RL_MJLAB}
MJLAB_PYTHON=${MJLAB_PYTHON:-python}
GPU=${GPU:-0}; NUM_ENVS=${NUM_ENVS:-2048}; MAX_ITERATIONS=${MAX_ITERATIONS:-19000}
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PARENT_RUN=huphy_00545_deploygains_release_parent_30000
mkdir -p "$UNITREE_RL_MJLAB/logs/rsl_rl/huphy_tracking/$PARENT_RUN"
cp -f "$PACKAGE_DIR/model_30000.pt" "$UNITREE_RL_MJLAB/logs/rsl_rl/huphy_tracking/$PARENT_RUN/model_30000.pt"
cd "$UNITREE_RL_MJLAB"
exec env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$MJLAB_PYTHON" scripts/train.py \
  HUPHY-Tracking-ProjectedGravity-UnionDR-DeployGains \
  --motion-file "$PACKAGE_DIR/motions/00545_interactionmesh_50hz.npz" \
  --env.scene.num-envs "$NUM_ENVS" --agent.seed 42 --agent.max-iterations "$MAX_ITERATIONS" \
  --agent.resume True --agent.load-run "$PARENT_RUN" --agent.load-checkpoint model_30000.pt \
  --agent.run-name huphy_00545_deploygains_reproduce_30000_to49000 --gpu-ids '[0]' "$@"
