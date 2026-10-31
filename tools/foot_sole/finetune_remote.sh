#!/bin/bash
# Foot-sole stage-0 fine-tune on the 3090 (run ON the 3090): resume Union-DR 49999 for MORE iterations with one foot variant.
#   bash tools/foot_sole/finetune_remote.sh <variant> [more_iterations]
#   variant: base (training foot, the control arm) or any <name> with pygmalion_mjlab/assets/pygmalion_legonly_ab_<name>.xml
#   optional env: FOOT_SOLREF=<timeconst> (foot contact solref; unset = 49999 value 0.02), SEED=<n> (unset = 49999 seed 43)
# Same task, seed, envs (2048), DR, rewards and sim settings as the 49999 run (checkpoints/terrain10_union_dr_49999/env.yaml);
# only the foot collision XML differs (PYG_FOOT_XML), plus the start pose translated up by the sole thickness (PYG_INIT_DZ).
# Logs: <branch>/mjlab_policy/logs/rsl_rl/pygmalion_velocity/<date>_sole_ft_<variant>_<more>, tensorboard (no W&B on the 3090).
set -euo pipefail
V="${1:?variant}"
MORE="${2:-2500}"
WS="$HOME/MikuchanRemote/Human-Pygmalion/Feet_design/Feet_Design_Workspace/mjcf_sim"
case "$V" in
  base) XML=pygmalion_legonly_ab.xml; DZ=0 ;;
  *) XML="pygmalion_legonly_ab_${V}.xml"; DZ=0.014 ;;   # every sole variant: sole 14 mm below the plate, start pose translated up (user-approved)
esac
cd "$WS/union_branch_49999/mjlab_policy"   # branch sole-mjcf-49999 = 49999 release 08a3208 + foot variants
[ -f "pygmalion_mjlab/assets/$XML" ] || { echo "no $XML" >&2; exit 2; }
export PYTHONPATH="$WS/stock_mjlab/src"
export PYG_FOOT_XML="$XML" PYG_INIT_DZ="$DZ"   # contact buffers stay at the 49999 training values (150 / 1500)
export CKPT=checkpoints/terrain10_union_dr_49999/model_49999.pt
SEED="${SEED:-43}"
TAG="${V}"
[ -n "${FOOT_SOLREF:-}" ] && { export PYG_FOOT_SOLREF="$FOOT_SOLREF"; TAG="${TAG}_sr${FOOT_SOLREF#0.}"; }
[ "$SEED" != 43 ] && TAG="${TAG}_s${SEED}"
export TARGET_ITER=$((49999 + MORE)) NUM_ENVS=2048 SEED RUN_NAME="sole_ft_${TAG}_${MORE}"
export MJLAB_PYTHON="$HOME/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/.venv/bin/python3"
mkdir -p "$WS/finetune"
setsid nohup bash scripts/train_generalized_union_resume.sh --agent.logger tensorboard --agent.save-interval 250 \
  > "$WS/finetune/train_${TAG}.log" 2>&1 < /dev/null &
echo "launched $TAG (more=$MORE seed=$SEED solref=${PYG_FOOT_SOLREF:-default}) pid $! log $WS/finetune/train_${TAG}.log"
