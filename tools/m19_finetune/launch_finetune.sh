#!/bin/bash
# Fine-tune Union-DR policy 49999 on the latest robot model (M19), everything else as in the original run.
# Runs ON the 3090 host from ~/pyg_m19_train/mjlab_policy. Launcher for the "every training records video + gets a note" rule:
# it always turns the training video on, snapshots the config next to the run, and prints the one-variable diff check.
#
#   bash launch_finetune.sh <run_name> <num_envs> <more_iterations> [extra train args]
#
# Single variable vs the parent run (2026-10-24_02-05-41_generalized_terrain10_uniondr_resume37400, iteration 49,999):
#   robot model  pygmalion_legonly_ab.xml (M10, 23.63 kg)  ->  pygmalion_legonly_ab_m19.xml (M19, 26.91 kg, foot v5, D455, reinforcement, aluminium parts)
#   spawn height +0.014 m (sole bottom of foot v5 is 14 mm lower; same value as the transfer study)
# Task, rewards, commands, DR, terrain curriculum, PPO settings, seed, environment count are those of the parent.
# Logger: tensorboard (the 3090 has no wandb login; logging backend does not change learning).
set -euo pipefail
RUN="${1:?run name}"; NUM_ENVS="${2:?num envs}"; MORE="${3:?more iterations}"; shift 3
HERE="${PYG_TRAIN_DIR:-$HOME/pyg_m19_train/mjlab_policy}"
R="$HOME/MikuchanRemote/Human-Pygmalion"
export PYTHONPATH="$R/Feet_design/Feet_Design_Workspace/mjcf_sim/stock_mjlab/src${PYTHONPATH:+:$PYTHONPATH}"
PY="$R/mujoco-sim/mjlab/.venv/bin/python3"
export PYG_FOOT_XML="${PYG_FOOT_XML:-pygmalion_legonly_ab_m19.xml}"
export PYG_INIT_DZ="${PYG_INIT_DZ:-0.014}"
export MUJOCO_GL=egl
CKPT="checkpoints/terrain10_union_dr_49999/model_49999.pt"
TASK="Pygmalion-Velocity-Generalized-Terrain10-UnionDR"
LOGS="logs/rsl_rl/pygmalion_velocity"
cd "$HERE"
[ -f "$CKPT" ] || { echo "missing $CKPT" >&2; exit 2; }
[ -f "pygmalion_mjlab/assets/$PYG_FOOT_XML" ] || { echo "missing model $PYG_FOOT_XML" >&2; exit 2; }
START=$("$PY" -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(c['iter'])" "$CKPT")
SRC="resume_src_union_model_${START}"
mkdir -p "$LOGS/$SRC"; cp -f "$CKPT" "$LOGS/$SRC/model_$START.pt"
echo "[m19-finetune] run=$RUN envs=$NUM_ENVS start_iter=$START more=$MORE target=$((START + MORE)) xml=$PYG_FOOT_XML dz=$PYG_INIT_DZ"
exec "$PY" scripts/train.py "$TASK" \
  --agent.seed 43 \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MORE" \
  --agent.resume True --agent.load-run "$SRC" --agent.load-checkpoint "model_$START.pt" \
  --agent.run-name "$RUN" \
  --agent.logger tensorboard \
  --video True --video-length 300 --video-interval 12000 \
  "$@"
