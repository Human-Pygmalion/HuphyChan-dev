#!/bin/bash
# Continue training from the released policy (or any full checkpoint) on Pygmalion-Velocity-Rough-Box,
# with training video on.
#
#   bash scripts/train_continue.sh resume    [more_iterations=4000]  [num_envs=4096] [run_name=continue]  [extra mjlab args...]
#   bash scripts/train_continue.sh warmstart [max_iterations=12000]  [num_envs=4096] [run_name=finetune]  [extra mjlab args...]
#
# resume     Full resume: actor, critic, optimizer and iteration counter (11,999 for the released
#            checkpoint) are loaded and training runs <more_iterations> further. The DR ramp follows
#            the absolute iteration, so it stays at full strength. Not restored: the terrain
#            curriculum levels (every env starts at level 0 again and climbs back) and the RNG state.
# warmstart  Actor only (weights, action std, observation normalizer); critic, optimizer and
#            iteration counter start new, the DR ramp starts from zero, and the actor is held
#            fixed for the first 300 PPO updates while the new critic learns. Use this after
#            changing the environment (e.g. adding latency or gain randomization) so the old
#            critic's values do not mislead the first updates.
#
# Environment variables:
#   MJLAB_PYTHON  python with mjlab installed (default: python)
#   CKPT          checkpoint to start from (default: checkpoints/rough_main_model_11999.pt).
#                 resume needs a full rsl_rl checkpoint (actor, critic, optimizer, iter);
#                 warmstart also accepts an actor-only file.
#   SEED          agent seed (default 43, the released runs' seed)
# Extra arguments after run_name go to mjlab / scripts/train.py, e.g. the optional randomization
# --obs-delay 1 --act-delay 1 --gain-dr 0.8 1.2, --joint-gains ..., or --torso-push ...
# (all off by default; README section 9),
# --agent.save-interval 200, or --video False.
# Logs: logs/rsl_rl/pygmalion_velocity/<date>_<run_name>/ (checkpoints, params/, videos/train/).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"; shift || true
PY="${MJLAB_PYTHON:-python}"
CKPT="${CKPT:-checkpoints/rough_main_model_11999.pt}"
SEED="${SEED:-43}"
LOGS=logs/rsl_rl/pygmalion_velocity
VIDEO=(--video True --video-interval 4800 --video-length 500)
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" >&2; exit 2; }

case "$MODE" in
  resume)
    ITER="${1:-4000}"; ENVS="${2:-4096}"; RUN="${3:-continue}"; shift $(( $# < 3 ? $# : 3 ))
    # mjlab resumes from <logs>/<load-run>/<load-checkpoint>: stage the checkpoint as a run dir.
    START="$("$PY" -c "import sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); \
missing=[k for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter') if k not in c]; \
print('MISSING:'+','.join(missing) if missing else c['iter'])" "$CKPT")"
    case "$START" in MISSING:*) echo "resume needs a full checkpoint; $CKPT lacks ${START#MISSING:} (use warmstart)" >&2; exit 2 ;; esac
    SRC="resume_src_$(basename "${CKPT%.pt}")"
    mkdir -p "$LOGS/$SRC"
    cp -f "$CKPT" "$LOGS/$SRC/model_$START.pt"
    echo "[train_continue] resume $CKPT (iteration $START) -> $ITER more iterations, run '$RUN'"
    exec "$PY" scripts/train.py Pygmalion-Velocity-Rough-Box --agent.seed "$SEED" \
      --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITER" \
      --agent.resume True --agent.load-run "$SRC" --agent.load-checkpoint "model_$START.pt" \
      --agent.run-name "$RUN" "${VIDEO[@]}" "$@"
    ;;
  warmstart)
    ITER="${1:-12000}"; ENVS="${2:-4096}"; RUN="${3:-finetune}"; shift $(( $# < 3 ? $# : 3 ))
    echo "[train_continue] actor-only warm start from $CKPT, $ITER iterations, run '$RUN'"
    exec "$PY" scripts/train.py Pygmalion-Velocity-Rough-Box --agent.seed "$SEED" \
      --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITER" \
      --warmstart-actor "$CKPT" --critic-warmup 300 \
      --agent.run-name "$RUN" "${VIDEO[@]}" "$@"
    ;;
  *)
    sed -n '2,27p' "$0"; exit 2 ;;
esac
