#!/bin/bash
# Train the released policy's three stages from scratch, in order, with training video on.
#
#   bash scripts/train_from_scratch.sh
#
# Stages (same as the original runs, see README §3):
#   p1     Pygmalion-Velocity-Flat-P1     from scratch, 16,384 envs, iterations 0..15,999
#   p2     Pygmalion-Velocity-Flat-P2     resume P1 model_15999 (actor+critic+optimizer+step
#                                         counter), iterations 15,999..29,998, entropy anneal
#   rough  Pygmalion-Velocity-Rough-Box   actor-only warm start from P2 model_29998, 4,096 envs,
#                                         critic warm-up 300, iterations 0..11,999
#
# Environment variables:
#   MJLAB_PYTHON   python with mjlab installed (default: python)
#   STAGES         which stages to run, in order (default: "p1 p2 rough")
#   P1_RUN         P1 run directory to resume P2 from (default: the one this script just trained)
#   P2_RUN         P2 run directory to warm-start rough from (default: the one just trained)
#   P1_ITERS / P2_ITERS / ROUGH_ITERS   iteration counts (defaults 16000 / 14000 / 12000;
#                  smaller values are for smoke tests - the DR and entropy schedules are in
#                  absolute iterations and only line up with the defaults)
#   TRAIN_EXTRA    extra arguments added to every stage, e.g. the optional randomization
#                  TRAIN_EXTRA="--obs-delay 1 --act-delay 1 --gain-dr 0.8 1.2" (all off by default;
#                  see scripts/train.py --help header and README section 9)
# Logs: logs/rsl_rl/pygmalion_velocity/<date>_<run name>/ (checkpoints, params/, videos/train/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${MJLAB_PYTHON:-python}"
STAGES="${STAGES:-p1 p2 rough}"
P1_ITERS="${P1_ITERS:-16000}"; P2_ITERS="${P2_ITERS:-14000}"; ROUGH_ITERS="${ROUGH_ITERS:-12000}"
LOGS=logs/rsl_rl/pygmalion_velocity
VIDEO=(--video True --video-interval 8000 --video-length 500)
read -r -a EXTRA <<< "${TRAIN_EXTRA:-}"
latest() { ls -d "$LOGS"/[0-9]*_"$1" 2>/dev/null | sort | tail -1; }  # date-named run dirs only

for stage in $STAGES; do
  case "$stage" in
    p1)
      "$PY" scripts/train.py Pygmalion-Velocity-Flat-P1 --agent.seed 43 \
        --env.scene.num-envs 16384 --agent.max-iterations "$P1_ITERS" \
        --agent.run-name flat_p1 "${VIDEO[@]}" "${EXTRA[@]}"
      P1_RUN="$(latest flat_p1)"
      ;;
    p2)
      P1_RUN="${P1_RUN:-$(latest flat_p1)}"
      CK="$(ls "$P1_RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)"
      echo "[train_from_scratch] P2 resumes $P1_RUN/model_$CK.pt"
      "$PY" scripts/train.py Pygmalion-Velocity-Flat-P2 --agent.seed 43 \
        --env.scene.num-envs 16384 --agent.max-iterations "$P2_ITERS" \
        --agent.resume True --agent.load-run "$(basename "$P1_RUN")" \
        --agent.load-checkpoint "model_$CK.pt" \
        --entropy-anneal 15999 29998 0.01 0.002 \
        --agent.run-name flat_p2 "${VIDEO[@]}" "${EXTRA[@]}"
      P2_RUN="$(latest flat_p2)"
      ;;
    rough)
      P2_RUN="${P2_RUN:-$(latest flat_p2)}"
      CK="$(ls "$P2_RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)"
      echo "[train_from_scratch] rough warm-starts from $P2_RUN/model_$CK.pt"
      "$PY" scripts/train.py Pygmalion-Velocity-Rough-Box --agent.seed 43 \
        --env.scene.num-envs 4096 --agent.max-iterations "$ROUGH_ITERS" \
        --warmstart-actor "$P2_RUN/model_$CK.pt" --critic-warmup 300 \
        --agent.run-name rough_main --video True --video-interval 4800 --video-length 500 "${EXTRA[@]}"
      ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
