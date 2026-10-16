#!/usr/bin/env bash
# Finished-run evaluation of the gain A/B arms (docs/experiments/2026-10-15_ab_rough_gain_compare.md).
# Same measurements as analysis/matched_set.sh, but one measurement per call so that a queue
# (xargs -P 3) can run them at most three at a time, and turn seeds are an argument.
# cwd = mujoco-sim/mjlab on the machine that holds the checkpoints; the training run's GPU lock is NOT taken.
#
#   gain_ab_eval.sh job     <tag> <run dir> <turn|stand|walk> <seed>        one CPU measurement
#   gain_ab_eval.sh verdict <tag> "<turn seeds>"                            turn / standing / walking verdicts
#   gain_ab_eval.sh probe   <run dir> <slope|stairs> <out json> <extra args...>   rough_load_probe.py (GPU, 256 envs)
#
# PYG_* flags come from the run's own repro/launch_manifest.json (so the crank gain of the run is the one built).
set -u
cd "$(dirname "$0")/../../mujoco-sim/mjlab" 2>/dev/null || true   # no-op when already in mjlab
MODE=$1; shift
pyg_env() { python3 -c "import json,sys;d=json.load(open('$1/repro/launch_manifest.json'))['environment'];print(' '.join(f'{k}={v}' for k,v in sorted(d.items()) if k.startswith('PYG_')))"; }
case $MODE in
  job)
    TAG=$1; D=$2; KIND=$3; S=$4; CK=model_11999.pt
    # claim the measurement (two queues may list the same one); the loser exits without running
    mkdir -p analysis/out/.claims; mkdir "analysis/out/.claims/${TAG}_${KIND}_${S}" 2>/dev/null || { echo "claimed elsewhere: ${TAG} ${KIND} ${S}"; exit 0; }
    E=$(pyg_env "$D")
    run() { env $E MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python3 "$@"; }
    case $KIND in
      turn)  run analysis/measure_turn.py  --run-dir "$D" --checkpoint "$D/$CK" --tag "m_${TAG}_turn_s${S}"  --device cpu --rollout-seed "$S" 2>&1 | tail -1 ;;
      stand) run analysis/measure_stand.py --run-dir "$D" --checkpoint "$D/$CK" --tag "m_${TAG}_stand_s${S}" --device cpu --rollout-seed "$S" 2>&1 | tail -1 ;;
      walk)  run analysis/gait_kinematics_probe.py --run-dir "$D" --checkpoint "$D/$CK" --tag "m_${TAG}_walk_s${S}" \
                 --vx 1.2 --seed "$S" --init-jitter 0.05 2>&1 | tail -1 ;;
    esac ;;
  verdict)
    TAG=$1; SEEDS=$2
    echo "=== verdict ${TAG} ==="
    for s in $SEEDS; do
      echo "=== m_${TAG}_turn_s${s} ==="; CUDA_VISIBLE_DEVICES="" .venv/bin/python3 analysis/turn_verdict.py analysis/out/m_${TAG}_turn_s${s}.npz 2>&1
    done
    CUDA_VISIBLE_DEVICES="" .venv/bin/python3 analysis/standing_verdict.py --npz "analysis/out/m_${TAG}_stand_s*.npz" --label "$TAG" 2>&1 | grep -E "foot off ground \(%\)|travelled|lift events"
    CUDA_VISIBLE_DEVICES="" .venv/bin/python3 analysis/walking_verdict.py --npz "analysis/out/m_${TAG}_walk_s*.npz" --label "$TAG" 2>&1 | grep -E "forward speed|L-R gap|heading drift" ;;
  probe)
    D=$1; SUITE=$2; OUT=$3; shift 3
    E=$(pyg_env "$D")
    env $E MUJOCO_GL=egl nice -n 10 .venv/bin/python3 ../../tools/robot_gains/rough_load_probe.py --run "$D" --ckpt model_11999.pt --suite "$SUITE" --out "$OUT" "$@" 2>&1 | tail -3 ;;
esac
