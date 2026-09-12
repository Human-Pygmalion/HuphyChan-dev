#!/bin/bash
# Two questions the idle GPU can answer, one arm each, both warm-started from the SAME
# checkpoint as bundleD1_AB so D1 is the control for both.
#   E1 = D1 recipe + the FIXED foot_loading_rate term (w 1e-4). That term has never actually
#        run: it was dead on a sign bug for three runs, and its old weight 0.002 was measured
#        at -0.664/step against a ~0.086/step budget, so it is newly calibrated too.
#   V4 = D1 recipe on the NEW 31.316 kg model. Mass-only change; asks whether a policy trained
#        at 35.347 kg survives the rebuild, which the V2 scratch plan needs to know.
set -euo pipefail
cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
D=$(ls -d logs/rsl_rl/pygmalion_velocity/*_ankleAB_c3)
CK=model_31999.pt
[ -f "$D/$CK" ] || { echo "!! $D/$CK missing"; exit 2; }
COMMON="PYG_V2=1 PYG_INIT_BENT=1 PYG_ARM_ABD_DEG=15 PYG_INERTIAL_DR=1 PYG_ANKLE_MODE=AB \
        PYG_DR_START_ITER=0 PYG_DR_END_ITER=1 PYG_TN=1 PYG_MOTOR_MEAS=1 PYG_SOFT_LANDING=1"
RECIPE="PYG_INIT_MID=1 PYG_KNEE_EXT=1 PYG_KNEE_EXT_W=2.0 PYG_KNEE_EXT_DEG=25"
launch () {
  local NAME="$1"; shift
  echo "== launch $NAME  $(date +%H:%M:%S)"
  env $COMMON $RECIPE "$@" nohup .venv/bin/python3 analysis/train_wandb_video.py \
      Mjlab-Velocity-Flat-Pygmalion --video True --video-interval 8000 --video-length 500 \
      --env.scene.num-envs 16384 --agent.max-iterations 800 --agent.run-name "$NAME" \
      --agent.logger wandb --agent.resume True --agent.load-run "$(basename $D)" \
      --agent.load-checkpoint "$CK" > "logs/${NAME}.log" 2>&1 &
  echo "   pid $! -> logs/${NAME}.log"
}
launch "bundleE1_AB" PYG_SOFT_LANDING_MODE=both
sleep 25
launch "bundleV4_AB" PYG_SOFT_LANDING_MODE=half PYG_MODEL_V4=1
