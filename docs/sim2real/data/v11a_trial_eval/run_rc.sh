#!/usr/bin/env bash
L=/home/syaro/pyg_fea/locks/gpu.lock
mkdir "$L" || { echo "GPU lock busy"; exit 1; }
trap "rmdir $L" EXIT
PKG=/home/syaro/pyg_share/fresh_test/vibecoded-audit-20261012
PY=/home/syaro/pyg_share/venv_fresh/bin/python3
S=/home/syaro/MikuchanRemote/Human-Pygmalion/tools/sim2real/real_condition_sim.py
R=/home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity
ONLY="nominal,delay,guards,loss2%,real timing (delay+guards+loss),0.8 +timing,0.7 +timing,0.5/0.5 +timing,1.0 +timing crane 1->0,imu grav_x bias +0.1,imu grav_x bias -0.1,real timing +bias +0.1,1.0 +timing +imu grav_x,0.05/0.3 +timing crane 1->0"
cd /home/syaro/v11a_eval
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nice -n 10 $PY $S --pkg $PKG --checkpoint $PKG/checkpoints/rough_main_model_11999.pt --out rc_base_brief_dec0.json --only "$ONLY" > rc_base_brief_dec0.log 2>&1
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nice -n 10 $PY $S --pkg $PKG --checkpoint $R/2026-10-14_18-24-35_rough_box_trial_v11a/model_11999.pt --crank-kp 55 --crank-kd 2.5 --out rc_trial_brief_dec0.json --only "$ONLY" > rc_trial_brief_dec0.log 2>&1
