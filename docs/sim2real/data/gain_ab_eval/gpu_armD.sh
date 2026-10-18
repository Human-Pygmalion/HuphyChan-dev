#!/usr/bin/env bash
cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
bash /home/syaro/v11a_eval/gain_ab/rc_armD.sh
export GAIN_AB_DROP=PYG_ACTION_DELAY GAIN_AB_PROBE_DROP=PYG_ACTION_DELAY
bash /home/syaro/MikuchanRemote/Human-Pygmalion/tools/sim2real/gain_ab_eval.sh probe logs/rsl_rl/pygmalion_velocity/2026-10-17_19-39-49_rough_gain_crank60_actdelay_v11a slope /home/syaro/v11a_eval/gain_ab/slope_armD.json --angles 20 --vx 0.3,0.6 > /home/syaro/v11a_eval/gain_ab/probe_slope_armD.log 2>&1
bash /home/syaro/MikuchanRemote/Human-Pygmalion/tools/sim2real/gain_ab_eval.sh probe logs/rsl_rl/pygmalion_velocity/2026-10-17_19-39-49_rough_gain_crank60_actdelay_v11a stairs /home/syaro/v11a_eval/gain_ab/stairs_armD.json --risers 0.15 --vx 0.3 > /home/syaro/v11a_eval/gain_ab/probe_stairs_armD.log 2>&1
