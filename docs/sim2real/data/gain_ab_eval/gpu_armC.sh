#!/usr/bin/env bash
cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
bash /home/syaro/v11a_eval/gain_ab/rc_armC.sh
export GAIN_AB_DROP=PYG_PD_GAIN_DR GAIN_AB_PROBE_DROP=PYG_PD_GAIN_DR
bash /home/syaro/MikuchanRemote/Human-Pygmalion/tools/sim2real/gain_ab_eval.sh probe logs/rsl_rl/pygmalion_velocity/2026-10-17_07-45-38_rough_gain_crank60_gaindr_v11a slope /home/syaro/v11a_eval/gain_ab/slope_armC.json --angles 20 --vx 0.3,0.6 > /home/syaro/v11a_eval/gain_ab/probe_slope_armC.log 2>&1
bash /home/syaro/MikuchanRemote/Human-Pygmalion/tools/sim2real/gain_ab_eval.sh probe logs/rsl_rl/pygmalion_velocity/2026-10-17_07-45-38_rough_gain_crank60_gaindr_v11a stairs /home/syaro/v11a_eval/gain_ab/stairs_armC.json --risers 0.15 --vx 0.3 > /home/syaro/v11a_eval/gain_ab/probe_stairs_armC.log 2>&1
