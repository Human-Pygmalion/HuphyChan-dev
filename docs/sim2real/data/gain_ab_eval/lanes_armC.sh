#!/usr/bin/env bash
cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
(xargs -P 2 -d "\n" -I{} bash -c "{}" < /home/syaro/v11a_eval/gain_ab/jobs_armC_cpu.txt) &
(bash /home/syaro/v11a_eval/gain_ab/gpu_armC.sh) &
wait
echo ALL_DONE_ARMC
