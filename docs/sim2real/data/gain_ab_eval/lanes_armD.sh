#!/usr/bin/env bash
cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
(xargs -P 3 -d "\n" -I{} bash -c "{}" < /home/syaro/v11a_eval/gain_ab/jobs_armD_cpu.txt) &
(bash /home/syaro/v11a_eval/gain_ab/gpu_armD.sh) &
wait
echo ALL_DONE_ARMD
