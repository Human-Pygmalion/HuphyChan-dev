#!/usr/bin/env bash
cd ~/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
(xargs -P 2 -d "\n" -I{} bash -c "{}" < /home/syaro/v11a_eval/gain_ab/jobs_armB_cpu.txt) &
(xargs -P 1 -d "\n" -I{} bash -c "{}" < /home/syaro/v11a_eval/gain_ab/jobs_armB_gpu.txt) &
wait
