#!/usr/bin/env python3
"""Show that the evaluation environment of a gain-randomised arm runs with NOMINAL gains.

Builds the rough task in play mode (the way analysis/measure_turn.py, measure_stand.py and
gait_kinematics_probe.py build it: 4 envs, CPU), lists the reset events, resets three times and prints the
spread (min / max over envs and resets) of the built Kp / Kd per joint.  With PYG_PD_GAIN_DR set, the
`pd_gains` event is in the list and the spread is wide; without it the spread is zero.
(2026-10-17, arm C rough_gain_crank60_gaindr_v11a.)

  cd mujoco-sim/mjlab
  env <PYG_* flags> CUDA_VISIBLE_DEVICES="" .venv/bin/python3 ../../tools/sim2real/gain_dr_eval_check.py
"""
import os

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

print("PYG_PD_GAIN_DR =", repr(os.environ.get("PYG_PD_GAIN_DR")))
cfg = load_env_cfg("Mjlab-Velocity-Rough-Pygmalion", play=True)
cfg.scene.num_envs = 4
print("events:", sorted(cfg.events))
print("curriculum:", sorted(cfg.curriculum))
env = ManagerBasedRlEnv(cfg, device="cpu")
lo, hi = {}, {}
for _ in range(3):
  env.reset()
  for act in env.scene["robot"].actuators:
    for j, name in enumerate(act._target_names):
      kp = act.stiffness[:, j]
      kd = act.damping[:, j]
      a = lo.setdefault(name, [1e9, 1e9])
      b = hi.setdefault(name, [-1e9, -1e9])
      a[0], a[1] = min(a[0], float(kp.min())), min(a[1], float(kd.min()))
      b[0], b[1] = max(b[0], float(kp.max())), max(b[1], float(kd.max()))
for n in lo:
  print(f"built {n:<22} Kp {lo[n][0]:8.3f} .. {hi[n][0]:8.3f}   Kd {lo[n][1]:7.4f} .. {hi[n][1]:7.4f}")
