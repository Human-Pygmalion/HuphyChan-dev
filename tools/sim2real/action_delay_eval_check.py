#!/usr/bin/env python3
"""Show whether the evaluation environment of the action-delay arm applies a command delay.

Builds the rough task in play mode (the way analysis/measure_turn.py, measure_stand.py and
gait_kinematics_probe.py build it: 4 envs, CPU), prints the class of the joint-position action term and its
max_delay / noise, then steps three times with random actions and prints the largest difference between the
action passed to env.step and the raw action the term stored (zero when no delay mixes the previous command).
(2026-10-18, arm D rough_gain_crank60_actdelay_v11a; counterpart of gain_dr_eval_check.py.)

  cd mujoco-sim/mjlab
  env <PYG_* flags> CUDA_VISIBLE_DEVICES="" .venv/bin/python3 ../../tools/sim2real/action_delay_eval_check.py
"""
import os

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

print("PYG_ACTION_DELAY =", repr(os.environ.get("PYG_ACTION_DELAY")))
cfg = load_env_cfg("Mjlab-Velocity-Rough-Pygmalion", play=True)
cfg.scene.num_envs = 4
print("curriculum:", sorted(cfg.curriculum))
print("action cfg class:", type(cfg.actions["joint_pos"]).__name__)
env = ManagerBasedRlEnv(cfg, device="cpu")
term = env.action_manager.get_term("joint_pos")
print("action term class:", type(term).__name__)
print("term.max_delay =", getattr(term, "max_delay", "no such attribute"), " term.noise =", getattr(term, "noise", "no such attribute"))
env.reset()
torch.manual_seed(0)
worst = 0.0
for _ in range(3):
  a = torch.randn(env.num_envs, term.action_dim)
  env.step(a)
  worst = max(worst, float((term._raw_actions - a).abs().max()))
print(f"max |raw action stored - action passed| over 3 steps = {worst:.6f}")
print("RESULT:", "no command delay (stored action equals passed action)" if worst == 0.0 else "command delay is active")
