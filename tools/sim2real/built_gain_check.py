#!/usr/bin/env python3
"""Print the per-joint Kp / Kd of the actuators in the evaluation environment.

Builds the rough task (2 envs, CPU) under the PYG_* flags of the calling shell, exactly like
analysis/measure_turn.py does, and lists stiffness / damping of every actuator.
Optionally compares them with the run's params/env.yaml (--run-dir) so that "the evaluation
environment has the trained gains" is a printed fact.

  cd mujoco-sim/mjlab
  env $(GAIN_AB_MANIFEST_DIR=<orig run> GAIN_AB_DROP="PYG_FRESH_STEPS PYG_CRITIC_WARMUP_ITERS" ...) \
    CUDA_VISIBLE_DEVICES="" .venv/bin/python3 ../../tools/sim2real/built_gain_check.py [--run-dir <run>]
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir")
a = ap.parse_args()

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-Velocity-Rough-Pygmalion", play=True)
cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg, device="cpu")
built = {}
for act in env.scene["robot"].actuators:
    for j, kp, kd in zip(act._target_names, act.stiffness[0].tolist(), act.damping[0].tolist()):
        built[j] = (round(kp, 4), round(kd, 4))
        print(f"built {j:<22} Kp={kp:8.3f} Kd={kd:7.4f}")
if a.run_dir:
    txt = open(a.run_dir + "/params/env.yaml").read()
    print("--- run params/env.yaml actuator gains, in file order ---")
    for m in re.finditer(r"^\s+(stiffness|damping): ([\d.]+)\s*$", txt, re.M):
        print(f"env.yaml {m.group(1)}: {m.group(2)}")
