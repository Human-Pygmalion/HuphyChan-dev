#!/usr/bin/env python3
"""Record a short deterministic rollout (actor observations, actions, base position) to .npz.

Used to show that this package on stock mjlab and the original training code produce the same
trajectory: run it once per setup with the same seed / command / policy, then compare the two
files with --compare.

  # this package, stock mjlab
  python scripts/rollout_trace.py --task Pygmalion-Velocity-Rough-Box --import pygmalion_mjlab \\
      --onnx checkpoints/rough_main_model_11999.onnx --out pkg.npz
  # original training code (needs its PYG_* flags; the manifest lists them)
  python scripts/rollout_trace.py --task Mjlab-Velocity-Rough-Pygmalion \\
      --manifest reference/launch_manifest_reference.json --onnx ... --out fork.npz
  python scripts/rollout_trace.py --compare pkg.npz fork.npz

The command is pinned (no resampling, no standing/heading envs); the terrain level is the
generator's own reset draw (same seed on both sides). CPU device for bit-stable physics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--task")
ap.add_argument("--import", dest="imp", default="")
ap.add_argument("--manifest", default="")
ap.add_argument("--onnx")
ap.add_argument("--out")
ap.add_argument("--envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--seed", type=int, default=43)
ap.add_argument("--cmd", default="0.6,0.0,0.0")
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()

import numpy as np  # noqa: E402

if a.compare:
  x, y = (np.load(p) for p in a.compare)
  for k in ("obs", "act", "pos"):
    d = np.abs(x[k] - y[k])
    per_step = d.reshape(d.shape[0], -1).max(axis=1)
    first = int(np.argmax(per_step > 1e-4)) if (per_step > 1e-4).any() else None
    print(f"{k:4s}: max |diff| {d.max():.3e}  mean {d.mean():.3e}  first step > 1e-4: {first}")
  print(f"base travel (pkg / ref), mean over envs: {np.linalg.norm(x['pos'][-1,:,:2]-x['pos'][0,:,:2],axis=1).mean():.3f} / "
        f"{np.linalg.norm(y['pos'][-1,:,:2]-y['pos'][0,:,:2],axis=1).mean():.3f} m")
  raise SystemExit(0)

if a.manifest:
  for k, v in json.load(open(a.manifest))["environment"].items():
    if k.startswith("PYG_"):
      os.environ[k] = str(v)
os.environ.setdefault("MUJOCO_GL", "egl")

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import mjlab  # noqa: E402
import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

if a.imp:
  importlib.import_module(a.imp)
print("mjlab from", mjlab.__file__, flush=True)

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.envs
cfg.seed = a.seed
tw = cfg.commands["twist"]
cmd = [float(v) for v in a.cmd.split(",")]
tw.ranges.lin_vel_x = (cmd[0], cmd[0])
tw.ranges.lin_vel_y = (cmd[1], cmd[1])
tw.ranges.ang_vel_z = (cmd[2], cmd[2])
tw.rel_standing_envs = 0.0
tw.rel_heading_envs = 0.0
tw.rel_forward_envs = 0.0
tw.resampling_time_range = (1e6, 1e6)
if tw.heading_command:
  tw.heading_command = False
  tw.ranges.heading = None
cfg.curriculum.pop("command_vel", None)  # the original code's gate would rewrite the ranges
cfg.observations["actor"].enable_corruption = False
env = ManagerBasedRlEnv(cfg, device="cpu")
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
inn = sess.get_inputs()[0].name
robot = env.scene["robot"]
obs, _ = env.reset()
O, A, P = [], [], []
for _ in range(a.steps):
  x = obs["actor"].numpy().astype(np.float32)
  act = np.concatenate([sess.run(None, {inn: x[i : i + 1]})[0] for i in range(len(x))])
  O.append(x)
  A.append(act)
  P.append(robot.data.root_link_pos_w.numpy().copy())
  obs, *_ = env.step(torch.as_tensor(act))
np.savez(a.out, obs=np.stack(O), act=np.stack(A), pos=np.stack(P))
print(f"saved {a.out}: {a.steps} steps x {a.envs} envs", flush=True)
env.close()
