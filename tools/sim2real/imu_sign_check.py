#!/usr/bin/env python3
"""Which IMU signal has the wrong sign in the real logs - gyro or gravity? Decided against the
leg kinematics, which do not involve the IMU.

In the real policy logs, d(gravity)/dt disagrees with gravity x gyro on the pitch axis (it fits
only if gyro y,z or gravity x is negated). Both fixes fit the IMU data equally well, so an
independent reference is needed: with the feet flat on the floor, body pitch and pitch rate are
functions of the leg joint angles.

1. Simulation (stock-mjlab package, flat ground, feet on the floor): a standing policy is kicked
   by random root-velocity pushes so the body pitches and rolls. Linear models are fitted:
     projected gravity x, y  ~ motor joint angles (12)
     gyro x, y               ~ motor joint velocities (12)
   The joint conventions are the robot's (user: motor direction and origin match the sim).
2. Real logs: the same models predict gravity x/y from the logged joint angles and gyro x/y from
   joint-angle differences, and are compared with what the IMU reported.

A positive correlation between "kinematic" and IMU values means that IMU channel has the sim sign.

  python tools/sim2real/imu_sign_check.py --pkg <vibecoded-audit-20261012> \
      --checkpoint <model.pt> --logs <HUPHY archivedlogs dir>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--logs", required=True)
ap.add_argument("--envs", type=int, default=64)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
sys.path.insert(0, a.pkg)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pygmalion_mjlab  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Pygmalion-Velocity-Flat-P2"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = a.envs
cfg.seed = 3
cfg.episode_length_s = a.seconds + 5
tw = cfg.commands["twist"]
tw.heading_command = False
tw.ranges.heading = None
tw.rel_standing_envs = 1.0  # stand: feet stay flat
tw.resampling_time_range = (1e6, 1e6)
env = ManagerBasedRlEnv(cfg, device=a.device)
agent = load_rl_cfg(TASK)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(TASK)(wenv, asdict(agent), device=a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
robot = env.scene["robot"]
# HUPHY order: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_a, ankle_b), then left leg.
names = [f"{s}_{j}" for s in ("R", "L") for j in
         ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint", "crank_A_joint", "crank_B_joint")]
jid = [robot.find_joints(n)[0][0] for n in names]
Q, QD, GR, GY = [], [], [], []
gen = torch.Generator(device=a.device).manual_seed(0)
steps = int(a.seconds / env.step_dt)
with torch.inference_mode():
  wenv.reset()
  obs = wenv.get_observations()
  for k in range(steps):
    if k % 40 == 20:  # kick: random pitch/roll rate and horizontal velocity
      v = robot.data.root_link_vel_w.clone()
      v[:, 0:2] += (torch.rand(v.shape[0], 2, device=a.device, generator=gen) - 0.5) * 0.6
      v[:, 3:5] += (torch.rand(v.shape[0], 2, device=a.device, generator=gen) - 0.5) * 2.0
      robot.write_root_link_velocity_to_sim(v)
    obs, _, dones, _ = wenv.step(policy(obs))
    o = obs["actor"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
    alive = ~dones.bool()
    Q.append(robot.data.joint_pos[:, jid][alive].cpu().numpy())
    QD.append(robot.data.joint_vel[:, jid][alive].cpu().numpy())
    GR.append(o[:, 3:6][alive].cpu().numpy())   # policy input: projected gravity
    GY.append(o[:, 0:3][alive].cpu().numpy())   # policy input: gyro (rad/s)
Q, QD, GR, GY = map(np.concatenate, (Q, QD, GR, GY))
print(f"sim samples {len(Q)}; gravity x range {GR[:,0].min():.2f}..{GR[:,0].max():.2f}, y {GR[:,1].min():.2f}..{GR[:,1].max():.2f}")


def fit(X, y):
  Xb = np.c_[X, np.ones(len(X))]
  n = len(X) // 5
  coef = np.linalg.lstsq(Xb[n:], y[n:], rcond=None)[0]
  pred = Xb[:n] @ coef
  r2 = 1 - ((y[:n] - pred) ** 2).sum() / ((y[:n] - y[:n].mean()) ** 2).sum()
  return coef, r2


models = {}
for label, X, y in (("grav_x", Q, GR[:, 0]), ("grav_y", Q, GR[:, 1]), ("gyro_x", QD, GY[:, 0]), ("gyro_y", QD, GY[:, 1])):
  coef, r2 = fit(X, y)
  models[label] = coef
  print(f"sim fit {label} ~ joints: held-out R^2 {r2:.3f}")

print("\nreal logs: corr(kinematic prediction, IMU value)  [+ = IMU has the sim sign]")
for f in sorted(glob.glob(os.path.join(a.logs, "*ob*.csv"))):
  d = np.genfromtxt(f, delimiter=",", skip_header=1)
  t, gyro, grav, pos = d[:, 0], d[:, 1:4], d[:, 4:7], np.radians(d[:, 70:82])
  dt = np.diff(t)
  ok = dt < 0.03
  qd = (np.diff(pos, axis=0) / dt[:, None])[ok]
  pm = (0.5 * (pos[1:] + pos[:-1]))[ok]
  gm = (0.5 * (gyro[1:] + gyro[:-1]))[ok]
  out = []
  for label, X, y in (("grav_x", pos, grav[:, 0]), ("grav_y", pos, grav[:, 1]), ("gyro_x", qd, gm[:, 0]), ("gyro_y", qd, gm[:, 1])):
    pred = np.c_[X, np.ones(len(X))] @ models[label]
    out.append(f"{label} {np.corrcoef(pred, y)[0, 1]:+.2f}")
  print(f"{os.path.basename(f):24s} " + "  ".join(out))
env.close()
