#!/usr/bin/env python3
"""What does the deployed policy do in simulation when every joint's Kp and Kd are scaled down?

The real-robot bring-up lowers Kp/Kd and raises them step by step towards the training value.
The policy was trained only at scale 1.0, so a low scale is outside its training data. This
sweep shows what the simulator predicts at each scale, so a real response can be read against
it ("expected at this scale" vs "real-only problem").

One env per (scale, command, repeat); gains are scaled per env on the live actuators
(IdealPdActuator.stiffness / damping are [num_envs, num_joints] tensors). Flat ground, the
policy's play config (no pushes, no observation noise unless --obs-noise), deterministic
(mean) action. Uses the stock-mjlab task package from the vibecoded-audit branch.

  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl <mjlab python> tools/sim2real/gain_scale_sweep.py \
      --pkg <path to vibecoded-audit-20261012> --checkpoint <model.pt> --out sweep.json

--hang H holds the base H m above its start pose every control step (the legs hang free), the
simulator counterpart of a robot lifted on a tether or stand.

Per env it reports: survival time (first non-timeout termination), mean base height, RMS of
the 12 motor-joint velocities over the last 5 s (shaking), RMS of the raw-action change per
step (chatter), and the dominant frequency of the right-knee angle in the last 5 s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True, help="vibecoded-audit-20261012 folder (pygmalion_mjlab)")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="Pygmalion-Velocity-Flat-P2")
ap.add_argument("--scales", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
ap.add_argument("--commands", default="0,0,0;0.3,0,0", help="vx,vy,wz;...")
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--seconds", type=float, default=15.0)
ap.add_argument("--obs-noise", action="store_true")
ap.add_argument("--hang", type=float, default=0.0,
                help="hold the base this many metres above its start pose every step (lifted robot); 0 = on the ground")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.pkg)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pygmalion_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

scales = [float(s) for s in a.scales.split(",")]
cmds = [tuple(float(v) for v in c.split(",")) for c in a.commands.split(";")]
grid = [(s, c, r) for s in scales for c in cmds for r in range(a.reps)]
n = len(grid)

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = n
cfg.seed = 7
cfg.episode_length_s = a.seconds + 5.0
cfg.observations["actor"].enable_corruption = a.obs_noise
tw = cfg.commands["twist"]
tw.heading_command = False
tw.ranges.heading = None
tw.rel_standing_envs = tw.rel_world_envs = tw.rel_forward_envs = 0.0
tw.init_velocity_prob = 0.0
tw.resampling_time_range = (1e6, 1e6)
env = ManagerBasedRlEnv(cfg, device=a.device)
agent_cfg = load_rl_cfg(a.task)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(a.task)(wenv, asdict(agent_cfg), device=a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)

robot = env.scene["robot"]
scale_t = torch.tensor([g[0] for g in grid], device=env.device).unsqueeze(1)
for act in robot.actuators:  # one gain factor per env, all joints
  if getattr(act, "stiffness", None) is not None:
    act.stiffness[:] = act.default_stiffness * scale_t
    act.damping[:] = act.default_damping * scale_t
cmd_t = torch.tensor([g[1] for g in grid], device=env.device, dtype=torch.float)
term = env.command_manager.get_term("twist")
motor_ids, _ = robot.find_joints(pygmalion_mjlab.robot.MOTOR_JOINTS)
knee_id = robot.find_joints("R_knee_joint")[0][0]

steps = int(round(a.seconds / env.step_dt))
tail = int(round(5.0 / env.step_dt))
alive = torch.ones(n, dtype=torch.bool, device=env.device)
t_alive = torch.zeros(n, device=env.device)
height, qd2, da2, knee = [], [], [], []
with torch.inference_mode():
  wenv.reset()
  term.vel_command_b[:] = cmd_t
  obs = wenv.get_observations()
  if a.hang > 0:
    root0 = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                       torch.zeros_like(robot.data.root_link_pos_w), torch.zeros_like(robot.data.root_link_pos_w)], dim=1).clone()
    root0[:, 2] += a.hang
  prev = None
  for k in range(steps):
    term.vel_command_b[:] = cmd_t
    if a.hang > 0:  # the base is held: position and orientation fixed, velocity zero
      robot.write_root_state_to_sim(root0)
    act = policy(obs)
    obs, _, dones, extras = wenv.step(act)
    timeout = env.termination_manager.time_outs
    fell = dones.bool() & ~timeout
    alive &= ~fell
    t_alive += alive.float() * env.step_dt
    height.append(robot.data.root_link_pos_w[:, 2].clone())
    if k >= steps - tail:
      qd2.append((robot.data.joint_vel[:, motor_ids] ** 2).mean(1))
      if prev is not None:
        da2.append(((act - prev) ** 2).mean(1))
      knee.append(robot.data.joint_pos[:, knee_id].clone())
    prev = act.clone()

H = torch.stack(height).cpu().numpy()
QD = torch.stack(qd2).cpu().numpy()
DA = torch.stack(da2).cpu().numpy()
K = torch.stack(knee).cpu().numpy()
f = np.fft.rfftfreq(K.shape[0], d=env.step_dt)
rows = []
for i, (s, c, r) in enumerate(grid):
  x = K[:, i] - K[:, i].mean()
  spec = np.abs(np.fft.rfft(x))
  band = (f >= 0.5) & (f <= 25.0)
  rows.append({
    "scale": s, "cmd": c, "rep": r,
    "survived_s": float(t_alive[i]), "fell": bool(not alive[i]),
    "base_h_mean": float(H[:, i].mean()),
    "qd_rms": float(np.sqrt(QD[:, i].mean())),
    "dact_rms": float(np.sqrt(DA[:, i].mean())),
    "knee_peak_hz": float(f[band][np.argmax(spec[band])]),
    "knee_std_deg": float(np.degrees(K[:, i].std())),
  })
json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print(f"{'scale':>5} {'cmd':>12} {'fell':>5} {'surv_s':>6} {'h':>5} {'qd_rms':>7} {'dact':>6} {'knee_hz':>7} {'knee_sd':>7}")
for s in scales:
  for c in cmds:
    rs = [x for x in rows if x["scale"] == s and x["cmd"] == c]
    m = lambda k: np.mean([x[k] for x in rs])  # noqa: E731
    print(f"{s:5.2f} {str(c):>12} {sum(x['fell'] for x in rs):>3}/{len(rs)} {m('survived_s'):6.1f} "
          f"{m('base_h_mean'):5.2f} {m('qd_rms'):7.2f} {m('dact_rms'):6.3f} {m('knee_peak_hz'):7.1f} {m('knee_std_deg'):7.2f}")
env.close()
