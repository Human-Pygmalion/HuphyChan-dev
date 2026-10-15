#!/usr/bin/env python3
"""How hard can the standing robot be pushed before the deployed policy diverges, in simulation?

Question from the real robot (2026-10-15): with the IMU sign fixed the policy stands, but a small
push makes it diverge. Is that expected from the policy (simulation diverges too) or a real-only
gap (gain too low, latency, ...)? This sweeps push size and direction under the real run's gain
scale and timing, so the real push response can be read against it.

Per env: stand 5 s (command 0), then one push = add dv to the base linear velocity (same mechanism
as training's push_by_setting_velocity; training pushes were up to 0.7 m/s in x and y), watch 6 s.
Factors: Kp/Kd scale (with the robot's 12-bit MIT encoding), timing (one-step stale joint
observation + 2 % lost replies + HUPHY guards), push size, push direction.

Reported per cell: fell, max |action| after the push, action RMS in the last 2 s vs before the push
("diverging" if > 2x), knee angle std after the push, peak base tilt.

  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl <stock mjlab python> tools/sim2real/push_recovery_sim.py \
      --pkg <vibecoded-audit-20261012> --checkpoint <model.pt> --out push.json
"""
from __future__ import annotations

import argparse, json, math, os, sys
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="Pygmalion-Velocity-Flat-P2")
ap.add_argument("--gains", default="1.0,0.8,0.6")
ap.add_argument("--pushes", default="0.1,0.2,0.3,0.5,0.7", help="m/s")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.pkg); os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

DIRS = {"+x (forward)": (1, 0), "-x (backward)": (-1, 0), "+y (to left)": (0, 1), "-y (to right)": (0, -1)}
gains = [float(g) for g in a.gains.split(",")]; pushes = [float(p) for p in a.pushes.split(",")]
grid = [(g, t, p, d, r) for g in gains for t in (0, 1) for p in pushes for d in DIRS for r in range(a.reps)]
n = len(grid)
T_PUSH, T_END = 5.0, 11.0

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = n; cfg.seed = 5; cfg.episode_length_s = T_END + 5
cfg.observations["actor"].enable_corruption = False
cfg.events.pop("push_robot", None)
tw = cfg.commands["twist"]; tw.heading_command = False; tw.ranges.heading = None
tw.rel_standing_envs = 1.0; tw.resampling_time_range = (1e6, 1e6)
env = ManagerBasedRlEnv(cfg, device=a.device)
ag = load_rl_cfg(a.task); wenv = RslRlVecEnvWrapper(env, clip_actions=ag.clip_actions)
runner = load_runner_cls(a.task)(wenv, asdict(ag), device=a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
dev = env.device; robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos"); jid = term.target_ids
jn = list(term.target_names); knee = [jn.index(x) for x in jn if "knee" in x]


def quant(x, xmax):
  return torch.floor(torch.clamp(x, 0, xmax) / xmax * 4095) * xmax / 4095


gs = torch.tensor([g[0] for g in grid], device=dev)
for act in robot.actuators:
  if getattr(act, "stiffness", None) is not None:
    act.stiffness[:] = quant(act.default_stiffness * gs[:, None], 5000.0)
    act.damping[:] = quant(act.default_damping * gs[:, None], 100.0)
timing = torch.tensor([g[1] for g in grid], device=dev).bool()
dv = torch.tensor([[DIRS[g[3]][0] * g[2], DIRS[g[3]][1] * g[2]] for g in grid], device=dev, dtype=torch.float)
default = robot.data.default_joint_pos[:, jid]; hard = robot.data.joint_pos_limits[:, jid]; m3 = math.radians(3)

steps = int(round(T_END / env.step_dt)); k_push = int(round(T_PUSH / env.step_dt))
alive = torch.ones(n, dtype=torch.bool, device=dev); t_fall = torch.full((n,), float("nan"), device=dev)
A, KN, TILT = [], [], []
with torch.inference_mode():
  wenv.reset(); obs = wenv.get_observations()
  q_seen = robot.data.joint_pos[:, jid].clone(); hist = [q_seen.clone()] * 3; raw_prev = torch.zeros(n, 12, device=dev)
  for k in range(steps):
    if k == k_push:
      v = robot.data.root_link_vel_w.clone(); v[:, 0:2] += dv; robot.write_root_link_velocity_to_sim(v)
    o = obs["actor"].clone()
    q_now = robot.data.joint_pos[:, jid]
    keep = (torch.rand(n, 12, device=dev) < 0.02) & timing[:, None]
    q_seen = torch.where(keep, q_seen, q_now); hist = [hist[1], hist[2], q_seen.clone()]
    o[:, 6:18] = torch.where(timing[:, None], hist[0], hist[1]) - default
    o[:, 18:30] = torch.where(timing[:, None], hist[1], hist[2]) - default
    o[:, 30:42] = raw_prev
    obs["actor"] = o
    raw = policy(obs)
    tgt = default + 0.25 * raw
    t1 = torch.minimum(torch.maximum(tgt, hard[..., 0] + m3), hard[..., 1] - m3)
    t2 = torch.minimum(torch.maximum(t1, q_seen - math.radians(50)), q_seen + math.radians(50))
    sent = torch.where(timing[:, None], t2, tgt)
    obs, _, dones, _ = wenv.step((sent - default) / 0.25)
    raw_prev = raw.clone()
    fell = dones.bool() & ~env.termination_manager.time_outs & alive
    t_fall[fell] = k * env.step_dt; alive &= ~fell
    A.append(torch.where(alive[:, None], raw, torch.nan)); KN.append(robot.data.joint_pos[:, jid][:, knee[0]].clone())
    g = robot.data.projected_gravity_b; TILT.append(torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))))
A = torch.stack(A).cpu().numpy(); KN = torch.stack(KN).cpu().numpy(); TILT = torch.stack(TILT).cpu().numpy()
tf = t_fall.cpu().numpy(); dt = env.step_dt; k2 = int(2 / dt)
rows = []
for i, (g, t, p, d, r) in enumerate(grid):
  pre = A[k_push - k2:k_push, i]; post = A[k_push:, i]; last = A[-k2:, i]
  rms = lambda x: float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else float("nan")  # noqa: E731
  fell = bool(np.isfinite(tf[i]))
  rows.append(dict(gain=g, timing=t, push=p, dir=d, rep=r, fell=fell, t_fall_after_push=float(tf[i] - T_PUSH) if fell else None,
                   act_rms_pre=rms(pre), act_rms_last2s=rms(last), act_max_post=float(np.nanmax(np.abs(post))) if np.isfinite(post).any() else None,
                   knee_std_post=float(np.degrees(np.nanstd(KN[k_push:, i]))), tilt_max_post=float(np.nanmax(TILT[k_push:, i]))))
json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)
print(f"{'gain':>4} {'timing':>6} {'push':>4} | " + " | ".join(f"{d:>14s}" for d in DIRS) + "   (fell/reps, act RMS last2s / pre)")
for g in gains:
  for t in (0, 1):
    for p in pushes:
      cells = []
      for d in DIRS:
        rs = [x for x in rows if x["gain"] == g and x["timing"] == t and x["push"] == p and x["dir"] == d]
        f = sum(x["fell"] for x in rs); ratio = np.nanmean([x["act_rms_last2s"] / x["act_rms_pre"] for x in rs if not x["fell"]] or [np.nan])
        cells.append(f"{f}/{len(rs)} x{ratio:4.1f}")
      print(f"{g:4.1f} {'real' if t else 'ideal':>6} {p:4.1f} | " + " | ".join(f"{c:>14s}" for c in cells))
env.close()
