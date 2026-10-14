#!/usr/bin/env python3
"""Reproduce the HUPHY real-robot run conditions in simulation, one factor at a time.

The deployed policy (rough_box_main_v11a checkpoint 11999) diverged on the real robot (legs
shaking at ~1.1-1.3 Hz, knee std 15-18 deg, |action| p99 6-8). The real runs differ from
training in several ways at once (docs/sim2real/2026-10-13_huphy_archivedlogs_conditions.md).
This script switches each one on separately and in combination, so the factor that reproduces
the real signature can be named.

Factors (per env, one simulation):
  gains    body / ankle Kp,Kd scale, sent through the same 12-bit MIT encoding as the robot
           (kp 0-5000, kd 0-100, int() truncation -> ankle Kp 0 at scale 0.05)
  delay    the joint-angle history the policy sees is one policy step old (the robot feeds the
           state received in the previous cycle's collect)
  guards   target clipped to the model limit minus 3 deg, then to current angle +-50 deg
           (safety/guards.py); last_action in the observation stays the raw model output
  loss     each joint's observed angle is not updated with probability p per step (lost reply)
  imu      sign candidates: gyro y,z negated, or gravity x negated
  crane    upward force on the base = fraction x robot weight; constant, or ramped 1 -> 0
The policy rate is a separate run: --decimation 5 gives 25 ms (40 Hz), as the --hz 100 runs.

Metrics are computed until the first non-timeout termination and match the real-log analysis:
|action| p99 and RMS, time until |action| first exceeds 3, knee angle std and dominant
frequency, gyro RMS, guard clip rates, survival time.

  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl <python> tools/sim2real/real_condition_sim.py \
      --pkg <vibecoded-audit-20261012> --checkpoint <model.pt> --out out.json [--decimation 5]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default="Pygmalion-Velocity-Flat-P2")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--decimation", type=int, default=0, help="0 = task default (4 -> 50 Hz)")
ap.add_argument("--only", default="", help="comma list of condition names")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--crank-kp", type=float, default=0.0, help="override ankle crank Kp before the gain scale (0 = package value)")
ap.add_argument("--crank-kd", type=float, default=0.0, help="override ankle crank Kd before the gain scale (0 = package value)")
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

REAL = dict(delay=1, guards=1, loss=0.02)  # timing/guard factors present in every real run
CONDITIONS = {
  # training conditions, then each factor alone at full gain
  "nominal": dict(),
  "delay": dict(delay=1),
  "guards": dict(guards=1),
  "loss2%": dict(loss=0.02),
  "imu gyro_yz flip": dict(imu="gyro_yz"),
  "imu grav_x flip": dict(imu="grav_x"),
  "real timing (delay+guards+loss)": dict(**REAL),
  # the real gain settings, with the real timing, on the ground and with crane support
  "0.05/0.3 +timing": dict(body=0.05, ankle=0.3, **REAL),
  "0.05/0.5 +timing": dict(body=0.05, ankle=0.5, **REAL),
  "0.05/1.0 +timing": dict(body=0.05, ankle=1.0, **REAL),
  "0.1/0.1 +timing": dict(body=0.1, ankle=0.1, **REAL),
  "0.05/0.3 +timing crane 0.5": dict(body=0.05, ankle=0.3, crane=0.5, **REAL),
  "0.05/0.3 +timing crane 0.9": dict(body=0.05, ankle=0.3, crane=0.9, **REAL),
  "0.05/0.3 +timing crane 1->0": dict(body=0.05, ankle=0.3, crane=-1, **REAL),
  "0.05/0.5 +timing crane 1->0": dict(body=0.05, ankle=0.5, crane=-1, **REAL),
  "0.1/0.1 +timing crane 1->0": dict(body=0.1, ankle=0.1, crane=-1, **REAL),
  # gains alone (no timing factors), to separate the two
  "0.05/0.3 only": dict(body=0.05, ankle=0.3),
  "0.05/0.3 crane 1->0 only": dict(body=0.05, ankle=0.3, crane=-1),
  "0.5/0.5 +timing": dict(body=0.5, ankle=0.5, **REAL),
  "1.0 +timing +imu grav_x": dict(imu="grav_x", **REAL),
  # IMU hypotheses: mount 180 deg about z (gyro x,y and gravity x,y negated), pitch offset of the
  # real start (gravity x = +0.06..0.12 while the sim stands at 0.00)
  "imu yaw180": dict(imu="yaw180"),
  "imu grav_x bias +0.1": dict(gbias=0.1),
  "imu grav_x bias -0.1": dict(gbias=-0.1),
  "real timing +bias +0.1": dict(gbias=0.1, **REAL),
  # the low-gain crane runs under each IMU hypothesis: which one looks like the real logs?
  "0.05/0.3 crane 1->0 +timing +gyro_yz": dict(body=0.05, ankle=0.3, crane=-1, imu="gyro_yz", **REAL),
  "0.05/0.3 crane 1->0 +timing +grav_x": dict(body=0.05, ankle=0.3, crane=-1, imu="grav_x", **REAL),
  "0.05/0.3 crane 1->0 +timing +yaw180": dict(body=0.05, ankle=0.3, crane=-1, imu="yaw180", **REAL),
  "0.05/0.3 crane 1->0 +timing +bias0.1": dict(body=0.05, ankle=0.3, crane=-1, gbias=0.1, **REAL),
  "0.05/0.5 crane 1->0 +timing +gyro_yz": dict(body=0.05, ankle=0.5, crane=-1, imu="gyro_yz", **REAL),
  "0.05/0.5 crane 1->0 +timing +grav_x": dict(body=0.05, ankle=0.5, crane=-1, imu="grav_x", **REAL),
  # a safer bring-up: full or near-full gain with the crane lowered, and how a sign error looks there
  "1.0 +timing crane 1->0": dict(crane=-1, **REAL),
  "0.7 +timing": dict(body=0.7, ankle=0.7, **REAL),
  "0.8 +timing": dict(body=0.8, ankle=0.8, **REAL),
  "0.7 +timing crane 1->0": dict(body=0.7, ankle=0.7, crane=-1, **REAL),
  "1.0 +timing crane 1->0 +gyro_yz": dict(crane=-1, imu="gyro_yz", **REAL),
  "1.0 +timing crane 1->0 +grav_x": dict(crane=-1, imu="grav_x", **REAL),
  "1.0 +timing crane 1->0 +bias0.1": dict(crane=-1, gbias=0.1, **REAL),
}
if a.only:
  keep = [s.strip() for s in a.only.split(",")]
  CONDITIONS = {k: v for k, v in CONDITIONS.items() if k in keep}
names = list(CONDITIONS)
grid = [(c, r) for c in names for r in range(a.reps)]
n = len(grid)
P = lambda key, default: torch.tensor([CONDITIONS[c].get(key, default) for c, _ in grid])  # noqa: E731

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = n
cfg.seed = 11
cfg.episode_length_s = a.seconds + 5.0
if a.decimation:
  cfg.decimation = a.decimation
cfg.observations["actor"].enable_corruption = False
tw = cfg.commands["twist"]
tw.heading_command = False
tw.ranges.heading = None
tw.rel_standing_envs = 1.0
tw.resampling_time_range = (1e6, 1e6)
env = ManagerBasedRlEnv(cfg, device=a.device)
agent_cfg = load_rl_cfg(a.task)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = load_runner_cls(a.task)(wenv, asdict(agent_cfg), device=a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
dev = env.device
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
jnames = list(term.target_names)
jid = term.target_ids
print("action joints:", jnames)
print(f"policy period {env.step_dt*1000:.1f} ms; envs {n}")


def quant(x, xmax):  # robot codec: int(x / xmax * 4095) * xmax / 4095
  return torch.floor(torch.clamp(x, 0, xmax) / xmax * 4095) * xmax / 4095


body = P("body", 1.0).to(dev).float()
ankle = P("ankle", 1.0).to(dev).float()
for act in robot.actuators:
  if getattr(act, "stiffness", None) is None:
    continue
  is_ankle = torch.tensor(["crank" in t for t in act.target_names], device=dev)
  if (a.crank_kp or a.crank_kd) and is_ankle.any():  # 2026-10-15: trial v11a trains with crank 55/2.5
    if a.crank_kp:
      act.default_stiffness[:, is_ankle] = a.crank_kp
    if a.crank_kd:
      act.default_damping[:, is_ankle] = a.crank_kd
  if is_ankle.any():
    print("crank Kp/Kd at scale 1.0:", float(act.default_stiffness[0, is_ankle][0]), float(act.default_damping[0, is_ankle][0]), flush=True)
  s = torch.where(is_ankle[None, :], ankle[:, None], body[:, None])
  act.stiffness[:] = quant(act.default_stiffness * s, 5000.0)
  act.damping[:] = quant(act.default_damping * s, 100.0)
  print(act.target_names, "Kp at scale 0.05 ->", quant(act.default_stiffness[0] * 0.05, 5000.0).tolist())

delay = P("delay", 0).to(dev).bool()
guards = P("guards", 0).to(dev).bool()
loss = P("loss", 0.0).to(dev).float()
imu = [CONDITIONS[c].get("imu", "") for c, _ in grid]
flip_g = torch.tensor([m == "gyro_yz" for m in imu], device=dev)
flip_x = torch.tensor([m == "grav_x" for m in imu], device=dev)
yaw180 = torch.tensor([m == "yaw180" for m in imu], device=dev)
gbias = P("gbias", 0.0).to(dev).float()
crane = P("crane", 0.0).to(dev).float()
base_id = robot.find_bodies("base_link")[0][0]
mass = float(env.sim.mj_model.body_mass[1:].sum())
print(f"robot mass {mass:.1f} kg")

lim = robot.data.soft_joint_pos_limits[:, jid] if hasattr(robot.data, "soft_joint_pos_limits") else None
hard = robot.data.joint_pos_limits[:, jid]
margin = math.radians(3.0)
default = robot.data.default_joint_pos[:, jid]

steps = int(round(a.seconds / env.step_dt))
alive = torch.ones(n, dtype=torch.bool, device=dev)
t_alive = torch.zeros(n, device=dev)
t_act3 = torch.full((n,), float("nan"), device=dev)
A, KN, GY, CL, CJ, H = [], [], [], [], [], []
knee_ids = [jnames.index(x) for x in jnames if "knee" in x]
with torch.inference_mode():
  wenv.reset()
  obs = wenv.get_observations()
  q_seen = robot.data.joint_pos[:, jid].clone()       # what the robot's state cache holds
  q_hist = [q_seen.clone(), q_seen.clone(), q_seen.clone()]
  raw_prev = torch.zeros(n, 12, device=dev)
  for k in range(steps):
    tsec = k * env.step_dt
    # crane: upward force on the base
    frac = torch.where(crane < 0, torch.clamp(torch.tensor(1.0 - tsec / 15.0, device=dev), min=0.0), crane)
    F = torch.zeros(n, 1, 3, device=dev)
    F[:, 0, 2] = frac * mass * 9.81
    robot.write_external_wrench_to_sim(F, torch.zeros_like(F), body_ids=[base_id])

    o = obs["actor"].clone()
    # joint history: the robot shows the state of the previous cycle (delay) and keeps an angle on a lost reply
    q_now = robot.data.joint_pos[:, jid]
    keep = torch.rand(n, 12, device=dev) < loss[:, None]
    q_seen = torch.where(keep, q_seen, q_now)
    q_hist = [q_hist[1], q_hist[2], q_seen.clone()]
    cur, prev = q_hist[2], q_hist[1]
    cur_d, prev_d = q_hist[1], q_hist[0]
    qc = torch.where(delay[:, None], cur_d, cur)
    qp = torch.where(delay[:, None], prev_d, prev)
    o[:, 6:18] = qp - default
    o[:, 18:30] = qc - default
    o[:, 30:42] = raw_prev
    o[:, 1:3] = torch.where(flip_g[:, None], -o[:, 1:3], o[:, 1:3])
    o[:, 3] = torch.where(flip_x, -o[:, 3], o[:, 3])
    o[:, 0:2] = torch.where(yaw180[:, None], -o[:, 0:2], o[:, 0:2])
    o[:, 3:5] = torch.where(yaw180[:, None], -o[:, 3:5], o[:, 3:5])
    gv = o[:, 3:6].clone(); gv[:, 0] += gbias
    o[:, 3:6] = gv / gv.norm(dim=1, keepdim=True)
    obs["actor"] = o
    raw = policy(obs)
    # guards on the target, as safety/guards.py does
    tgt = default + 0.25 * raw
    g_lo = hard[..., 0] + margin
    g_hi = hard[..., 1] - margin
    t1 = torch.minimum(torch.maximum(tgt, g_lo), g_hi)
    t2 = torch.minimum(torch.maximum(t1, q_seen - math.radians(50)), q_seen + math.radians(50))
    cl = (t1 != tgt) & guards[:, None]
    cj = (t2 != t1) & guards[:, None]
    sent = torch.where(guards[:, None], t2, tgt)
    obs, _, dones, _ = wenv.step((sent - default) / 0.25)
    raw_prev = raw.clone()
    fell = dones.bool() & ~env.termination_manager.time_outs
    alive &= ~fell
    t_alive += alive.float() * env.step_dt
    big = (raw.abs().max(1).values > 3) & torch.isnan(t_act3) & alive
    t_act3[big] = tsec
    A.append(torch.where(alive[:, None], raw, torch.nan))
    KN.append(robot.data.joint_pos[:, jid][:, knee_ids[0]].clone())
    GY.append(o[:, 0:3].clone())
    CL.append(torch.where(alive[:, None], cl.float(), torch.nan))
    CJ.append(torch.where(alive[:, None], cj.float(), torch.nan))
    H.append(robot.data.root_link_pos_w[:, 2].clone())

A = torch.stack(A).cpu().numpy(); KN = torch.stack(KN).cpu().numpy(); GY = torch.stack(GY).cpu().numpy()
CL = torch.stack(CL).cpu().numpy(); CJ = torch.stack(CJ).cpu().numpy(); H = torch.stack(H).cpu().numpy()
ta = t_alive.cpu().numpy(); t3 = t_act3.cpu().numpy()
rows = []
for i, (c, r) in enumerate(grid):
  m = int(round(ta[i] / env.step_dt))
  m = max(m, 2)
  a_ = np.abs(A[:m, i]); kn = KN[:m, i]; gy = GY[:m, i]
  x = kn - kn.mean()
  f = np.fft.rfftfreq(len(x), env.step_dt); band = (f > 0.3) & (f < 12)
  pk = float(f[band][np.argmax(np.abs(np.fft.rfft(x * np.hanning(len(x))))[band])]) if band.any() else float("nan")
  rows.append(dict(cond=c, rep=r, survived_s=float(ta[i]), fell=bool(ta[i] < a.seconds - env.step_dt / 2),
                   act_p99=float(np.nanpercentile(a_, 99)), act_rms=float(np.sqrt(np.nanmean(a_ ** 2))),
                   t_act3=float(t3[i]), knee_std_deg=float(np.degrees(kn.std())), knee_hz=pk,
                   gyro_rms_dps=np.degrees(gy.std(0)).tolist(),
                   clip_limit=float(np.nanmean(CL[:m, i])), clip_jump=float(np.nanmean(CJ[:m, i])),
                   base_h=float(H[:m, i].mean())))
json.dump({"args": vars(a), "step_dt": env.step_dt, "conditions": CONDITIONS, "rows": rows}, open(a.out, "w"), indent=1)
print(f"{'condition':36s} {'fell':>5} {'surv':>5} {'act99':>6} {'actRMS':>6} {'t|a|>3':>6} {'kneeSD':>6} {'kneeHz':>6} {'gyro x/y/z dps':>15} {'lim%':>5} {'jump%':>5}")
for c in names:
  rs = [x for x in rows if x["cond"] == c]
  mm = lambda k: np.nanmean([x[k] for x in rs])  # noqa: E731
  g = np.mean([x["gyro_rms_dps"] for x in rs], 0)
  print(f"{c:36s} {sum(x['fell'] for x in rs):>2}/{len(rs)} {mm('survived_s'):5.1f} {mm('act_p99'):6.2f} {mm('act_rms'):6.2f} "
        f"{mm('t_act3'):6.2f} {mm('knee_std_deg'):6.1f} {mm('knee_hz'):6.2f} {g[0]:4.0f}/{g[1]:4.0f}/{g[2]:4.0f}  "
        f"{100*mm('clip_limit'):5.1f} {100*mm('clip_jump'):5.1f}")
env.close()
