#!/usr/bin/env python3
"""Closed-loop check of deploy_asset/huphy_squat against the training env.

  cd huphy_squat && /home/roy/huphy/mjlab/.venv/bin/python scripts/verify_deploy.py [--iters 5000 4500 4000]

The env runs the torch policy on the nominal model (no DR, noise, lags, encoder bias or pushes)
with an h* schedule that changes mid-run. Every step one deploy runner per env receives the raw
sim sensors (IMU gyro, IMU gravity = -up vector, motor encoder angles by name, h*) and builds
its own observation; compared each step: deploy obs vs env actor obs, ONNX output vs torch
actor output, deploy motor targets vs the env's processed joint targets. Pass: all < --tol.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[2] / "deploy_asset" / "huphy_squat"
SCHEDULE = (0.8914, 0.80, 0.70, 0.85, 0.65, 0.75, 0.895)  # h*, 150 steps (3 s) each
KEEP_EVENTS = ("reset_pose", "motor_step_end", "obs_lag")


def run(it: int, num_envs: int, tol: float) -> bool:
  import huphy_squat  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from huphy_squat.mdp.events import get_motor

  sys.path.insert(0, str(PKG))
  from policy_runner import SquatPolicy

  cfg = load_env_cfg("Huphy-Squat-Flat-v0", play=True)
  cfg.scene.num_envs = num_envs
  for k in list(cfg.events):
    if k not in KEEP_EVENTS:
      cfg.events.pop(k)
  cfg.events["obs_lag"].params["max_lag"] = 0
  cfg.commands["height"].resampling_time_range = (1e9, 1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  buf = get_motor(env)._delay_buffer  # noqa: SLF001
  if buf is not None:
    buf.min_lag = buf.max_lag = 0
  rl = load_rl_cfg("Huphy-Squat-Flat-v0")
  w = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  runner = MjlabOnPolicyRunner(w, asdict(rl), device="cuda:0")
  runner.load(str(PKG / f"policy_{it}.pt"), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")

  robot = env.scene["robot"]
  names = list(get_motor(env).target_names)
  jids = torch.tensor([robot.joint_names.index(j) for j in names], device=env.device)
  cmd = env.command_manager.get_term("height")
  term = env.action_manager.get_term("joint_pos")  # _processed_actions: offset + scale * a, clipped
  deps = [SquatPolicy.from_package(PKG, checkpoint=it, warn_period=False) for _ in range(num_envs)]
  for d in deps:
    d.reset()

  def h_at(t: int) -> float:
    return SCHEDULE[min(t // 150, len(SCHEDULE) - 1)]

  obs = w.get_observations()  # computed at reset with a random h*; the runners get that value at t = 0
  h0 = obs["actor"][:, -1].cpu().numpy().astype(np.float64)
  err = {"obs": 0.0, "action": 0.0, "target": 0.0}
  steps, fell = 150 * len(SCHEDULE), 0
  for t in range(steps):
    h = h_at(t)
    gyro = env.scene["robot/imu_ang_vel"].data.cpu().numpy()
    grav = (-env.scene["robot/imu_upvector"].data).cpu().numpy()
    q = robot.data.joint_pos_biased[:, jids].cpu().numpy()
    with torch.inference_mode():
      a = pol(obs)
    actor = obs["actor"].cpu().numpy()
    targets = []
    for i, d in enumerate(deps):
      o = d.adapter.build(gyro=gyro[i], projected_gravity=grav[i], joint_pos=dict(zip(names, q[i], strict=True)),
                          last_action=d._last_raw, h_star=h0[i] if t == 0 else h)  # noqa: SLF001
      raw = d.session.run([d.out_name], {d.in_name: o[None]})[0][0]
      d._last_raw = np.asarray(raw, dtype=np.float64)  # noqa: SLF001
      targets.append([d.adapter.action_to_target(d._last_raw)[j] for j in names])  # noqa: SLF001
      err["obs"] = max(err["obs"], float(np.abs(o - actor[i]).max()))
      err["action"] = max(err["action"], float(np.abs(raw - a[i].cpu().numpy()).max()))
    cmd.target[:, 0] = h_at(t + 1)  # read by the observation computed at the end of this step
    obs, _, dones, _ = w.step(a)
    fell += int(dones.sum())
    err["target"] = max(err["target"], float(np.abs(np.array(targets) - term._processed_actions.cpu().numpy()).max()))  # noqa: SLF001
    if bool(dones.any()):  # a reset env restarts its history; restart the matching runners
      for i in torch.nonzero(dones).flatten().tolist():
        deps[i].reset()
  env.close()
  ok = all(v < tol for v in err.values())
  print(f"checkpoint {it}: {steps} steps x {num_envs} envs, resets {fell}, max |diff| "
        + ", ".join(f"{k} {v:.2e}" for k, v in err.items()) + f" (tol {tol:g}) -> {'PASS' if ok else 'FAIL'}")
  return ok


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--iters", type=int, nargs="+", default=[5000, 4500, 4000])
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--tol", type=float, default=1e-4)
  args = ap.parse_args()
  ok = [run(it, args.num_envs, args.tol) for it in args.iters]
  sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
  main()
