#!/usr/bin/env python3
"""Quantify jitter of a Huphy-Squat-Flat-v0 policy holding a fixed height (deterministic actor).

  eval_jitter.py CKPT [--heights 0.75 0.80 0.85 0.89] [--envs-per-height 64] [--out jitter.json]

Per condition (clean + obs noise / clean, no noise / train DR + noise; never pushes), each env
gets a fixed h*, settles for 3 s, then 8 s are measured at 50 Hz:
  act_rate     mean ||a_t - a_t-1||^2               (the action_rate reward term)
  dtarget_rms  RMS of the per-step change of the 12 joint targets [mrad]
  qd_rms       RMS motor joint speed [rad/s]
  tau_std      std over time of the applied torque, mean over motors [N*m]
  dtau_rms     RMS per-step change of the applied torque [N*m]
  angvel_rms   RMS pelvis angular speed [rad/s]
  z_std        std of the pelvis height [mm]
  hf_frac      share of the action variance above 5 Hz
  h_err        mean |h - h*| [mm]
plus the per-joint RMS of the target change. Works with any version of the huphy_squat
package on the path (to compare an old checkpoint: PYTHONPATH=<old checkout>/huphy_squat).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

SETTLE_S, MEASURE_S = 3.0, 8.0
CLEAN_KEEP = ("reset_pose", "motor_step_end", "obs_lag")


def build(condition: str, heights, per_h: int):
  import huphy_squat  # noqa: F401
  from huphy_squat.mdp.events import get_motor
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Huphy-Squat-Flat-v0", play=False)
  cfg.scene.num_envs = per_h * len(heights)
  cfg.episode_length_s = 1e9
  cfg.events.pop("push")
  cfg.commands["height"].resampling_time_range = (1e9, 1e9)
  rp = cfg.events["reset_pose"].params
  if "spread_prob" in rp:
    rp["spread_prob"] = 0.0
  if condition.startswith("clean"):
    for k in list(cfg.events):
      if k not in CLEAN_KEEP:
        cfg.events.pop(k)
    cfg.events["obs_lag"].params["max_lag"] = 0
  if condition == "clean_nonoise":
    cfg.observations["actor"].enable_corruption = False
  only = {"clean_gyro_noise": "base_ang_vel", "clean_gravity_noise": "projected_gravity",
          "clean_encoder_noise": "motor_pos"}.get(condition)
  if only:  # noise on one actor term only
    for name, term in cfg.observations["actor"].terms.items():
      if name != only:
        term.noise = None
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  if condition.startswith("clean"):
    buf = get_motor(env)._delay_buffer  # noqa: SLF001
    if buf is not None:
      buf.min_lag = buf.max_lag = 0
  return env


def run(ckpt: str, condition: str, heights, per_h: int) -> dict:
  from huphy_squat import robot
  from huphy_squat.mdp.commands import base_height
  from huphy_squat.mdp.events import get_motor
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  env = build(condition, heights, per_h)
  rl = load_rl_cfg("Huphy-Squat-Flat-v0")
  w = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  runner = MjlabOnPolicyRunner(w, asdict(rl), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  pol = runner.get_inference_policy(device="cuda:0")
  dev, n = env.device, env.num_envs
  h_star = torch.tensor(heights, device=dev).repeat_interleave(per_h)
  cmd = env.command_manager.get_term("height")
  obs = w.get_observations()
  cmd.target[:, 0] = h_star
  obs = w.get_observations()
  robot_e, m = env.scene["robot"], get_motor(env)
  mids = torch.tensor([robot_e.joint_names.index(j) for j in m.target_names], device=dev)
  n_settle, n_meas = int(SETTLE_S / env.step_dt), int(MEASURE_S / env.step_dt)
  A, TAU, QD, W, Z = [], [], [], [], []
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  for t in range(n_settle + n_meas):
    with torch.inference_mode():
      obs, _, dones, _ = w.step(pol(obs))
    fell |= dones.bool()
    cmd.target[:, 0] = h_star  # a reset after a fall would redraw h*
    if t >= n_settle:
      A.append(env.action_manager.action.clone())
      TAU.append(m.applied_effort.clone())
      QD.append(robot_e.data.joint_vel[:, mids].clone())
      W.append(robot_e.data.root_link_ang_vel_w.norm(dim=1))
      Z.append(base_height(env) - h_star)
  A, TAU, QD = torch.stack(A, 1), torch.stack(TAU, 1), torch.stack(QD, 1)  # [B, T, 12]
  W, Z = torch.stack(W, 1), torch.stack(Z, 1)
  ok = ~fell
  dA = A[:, 1:] - A[:, :-1]
  dtarget = dA * robot.ACTION_SCALE
  spec = torch.fft.rfft(A - A.mean(1, keepdim=True), dim=1).abs().square()
  freqs = torch.fft.rfftfreq(A.shape[1], d=env.step_dt).to(dev)
  hf = spec[:, freqs > 5.0].sum((1, 2)) / spec.sum((1, 2)).clamp(min=1e-12)
  per_env = {
    "act_rate": dA.square().sum(-1).mean(1),
    "dtarget_rms": 1000 * dtarget.square().mean((1, 2)).sqrt(),
    "qd_rms": QD.square().mean((1, 2)).sqrt(),
    "tau_std": TAU.std(1).mean(-1),
    "dtau_rms": (TAU[:, 1:] - TAU[:, :-1]).square().mean((1, 2)).sqrt(),
    "angvel_rms": W.square().mean(1).sqrt(),
    "z_std": 1000 * Z.std(1),
    "hf_frac": hf,
    "h_err": 1000 * Z.abs().mean(1),
  }
  out = {"condition": condition, "falls": int(fell.sum()), "by_height": {}}
  for i, h in enumerate(heights):
    sel = slice(i * per_h, (i + 1) * per_h)
    mask = ok[sel]
    out["by_height"][f"{h:.3f}"] = {k: round(float(v[sel][mask].mean()), 4) if mask.any() else None
                                    for k, v in per_env.items()}
  out["all"] = {k: round(float(v[ok].mean()), 4) for k, v in per_env.items()}
  joint = 1000 * dtarget[ok].square().mean((0, 1)).sqrt()
  out["dtarget_rms_per_joint"] = {j: round(float(x), 3) for j, x in zip(m.target_names, joint, strict=True)}
  env.close()
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("ckpt")
  ap.add_argument("--heights", type=float, nargs="+", default=[0.75, 0.80, 0.85, 0.89])
  ap.add_argument("--envs-per-height", type=int, default=64)
  ap.add_argument("--conditions", nargs="+", default=["clean_noise", "clean_nonoise", "train_dr"],
                  help="also: clean_gyro_noise, clean_gravity_noise, clean_encoder_noise (one noise source)")
  ap.add_argument("--out")
  args = ap.parse_args()
  res = {"ckpt": args.ckpt, "results": [run(args.ckpt, c, args.heights, args.envs_per_height) for c in args.conditions]}
  for r in res["results"]:
    print(f"== {r['condition']}  falls {r['falls']}")
    print("   all   " + "  ".join(f"{k} {v}" for k, v in r["all"].items()))
    for h, row in r["by_height"].items():
      print(f"   {h}  " + "  ".join(f"{k} {v}" for k, v in row.items()))
    print("   dtarget_rms per joint [mrad]:", r["dtarget_rms_per_joint"])
  if args.out:
    with open(args.out, "w") as f:
      json.dump(res, f, indent=1)


if __name__ == "__main__":
  main()
