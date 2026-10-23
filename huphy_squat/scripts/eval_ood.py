#!/usr/bin/env python3
"""Robustness sweep outside the training ranges, one factor at a time.

Everything else stays at the training distribution (full DR, training pushes, random h*
commands every 3-6 s). Per condition: 512 envs x 20 s, deterministic policy.

  eval_ood.py --ckpt PATH [--num-envs 512] [--only action_delay obs_delay friction payload push]
      [--out ood.json]

Factors
  action_delay_ms  command delay fixed to the value (training: 0-20 ms in 5 ms steps)
  obs_delay_ms     IMU + encoder delay fixed to the value (training: 0 / 20 / 40 ms)
  foot_friction    fixed foot friction (training: U(0.3, 1.2))
  payload_kg       fixed pelvis payload (training: U(-1, 8))
  push_N           directed pelvis push F x 0.2 s, 8 directions, at h* 0.72 / 0.80 / 0.89
                   (training: U(0, 60) N x 0.1-0.2 s); no random pushes in this test
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace

import torch

ACTION_DELAY_MS = (0, 10, 20, 30, 40)
OBS_DELAY_MS = (0, 20, 40, 60, 80)
FRICTION = (0.15, 0.2, 0.3, 0.7, 1.2, 1.5)
PAYLOAD = (-1.0, 0.0, 8.0, 12.0, 15.0, 20.0)
PUSH_N = (60.0, 80.0, 100.0, 120.0)
PUSH_HEIGHTS = (0.72, 0.80, 0.89)


def make(edit, num_envs):
  import huphy_squat  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Huphy-Squat-Flat-v0")
  cfg.scene.num_envs = num_envs
  edit(cfg)
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def policy(env, ckpt):
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  rl = load_rl_cfg("Huphy-Squat-Flat-v0")
  w = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  runner = MjlabOnPolicyRunner(w, asdict(rl), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return w, runner.get_inference_policy(device="cuda:0")


def rollout(env, w, pol, seconds=20.0, fixed_obs_lag=None):
  from huphy_squat.mdp import commands
  from huphy_squat.mdp.events import get_motor

  obs = w.get_observations()
  n = int(seconds / env.step_dt)
  fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  falls = 0
  errs, ratio, arate = [], [], []
  m = get_motor(env)
  c = env.command_manager.get_term("height")
  for _ in range(n):
    if fixed_obs_lag is not None:
      env.huphy_obs_lag[:] = fixed_obs_lag
    a = pol(obs)
    obs, _, d, _ = w.step(a)
    f = d.bool() & ~env.termination_manager.time_outs
    falls += int(f.sum())
    fell |= f
    settled = c._since_resample >= 2.0  # noqa: SLF001
    errs.append((commands.base_height(env) - c.command[:, 0]).abs()[settled])
    ratio.append(m.step_max_ratio.amax(1))
    am = env.action_manager
    arate.append((am.action - am.prev_action).square().sum(1))
  e = torch.cat(errs)
  return {
    "falls_per_env_minute": round(falls / env.num_envs / (seconds / 60), 4),
    "envs_with_fall_frac": round(float(fell.float().mean()), 4),
    "settled_err_mm_mean": round(float(e.mean()) * 1e3, 2),
    "settled_err_mm_p90": round(float(e.quantile(0.9)) * 1e3, 2),
    "torque_ratio_p99": round(float(torch.stack(ratio).flatten().quantile(0.99)), 3),
    "action_rate_mean": round(float(torch.stack(arate).mean()), 4),
  }


def run_condition(name, value, edit, ckpt, num_envs, fixed_obs_lag=None):
  env = make(edit, num_envs)
  w, pol = policy(env, ckpt)
  r = rollout(env, w, pol, fixed_obs_lag=fixed_obs_lag)
  r = {"factor": name, "value": value, **r}
  print(json.dumps(r), flush=True)
  del env, w, pol
  torch.cuda.empty_cache()
  return r


def push_sweep(ckpt, num_envs):
  from huphy_squat import robot

  def edit(cfg):
    cfg.events.pop("push")
  env = make(edit, num_envs)
  w, pol = policy(env, ckpt)
  r = env.scene["robot"]
  pel = [r.body_names.index(robot.BASE_BODY)]
  feet = [list(r.site_names).index(s) for s in robot.FOOT_SITES]
  ang = torch.arange(num_envs, device=env.device) % 8 * (2 * math.pi / 8)
  c = env.command_manager.get_term("height")
  out = []
  for h in PUSH_HEIGHTS:
    for F in PUSH_N:
      env.reset()
      obs = w.get_observations()
      fell = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
      t0, t1 = int(3.0 / env.step_dt), int(3.2 / env.step_dt)
      for i in range(int(6.0 / env.step_dt)):
        c.target[:, 0] = h
        c.time_left[:] = 1e9
        if i == t0:
          f0 = r.data.site_pos_w[:, feet, :2].clone()
        f = torch.zeros(num_envs, 1, 3, device=env.device)
        if t0 <= i < t1:
          f[:, 0, 0], f[:, 0, 1] = F * torch.cos(ang), F * torch.sin(ang)
        r.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=pel)
        obs, _, d, _ = w.step(pol(obs))
        if i >= t0:
          fell |= d.bool() & ~env.termination_manager.time_outs
      moved = (r.data.site_pos_w[:, feet, :2] - f0).norm(dim=-1).amax(1) > 0.02
      res = {"factor": "push_N", "h_star": h, "value": F, "impulse_Ns": F * 0.2,
             "fall_frac": round(float(fell.float().mean()), 3),
             "stepped_frac_of_survivors": round(float(moved[~fell].float().mean()), 3) if (~fell).any() else None,
             "fall_frac_by_direction": [round(float(fell[ang == a].float().mean()), 2) for a in ang[:8]]}
      print(json.dumps(res), flush=True)
      out.append(res)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--out")
  ap.add_argument("--only", nargs="*", default=None)
  a = ap.parse_args()
  ckpt, n = a.ckpt, a.num_envs
  want = lambda k: a.only is None or k in a.only  # noqa: E731
  res = []
  with torch.inference_mode():
    if want("action_delay"):
      for ms in ACTION_DELAY_MS:
        lag = ms // 5

        def edit(cfg, lag=lag):
          rb = cfg.scene.entities["robot"]
          rb.articulation.actuators = tuple(replace(x, delay_min_lag=lag, delay_max_lag=lag) for x in rb.articulation.actuators)
        res.append(run_condition("action_delay_ms", ms, edit, ckpt, n))
    if want("obs_delay"):
      from huphy_squat.mdp.observations import _SharedLagTerm

      train_max_lag = _SharedLagTerm.max_lag
      _SharedLagTerm.max_lag = max(OBS_DELAY_MS) // 20  # longer sensor ring buffer for this sweep only
      try:
        for ms in OBS_DELAY_MS:
          res.append(run_condition("obs_delay_ms", ms, lambda cfg: None, ckpt, n, fixed_obs_lag=ms // 20))
      finally:
        _SharedLagTerm.max_lag = train_max_lag
    if want("friction"):
      for mu in FRICTION:
        def edit(cfg, mu=mu):
          cfg.events["foot_friction"].params["ranges"] = (mu, mu)
        res.append(run_condition("foot_friction", mu, edit, ckpt, n))
    if want("payload"):
      for kg in PAYLOAD:
        def edit(cfg, kg=kg):
          cfg.events["payload"].params["mass_range"] = (kg, kg)
        res.append(run_condition("payload_kg", kg, edit, ckpt, n))
    if want("push"):
      res += push_sweep(ckpt, min(n, 256))
  if a.out:
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
