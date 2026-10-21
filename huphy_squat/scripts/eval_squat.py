#!/usr/bin/env python3
"""Evaluate a Huphy-Squat-Flat-v0 policy (or the zero / random action baseline).

  eval_squat.py liveness [--ckpt PATH] [--num-envs 512] [--seconds 20]
      training cfg (full DR + pushes): per reward term mean w*f and how often it is non-zero,
      terminations by cause, episode length, ankle / torque / linkage margins.
  eval_squat.py track --ckpt PATH [--no-dr]
      fixed h* sequence (5 s each), no pushes: steady error, time to +-2 cm, peak |vz|.
  eval_squat.py push --ckpt PATH
      per target height: settle 3 s, then one horizontal pelvis push (F x 0.2 s) in 8 directions;
      falls within 3 s, peak pelvis displacement, how far the feet moved.
  eval_squat.py all --ckpt PATH --out results.json

Prints a summary and writes everything to --out (json).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

import torch

TRACK_SEQUENCE = (0.895, 0.80, 0.72, 0.85, 0.75, 0.895, 0.72)
PUSH_HEIGHTS = (0.72, 0.80, 0.89)
PUSH_FORCES = (20.0, 35.0, 50.0)


def build(play: bool, num_envs: int, no_dr: bool = False, no_push: bool = False):
  import huphy_squat  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Huphy-Squat-Flat-v0", play=play)
  cfg.scene.num_envs = num_envs
  if no_dr:
    keep = ("reset_pose", "motor_step_end", "obs_lag", "push")
    for k in list(cfg.events):
      if k not in keep:
        cfg.events.pop(k)
  if no_push:
    cfg.events.pop("push", None)
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def policy_for(env, ckpt: str | None, mode: str = "zero"):
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  rl = load_rl_cfg("Huphy-Squat-Flat-v0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  if ckpt is None:
    if mode == "random":
      return wrapped, lambda obs: torch.randn(env.num_envs, 12, device=env.device)
    return wrapped, lambda obs: torch.zeros(env.num_envs, 12, device=env.device)
  runner = MjlabOnPolicyRunner(wrapped, asdict(rl), device="cuda:0")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return wrapped, runner.get_inference_policy(device="cuda:0")


def _robot_ids(env, names):
  r = env.scene["robot"]
  return [r.joint_names.index(n) for n in names]


def liveness(ckpt, num_envs, seconds, baseline="zero"):
  from huphy_squat import robot
  from huphy_squat.mdp import commands
  from huphy_squat.mdp.events import get_motor
  from huphy_squat.mdp.observations import loop_distances

  env = build(play=False, num_envs=num_envs)
  wrapped, pol = policy_for(env, ckpt, baseline)
  obs = wrapped.get_observations()
  rm, tm = env.reward_manager, env.termination_manager
  names = list(rm.active_terms)
  n = int(seconds / env.step_dt)
  sum_wf = torch.zeros(len(names), device=env.device)
  nonzero = torch.zeros(len(names), device=env.device)
  term_counts = {t: 0 for t in tm.active_terms}
  r = env.scene["robot"]
  m = get_motor(env)
  ai = _robot_ids(env, robot.ANKLE_JOINT_NAMES)
  lim = r.data.soft_joint_pos_limits[:, ai]
  ankle_out, tau90, rod, steps = 0.0, 0.0, 0.0, 0
  loop_q = []
  err_settled = []
  qd_max = torch.zeros(12, device=env.device)

  for _ in range(n):
    with torch.inference_mode():
      obs, _, dones, _ = wrapped.step(pol(obs))
    st = rm._step_reward  # noqa: SLF001  [B, T] = w * f
    sum_wf += st.sum(0)
    nonzero += (st != 0).float().sum(0)
    for t in tm.active_terms:
      term_counts[t] += int(tm.get_term(t).sum())
    q = r.data.joint_pos[:, ai]
    ankle_out += float(((q < lim[..., 0]) | (q > lim[..., 1])).float().mean())
    tau90 += float((m.step_max_ratio > 0.9).float().mean())
    rod += float((env.scene["rod_contact"].data.found > 0).any(dim=1).float().mean())
    loop_q.append(loop_distances(env).max(dim=1).values)
    qd_max = torch.maximum(qd_max, r.data.joint_vel[:, m.target_ids].abs().amax(0))
    c = env.command_manager.get_term("height")
    settled = c._since_resample >= 2.0  # noqa: SLF001
    if settled.any():
      err_settled.append((commands.base_height(env) - c.command[:, 0]).abs()[settled])
    steps += 1
  loops = torch.cat(loop_q)
  es = torch.cat(err_settled) if err_settled else torch.zeros(1, device=env.device)
  total = num_envs * steps
  pos = float(sum_wf[sum_wf > 0].sum() / total)
  neg = float(sum_wf[sum_wf < 0].sum() / total)
  terms = {nm: {"mean_wf": round(float(sum_wf[i] / total), 5), "frac_nonzero": round(float(nonzero[i] / total), 4)}
           for i, nm in enumerate(names)}
  episodes = sum(term_counts.values())
  out = {
    "ckpt": ckpt, "baseline": None if ckpt else baseline, "num_envs": num_envs, "seconds": seconds,
    "reward_terms": terms, "positive_sum_wf": round(pos, 4), "negative_sum_wf": round(neg, 4),
    "terminations": term_counts, "episodes_ended": episodes,
    "ankle_outside_soft_frac": round(ankle_out / steps, 5), "torque_ratio_gt_0p9_frac": round(tau90 / steps, 5),
    "rod_contact_env_step_frac": round(rod / steps, 5),
    "loop_residual_mm": {"p50": round(float(loops.median()) * 1e3, 4), "p99": round(float(loops.quantile(0.99)) * 1e3, 4),
                         "max": round(float(loops.max()) * 1e3, 4)},
    "motor_speed_max_rad_s": dict(zip(robot.MOTOR_JOINT_NAMES, [round(x, 2) for x in qd_max.tolist()], strict=True)),
    "height_error_settled_m": {"mean": round(float(es.mean()), 4), "p90": round(float(es.quantile(0.9)), 4)},
  }
  return out


def _set_target(env, h):
  c = env.command_manager.get_term("height")
  c.target[:, 0] = h
  c.time_left[:] = 1e9
  c._since_resample[:] = 0.0  # noqa: SLF001


def track(ckpt, num_envs, no_dr):
  from huphy_squat import robot
  from huphy_squat.mdp import commands
  from huphy_squat.mdp.events import get_motor

  env = build(play=True, num_envs=num_envs, no_dr=no_dr, no_push=True)
  wrapped, pol = policy_for(env, ckpt)
  obs = wrapped.get_observations()
  seg_steps = int(5.0 / env.step_dt)
  r = env.scene["robot"]
  m = get_motor(env)
  ap = [r.joint_names.index(n) for n in robot.ANKLE_JOINT_NAMES if "pitch" in n]
  ap_hi = r.data.soft_joint_pos_limits[0, ap, 1]
  rows, falls = [], 0
  for h_star in TRACK_SEQUENCE:
    _set_target(env, h_star)
    hs, vz, ank, ratio = [], [], [], []
    for _ in range(seg_steps):
      with torch.inference_mode():
        obs, _, dones, _ = wrapped.step(pol(obs))
      env.command_manager.get_term("height").target[:, 0] = h_star  # resets re-draw h*: pin it
      env.command_manager.get_term("height").time_left[:] = 1e9
      falls += int((dones.bool() & ~env.termination_manager.time_outs).sum())
      hs.append(commands.base_height(env))
      vz.append(r.data.root_link_lin_vel_w[:, 2])
      ank.append(r.data.joint_pos[:, ap].amax(1))
      ratio.append(m.step_max_ratio.amax(1))
    H = torch.stack(hs)  # [T, B]
    err = (H - h_star).abs()
    within = err < 0.02
    reach = torch.where(within.any(0), within.float().argmax(0).float() * env.step_dt, torch.full_like(err[0], float("nan")))
    rows.append({
      "h_star": h_star,
      "steady_abs_err_mean_m": round(float(err[-100:].mean()), 4),
      "steady_err_signed_m": round(float((H[-100:] - h_star).mean()), 4),
      "time_to_2cm_s_median": round(float(reach.nanmedian()), 2) if not torch.isnan(reach).all() else None,
      "reached_2cm_frac": round(float(within.any(0).float().mean()), 3),
      "peak_abs_vz_median": round(float(torch.stack(vz).abs().amax(0).median()), 3),
      "ankle_pitch_max_rad": round(float(torch.stack(ank).max()), 3),
      "ankle_pitch_soft_limit_rad": round(float(ap_hi.max()), 3),
      "ankle_over_soft_frac": round(float((torch.stack(ank) > ap_hi.max()).float().mean()), 4),
      "torque_ratio_p99": round(float(torch.stack(ratio).flatten().quantile(0.99)), 3),
    })
  return {"ckpt": ckpt, "dr": not no_dr, "segments": rows, "falls": falls}


def push(ckpt, num_envs):
  from huphy_squat import robot
  from huphy_squat.mdp import commands

  env = build(play=True, num_envs=num_envs, no_push=True)
  wrapped, pol = policy_for(env, ckpt)
  r = env.scene["robot"]
  pel = [r.body_names.index(robot.BASE_BODY)]
  feet = [list(r.site_names).index(s) for s in robot.FOOT_SITES]
  ang = torch.arange(num_envs, device=env.device) % 8 * (2 * math.pi / 8)
  res = []
  for h_star in PUSH_HEIGHTS:
    for F in PUSH_FORCES:
      env.reset()
      obs = wrapped.get_observations()
      _set_target(env, h_star)
      fell = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
      for i in range(int(6.0 / env.step_dt)):
        if i == int(3.0 / env.step_dt):
          p0 = r.data.root_link_pos_w[:, :2].clone()
          f0 = r.data.site_pos_w[:, feet, :2].clone()
          dmax = torch.zeros(num_envs, device=env.device)
        f = torch.zeros(num_envs, 1, 3, device=env.device)
        if int(3.0 / env.step_dt) <= i < int(3.2 / env.step_dt):
          f[:, 0, 0], f[:, 0, 1] = F * torch.cos(ang), F * torch.sin(ang)
        r.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=pel)
        with torch.inference_mode():
          obs, _, dones, _ = wrapped.step(pol(obs))
        env.command_manager.get_term("height").target[:, 0] = h_star
        env.command_manager.get_term("height").time_left[:] = 1e9
        fell |= dones.bool() & ~env.termination_manager.time_outs
        if i >= int(3.0 / env.step_dt):
          dmax = torch.maximum(dmax, (r.data.root_link_pos_w[:, :2] - p0).norm(dim=1))
      foot_move = (r.data.site_pos_w[:, feet, :2] - f0).norm(dim=-1).amax(1)
      ok = ~fell
      res.append({"h_star": h_star, "force_N": F, "impulse_Ns": F * 0.2, "fall_frac": round(float(fell.float().mean()), 3),
                  "pelvis_disp_max_m_median": round(float(dmax[ok].median()), 3) if ok.any() else None,
                  "foot_moved_gt_2cm_frac": round(float((foot_move[ok] > 0.02).float().mean()), 3) if ok.any() else None,
                  "final_height_err_median": round(float((commands.base_height(env) - h_star).abs()[ok].median()), 4) if ok.any() else None})
  return {"ckpt": ckpt, "results": res}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("liveness", "track", "push", "all"))
  ap.add_argument("--ckpt")
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--baseline", default="zero", choices=("zero", "random"))
  ap.add_argument("--no-dr", action="store_true")
  ap.add_argument("--out")
  a = ap.parse_args()
  out = {}
  with torch.inference_mode():  # env tensors created during inference steps cannot be edited outside it
    run(a, out)
  print(json.dumps(out, indent=1))
  if a.out:
    json.dump(out, open(a.out, "w"), indent=1)


def run(a, out: dict) -> None:
  if a.mode in ("liveness", "all"):
    out["liveness"] = liveness(a.ckpt, a.num_envs, a.seconds, a.baseline)
  if a.mode in ("track", "all"):
    out["track"] = track(a.ckpt, min(a.num_envs, 256), a.no_dr)
    if a.mode == "all":
      out["track_no_dr"] = track(a.ckpt, min(a.num_envs, 256), True)
  if a.mode in ("push", "all"):
    out["push"] = push(a.ckpt, min(a.num_envs, 256))


if __name__ == "__main__":
  main()
