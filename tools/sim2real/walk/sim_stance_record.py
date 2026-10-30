#!/usr/bin/env python3
"""Replay a logged real command trace (or a segment of it) through a simulated `Pygmalion-Velocity-Generalized-Terrain10-UnionDR`
policy and write the result in the real-log CSV layout, with optional ONE-variable changes of the simulator (stance-width study,
docs/sim2real/2026-10-30_union49999_stance_width.md).

Setup (same as sim_walk_record.py; stock mjlab is NOT the mujoco-sim/mjlab fork, the fork venv supplies torch / mujoco-warp):
  git worktree add --detach <dir>/union_branch origin/generalized-union-dr
  git -C mujoco-sim/mjlab worktree add --detach <dir>/stock_mjlab 20f10e96
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl nice -n 10 mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/sim_stance_record.py \
      --branch-root <dir>/union_branch/mjlab_policy --mjlab-src <dir>/stock_mjlab/src \
      --trace Expr_Data/huphy_main_90bbd1f/1028_teleop_ob.csv --name T1_nominal --out <outdir> [--mode train] [--variant payload8]

Command: vx, wz from columns obs/cmd/vx, obs/cmd/wz of the trace, sampled at policy step k with t = k * 0.02 s (previous-row hold).
--seg t0:t1 cuts the trace (time zero = t0). The robot starts from the default pose like in training (joint offset +-0.05 rad,
base velocity +-0.2 m/s), position and heading forced to 0, flat 60 x 60 m strip, mean action, pushes OFF.
Modes: nominal = no DR, no noise, no lag, no delay (events: reset only); train = all Union-DR events of the task (pushes off).
Variants (one change each, applied on top of the mode; all via reset events of the task's own DR functions):
  payloadN         pelvis point mass +N kg                         kpR_<f>   hip_roll Kp x f        kdR_<f>   hip_roll Kd x f
  frictR_<f>       hip_roll joint friction loss x f                encM<d>   hip_roll encoder offset, adduction direction, d deg
  imuR<d>          IMU mount roll +d deg                           footmu<f> foot friction coefficient f      torqR_<f> hip_roll torque limit x f
  rollout<d>       DEPLOY FIX, target only: +d deg on the right hip_roll target and -d deg on the left one (right + = outward, left - = outward)
                   after the policy (after scale/offset/clip, before the PD controller). The policy observation (joint angles, last action) is unchanged.
Combined variants: parts joined by "+" (e.g. payload6+kpR_1.3, kpR_0.7+rollout2) are applied in order. Gain factors COMPOSE BY PRODUCT:
the task's pd_gains event scales the DEFAULT gain (it does not stack), so all kpR / kdR / kpkdR factors of one variant are multiplied first
and written as ONE event (kpR_0.7+kpR_1.4 -> Kp x 0.98; kdR is multiplied separately; Kp and Kd never reset each other). rolloutN offsets add up.
Every other part is an independent reset event with its own key.
Output: <out>/<name>.npz (keys as sim_walk_record.py plus q_true16, foot_contact, base_quat, base_pos) and <out>/csv/<name>_env<k>.csv
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--branch-root", required=True)
ap.add_argument("--mjlab-src", required=True)
ap.add_argument("--checkpoint", default="")
ap.add_argument("--task", default="Pygmalion-Velocity-Generalized-Terrain10-UnionDR")
ap.add_argument("--trace", required=True, help="real-layout CSV with obs/cmd/vx, obs/cmd/wz")
ap.add_argument("--seg", default="", help="t0:t1 seconds of the trace to use")
ap.add_argument("--mode", default="nominal", choices=["nominal", "train"])
ap.add_argument("--variant", default="")
ap.add_argument("--variants", default="", help="comma list: run each variant in this process as <name>_<variant> (saves the start-up time)")
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

sys.path.insert(0, args.mjlab_src)
sys.path.insert(0, args.branch_root)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab  # noqa: E402
import pygmalion_mjlab  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg  # noqa: E402
from pygmalion_mjlab.terrain10 import StripTerrainCfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_walk_export_csv import export  # noqa: E402

DT = 0.02
MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
PASSIVE = [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]
JOINTS16 = MOTORS + PASSIVE
ROLL = ("R_hip_roll_joint", "L_hip_roll_joint")
ROLL_OUT_DEG = [0.0]  # set by apply_variant (rolloutN), used by install_roll_offset


def load_trace(path, seg):
  t, vx, wz = [], [], []
  hdr = None
  for ln in open(path, encoding="utf-8"):
    if ln.startswith("#"):
      continue
    p = ln.rstrip("\n").split(",")
    if hdr is None:
      hdr = p
      it, ix, iz = hdr.index("t"), hdr.index("obs/cmd/vx"), hdr.index("obs/cmd/wz")
      continue
    t.append(float(p[it])); vx.append(float(p[ix])); wz.append(float(p[iz]))
  t, vx, wz = map(np.array, (t, vx, wz))
  if seg:
    a, b = map(float, seg.split(":"))
    m = (t >= a) & (t < b)
    t, vx, wz = t[m] - a, vx[m], wz[m]
  n = int(math.floor(t[-1] / DT))
  tk = np.arange(n) * DT
  idx = np.clip(np.searchsorted(t, tk, side="right") - 1, 0, len(t) - 1)
  return vx[idx].astype(np.float32), wz[idx].astype(np.float32)


@dataclass(kw_only=True)
class TraceCommandCfg(UniformVelocityCommandCfg):
  vx_trace: tuple = ()
  wz_trace: tuple = ()

  def build(self, env):
    return TraceCommand(self, env)


class TraceCommand(UniformVelocityCommand):
  """Command = function of episode time read from the logged trace (vy = 0); holds the last value after the end."""

  def _apply(self, ids):
    k = self._env.episode_length_buf[ids].long().clamp(max=len(self.cfg.vx_trace) - 1)
    vx = torch.tensor(self.cfg.vx_trace, device=self.device)[k]
    wz = torch.tensor(self.cfg.wz_trace, device=self.device)[k]
    self.vel_command_b[ids] = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)

  def _resample_command(self, env_ids):
    self._apply(env_ids)

  def _update_command(self):
    self._apply(torch.arange(self.num_envs, device=self.device))


def hiproll_strength(env, env_ids, scale, asset_cfg=SceneEntityCfg("robot")):
  """Scale the torque envelope of the two hip_roll motors only (episode-fixed)."""
  asset = env.scene[asset_cfg.name]
  for a in asset.actuators:
    if not hasattr(a, "strength_scale"):
      continue
    for j, n in enumerate(a.target_names):
      if n in ROLL:
        a.strength_scale[env_ids, j] = scale


def _apply_one(cfg, v, acc):
  ev = cfg.events
  roll = SceneEntityCfg("robot", joint_names=ROLL, preserve_order=True)
  rollre = SceneEntityCfg("robot", joint_names=(".*_hip_roll_joint",))
  if not v or v == "base":
    return
  m = re.fullmatch(r"payload(\d+(?:\.\d+)?)", v)
  if m:
    import pygmalion_mjlab.mdp.events as pe
    kg = float(m.group(1))
    ev["payload"] = EventTermCfg(func=pe.add_payload_mass, mode="reset",
                                 params={"mass_range": (kg, kg), "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))})
    return
  m = re.fullmatch(r"(kp|kd)R_(\d+(?:\.\d+)?)", v)
  if m:
    acc[m.group(1)] *= float(m.group(2))
    return
  m = re.fullmatch(r"kpkdR_(\d+(?:\.\d+)?)", v)
  if m:
    acc["kp"] *= float(m.group(1))
    acc["kd"] *= float(m.group(1))
    return
  m = re.fullmatch(r"rollout(-?\d+(?:\.\d+)?)", v)
  if m:
    ROLL_OUT_DEG[0] += float(m.group(1))
    return
  m = re.fullmatch(r"frictR_(\d+(?:\.\d+)?)", v)
  if m:
    f = float(m.group(1))
    ev["fric_" + v] = EventTermCfg(func=dr.joint_friction, mode="reset", params={"ranges": (f, f), "operation": "scale", "asset_cfg": rollre})
    return
  m = re.fullmatch(r"frictAbsR_(\d+(?:\.\d+)?)", v)  # absolute friction loss [N m]
  if m:
    f = float(m.group(1))
    ev["fric_" + v] = EventTermCfg(func=dr.joint_friction, mode="reset", params={"ranges": (f, f), "operation": "abs", "asset_cfg": rollre})
    return
  m = re.fullmatch(r"encM(-?\d+(?:\.\d+)?)", v)
  if m:
    d = math.radians(float(m.group(1)))
    # reported = true + bias.  Adduction of the true leg by d  <=>  reported angle is d larger in the abduction direction:
    # right abduction = +, left abduction = - (see the study); reported = true + bias, so bias_R = +d, bias_L = -d means
    # the encoders read d MORE abduction than the leg has, i.e. the true leg sits d more ADDUCTED than the controller thinks.
    sgn = {"R_hip_roll_joint": +1.0, "L_hip_roll_joint": -1.0}
    # the task's motor_pos_history observation reads the TRUE angle (joint_pos_rel, biased=False): the stock bias would only offset
    # the target, invisible to the policy. A real encoder zero error is seen by the policy, so the actor observation is switched to
    # the biased reading here (reported = true + bias), like on the robot.
    cfg.observations["actor"].terms["motor_pos_history"].params["biased"] = True
    for jn, s in sgn.items():
      ev[f"enc_{v}_{jn}"] = EventTermCfg(func=dr.encoder_bias, mode="reset",
                                         params={"bias_range": (s * d, s * d), "asset_cfg": SceneEntityCfg("robot", joint_names=(jn,))})
    return
  m = re.fullmatch(r"imuR(-?\d+(?:\.\d+)?)", v)
  if m:
    d = math.radians(float(m.group(1)))
    ev["imu_roll_" + v] = EventTermCfg(func=dr.site_quat, mode="reset", params={
      "roll_range": (d, d), "pitch_range": (0.0, 0.0), "yaw_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", site_names=("imu_in_base",))})
    return
  m = re.fullmatch(r"footmu(\d+(?:\.\d+)?)", v)
  if m:
    f = float(m.group(1))
    import pygmalion_mjlab.robot as pr
    ev["foot_mu_" + v] = EventTermCfg(func=dr.geom_friction, mode="reset", params={
      "ranges": (f, f), "operation": "abs", "shared_random": True, "asset_cfg": SceneEntityCfg("robot", geom_names=pr.FOOT_GEOMS)})
    return
  m = re.fullmatch(r"torqR_(\d+(?:\.\d+)?)", v)
  if m:
    ev["torq_" + v] = EventTermCfg(func=hiproll_strength, mode="reset", params={"scale": float(m.group(1))})
    return
  raise SystemExit(f"unknown variant {v}")


def apply_variant(cfg, v):
  """Apply a variant, or several joined by "+" in order (see the module docstring for how the parts compose)."""
  ROLL_OUT_DEG[0] = 0.0
  acc = {"kp": 1.0, "kd": 1.0}
  for part in (v.split("+") if v else []):
    _apply_one(cfg, part, acc)
  if acc["kp"] != 1.0 or acc["kd"] != 1.0:
    rollre = SceneEntityCfg("robot", joint_names=(".*_hip_roll_joint",))
    cfg.events["gain_" + v] = EventTermCfg(func=dr.pd_gains, mode="reset", params={
      "kp_range": (acc["kp"], acc["kp"]), "kd_range": (acc["kd"], acc["kd"]), "operation": "scale", "asset_cfg": rollre})


def install_roll_offset(at):
  """Deploy fix: add +d deg (right) / -d deg (left) to the hip_roll TARGET after the policy action term (post scale/offset/clip)."""
  d = math.radians(ROLL_OUT_DEG[0])
  if d == 0.0:
    return
  off = torch.zeros(1, len(MOTORS), device=at._processed_actions.device)
  off[0, MOTORS.index(ROLL[0])] = +d
  off[0, MOTORS.index(ROLL[1])] = -d
  orig = at.process_actions

  def process_actions(actions):
    orig(actions)
    at._processed_actions = at._processed_actions + off

  at.process_actions = process_actions


def build_cfg(vx, wz, variant):
  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  side = 60.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=89500, curriculum=False, size=(side, side), num_rows=1, num_cols=1, border_width=0.0, color_scheme="none",
      sub_terrains={"level1_flat": StripTerrainCfg(size=(side, side), slope_deg=0.0, ridge_height_range=(0.0, 0.0))},
    ),
    max_init_terrain_level=0,
  )
  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  tw = cfg.commands["twist"]
  cfg.commands["twist"] = TraceCommandCfg(**{f.name: getattr(tw, f.name) for f in fields(tw)}, vx_trace=tuple(vx.tolist()), wz_trace=tuple(wz.tolist()))
  cfg.commands["twist"].rel_standing_envs = 0.0
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.05), "yaw": (0.0, 0.0)}
  cfg.events.pop("torso_push", None)
  if args.mode == "nominal":
    cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints")}
    cfg.observations["actor"].enable_corruption = False
    for nm in ("base_ang_vel", "projected_gravity", "motor_pos_history"):
      th = cfg.observations["actor"].terms[nm]
      th.delay_min_lag = th.delay_max_lag = 0
    from dataclasses import replace
    ent = cfg.scene.entities["robot"]
    ent.articulation.actuators = tuple(replace(a, delay_min_lag=0, delay_max_lag=0) for a in ent.articulation.actuators)
  apply_variant(cfg, variant)
  return cfg


def run_one(name, variant):
  vx, wz = load_trace(args.trace, args.seg)
  T = len(vx)
  cfg = build_cfg(vx, wz, variant)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=args.device)
  ckpt = Path(args.checkpoint or Path(args.branch_root) / "checkpoints/terrain10_union_dr_49999/model_49999.pt")
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  torch.manual_seed(args.seed)
  env.seed(args.seed)
  obs, _ = wenv.reset()
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  idx16 = torch.tensor([names.index(n) for n in JOINTS16], device=env.device)
  at = env.action_manager.get_term("joint_pos")
  assert list(at.target_names) == MOTORS
  install_roll_offset(at)  # recorded D["target"] then INCLUDES the rolloutN offset (what the motors are commanded)
  default12 = robot.data.default_joint_pos[:, idx16[:12]].cpu().numpy()
  sensor = env.scene["feet_ground_contact"]
  pn = [p.split("/")[-1] for p in sensor.primary_names]
  fo = [pn.index("R_foot_link"), pn.index("L_foot_link")]
  E = env.num_envs
  f = lambda *s: np.full(s, np.nan, np.float32)  # noqa: E731
  D = dict(obs=f(T, E, 44), action=f(T, E, 12), target=f(T, E, 12), q_true16=f(T, E, 16), tau=f(T, E, 12), command=f(T, E, 3),
           base_pos=f(T, E, 3), base_quat=f(T, E, 4), foot_contact=np.zeros((T, E, 2), bool), foot_force_w=f(T, E, 2, 3),
           terminated=np.zeros((T, E), bool))
  acts = []
  for a in robot.actuators:
    cols = [MOTORS.index(n) for n in a.target_names]
    acts.append((a, cols))
  t0 = time.time()
  with torch.inference_mode():
    for i in range(T):
      d = robot.data
      D["obs"][i] = obs["actor"].cpu().numpy()
      D["q_true16"][i] = d.joint_pos[:, idx16].cpu().numpy()
      D["command"][i] = env.command_manager.get_command("twist").cpu().numpy()
      D["base_pos"][i] = d.root_link_pos_w.cpu().numpy()
      D["base_quat"][i] = d.root_link_quat_w.cpu().numpy()
      s = sensor.data
      D["foot_contact"][i] = s.found[:, fo].cpu().numpy() > 0
      D["foot_force_w"][i] = s.force[:, fo].cpu().numpy()
      act = policy(obs)
      obs, _, dones, ext = wenv.step(act)
      D["action"][i] = act.cpu().numpy()
      D["target"][i] = at._processed_actions.cpu().numpy()
      for a, cols in acts:
        D["tau"][i][:, cols] = a.applied_effort.cpu().numpy()
      D["terminated"][i] = env.reset_terminated.cpu().numpy()
      if i % 250 == 0:
        print(f"[{name}] step {i}/{T}  {time.time() - t0:.0f}s", flush=True)
  out = Path(args.out)
  (out / "csv").mkdir(parents=True, exist_ok=True)
  D["t"] = (np.arange(T) * DT).astype(np.float32)
  D["default_q12"] = default12[0].astype(np.float32)
  np.savez_compressed(out / f"{name}.npz", **D)
  export(out / f"{name}.npz", out / "csv", name)
  # same export with the TRUE joint angles in place of the reported ones (differs only when an encoder bias is active)
  D2 = dict(D)
  D2["obs"] = D["obs"].copy()
  D2["obs"][:, :, 18:30] = D["q_true16"][:, :, :12] - D["default_q12"]
  np.savez_compressed(out / f"{name}_true.npz", **D2)
  export(out / f"{name}_true.npz", out / "csv", name + "_true")
  nfall = int(D["terminated"].any(0).sum())
  fall_t = [float(D["t"][np.argmax(D["terminated"][:, e])]) if D["terminated"][:, e].any() else None for e in range(E)]
  meta = dict(name=name, mode=args.mode, variant=variant, trace=args.trace, seg=args.seg, seed=args.seed, num_envs=E,
              steps=T, duration_s=T * DT, checkpoint=str(ckpt), falls=nfall, fall_times_s=fall_t,
              kp=None, wall_s=round(time.time() - t0, 1), mjlab=getattr(mjlab, "__version__", "?"))
  (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
  print(f"[{name}] done: falls={nfall}/{E} wall={time.time() - t0:.0f}s", flush=True)
  try:
    env.close()
  except Exception:
    pass


def main():
  if args.variants:
    import traceback
    for v in args.variants.split(","):
      try:
        run_one(f"{args.name}_{v}", v)
      except Exception:  # keep going: one broken variant must not cost the others
        print(f"[{args.name}_{v}] FAILED", flush=True)
        traceback.print_exc()
  else:
    run_one(args.name, args.variant)


if __name__ == "__main__":
  main()
