#!/usr/bin/env python3
"""Run the deployed Union-DR 49999 policy with a swapped foot collision model and record contact / CoP / GRF / tracking data.

Foot variants are the XMLs written by tools/foot_sole/v3_collision.py next to pygmalion_legonly_ab.xml:
  base (training model: 3 boxes = bare plate), v3full, v3same, v3sph.
The policy is NOT retrained: every v3 variant is a zero-shot test of the policy trained on the base foot.

Setup (same worktrees as tools/sim2real/walk/sim_stance_record.py):
  git worktree add --detach <dir>/union_branch origin/generalized-union-dr
  git -C mujoco-sim/mjlab worktree add --detach <dir>/stock_mjlab 20f10e96
  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/foot_variant_eval.py \
     --branch-root <dir>/union_branch/mjlab_policy --mjlab-src <dir>/stock_mjlab/src --variant v3full --out <outdir>

Command program (15 s per command, vy = 0): stand, vx 0.3 / 0.6 / 1.0 / 1.5, vx -0.5, wz +0.6, wz -0.6, vx 0.6 + wz 0.5.
Flat 60 x 60 m strip, nominal mode (no DR, no noise, no delay), mean action, pushes off, reset joint offset +-0.05 rad.
Per step and foot the script reduces the sole contacts (up to 64 strongest per foot) to: normal force, tangential force,
CoP in the foot_link frame (x from the ankle, y left), contact count, contact-point extent, deepest penetration.
Output: <out>/<variant>.npz + <variant>.json (meta).
"""
from __future__ import annotations

import argparse
import os
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--branch-root", required=True)
ap.add_argument("--mjlab-src", required=True)
ap.add_argument("--variant", required=True, help="base, or <name> of pygmalion_legonly_ab_<name>.xml (v3full, v5full, v5fullm, v5_* ...)")
ap.add_argument("--terrain", default="flat", choices=["flat", "ridge", "slope_rough", "stones", "stones_lo", "pebbles", "hf_flat"],
                help="flat: 60 m strip (as the zero-shot runs); ridge: training level 7 (capsule ridges 1-2 cm every 10-20 cm); "
                     "slope_rough: training level 10 (6 deg up + ridges); stones / stones_lo: scattered 2 cm / 1 cm boxes 10-20 cm wide "
                     "(not in training); pebbles: dense 3-8 mm boxes 1-3 cm wide in a 6 m corridor along +x (not in training); "
                     "hf_flat: a FLAT HEIGHT FIELD, only to check the Warp height-field contact defect against flat (never for results)")
ap.add_argument("--task", default="Pygmalion-Velocity-Generalized-Terrain10-UnionDR")
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seconds-per-cmd", type=float, default=15.0)
ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N policy steps")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--checkpoint", default="", help="policy checkpoint (default: the deployed 49999); e.g. a fine-tuned model_52499.pt")
ap.add_argument("--tag", default="", help="output file stem (default: the variant name)")
args = ap.parse_args()
TAG = args.tag or (args.variant if args.terrain == "flat" else f"{args.variant}_{args.terrain}")

sys.path.insert(0, args.mjlab_src)
sys.path.insert(0, args.branch_root)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pygmalion_mjlab  # noqa: E402,F401
import pygmalion_mjlab.robot as pr  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg  # noqa: E402
from mjlab.utils.spec_config import CollisionCfg  # noqa: E402
from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg  # noqa: E402
from sole_terrains import ScatterBoxTerrainCfg  # noqa: E402
from pygmalion_mjlab.terrain10 import StripTerrainCfg  # noqa: E402

DT = 0.02
PROGRAM = [("stand", 0.0, 0.0), ("vx0.3", 0.3, 0.0), ("vx0.6", 0.6, 0.0), ("vx1.0", 1.0, 0.0), ("vx1.5", 1.5, 0.0),
           ("vx-0.5", -0.5, 0.0), ("wz+0.6", 0.0, 0.6), ("wz-0.6", 0.0, -0.6), ("vx0.6wz0.5", 0.6, 0.5)]
MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
JOINTS16 = MOTORS + [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]
XMLS = {"base": "pygmalion_legonly_ab.xml", "v3full": "pygmalion_legonly_ab_v3full.xml",
        "v3same": "pygmalion_legonly_ab_v3same.xml", "v3sph": "pygmalion_legonly_ab_v3sph.xml", "v3lat": "pygmalion_legonly_ab_v3lat.xml", "v5full": "pygmalion_legonly_ab_v5full.xml",
        "v5fullm": "pygmalion_legonly_ab_v5fullm.xml"}
NSLOT = 64
SIGN = float(__import__("os").environ.get("SOLE_FN_SIGN", "1"))  # +1: sensor force along the primary->secondary normal = push on the foot (checked against body weight)

n_per = int(round(args.seconds_per_cmd / DT))
VX = np.concatenate([np.full(n_per, vx, np.float32) for _, vx, _ in PROGRAM])
WZ = np.concatenate([np.full(n_per, wz, np.float32) for _, _, wz in PROGRAM])


@dataclass(kw_only=True)
class TraceCommandCfg(UniformVelocityCommandCfg):
  vx_trace: tuple = ()
  wz_trace: tuple = ()

  def build(self, env):
    return TraceCommand(self, env)


class TraceCommand(UniformVelocityCommand):
  def _apply(self, ids):
    k = self._env.episode_length_buf[ids].long().clamp(max=len(self.cfg.vx_trace) - 1)
    vx = torch.tensor(self.cfg.vx_trace, device=self.device)[k]
    wz = torch.tensor(self.cfg.wz_trace, device=self.device)[k]
    self.vel_command_b[ids] = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)

  def _resample_command(self, env_ids):
    self._apply(env_ids)

  def _update_command(self):
    self._apply(torch.arange(self.num_envs, device=self.device))


def build_cfg():
  cfg = load_env_cfg(args.task, play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  side = 60.0
  sub = {
    "flat": lambda: StripTerrainCfg(size=(side, side), slope_deg=0.0, ridge_height_range=(0.0, 0.0)),
    "ridge": lambda: StripTerrainCfg(size=(side, side), slope_deg=0.0, ridge_height_range=(0.010, 0.020)),
    "slope_rough": lambda: StripTerrainCfg(size=(side, side), slope_deg=6.0, ridge_height_range=(0.010, 0.020)),
    "stones": lambda: ScatterBoxTerrainCfg(size=(side, side), n_boxes=6000, width_range=(0.10, 0.20), height_range=(0.02, 0.02)),
    "stones_lo": lambda: ScatterBoxTerrainCfg(size=(side, side), n_boxes=6000, width_range=(0.10, 0.20), height_range=(0.01, 0.01)),
    "pebbles": lambda: ScatterBoxTerrainCfg(size=(side, side), n_boxes=9000, width_range=(0.01, 0.03), height_range=(0.003, 0.008),
                                            corridor_half_width=3.0),
    "hf_flat": lambda: HfRandomUniformTerrainCfg(size=(side, side), noise_range=(0.0, 0.0), noise_step=0.005, downsampled_scale=0.1,
                                                 horizontal_scale=0.1, vertical_scale=0.005),
  }[args.terrain]()
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=89500, curriculum=False, size=(side, side), num_rows=1, num_cols=1, border_width=0.0, color_scheme="none",
      sub_terrains={args.terrain: sub}),
    max_init_terrain_level=0)
  cfg.curriculum = {}
  cfg.episode_length_s = int(1e9)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  tw = cfg.commands["twist"]
  cfg.commands["twist"] = TraceCommandCfg(**{f.name: getattr(tw, f.name) for f in fields(tw)},
                                          vx_trace=tuple(VX.tolist()), wz_trace=tuple(WZ.tolist()))
  cfg.commands["twist"].rel_standing_envs = 0.0
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  # flat: every env starts at the centre facing +x (as the zero-shot runs); rough: spread the starts over the terrain and the
  # heading over the full circle so the envs sample different ground under the feet
  # (slope: spread only across the slope, along y, where the ground height does not change; rough: start 3-6 cm up so the feet
  # cannot start inside a 2 cm ridge, stone or gravel cell)
  if args.terrain in ("flat", "hf_flat"):
    pr_ = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.05), "yaw": (0.0, 0.0)}
  elif args.terrain == "slope_rough":
    pr_ = {"x": (0.0, 0.0), "y": (-6.0, 6.0), "z": (0.03, 0.06), "yaw": (-math.pi, math.pi)}
  elif args.terrain == "ridge":
    pr_ = {"x": (-6.0, 6.0), "y": (-6.0, 6.0), "z": (0.03, 0.06), "yaw": (-math.pi, math.pi)}
  else:
    # scattered boxes: start at the -x end facing +x, so the 135 s program (about 50 m of forward walking) crosses the whole tile
    pr_ = {"x": (-27.0, -25.0), "y": (-1.5, 1.5) if args.terrain == "pebbles" else (-20.0, 20.0), "z": (0.03, 0.06), "yaw": (0.0, 0.0)}
  cfg.events["reset_base"].params["pose_range"] = pr_
  cfg.events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints")}
  cfg.observations["actor"].enable_corruption = False
  for nm in ("base_ang_vel", "projected_gravity", "motor_pos_history"):
    th = cfg.observations["actor"].terms[nm]
    th.delay_min_lag = th.delay_max_lag = 0
  ent = cfg.scene.entities["robot"]
  ent.articulation.actuators = tuple(replace(a, delay_min_lag=0, delay_max_lag=0) for a in ent.articulation.actuators)
  # foot variant: geometry from the variant XML, foot collision settings on every <side>_foot<k>_collision (not only 1-7)
  xml = Path(args.branch_root) / "pygmalion_mjlab/assets" / XMLS.get(args.variant, f"pygmalion_legonly_ab_{args.variant}.xml")
  ent.spec_fn = lambda: mujoco.MjSpec.from_file(str(xml))
  if args.variant not in ("base", "v3same"):  # sole bottom 14 mm lower than the training foot: spawn 14 mm higher
    ent.init_state = replace(ent.init_state, pos=(ent.init_state.pos[0], ent.init_state.pos[1], ent.init_state.pos[2] + 0.014))
  # 47 convex sole pieces per foot give up to ~200 contacts per world (box feet: <= 24)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 640)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 3072)
  ent.collisions = (CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(L|R)_foot\d+_collision$": 3, ".*_collision": 1},
    priority={r"^(L|R)_foot\d+_collision$": 1},
    friction={r"^(L|R)_foot\d+_collision$": (0.6,)},
    solref=({r"^(L|R)_foot\d+_collision$": (float(os.environ["PYG_FOOT_SOLREF"]), 1.0)}  # same switch as robot.py
            if os.environ.get("PYG_FOOT_SOLREF") else None)),)
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (ContactSensorCfg(
    name="sole_contacts",
    primary=ContactMatch(mode="subtree", pattern=r"^(R_foot_link|L_foot_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos", "normal", "tangent", "dist"), reduce="maxforce", num_slots=NSLOT, global_frame=True),
    # net force per foot at every physics substep (5 ms): impact peaks and loading rate need more than the 50 Hz policy rate
    ContactSensorCfg(name="sole_net",
                     primary=ContactMatch(mode="subtree", pattern=r"^(R_foot_link|L_foot_link)$", entity="robot"),
                     secondary=ContactMatch(mode="body", pattern="terrain"),
                     fields=("found", "force"), reduce="netforce", history_length=cfg.decimation),)
  return cfg, xml


def quat_to_mat(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def main():
  cfg, xml = build_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(wenv, asdict(agent_cfg), device=args.device)
  ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.branch_root) / "checkpoints/terrain10_union_dr_49999/model_49999.pt"
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  torch.manual_seed(args.seed)
  env.seed(args.seed)
  obs, _ = wenv.reset()
  robot = env.scene["robot"]
  names = list(robot.joint_names)
  idx16 = torch.tensor([names.index(n) for n in JOINTS16], device=env.device)
  bn = list(robot.body_names)
  fbody = torch.tensor([bn.index("R_foot_link"), bn.index("L_foot_link")], device=env.device)
  sens = env.scene["sole_contacts"]
  pn = [p.split("/")[-1] for p in sens.primary_names]
  order = [pn.index("R_foot_link"), pn.index("L_foot_link")]
  acts = [(a, [MOTORS.index(n) for n in a.target_names]) for a in robot.actuators]
  T = len(VX) if not args.max_steps else min(args.max_steps, len(VX))
  E = env.num_envs
  f = lambda *s: np.full(s, np.nan, np.float32)  # noqa: E731
  # per foot: Fn, Ft, cop_x, cop_y, n_contacts, x_min, x_max, y_min, y_max, max_penetration
  D = dict(sole=f(T, E, 2, 10), base_pos=f(T, E, 3), base_quat=f(T, E, 4), base_lin_vel_b=f(T, E, 3), base_ang_vel_b=f(T, E, 3),
           command=f(T, E, 3), q16=f(T, E, 16), qd16=f(T, E, 16), tau=f(T, E, 12), foot_pos_w=f(T, E, 2, 3), foot_quat_w=f(T, E, 2, 4),
           foot_lin_vel_w=f(T, E, 2, 3), terminated=np.zeros((T, E), bool), grf_sub=f(T, E, 2, cfg.decimation, 3))
  D["zone"] = f(T, E, 2, 5)      # share of the foot's normal force in: heel zone, toe zone, medial arch zone, outer edge, heel+toe split
  D["tau_unc"] = f(T, E, 12)     # motor torque before the T-N curve clipping (saturation = |unclipped| > |applied|)
  side_sign = np.array([1.0, -1.0])  # R, L: medial = +y on the right foot, -y on the left foot (foot_link frame)
  NQ = min(E, 4)  # full qpos of the first envs for the comparison videos (render_eval_video.py)
  D["qpos"] = f(T, NQ, env.sim.mj_model.nq)
  snet = env.scene["sole_net"]
  pn_net = [p.split("/")[-1] for p in snet.primary_names]
  order_net = [pn_net.index("R_foot_link"), pn_net.index("L_foot_link")]
  t0 = time.time()
  with torch.inference_mode():
    for i in range(T):
      d = robot.data
      D["base_pos"][i] = d.root_link_pos_w.cpu().numpy()
      D["base_quat"][i] = d.root_link_quat_w.cpu().numpy()
      D["base_lin_vel_b"][i] = d.root_link_lin_vel_b.cpu().numpy()
      D["base_ang_vel_b"][i] = d.root_link_ang_vel_b.cpu().numpy()
      D["command"][i] = env.command_manager.get_command("twist").cpu().numpy()
      D["q16"][i] = d.joint_pos[:, idx16].cpu().numpy()
      D["qd16"][i] = d.joint_vel[:, idx16].cpu().numpy()
      fp = d.body_link_pos_w[:, fbody].cpu().numpy()
      fq = d.body_link_quat_w[:, fbody].cpu().numpy()
      D["foot_pos_w"][i], D["foot_quat_w"][i] = fp, fq
      D["foot_lin_vel_w"][i] = d.body_link_lin_vel_w[:, fbody].cpu().numpy()
      s = sens.data
      F = s.force.cpu().numpy().reshape(E, len(pn), NSLOT, 3)[:, order]
      Pp = s.pos.cpu().numpy().reshape(E, len(pn), NSLOT, 3)[:, order]
      Nn = s.normal.cpu().numpy().reshape(E, len(pn), NSLOT, 3)[:, order]
      dist = s.dist.cpu().numpy().reshape(E, len(pn), NSLOT)[:, order]
      found = s.found.cpu().numpy().reshape(E, len(pn), NSLOT)[:, order] > 0
      # contact force on the FOOT from the ground: normal points primary -> secondary, so the ground pushes along -normal
      fn = SIGN * np.sum(F * Nn, -1)
      fn = np.where(found, fn, 0.0)
      ft = np.linalg.norm(np.where(found[..., None], F + fn[..., None] * Nn, 0.0), axis=-1)
      R = quat_to_mat(fq)  # E,2,3,3 foot -> world
      loc = np.einsum("efji,efkj->efki", R, Pp - fp[:, :, None, :])  # contact points in foot frame
      W = np.clip(fn, 0, None)
      Ws = W.sum(-1)
      with np.errstate(invalid="ignore", divide="ignore"):
        cop = (loc[..., :2] * W[..., None]).sum(-2) / Ws[..., None]
      act = (found & (W > 1.0))
      big = np.where(act[..., None], loc[..., :2], np.nan)
      with np.errstate(invalid="ignore", divide="ignore"):
        px, py = loc[..., 0], loc[..., 1] * side_sign[None, :, None]   # contact points in the foot_link frame, medial +
        wsum = np.where(Ws > 1.0, Ws, np.nan)
        z_heel = (W * (px < -0.060)).sum(-1) / wsum                    # foot x < 20 mm from the heel tip (ankle is 80 mm in)
        z_toe = (W * (px > 0.135)).sum(-1) / wsum                      # foot x > 215 mm: the forefoot rocker
        z_arch = (W * ((px >= -0.020) & (px <= 0.085) & (py > 0.020))).sum(-1) / wsum   # foot x 60-165 mm, medial side
        z_lat = (W * (py < -0.040)).sum(-1) / wsum                     # outer 17 mm of the sole
        f_back, f_front = (W * (px < -0.030)).sum(-1) / wsum, (W * (px > 0.070)).sum(-1) / wsum
        z_split = ((f_back >= 0.25) & (f_front >= 0.25) & (1.0 - f_back - f_front < 0.15)).astype(np.float32)
      D["zone"][i, :, :, 0], D["zone"][i, :, :, 1], D["zone"][i, :, :, 2], D["zone"][i, :, :, 3] = z_heel, z_toe, z_arch, z_lat
      D["zone"][i, :, :, 4] = np.where(Ws > 1.0, z_split, np.nan)
      D["sole"][i, :, :, 0] = Ws
      D["sole"][i, :, :, 1] = np.where(found, ft, 0.0).sum(-1)
      D["sole"][i, :, :, 2:4] = cop
      D["sole"][i, :, :, 4] = act.sum(-1)
      with np.errstate(all="ignore"):
        D["sole"][i, :, :, 5] = np.nanmin(big[..., 0], -1)
        D["sole"][i, :, :, 6] = np.nanmax(big[..., 0], -1)
        D["sole"][i, :, :, 7] = np.nanmin(big[..., 1], -1)
        D["sole"][i, :, :, 8] = np.nanmax(big[..., 1], -1)
        D["sole"][i, :, :, 9] = np.nanmax(np.where(found, -dist, np.nan), -1)
      D["qpos"][i] = env.sim.data.qpos[:NQ].cpu().numpy()
      act_ = policy(obs)
      obs, _, dones, ext = wenv.step(act_)
      for a, cols in acts:
        D["tau"][i][:, cols] = a.applied_effort.cpu().numpy()
        unc = getattr(a, "unclipped_effort", None)
        D["tau_unc"][i][:, cols] = (unc if unc is not None else a.applied_effort).cpu().numpy()
      D["terminated"][i] = env.reset_terminated.cpu().numpy()
      # history index 0 = latest substep -> store oldest first; world frame, sign fixed in the analysis (stand: +body weight)
      D["grf_sub"][i] = snet.data.force_history.cpu().numpy()[:, order_net][:, :, ::-1]
      if i % 500 == 0:
        print(f"[{args.variant}] step {i}/{T}  {time.time() - t0:.0f}s", flush=True)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  D["t"] = (np.arange(T) * DT).astype(np.float32)
  np.savez_compressed(out / f"{TAG}.npz", **D)
  m = env.sim.mj_model
  mujoco.mj_saveModel(m, str(out / f"{TAG}.mjb"))  # terrain + robot exactly as simulated, for rendering the recorded qpos
  lim = robot.data.joint_pos_limits[0].cpu().numpy()
  meta = dict(variant=args.variant, tag=TAG, terrain=args.terrain, joint_range_16=lim[idx16.cpu().numpy()].tolist(), joints_16=JOINTS16, foot_solref=os.environ.get("PYG_FOOT_SOLREF") or "default 0.02", xml=str(xml), program=PROGRAM, seconds_per_cmd=args.seconds_per_cmd, num_envs=E, steps=T,
              seed=args.seed, checkpoint=str(ckpt), robot_mass_kg=float(sum(m.body_mass[b] for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("robot/"))),
              falls=int(D["terminated"].any(0).sum()), wall_s=round(time.time() - t0, 1),
              foot_geoms=int(sum(1 for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("R_foot", "L_foot")) and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_collision"))))
  (out / f"{TAG}.json").write_text(json.dumps(meta, indent=1))
  print(f"[{args.variant}] done falls={meta['falls']}/{E} wall={meta['wall_s']}s mass={meta['robot_mass_kg']:.2f}", flush=True)


if __name__ == "__main__":
  main()
