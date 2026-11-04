#!/usr/bin/env python3
"""Play the colleague's stairs_slopes12_union_dr_99998 policy on the combined v5-foot + D455 model, in the viser viewer, on one course
that has flat ground, slopes and stairs; the D455 depth image is shown next to the 3D view.

What is swapped, nothing else:
  * robot MJCF  -> pygmalion_locomotion/assets/pygmalion_v2/LegOnly_..._footv5_d455_loop.xml   (docs/185)
  * foot contact regex of the colleague's robot.py (`^(L|R)_foot[1-7]_collision$`) -> `\\d+`, because the new foot has 32 collision geoms per foot
    (foot1..foot32); without this only foot1..7 would get condim 3 / friction 0.6 / priority 1
  * robot start height +14 mm (the sole is 14 mm lower than the box foot the policy was trained on)
  * terrain: one 2 m wide course (see CourseCfg) instead of the curriculum grid
  * a CameraSensor on the MJCF camera `d455_depth` (depth only, display only: the policy does not see it, its observation has 44 entries)
The colleague's task, observation, actuators and the checkpoint are used as they are (a git worktree of origin/generalized-union-dr).

Course along +x (default): flat 3 m, slope up 15 deg (to 0.60 m), landing 1 m, 5 stairs down 12 cm, flat 1.5 m, 5 stairs up 12 cm, landing 1 m,
slope down 15 deg, flat 8 m. Commands: vx 0.5 m/s for 5 s, then 1.0 m/s (the colleague's --staged-forward); the episode restarts every
--episode-s seconds. The viewer's own controls (pause, reset, speed, camera) are mjlab's.

  PYTHONPATH=<stock mjlab 20f10e96>/src  mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/viser_policy_footv5_d455.py [--slope-deg 15] [--step-cm 12]
Viser listens on 0.0.0.0:8080 (viser default).
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--branch-root", default=str(REPO / ".claude/worktrees/union_dr_99998/mjlab_policy"))
ap.add_argument("--checkpoint", default="checkpoints/stairs_slopes12_union_dr_99998/model_99998.pt")
ap.add_argument("--model", default="LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_loop",
                help="MJCF in pygmalion_locomotion/assets/pygmalion_v2 (use ..._proxyfix_loop to see the unmodified box foot)")
ap.add_argument("--slope-deg", type=float, default=15.0)
ap.add_argument("--step-cm", type=float, default=12.0)
ap.add_argument("--num-steps", type=int, default=5)
ap.add_argument("--episode-s", type=float, default=15.0, help="restart before the robot reaches the end of the course (the 8 m exit flat is not a loop)")
ap.add_argument("--start-dz", type=float, default=0.014, help="extra start height [m] for the lower sole")
ap.add_argument("--cam-width", type=int, default=424)
ap.add_argument("--cam-height", type=int, default=240)
ap.add_argument("--device", default="cpu")
ap.add_argument("--no-camera", action="store_true")
ap.add_argument("--rollout", type=float, default=0.0, help="no viewer: run this many seconds, save trajectory + depth frames to --out")
ap.add_argument("--out", default="")
args, rest = ap.parse_known_args()
sys.argv = [sys.argv[0]] + rest
BR = Path(args.branch_root)
sys.path.insert(0, str(BR))
sys.path.insert(0, str(BR / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import pygmalion_mjlab  # noqa: E402,F401  (registers the tasks)
from pygmalion_mjlab import robot  # noqa: E402
import play as colleague_play  # noqa: E402  (scripts/play.py of the colleague: staged forward command)
import mjlab.scripts.play as mjlab_play  # noqa: E402
from mjlab.sensor import CameraSensorCfg  # noqa: E402
from mjlab.terrains import TerrainGeneratorCfg  # noqa: E402
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402

TASK = "Pygmalion-Velocity-Generalized-StairsSlopes12-UnionDR"
XML = REPO / "pygmalion_locomotion/assets/pygmalion_v2" / f"{args.model}.xml"
assert XML.exists(), XML
robot.XML = XML
robot.COLLISION = replace(
  robot.COLLISION,
  condim={r"^(L|R)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(L|R)_foot\d+_collision$": 1},
  friction={r"^(L|R)_foot\d+_collision$": (0.6,)},
)


@dataclass(kw_only=True)
class CourseCfg(SubTerrainCfg):
  """flat - slope up - landing - stairs down - flat - stairs up - landing - slope down - flat, one straight lane along +x"""

  slope_deg: float = 15.0
  step_height: float = 0.12
  num_steps: int = 5
  tread_depth: float = 0.38
  approach: float = 3.0
  landing: float = 1.0
  middle_flat: float = 1.5
  base_depth: float = 0.20

  def length(self) -> float:
    top = self.num_steps * self.step_height
    ramp = top / math.tan(math.radians(self.slope_deg))
    return self.approach + ramp + self.landing + self.num_steps * self.tread_depth + self.middle_flat \
      + self.num_steps * self.tread_depth + self.landing + ramp

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    top = self.num_steps * self.step_height
    ramp = top / math.tan(math.radians(self.slope_deg))
    width, cy = self.size[1], 0.5 * self.size[1]
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    flat, stair, slope = (0.30, 0.38, 0.46, 1.0), (0.28, 0.48, 0.72, 1.0), (0.72, 0.42, 0.20, 1.0)

    def box(x0: float, x1: float, h: float, color) -> None:
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.5 * (x1 - x0), 0.5 * width, 0.5 * (h + self.base_depth)),
                        pos=(0.5 * (x0 + x1), cy, 0.5 * (h - self.base_depth)))
      geoms.append(TerrainGeometry(geom=g, color=color))

    def ramp_box(x0: float, x1: float, z0: float, z1: float) -> None:
      th = math.atan2(z1 - z0, x1 - x0)
      g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.5 * (x1 - x0) / math.cos(th), 0.5 * width, 0.5 * self.base_depth),
                        pos=(0.5 * (x0 + x1) - 0.5 * self.base_depth * math.sin(th), cy, 0.5 * (z0 + z1) - 0.5 * self.base_depth * math.cos(th)),
                        quat=(math.cos(-0.5 * th), 0.0, math.sin(-0.5 * th), 0.0))
      geoms.append(TerrainGeometry(geom=g, color=slope))

    x = 0.0
    box(x, x + self.approach, 0.0, flat); x += self.approach                       # flat start
    ramp_box(x, x + ramp, 0.0, top); x += ramp                                      # slope up
    box(x, x + self.landing, top, flat); x += self.landing                          # landing on top
    for i in range(self.num_steps):                                                 # stairs down
      box(x, x + self.tread_depth, top - (i + 1) * self.step_height, stair); x += self.tread_depth
    box(x, x + self.middle_flat, 0.0, flat); x += self.middle_flat                  # flat in the middle
    for i in range(self.num_steps):                                                 # stairs up
      box(x, x + self.tread_depth, (i + 1) * self.step_height, stair); x += self.tread_depth
    box(x, x + self.landing, top, flat); x += self.landing                          # landing on top
    ramp_box(x, x + ramp, top, 0.0); x += ramp                                      # slope down
    box(x, self.size[0], 0.0, flat)                                                 # flat exit
    return TerrainOutput(origin=np.array((1.0, cy, 0.0)), geometries=geoms)


def course_generator() -> TerrainGeneratorCfg:
  c = CourseCfg(size=(1.0, 2.0), slope_deg=args.slope_deg, step_height=args.step_cm / 100.0, num_steps=args.num_steps)
  size = (c.length() + 8.0, 2.0)
  c = replace(c, size=size)
  print(f"[viser-play] course {size[0]:.1f} m x {size[1]:.1f} m: slope {args.slope_deg:g} deg, stairs {args.num_steps} x {args.step_cm:g} cm", flush=True)
  return TerrainGeneratorCfg(seed=42, size=size, num_rows=1, num_cols=1, border_width=0.0, curriculum=False, color_scheme="none",
                             sub_terrains={"flat_slope_stairs": c})


_original = mjlab_play.load_env_cfg


def load_env_cfg(task_id, *a, **k):
  k["play"] = True
  cfg = _original(task_id, *a, **k)
  cfg.scene.terrain.terrain_generator = course_generator()
  cfg.curriculum = {}
  cfg.episode_length_s = args.episode_s
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("course_stagnation", None)
  rb = cfg.events["reset_base"].params["pose_range"]
  cfg.events["reset_base"].params["pose_range"] = {k_: ((0.0, 0.0) if k_ in ("x", "y", "yaw") else v) for k_, v in rb.items()}
  ent = cfg.scene.entities["robot"]
  p = ent.init_state.pos
  ent.init_state = replace(ent.init_state, pos=(p[0], p[1], p[2] + args.start_dz))
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  from dataclasses import fields  # noqa: PLC0415
  staged = colleague_play.StagedForwardVelocityCommandCfg(**{f.name: getattr(twist, f.name) for f in fields(twist)})
  staged.resampling_time_range = (5.0, 5.0)
  staged.heading_command = False
  staged.ranges.heading = None
  for n in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs", "init_velocity_prob"):
    setattr(staged, n, 0.0)
  staged.ranges.lin_vel_y = (-0.1, 0.1)   # viser cannot build a slider from a zero-width range; the staged command writes vy = wz = 0
  staged.ranges.ang_vel_z = (-0.1, 0.1)
  cfg.commands["twist"] = staged
  if not args.no_camera:
    cam = CameraSensorCfg(name="d455", camera_name="robot/d455_depth", width=args.cam_width, height=args.cam_height,
                          data_types=("depth",), enabled_geom_groups=(0,), use_textures=False, use_shadows=False)
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (cam,)
  print(f"[viser-play] robot {XML.name}, start z +{args.start_dz * 1000:g} mm, camera {'off' if args.no_camera else f'{args.cam_width}x{args.cam_height}'}", flush=True)
  return cfg


mjlab_play.load_env_cfg = load_env_cfg

def rollout() -> None:
  """headless run of the same environment: trajectory, falls, and D455 depth frames (for checking and for figures)"""
  import torch  # noqa: PLC0415
  from dataclasses import asdict  # noqa: PLC0415
  from mjlab.envs import ManagerBasedRlEnv  # noqa: PLC0415
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls  # noqa: PLC0415

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  ckpt = Path(args.checkpoint)
  ckpt = ckpt if ckpt.is_absolute() else BR / ckpt
  env_cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  genv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(genv, asdict(agent_cfg), device=args.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  obs = genv.get_observations()
  robot_ent = env.scene["robot"]
  cam = None if args.no_camera else env.scene["d455"]
  dt = env.step_dt
  n = int(args.rollout / dt)
  rec, frames, resets = [], {}, []
  every = int(1.0 / dt)
  for k in range(n):
    with torch.no_grad():
      act = policy(obs)
    obs, _, dones, _ = genv.step(act)
    pos = robot_ent.data.root_link_pos_w[0].cpu().numpy()
    q = robot_ent.data.root_link_quat_w[0].cpu().numpy()
    rec.append([k * dt, *pos, *q])
    if bool(dones[0]):
      resets.append(k * dt)
    if cam is not None and k % every == 0:
      env.sim.sense()   # camera sensors render only on sim.sense() (the viewer calls it for its camera feed)
      frames[f"t{k * dt:05.1f}"] = cam.data.depth[0, ..., 0].cpu().numpy().copy()   # copy: on CPU the tensor shares memory with the sensor buffer
  np.savez_compressed(out / "rollout.npz", traj=np.array(rec), resets=np.array(resets), **frames)
  print(f"[rollout] {n} steps ({args.rollout:g} s), resets at {resets}, final x {rec[-1][1]:.2f} m, z {rec[-1][3]:.3f} m", flush=True)


if __name__ == "__main__":
  if args.rollout > 0:
    rollout()
    sys.exit(0)
  ckpt = Path(args.checkpoint)
  ckpt = ckpt if ckpt.is_absolute() else BR / ckpt
  assert ckpt.exists(), ckpt
  sys.argv += [TASK, "--checkpoint-file", str(ckpt), "--viewer", "viser", "--device", args.device, "--num-envs", "1"]
  lan = os.popen("hostname -I").read().split()
  print(f"[viser-play] viser prints its port below (8080, or the next free one): http://{lan[0] if lan else 'localhost'}:<port>", flush=True)
  mjlab_play.main()
