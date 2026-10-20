"""Pygmalion velocity tasks on stock mjlab: the three training stages of the released policy.

Built on stock `make_velocity_env_cfg()` (mjlab 1.5.0, upstream commit 20f10e96) plus the
overrides used by the training runs. Every value is written out here; nothing reads
environment variables. scripts/verify_config.py compares each resolved config with the
run's own dump in reference/<stage>/env.yaml.

  stage       run                                                    builder
  flat_p1     2026-10-01_17-15-50_legonly_ab_v11a_stand01_s43_p1    pygmalion_flat_env_cfg("p1")
  flat_p2     2026-10-02_23-52-14_legonly_ab_v11a_stand01_s43_p2    pygmalion_flat_env_cfg("p2")
  rough_main  2026-10-11_23-35-20_rough_box_main_v11a                pygmalion_rough_env_cfg()

The flat stages and the rough stage share robot, actuators, observations, rewards and push /
friction / encoder DR. Flat differs in: plane terrain (no height scan, no terrain curriculum),
a performance-gated command curriculum, absolute base-height termination, no NaN / blow-up /
stagnation terminations, no reward clamp, base CoM DR (P1) or per-link inertial DR (P2), and
16,384 envs. The section comments below mark each difference.

Differences from the rough run that are intentional:
  * command range: the run's gated curriculum was set to promote with no dwell
    (min/max dwell 0) and reached its last stage in the first iteration. Here the twist
    command uses that last-stage range directly and the command curriculum term is removed.
    (Only the first command of each env differs: the run drew it from the stage-1 range.)
  * per-term reward clamp: the run clamped each weighted reward term to +-50*dt inside a
    patched RewardManager. Here each term function is wrapped (`clamped`) so stock
    RewardManager gives the same values (see `clamped`).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  pyramid_stairs,
  pyramid_stairs_inv,
  tilted_grid,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from . import robot
from .mdp import commands as pyg_cmd
from .mdp import curriculums as pyg_cur
from .mdp import events as pyg_events
from .mdp import observations as pyg_obs
from .mdp import rewards as pyg_rew
from .mdp import terminations as pyg_term

REWARD_TERM_CLAMP = 50.0  # per step, before the dt scale (rough stage only)

# Gated command curriculum of the flat stages (fork env_cfgs.py, PYG_GATED_CURRICULUM=1,
# PYG_CMD_VY_STAGES=1). "step" is ignored by the gate (kept for the fork's cfg format).
VELOCITY_STAGES = [
  {"step": 0, "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.5, 0.5), "lin_vel_y": (-0.5, 0.5)},
  {"step": 96000, "lin_vel_x": (-1.2, 1.2), "ang_vel_z": (-0.8, 0.8), "lin_vel_y": (-0.6, 0.6)},
  {"step": 192000, "lin_vel_x": (-1.6, 1.6), "ang_vel_z": (-1.0, 1.0), "lin_vel_y": (-0.7, 0.7)},
  {"step": 288000, "lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.0, 1.0), "lin_vel_y": (-0.85, 0.85)},
  {"step": 384000, "lin_vel_x": (-2.0, 2.5), "ang_vel_z": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0)},
]
# P1: real gate (dwell 800..3000 iterations). P2: dwell 0 - the gate state is not saved in
# checkpoints, so the resumed run promotes on every call and is at the last stage from its
# first iteration.
GATE = {"window_iters": 100, "err_ratio": 1.1, "fell_max": 0.005, "min_episodes": 64, "steps_per_iter": 24}
GATE_DWELL = {"p1": (800, 3000), "p2": (0, 0)}
# DR ramp window in iterations (x24 env steps). P1 never reaches it (DR off).
FLAT_DR_WINDOW = {"p1": (100_000_000, 100_000_001), "p2": (15_999, 25_999)}

# Box-only rough terrain. MuJoCo-Warp heightfield contact made the walking foot slip even on
# a flat heightfield tile, so no heightfield sub-terrain is used.
BOX_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": flat(proportion=0.2),
    "pyramid_stairs": pyramid_stairs(proportion=0.2, step_height_range=(0.0, 0.1)),
    "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.2, step_height_range=(0.0, 0.1)),
    "box_random_grid": box_random_grid(proportion=0.2, grid_height_range=(0.0, 0.12)),
    "tilted_grid": tilted_grid(proportion=0.2, tilt_range_deg=15.0, height_range=0.2),
  },
  add_lights=True,
)

# Command range (= last stage of the run's command curriculum).
COMMAND_RANGE = {"lin_vel_x": (-2.0, 2.5), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)}

PUSH_RANGE = {
  "x": (-0.7, 0.7),
  "y": (-0.7, 0.7),
  "z": (-0.4, 0.4),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

# Link inertial DR (mass e^(2*alpha), CoM shift t1..t3, shape d), from the CAD mass
# uncertainty propagation (mass_dr_legonly_fastener50_prototype-tempmass.json).
INERTIAL_DR = {
  "pelvis": (("base_link",), dict(alpha_range=(-0.0256, 0.0699), t1_range=(-0.02, 0.02), t2_range=(-0.02, 0.02), t3_range=(-0.02, 0.02), d_range=(-0.0154, 0.0154))),
  "hip_pitch_link": (("L_hip_pitch_link", "R_hip_pitch_link"), dict(alpha_range=(-0.0256, 0.0364), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "hip_roll_link": (("L_hip_roll_link", "R_hip_roll_link"), dict(alpha_range=(-0.0256, 0.0378), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "thigh": (("L_thigh_link", "R_thigh_link"), dict(alpha_range=(-0.0256, 0.0455), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.0066), d_range=(-0.0146, 0.0146))),
  "shin": (("L_shin_link", "R_shin_link"), dict(alpha_range=(-0.0256, 0.0508), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "ankle_pitch_link": (("L_ankle_pitch_link", "R_ankle_pitch_link"), dict(alpha_range=(-0.059, 0.0554), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "foot": (("L_foot_link", "R_foot_link"), dict(alpha_range=(-0.0256, 0.0647), t1_range=(-0.0053, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
}


def clamped(func, weight: float, limit: float = REWARD_TERM_CLAMP):
  """Wrap a reward term so stock RewardManager reproduces the run's per-term clamp.

  The run computed v = nan_to_num(f * w * dt) and then clamped v to +-limit*dt.
  For w != 0 that equals nan_to_num(f) clamped to +-limit/|w|, then multiplied by w*dt.
  Works for function terms and for class terms (instantiated by the manager).
  """
  bound = limit / abs(weight)

  def _clamp(v: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-bound, bound)

  if inspect.isclass(func):

    class ClampedTerm:
      wrapped = func

      def __init__(self, cfg, env):
        self.inner = func(cfg=cfg, env=env)

      def __call__(self, env, **params):
        return _clamp(self.inner(env, **params))

      def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
          return self.inner.reset(*args, **kwargs)
        return None

      def __getattr__(self, name):
        return getattr(self.inner, name)

    ClampedTerm.__name__ = ClampedTerm.__qualname__ = f"clamped_{func.__name__}"
    return ClampedTerm

  def term(env, **params):
    return _clamp(func(env, **params))

  term.__name__ = term.__qualname__ = f"clamped_{getattr(func, '__name__', 'term')}"
  term.wrapped = func  # type: ignore[attr-defined]
  return term


def pygmalion_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough stage (box terrain). The released policy's last 12,000 iterations."""
  return _pygmalion_env_cfg("rough", play)


def pygmalion_flat_env_cfg(stage: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat stage "p1" (from scratch, 16,000 iterations) or "p2" (resumed from P1, to 29,998)."""
  assert stage in ("p1", "p2"), stage
  return _pygmalion_env_cfg(f"flat_{stage}", play)


def _pygmalion_env_cfg(variant: str, play: bool) -> ManagerBasedRlEnvCfg:
  rough = variant == "rough"
  stage = variant.removeprefix("flat_")
  cfg = make_velocity_env_cfg()

  # --- simulation / scene ---------------------------------------------------------------
  cfg.scene.entities = {"robot": robot.robot_cfg()}
  if rough:
    cfg.sim.mujoco.ccd_iterations = 100
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 150
    cfg.scene.terrain.terrain_generator = replace(BOX_TERRAINS_CFG)
    cfg.scene.terrain.terrain_generator.curriculum = True
  else:  # flat: plane, no terrain scan
    cfg.scene.num_envs = 16384
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None
    cfg.sim.njmax = 300
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors or () if s.name != "terrain_scan")
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.curriculum.pop("terrain_levels", None)

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in robot.FOOT_SITES)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(L_foot_link|R_foot_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, self_collision)

  # --- action ----------------------------------------------------------------------------
  act = cfg.actions["joint_pos"]
  assert isinstance(act, JointPositionActionCfg)
  act.scale = {n: robot.ACTION_SCALE for a in robot.ACTUATORS for n in a.target_names_expr}
  act.clip = dict(robot.ACTION_CLIP)

  # --- observations ----------------------------------------------------------------------
  # Actor: deployable proprioception only (stock actor also has a height scan; removed).
  cfg.observations["actor"].terms.pop("height_scan", None)
  for grp in ("actor", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[grp].terms[term].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.OBS_JOINTS)
      }
  del cfg.observations["actor"].terms["base_lin_vel"]
  critic = cfg.observations["critic"].terms
  cfg.observations["actor"].terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=critic["base_ang_vel"].noise,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity_from_sensor,
      params={"sensor_name": "robot/imu_upvector"},
      noise=critic["projected_gravity"].noise,
    ),
    "motor_pos_history": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS)},
      noise=critic["joint_pos"].noise,
      history_length=2,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
  }
  cfg.viewer.body_name = "base_link"

  # --- command ---------------------------------------------------------------------------
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.viz.z_offset = 1.0
  twist.rel_standing_envs = 0.1
  twist.rel_heading_envs = 0.3
  if rough:
    twist.ranges.lin_vel_x = COMMAND_RANGE["lin_vel_x"]
    twist.ranges.lin_vel_y = COMMAND_RANGE["lin_vel_y"]
    twist.ranges.ang_vel_z = COMMAND_RANGE["ang_vel_z"]
    cfg.curriculum.pop("command_vel", None)
  else:  # flat: start at stage 0, the gate widens the range
    s0 = VELOCITY_STAGES[0]
    twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z = (
      s0["lin_vel_x"], s0["lin_vel_y"], s0["ang_vel_z"]
    )
    cfg.commands["twist"] = pyg_cmd.MetricVelocityCommandCfg(
      **{f.name: getattr(twist, f.name) for f in fields(twist)}
    )
    min_dwell, max_dwell = GATE_DWELL[stage]
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=pyg_cur.commands_vel_gated,
      params={
        "command_name": "twist",
        "velocity_stages": [dict(s) for s in VELOCITY_STAGES],
        "min_dwell_iters": min_dwell,
        "max_dwell_iters": max_dwell,
        **GATE,
      },
    )

  # --- rewards ---------------------------------------------------------------------------
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=robot.POSE_JOINTS)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3, r".*hip_roll.*": 0.4, r".*hip_yaw.*": 0.15,
    r".*knee.*": 1.2, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.2, r".*hip_yaw.*": 0.2,
    r".*knee.*": 1.5, r".*ankle_pitch.*": 0.35, r".*ankle_roll.*": 0.15,
  }
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = robot.FOOT_SITES
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["track_lin_vel_progress"] = RewardTermCfg(
    func=pyg_rew.track_lin_vel_progress, weight=1.0, params={"command_name": "twist"}
  )
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.75)
  cfg.rewards["stand_still_penalty"] = RewardTermCfg(
    func=pyg_rew.stand_still_penalty, weight=-1.0, params={"command_name": "twist"}
  )
  cfg.rewards["knee_overspeed"] = RewardTermCfg(
    func=pyg_rew.knee_overspeed, weight=-0.5, params={"speed_limit": 19.9}
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )
  cfg.rewards["torque_limit"] = RewardTermCfg(  # weight 0: logged only, as in the run
    func=pyg_rew.torque_limit,
    weight=-0.0,
    params={"asset_cfg": SceneEntityCfg("robot"), "soft_factor": 0.7},
  )
  cfg.rewards["contact_force_cap"] = RewardTermCfg(
    func=pyg_rew.contact_force_cap,
    weight=-0.01,
    params={"sensor_name": "feet_ground_contact", "threshold": 420.0, "clip": 560.0},
  )
  cfg.rewards["foot_impact_velocity"] = RewardTermCfg(
    func=pyg_rew.foot_impact_velocity,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=robot.FOOT_SITES),
      "height_threshold": 0.10,
      "power": 2.0,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["thermal_effort"] = RewardTermCfg(
    func=pyg_rew.thermal_effort,
    weight=-0.02,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "rated": {
        ".*_hip_pitch_joint": 40.0,
        ".*_hip_roll_joint": 40.0,
        ".*_hip_yaw_joint": 20.0,
        ".*_knee_joint": 40.0,
        ".*_ankle_pitch_joint": 20.0,
        ".*_ankle_roll_joint": 5.0,
        ".*_waist_yaw_joint": 40.0,
        ".*_shoulder_(pitch|roll)_joint": 20.0,
        ".*_crank_[AB]_joint": 20.0,
      },
    },
  )
  cfg.rewards["stance_knee_extension"] = RewardTermCfg(
    func=pyg_rew.stance_knee_extension,
    weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
      "target_deg": 25.0,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  )
  if rough:
    for term in cfg.rewards.values():  # per-term clamp (see `clamped`)
      if term.weight != 0.0:
        term.func = clamped(term.func, term.weight)

  # --- events / domain randomization -----------------------------------------------------
  cfg.events["push_robot"].params["velocity_range"] = dict(PUSH_RANGE)
  for name in ("foot_friction", "encoder_bias", "base_com"):
    cfg.events[name].mode = "reset"
  cfg.curriculum["dr_levels"] = CurriculumTermCfg(
    func=pyg_cur.dr_levels,
    params={
      # linear 0 -> 1 between iterations start and end (24 env steps per iteration);
      # rough: 2,000 -> 8,000, flat P2: 15,999 -> 25,999, flat P1: never
      "start_step": (2000 if rough else FLAT_DR_WINDOW[stage][0]) * 24,
      "end_step": (8000 if rough else FLAT_DR_WINDOW[stage][1]) * 24,
      "push_max": dict(PUSH_RANGE),
      "friction_max": (0.3, 1.2),
      "encoder_max": (-0.015, 0.015),
      "com_max": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
    },
  )
  if variant != "flat_p1":  # P1 keeps stock base CoM DR; P2 and rough use per-link inertial DR
    cfg.events.pop("base_com", None)  # replaced by the per-link inertial DR below
    inertial_max = {}
    for link, (bodies, ranges) in INERTIAL_DR.items():
      name = f"inertial_dr_{link}"
      inertial_max[name] = dict(ranges)
      cfg.events[name] = EventTermCfg(
        mode="reset",
        func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies), **ranges},
      )
    cfg.curriculum["dr_levels"].params["inertial_max"] = inertial_max

  # Critic-only view of the DR draw.
  critic = cfg.observations["critic"].terms
  critic["dr_friction"] = ObservationTermCfg(
    func=pyg_obs.dr_foot_friction,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=robot.FOOT_GEOMS)},
  )
  critic["dr_mass"] = ObservationTermCfg(
    func=pyg_obs.dr_mass_offset, params={"asset_cfg": SceneEntityCfg("robot")}
  )
  critic["dr_com"] = ObservationTermCfg(
    func=pyg_obs.dr_com_offset,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
  )
  critic["dr_push"] = ObservationTermCfg(func=pyg_obs.dr_push_state)
  critic["dr_factor"] = ObservationTermCfg(func=pyg_obs.dr_factor)
  cfg.events["push_robot"].func = pyg_events.push_by_setting_velocity_logged
  cfg.events["pyg_push_tick"] = EventTermCfg(
    mode="step", func=pyg_events.push_age_tracker, params={"action": "tick"}
  )
  cfg.events["pyg_push_clear"] = EventTermCfg(
    mode="reset", func=pyg_events.push_age_tracker, params={"action": "clear"}
  )

  # --- terminations ----------------------------------------------------------------------
  if not rough:  # flat: absolute base height, stock terminations only
    cfg.terminations["low_base"] = TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)
    if play:
      cfg.episode_length_s = int(1e9)
      cfg.observations["actor"].enable_corruption = False
      cfg.events.pop("push_robot", None)
      cfg.curriculum = {}
      s4, r = VELOCITY_STAGES[-1], cfg.commands["twist"].ranges
      r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = s4["lin_vel_x"], s4["lin_vel_y"], s4["ang_vel_z"]
    return cfg
  cfg.terminations["low_base"] = TerminationTermCfg(
    func=pyg_term.base_height_above_terrain_below,
    params={"minimum_height": 0.7, "sensor_name": "terrain_scan"},
  )
  cfg.terminations["nan_detection"] = TerminationTermCfg(func=envs_mdp.nan_detection)
  cfg.terminations["physics_blowup"] = TerminationTermCfg(func=pyg_term.physics_blowup)
  for group in ("actor", "critic"):
    cfg.observations[group].nan_policy = "warn"
    cfg.observations[group].nan_check_per_term = True
  cfg.terminations["stagnation"] = TerminationTermCfg(
    func=pyg_term.stagnation,
    params={
      "command_name": "twist",
      "window_s": 3.0,
      "min_cmd_speed": 0.2,
      "min_progress_m": 0.10,
      "min_progress_frac": 0.2,
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    tg = cfg.scene.terrain.terrain_generator
    tg.curriculum = False
    tg.num_cols = 5
    tg.num_rows = 5
    tg.border_width = 10.0
  return cfg
