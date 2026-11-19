"""One-stage, from-scratch HUPHY velocity task with general rewards and moderate DR."""

from __future__ import annotations

import copy
import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from . import obstacle_course, robot, stairs_slopes, terrain10, unitree_rough
from .env_cfg import pygmalion_rough_env_cfg
from .mdp import events as pyg_events
from .mdp import observations as pyg_obs
from .mdp import rewards as pyg_rew
from .mdp import terminations as pyg_term
from .mdp.commands import VxMixtureVelocityCommandCfg


def vx_wz_commands(env, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, (0, 2)]


def _general_rewards() -> dict[str, RewardTermCfg]:
  pose_params = {
    "asset_cfg": SceneEntityCfg("robot", joint_names=robot.POSE_JOINTS),
    "command_name": "twist",
    "std_standing": {".*": 0.05},
    "std_walking": {
      ".*_hip_pitch.*": 0.3, ".*_hip_roll.*": 0.4, ".*_hip_yaw.*": 0.15,
      ".*_knee.*": 1.2, ".*_ankle_pitch.*": 0.25, ".*_ankle_roll.*": 0.1,
    },
    "std_running": {
      ".*_hip_pitch.*": 0.5, ".*_hip_roll.*": 0.2, ".*_hip_yaw.*": 0.2,
      ".*_knee.*": 1.5, ".*_ankle_pitch.*": 0.35, ".*_ankle_roll.*": 0.15,
    },
    "walking_threshold": 0.1,
    "running_threshold": 1.5,
  }
  return {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity, weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity, weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=pyg_rew.body_orientation_l2, weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "pose": RewardTermCfg(func=mdp.variable_posture, weight=0.5, params=pose_params),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty, weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty, weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time, weight=0.5,
      params={
        "sensor_name": "feet_ground_contact", "threshold_min": 0.05,
        "threshold_max": 0.5, "command_name": "twist", "command_threshold": 0.1,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance, weight=-1.0,
      params={
        "target_height": 0.1, "height_sensor_name": "foot_height_scan",
        "command_name": "twist", "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=robot.FOOT_SITES),
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip, weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact", "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=robot.FOOT_SITES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing, weight=-1.0e-3,
      params={
        "sensor_name": "feet_ground_contact", "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=pyg_rew.stand_still, weight=-1.0,
      params={
        "command_name": "twist", "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost, weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "actuator_clipping_fraction": RewardTermCfg(
      func=pyg_rew.actuator_clipping_fraction, weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "negative_mechanical_power": RewardTermCfg(
      func=pyg_rew.negative_mechanical_power, weight=-0.1,
      params={"free_fraction": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    ),
  }


def _general_events() -> dict[str, EventTermCfg]:
  density_alpha = (0.5 * math.log(0.9), 0.5 * math.log(1.1))
  return {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform, mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset, mode="reset",
      params={
        "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS),
      },
    ),
    "inertial_density": EventTermCfg(
      func=dr.pseudo_inertia, mode="reset",
      params={
        "alpha_range": density_alpha,
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      },
    ),
    "base_com": EventTermCfg(
      func=dr.body_com_offset, mode="reset",
      params={
        "operation": "add",
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "pd_gains": EventTermCfg(
      func=dr.pd_gains, mode="reset",
      params={
        "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1), "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "motor_strength": EventTermCfg(
      func=pyg_events.randomize_motor_strength, mode="reset",
      params={
        "strength_range": (0.8, 1.2),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "joint_friction": EventTermCfg(
      func=dr.joint_friction, mode="reset",
      params={
        "ranges": (0.8, 1.2), "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS),
      },
    ),
    "joint_damping": EventTermCfg(
      func=dr.joint_damping, mode="reset",
      params={
        "ranges": (0.8, 1.2), "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS),
      },
    ),
    "foot_friction": EventTermCfg(
      func=dr.geom_friction, mode="reset",
      params={
        "ranges": (0.3, 1.2), "operation": "abs", "shared_random": True,
        "asset_cfg": SceneEntityCfg("robot", geom_names=robot.FOOT_GEOMS),
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias, mode="reset",
      params={"bias_range": (-0.01, 0.01), "asset_cfg": SceneEntityCfg("robot")},
    ),
    "torso_push": EventTermCfg(
      func=pyg_events.apply_horizontal_body_push, mode="step",
      params={
        "magnitude_range": (50.0, 100.0), "duration_s": 0.1,
        "cooldown_s": (5.0, 10.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
  }


_WALK_ALPHA = (0.5 * math.log(0.9), 0.5 * math.log(1.1))
_UNION_INERTIAL_DR = {
  "pelvis": (
    ("base_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0699),
      t1_range=(-0.03, 0.03), t2_range=(-0.03, 0.03), t3_range=(-0.03, 0.03),
      d_range=(-0.0154, 0.0154),
    ),
  ),
  "hip_pitch": (
    (".*_hip_pitch_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005),
      d_range=(-0.02, 0.02),
    ),
  ),
  "hip_roll": (
    (".*_hip_roll_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005),
      d_range=(-0.02, 0.02),
    ),
  ),
  "thigh": (
    (".*_thigh_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.0066),
      d_range=(-0.0146, 0.0146),
    ),
  ),
  "shin": (
    (".*_shin_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0508),
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005),
      d_range=(-0.02, 0.02),
    ),
  ),
  "ankle_pitch": (
    (".*_ankle_pitch_link",),
    dict(
      alpha_range=(-0.059, 0.0554),
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005),
      d_range=(-0.02, 0.02),
    ),
  ),
  "foot": (
    (".*_foot_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0647),
      t1_range=(-0.0053, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005),
      d_range=(-0.02, 0.02),
    ),
  ),
  "crank_rod": (
    (".*_crank_[AB]", ".*_rod_[AB]"),
    dict(alpha_range=_WALK_ALPHA),
  ),
}


def _apply_union_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply the envelope of the walking and Huphy-squat DR configurations."""
  events = cfg.events
  events.pop("inertial_density")
  events.pop("base_com")

  for name, (body_names, ranges) in _UNION_INERTIAL_DR.items():
    events[f"union_inertia_{name}"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=body_names),
        **ranges,
      },
    )

  events["payload"] = EventTermCfg(
    func=pyg_events.add_payload_mass,
    mode="reset",
    params={
      "mass_range": (-1.0, 8.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  events["pd_gains"].params.update(
    {"kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2)}
  )
  events["motor_armature"] = EventTermCfg(
    func=dr.joint_armature,
    mode="reset",
    params={
      "ranges": (0.8, 1.2), "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", joint_names=robot.MOTOR_JOINTS),
    },
  )
  passive_joints = (".*_rod_[AB]_u[12]", ".*_ankle_(pitch|roll)_joint")
  events["passive_friction"] = EventTermCfg(
    func=dr.joint_friction,
    mode="reset",
    params={
      "ranges": (0.0, 0.5), "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=passive_joints),
    },
  )
  events["passive_damping"] = EventTermCfg(
    func=dr.joint_damping,
    mode="reset",
    params={
      "ranges": (0.0, 0.05), "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=passive_joints),
    },
  )
  events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
  imu_angle = math.radians(2.0)
  events["imu_mount"] = EventTermCfg(
    func=dr.site_quat,
    mode="reset",
    params={
      "roll_range": (-imu_angle, imu_angle),
      "pitch_range": (-imu_angle, imu_angle),
      "yaw_range": (-imu_angle, imu_angle),
      "asset_cfg": SceneEntityCfg("robot", site_names=("imu_in_base",)),
    },
  )
  events["reset_hip_spread"] = EventTermCfg(
    func=pyg_events.add_hip_roll_reset_spread,
    mode="reset",
    params={
      "spread_range": (-0.10, 0.06), "spread_prob": 0.5,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("R_hip_roll_joint", "L_hip_roll_joint"),
        preserve_order=True,
      ),
    },
  )
  events["torso_push"].params.update(
    {
      "magnitude_range": (0.0, 100.0),
      "duration_s": (0.1, 0.2),
      "cooldown_s": (2.0, 10.0),
    }
  )

  for term_name in ("base_ang_vel", "projected_gravity", "motor_pos_history"):
    term = cfg.observations["actor"].terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = 2
    term.delay_per_env = True
    term.delay_update_period = 1_000_000
    term.delay_per_env_phase = False

  for actuator in cfg.scene.entities["robot"].articulation.actuators:
    actuator.delay_min_lag = 0
    actuator.delay_max_lag = 4
    actuator.delay_update_period = 1_000_000
    actuator.delay_per_env_phase = False


def pygmalion_generalized_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pygmalion_rough_env_cfg(play=False)
  cfg.scene.entities["robot"] = robot.generalized_robot_cfg()
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=89500,
      curriculum=True,
      size=(10.0, 10.0),
      num_rows=1,
      num_cols=10,
      border_width=0.0,
      color_scheme="none",
      sub_terrains={
        name: terrain10.StripTerrainCfg(
          size=(10.0, 10.0), slope_deg=slope, ridge_height_range=ridges
        )
        for name, slope, ridges in terrain10.LEVEL_SPECS
      },
    ),
    max_init_terrain_level=0,
  )
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1024)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch or 0, 128)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.resampling_time_range = (3.0, 8.0)
  twist.rel_standing_envs = 0.05
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.lin_vel_x = (-1.0, 2.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.ranges.heading = None
  for group in ("actor", "critic"):
    cfg.observations[group].terms["command"] = ObservationTermCfg(
      func=vx_wz_commands, params={"command_name": "twist"}
    )
  motor_history = cfg.observations["actor"].terms["motor_pos_history"]
  motor_history.delay_min_lag = 0
  motor_history.delay_max_lag = 1
  motor_history.delay_per_env = True

  cfg.rewards = _general_rewards()
  cfg.events = _general_events()
  cfg.terminations.pop("stagnation", None)
  cfg.curriculum = {
    "terrain_5k_unlock": CurriculumTermCfg(
      func=terrain10.fixed_step_curriculum,
      params={"iterations_per_level": 5000, "steps_per_iteration": 24, "num_levels": 10},
    )
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events = {
      "reset_base": cfg.events["reset_base"],
      "reset_robot_joints": cfg.events["reset_robot_joints"],
      "randomize_terrain": EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={}),
    }
    cfg.curriculum = {}
    cfg.terminations.pop("out_of_terrain_bounds", None)
  return cfg


def pygmalion_generalized_unitree_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = pygmalion_generalized_env_cfg(play=False)
  terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)
  terrain_generator.seed = 89500
  terrain_generator.curriculum = True
  terrain_generator.num_rows = 10
  terrain_generator.num_cols = len(terrain_generator.sub_terrains)
  terrain_generator.color_scheme = "height"
  for sub_terrain in terrain_generator.sub_terrains.values():
    sub_terrain.proportion = 1.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=terrain_generator,
    max_init_terrain_level=1,
  )
  cfg.curriculum = {
    "unitree_rough_uniform": CurriculumTermCfg(
      func=unitree_rough.staged_uniform_curriculum,
      params={
        "start_iteration": 19_500,
        "iterations_per_level": 2_000,
        "steps_per_iteration": 24,
        "num_levels": 10,
        "start_max_level": 0,
      },
    )
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events = {
      "reset_base": cfg.events["reset_base"],
      "reset_robot_joints": cfg.events["reset_robot_joints"],
      "randomize_terrain": EventTermCfg(
        func=envs_mdp.randomize_terrain, mode="reset", params={}
      ),
    }
    cfg.curriculum = {}
    cfg.terminations.pop("out_of_terrain_bounds", None)
  return cfg


def pygmalion_generalized_terrain10_2k_resume_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = pygmalion_generalized_env_cfg(play=play)
  if not play:
    cfg.curriculum["terrain_5k_unlock"] = CurriculumTermCfg(
      func=terrain10.fixed_step_curriculum,
      params={
        "iterations_per_level": 2_000,
        "steps_per_iteration": 24,
        "num_levels": 10,
        "start_iteration": 19_700,
        "start_max_level": 3,
      },
    )
  return cfg


def pygmalion_generalized_union_dr_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Terrain10 resume task with the envelope of walking and squat DR."""
  cfg = pygmalion_generalized_terrain10_2k_resume_env_cfg(play=play)
  if not play:
    _apply_union_dr(cfg)
  return cfg


def pygmalion_flat_union_dr_from_scratch_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat-ground Union-DR task with full planar velocity commands."""
  cfg = pygmalion_generalized_union_dr_env_cfg(play=play)

  # Keep the foot-local height sensor used by the clearance reward, but remove
  # terrain-map observations and curriculum state that only apply to generated
  # rough terrain.
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="plane",
    terrain_generator=None,
  )
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.curriculum = {}
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["low_base"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.events.pop("randomize_terrain", None)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-1.5, 1.5)
  twist.ranges.lin_vel_y = (-1.0, 1.0)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.ranges.heading = None
  twist.heading_command = False
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  for group in ("actor", "critic"):
    cfg.observations[group].terms["command"] = ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    )

  cfg.rewards["foot_clearance"].params["target_height"] = 0.15

  # Match the final narrow-course run's widened deployment envelope.
  if not play:
    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.7)
    cfg.events["payload"].params["mass_range"] = (-1.0, 15.0)
  return cfg


def pygmalion_flat_union_dr_knee_resume_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat Union-DR fine-tune with symmetric swing-knee shaping."""
  cfg = pygmalion_flat_union_dr_from_scratch_env_cfg(play=play)

  cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  cfg.rewards["self_collisions"].weight = -10.0

  ordered_knees = SceneEntityCfg(
    "robot",
    # feet_ground_contact resolves primaries in MuJoCo body order: right, left.
    joint_names=("R_knee_joint", "L_knee_joint"),
    preserve_order=True,
  )
  cfg.rewards["swing_knee_flexion"] = RewardTermCfg(
    func=pyg_rew.swing_knee_flexion_cost,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "target_flexion": 0.55,
      "max_air_time": 0.5,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": ordered_knees,
    },
  )
  cfg.rewards["landing_peak_knee_flexion"] = RewardTermCfg(
    func=pyg_rew.landing_peak_knee_flexion_reward,
    weight=0.3,
    params={
      "sensor_name": "feet_ground_contact",
      "target_flexion": 0.7,
      "sigma": 0.2,
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": ordered_knees,
    },
  )
  return cfg


def pygmalion_flat_union_dr_knee_lr2x_resume_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Continue the knee-shaped flat policy with separate 2x left/right terms."""
  cfg = pygmalion_flat_union_dr_knee_resume_env_cfg(play=play)
  cfg.rewards.pop("swing_knee_flexion")
  cfg.rewards.pop("landing_peak_knee_flexion")

  ordered_knees = SceneEntityCfg(
    "robot",
    # feet_ground_contact primaries are in MuJoCo body order: right, left.
    joint_names=("R_knee_joint", "L_knee_joint"),
    preserve_order=True,
  )
  common_swing = {
    "sensor_name": "feet_ground_contact",
    "target_flexion": 0.55,
    "max_air_time": 0.5,
    "command_name": "twist",
    "command_threshold": 0.1,
    "asset_cfg": ordered_knees,
  }
  common_landing = {
    "sensor_name": "feet_ground_contact",
    "target_flexion": 0.7,
    "sigma": 0.2,
    "command_name": "twist",
    "command_threshold": 0.1,
    "asset_cfg": ordered_knees,
  }
  for side, side_index in (("right", 0), ("left", 1)):
    cfg.rewards[f"swing_knee_flexion_{side}"] = RewardTermCfg(
      func=pyg_rew.swing_knee_flexion_cost,
      weight=-0.2,
      params={
        **common_swing,
        "side_index": side_index,
        "metric_suffix": f"_{side}",
      },
    )
    cfg.rewards[f"landing_peak_knee_flexion_{side}"] = RewardTermCfg(
      func=pyg_rew.landing_peak_knee_flexion_reward,
      weight=0.6,
      params={
        **common_landing,
        "side_index": side_index,
        "metric_suffix": f"_{side}",
      },
    )
  return cfg


def pygmalion_flat_union_dr_knee_lr2x_vx_mix_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """LR2x task with 60% general and 40% high-speed straight vx commands."""
  cfg = pygmalion_flat_union_dr_knee_lr2x_resume_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = VxMixtureVelocityCommandCfg(
    resampling_time_range=twist.resampling_time_range,
    debug_vis=twist.debug_vis,
    entity_name=twist.entity_name,
    heading_command=False,
    heading_control_stiffness=twist.heading_control_stiffness,
    rel_standing_envs=twist.rel_standing_envs,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    init_velocity_prob=twist.init_velocity_prob,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.5, 1.5),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=None,
    ),
    viz=twist.viz,
    high_speed_fraction=0.4,
    high_speed_vx_range=(1.5, 2.0),
    high_speed_straight=True,
  )
  return cfg


def pygmalion_generalized_stairs_slopes_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Union-DR task with a progressive 12-stage stairs/slopes curriculum."""
  cfg = pygmalion_generalized_union_dr_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=stairs_slopes.curriculum_terrain_cfg(),
    max_init_terrain_level=0,
  )

  # This is a fixed world-frame course, not a free-heading velocity task.  Keep
  # commands forward along world +X and start close enough to the target heading
  # that the 45 degree course boundary is meaningful from the first step.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.1, 2.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_world_envs = 0.0
  twist.rel_forward_envs = 1.0
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (
    -math.radians(15.0), math.radians(15.0)
  )

  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["course_forward_progress"] = RewardTermCfg(
    func=pyg_rew.course_forward_progress,
    weight=2.0,
    params={
      "command_name": "twist",
      "target_heading": 0.0,
      "heading_gain": 2.0,
      "reverse_scale": 0.5,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["course_heading_error"] = TerminationTermCfg(
    func=pyg_term.course_heading_error,
    params={
      "max_error": math.radians(45.0),
      "target_heading": 0.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["course_stagnation"] = TerminationTermCfg(
    func=pyg_term.course_stagnation,
    params={
      "command_name": "twist",
      "window_s": 1.0,
      "failure_hold_s": 0.5,
      "min_cmd_speed": 0.2,
      "min_progress_m": 0.10,
      "target_heading": 0.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  if not play:
    cfg.curriculum = {
      "terrain_2k_unlock": CurriculumTermCfg(
        func=stairs_slopes.progressive_uniform_curriculum,
        params={
          "start_step": 1_200_048,
          "iterations_per_level": 2_000,
          "steps_per_iteration": 24,
          "num_levels": 12,
        },
      )
    }
  return cfg


def pygmalion_narrow_bidirectional_course_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Union-DR fine-tune on the 1 m-wide stairs/landing/down-slope course."""
  approach_length = 10.0
  exit_length = 10.0
  stair_run = 4 * 0.30
  landing_length = 0.70
  ramp_run = 4 * 0.15 / math.tan(math.radians(20.0))
  terrain_origin_x = 0.5 * approach_length
  forward_obstacle_x = approach_length - terrain_origin_x
  reverse_obstacle_x = (
    approach_length + stair_run + landing_length + ramp_run - terrain_origin_x
  )
  course_length = approach_length + stair_run + landing_length + ramp_run + exit_length
  cfg = pygmalion_generalized_union_dr_env_cfg(play=play)
  cfg.episode_length_s = 30.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=obstacle_course.four_step_downslope_terrain_cfg(
      landing_length=landing_length,
      tread_depth=0.30,
      course_width=1.0,
      approach_length=approach_length,
      exit_length=exit_length,
    ),
    max_init_terrain_level=0,
  )
  cfg.curriculum = {}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (0.1, 1.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  # The sampled vx is already positive; avoid the stock forward-only clamp to 0.3 m/s.
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  for group in ("actor", "critic"):
    cfg.observations[group].terms["command"] = ObservationTermCfg(
      func=pyg_obs.course_velocity_command,
      params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot")},
    )

  cfg.events["reset_base"] = EventTermCfg(
    func=pyg_events.reset_bidirectional_course,
    mode="reset",
    params={
      "forward_obstacle_x": forward_obstacle_x,
      "reverse_obstacle_x": reverse_obstacle_x,
      "obstacle_distance": (0.30, 1.00),
      # Nominal foremost capsule7plus5 collision relative to base_link.
      "foot_front_offset": 0.0853,
      "y_jitter": (-0.15, 0.15),
      "z_jitter": (0.01, 0.05),
      "safe_lateral_offset": 0.25,
      "max_yaw_offset": math.radians(30.0),
      # Oversample the harder stairs-up direction while retaining slope-up
      # rollouts to reduce catastrophic forgetting.
      "forward_probability": 0.80,
      "velocity_range": {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "yaw": (-0.2, 0.2),
      },
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["course_forward_progress"] = RewardTermCfg(
    func=pyg_rew.bidirectional_course_forward_progress,
    weight=2.0,
    params={
      "command_name": "twist",
      "heading_gain": 2.0,
      "reverse_scale": 0.5,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["course_lateral_velocity_l2"] = RewardTermCfg(
    func=pyg_rew.course_lateral_velocity_l2,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["course_overspeed_l2"] = RewardTermCfg(
    func=pyg_rew.course_overspeed_l2,
    weight=-1.0,
    params={"max_speed": 1.0, "asset_cfg": SceneEntityCfg("robot")},
  )

  cfg.terminations["course_heading_error"] = TerminationTermCfg(
    func=pyg_term.course_heading_error,
    params={
      "max_error": math.radians(45.0),
      "bidirectional": True,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["course_stagnation"] = TerminationTermCfg(
    func=pyg_term.course_stagnation,
    params={
      "command_name": "twist",
      "window_s": 1.0,
      "failure_hold_s": 0.5,
      "min_cmd_speed": 0.2,
      "min_progress_m": 0.10,
      "bidirectional": True,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["course_completed"] = TerminationTermCfg(
    func=pyg_term.bidirectional_course_completed,
    time_out=True,
    params={
      # Finish 0.5 m before either end, after almost all 10 m of exit flat.
      "forward_target_x": course_length - terrain_origin_x - 0.50,
      "reverse_target_x": -terrain_origin_x + 0.50,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["course_side_fall"] = TerminationTermCfg(
    func=pyg_term.course_lateral_out_of_bounds,
    params={"max_offset": 0.45, "asset_cfg": SceneEntityCfg("robot")},
  )

  if not play:
    cfg.events["foot_friction"].params["ranges"] = (0.3, 1.7)
    cfg.events["payload"].params["mass_range"] = (-1.0, 15.0)
  return cfg
