"""Huphy-Squat-Flat-v0: in-place squat with pelvis-height control on flat ground.

Written from scratch on stock mjlab (no velocity-task base cfg); design record:
docs/task_design/decisions.md, reward_design.md, height_range.md.

  policy        50 Hz (decimation 4 x 5 ms physics), 20 s episodes
  command       target pelvis height h* in [0.72, 0.895] m, new value every 3-6 s (20 % keep)
  actor obs     IMU gyro, IMU gravity, 12 encoder angles (2 frames), last action, h*
  action        12 motor position targets = default + 0.25 * a, clipped, sent to explicit
                PD + T-N motors (real-robot gains)
  DR            all at full strength from the start, drawn at every reset
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import robot
from .mdp import commands as cmd
from .mdp import events as ev
from .mdp import observations as obs
from .mdp import rewards as rew
from .mdp import terminations as term

HEIGHT_RANGE = (0.72, 0.895)

# Per-link inertial DR (mass x e^(2 alpha), CoM shift t, shape d) from the CAD mass uncertainty
# (same ranges as mjlab_policy Rough-Box). Cranks and rods are not randomized.
INERTIAL_DR = {
  "pelvis": (("base_link",), dict(alpha_range=(-0.0256, 0.0699), t1_range=(-0.02, 0.02), t2_range=(-0.02, 0.02), t3_range=(-0.02, 0.02), d_range=(-0.0154, 0.0154))),
  "hip_pitch_link": (("L_hip_pitch_link", "R_hip_pitch_link"), dict(alpha_range=(-0.0256, 0.0364), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "hip_roll_link": (("L_hip_roll_link", "R_hip_roll_link"), dict(alpha_range=(-0.0256, 0.0378), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "thigh": (("L_thigh_link", "R_thigh_link"), dict(alpha_range=(-0.0256, 0.0455), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.0066), d_range=(-0.0146, 0.0146))),
  "shin": (("L_shin_link", "R_shin_link"), dict(alpha_range=(-0.0256, 0.0508), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "ankle_pitch_link": (("L_ankle_pitch_link", "R_ankle_pitch_link"), dict(alpha_range=(-0.059, 0.0554), t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
  "foot": (("L_foot_link", "R_foot_link"), dict(alpha_range=(-0.0256, 0.0647), t1_range=(-0.0053, 0.005), t2_range=(-0.005, 0.005), t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02))),
}

# Horizontal pelvis push: impulse <= 60 N x 0.2 s = 12 N*s -> delta v <= 0.51 m/s (23.6 kg),
# 0.38 m/s (+8 kg). (First version: 35 N; widened 2026-10-21 after the out-of-range eval:
# the 35 N policy starts to fall at 50 N and mostly falls at 65 N.)
PUSH = dict(force_range=(0.0, 60.0), duration_s=(0.1, 0.2), interval_s=(2.0, 5.0))
FOOT_FRICTION = (0.3, 1.2)  # first version (0.4, 1.0)
PAYLOAD_KG = (-1.0, 8.0)  # first version (-1, 5)
OBS_MAX_LAG = 2  # policy steps -> 0 / 20 / 40 ms; first version 1 (0 / 20 ms)
# Command delay 0-20 ms: robot.ACTUATOR delay_max_lag = 4 physics steps (first version 2).

REWARD_WEIGHTS = {
  "track_height": 2.0, "vz_speed": -600.0, "knee_guidance": -0.2, "squat_posture": 1.0,
  "pitch_symmetry": 0.5, "upright": 1.5, "body_ang_vel": -0.05, "angular_momentum": -0.02,
  "lin_vel_xy": -0.5, "ang_vel_z": -0.2, "motor_limits": -5.0, "ankle_limits": -10.0,
  "loop_residual": -1.0, "rod_collision": -2.0, "torque_margin": -2.0, "thermal_effort": -0.02,
  "action_rate": -0.1, "joint_acc": -2.5e-7, "self_collisions": -1.0, "foot_slip": -0.5,
  "foot_impact": -0.5,
}


def _robot(**kw) -> SceneEntityCfg:
  return SceneEntityCfg("robot", **kw)


MOTORS = _robot(joint_names=robot.MOTOR_JOINT_NAMES)
ANKLES = _robot(joint_names=robot.ANKLE_JOINT_NAMES)
PASSIVE = _robot(joint_names=robot.ROD_JOINT_NAMES + robot.ANKLE_JOINT_NAMES)
MOTORS_AND_ANKLES = _robot(joint_names=robot.MOTOR_JOINT_NAMES + robot.ANKLE_JOINT_NAMES)
FEET = _robot(site_names=robot.FOOT_SITES)
FOOT_GEOMS = _robot(geom_names=(robot.FOOT_GEOM_EXPR,))
PELVIS = _robot(body_names=(robot.BASE_BODY,))


def sensors() -> tuple:
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(R_foot_link|L_foot_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    history_length=4,
  )
  nonfoot = ContactSensorCfg(
    name="nonfoot_ground_contact",
    primary=ContactMatch(mode="geom", pattern=r".*_collision$", entity="robot", exclude=(robot.FOOT_GEOM_EXPR,)),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    history_length=4,
  )
  self_col = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  rods = ContactSensorCfg(
    name="rod_contact",
    primary=ContactMatch(mode="geom", pattern=robot.ROD_GEOM_EXPR, entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
    history_length=4,
  )
  return (feet, nonfoot, self_col, rods)


def observations() -> dict[str, ObservationGroupCfg]:
  actor = {
    "base_ang_vel": ObservationTermCfg(func=obs.imu_gyro, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(func=obs.imu_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "motor_pos": ObservationTermCfg(func=obs.motor_encoder, params={"asset_cfg": MOTORS},
                                    noise=Unoise(n_min=-0.01, n_max=0.01), history_length=2),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "height_command": ObservationTermCfg(func=envs_mdp.generated_commands, params={"command_name": "height"}),
  }
  T = ObservationTermCfg
  critic = {
    "height_command": T(func=envs_mdp.generated_commands, params={"command_name": "height"}),
    "height": T(func=obs.height),
    "height_error": T(func=obs.height_error),
    "base_lin_vel_w": T(func=obs.root_lin_vel_w),
    "base_ang_vel": T(func=obs.root_ang_vel_b),
    "projected_gravity": T(func=obs.root_gravity_b),
    "joint_pos": T(func=obs.joint_pos_true, params={"asset_cfg": MOTORS_AND_ANKLES}),
    "joint_vel": T(func=obs.joint_vel_true, params={"asset_cfg": MOTORS_AND_ANKLES}, scale=0.1),
    "actions": T(func=envs_mdp.last_action),
    "torque": T(func=obs.applied_torque),
    "foot_contact": T(func=obs.foot_contact, params={"sensor_name": "feet_ground_contact"}),
    "foot_force": T(func=obs.foot_force, params={"sensor_name": "feet_ground_contact"}),
    "dr_pd_gains": T(func=obs.pd_gain_scale),
    "dr_encoder_bias": T(func=obs.encoder_bias, params={"asset_cfg": MOTORS}),
    "dr_lags": T(func=obs.lags),
    "dr_foot_friction": T(func=obs.foot_friction, params={"asset_cfg": FOOT_GEOMS}),
    "dr_mass": T(func=obs.mass_offset),
    "dr_pelvis_com": T(func=obs.com_offset, params={"asset_cfg": PELVIS}),
    "dr_imu_mount": T(func=obs.imu_mount, params={"asset_cfg": _robot(site_names=(robot.IMU_SITE,))}),
    "dr_push": T(func=obs.push_state),
    "dr_motor_friction": T(func=obs.dof_param, params={"field": "dof_frictionloss", "asset_cfg": MOTORS}),
    "dr_motor_damping": T(func=obs.dof_param, params={"field": "dof_damping", "asset_cfg": MOTORS}),
    "dr_motor_armature": T(func=obs.dof_param, params={"field": "dof_armature", "asset_cfg": MOTORS}),
    "dr_passive_friction": T(func=obs.dof_param, params={"field": "dof_frictionloss", "asset_cfg": PASSIVE, "relative": False, "scale": 2.0}),
    "dr_passive_damping": T(func=obs.dof_param, params={"field": "dof_damping", "asset_cfg": PASSIVE, "relative": False, "scale": 20.0}),
  }
  groups = {
    "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }
  for g in groups.values():
    g.nan_policy = "warn"
    g.nan_check_per_term = True
  return groups


def events() -> dict[str, EventTermCfg]:
  E = EventTermCfg
  out: dict[str, EventTermCfg] = {}
  for link, (bodies, ranges) in INERTIAL_DR.items():
    out[f"inertia_{link}"] = E(mode="reset", func=dr.pseudo_inertia, params={"asset_cfg": _robot(body_names=bodies), **ranges})
  out["payload"] = E(mode="reset", func=ev.payload_mass, params={"mass_range": PAYLOAD_KG, "asset_cfg": PELVIS})  # after inertia_pelvis
  for field_name, fn in (("friction", dr.joint_friction), ("damping", dr.joint_damping), ("armature", dr.joint_armature)):
    out[f"motor_{field_name}"] = E(mode="reset", func=fn, params={"asset_cfg": MOTORS, "operation": "scale", "ranges": (0.8, 1.2)})
  out["passive_friction"] = E(mode="reset", func=dr.joint_friction, params={"asset_cfg": PASSIVE, "operation": "abs", "ranges": (0.0, 0.5)})
  out["passive_damping"] = E(mode="reset", func=dr.joint_damping, params={"asset_cfg": PASSIVE, "operation": "abs", "ranges": (0.0, 0.05)})
  out["foot_friction"] = E(mode="reset", func=dr.geom_friction, params={"asset_cfg": FOOT_GEOMS, "operation": "abs", "ranges": FOOT_FRICTION, "shared_random": True})
  out["pd_gains"] = E(mode="reset", func=dr.pd_gains, params={"asset_cfg": _robot(), "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2), "operation": "scale"})
  out["encoder_bias"] = E(mode="reset", func=dr.encoder_bias, params={"asset_cfg": MOTORS, "bias_range": (-0.015, 0.015)})
  deg2 = math.radians(2.0)
  out["imu_mount"] = E(mode="reset", func=dr.site_quat, params={"asset_cfg": _robot(site_names=(robot.IMU_SITE,)), "roll_range": (-deg2, deg2), "pitch_range": (-deg2, deg2), "yaw_range": (-deg2, deg2)})
  out["obs_lag"] = E(mode="reset", func=ev.sample_obs_lag, params={"max_lag": OBS_MAX_LAG})
  out["reset_pose"] = E(mode="reset", func=ev.reset_squat_pose, params={"motor_noise": 0.05})
  out["push"] = E(mode="step", func=ev.push_pelvis, params={**PUSH, "asset_cfg": PELVIS})
  out["motor_step_end"] = E(mode="step", func=ev.motor_step_end)
  return out


def rewards() -> dict[str, RewardTermCfg]:
  P = {
    "track_height": (rew.track_height, {"std": 0.05}),
    "vz_speed": (rew.vz_speed, {"threshold": 0.1, "grace_s": 0.2}),
    "knee_guidance": (rew.knee_guidance, {"gate": 0.05}),
    "squat_posture": (rew.squat_posture, {"std": {r".*_hip_roll_joint": 0.08, r".*_hip_yaw_joint": 0.08, r".*_ankle_roll_joint": 0.08,
                                                 r".*_hip_pitch_joint": 0.25, r".*_knee_joint": 0.25, r".*_ankle_pitch_joint": 0.15}}),
    "pitch_symmetry": (rew.pitch_symmetry, {"std": 0.10}),
    "upright": (rew.upright, {"std": math.sqrt(0.05)}),
    "body_ang_vel": (rew.ang_vel_xy, {}),
    "angular_momentum": (rew.angular_momentum, {}),
    "lin_vel_xy": (rew.lin_vel_xy, {}),
    "ang_vel_z": (rew.ang_vel_z, {}),
    "motor_limits": (rew.joint_soft_limits, {"asset_cfg": MOTORS}),
    "ankle_limits": (rew.joint_soft_limits, {"asset_cfg": ANKLES}),
    "loop_residual": (rew.loop_residual, {"r0": 0.0005}),
    "rod_collision": (rew.contact_count, {"sensor_name": "rod_contact", "force_threshold": 1.0}),
    "torque_margin": (rew.torque_margin, {"soft_ratio": 0.9}),
    "thermal_effort": (rew.thermal_effort, {"rated": robot.RATED_TORQUE}),
    "action_rate": (rew.action_rate, {}),
    "joint_acc": (rew.joint_acc, {"asset_cfg": MOTORS_AND_ANKLES}),
    "self_collisions": (rew.contact_count, {"sensor_name": "self_collision", "force_threshold": 10.0}),
    "foot_slip": (rew.foot_slip, {"sensor_name": "feet_ground_contact", "asset_cfg": FEET}),
    "foot_impact": (rew.foot_impact, {"sensor_name": "feet_ground_contact", "asset_cfg": FEET}),
  }
  assert P.keys() == REWARD_WEIGHTS.keys()
  return {k: RewardTermCfg(func=rew.clamped(f, REWARD_WEIGHTS[k]), weight=REWARD_WEIGHTS[k], params=p) for k, (f, p) in P.items()}


def terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
    "nonfoot_contact": TerminationTermCfg(func=illegal_contact, params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 1.0}),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "physics_blowup": TerminationTermCfg(func=term.physics_blowup),
  }


def huphy_squat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=4096,
      env_spacing=2.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot.robot_cfg()},
      sensors=sensors(),
    ),
    observations=observations(),
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=robot.MOTOR_JOINT_NAMES, scale=robot.ACTION_SCALE,
      clip=dict(robot.ACTION_CLIP), use_default_offset=True)},
    commands={"height": cmd.HeightCommandCfg(height_range=HEIGHT_RANGE, resampling_time_range=(3.0, 6.0),
                                              keep_prob=0.2, debug_vis=True)},
    events=events(),
    rewards=rewards(),
    terminations=terminations(),
    curriculum={},
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
                        distance=2.5, elevation=-5.0, azimuth=90.0),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
  if play:
    cfg.scene.num_envs = 16
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push")
  return cfg
