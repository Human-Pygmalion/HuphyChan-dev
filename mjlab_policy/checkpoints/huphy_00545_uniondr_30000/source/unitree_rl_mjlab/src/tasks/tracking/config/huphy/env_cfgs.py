"""HUPHY leg-only motion-mimic environments."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.assets.robots.huphy import HUPHY_ACTION_SCALE, get_huphy_robot_cfg
import src.tasks.tracking.mdp as mdp
from src.tasks.tracking.mdp import MotionCommandCfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from . import union_dr_events


TRACKED_BODIES = (
  "base_link",
  "L_hip_pitch_link",
  "R_hip_pitch_link",
  "L_shin_link",
  "R_shin_link",
  "L_ankle_pitch_link",
  "R_ankle_pitch_link",
  "L_foot_link",
  "R_foot_link",
)

MOTOR_JOINTS = (
  r".*_crank_[AB]_joint",
  r".*_hip_(yaw|pitch|roll)_joint",
  r".*_knee_joint",
)

_WALK_ALPHA = (0.5 * math.log(0.9), 0.5 * math.log(1.1))
_UNION_INERTIAL_DR = {
  "pelvis": (
    ("base_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0699),
      t1_range=(-0.03, 0.03), t2_range=(-0.03, 0.03),
      t3_range=(-0.03, 0.03), d_range=(-0.0154, 0.0154),
    ),
  ),
  "hip_pitch": (
    (r".*_hip_pitch_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02),
    ),
  ),
  "hip_roll": (
    (r".*_hip_roll_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02),
    ),
  ),
  "thigh": (
    (r".*_thigh_link",),
    dict(
      alpha_range=_WALK_ALPHA,
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.0066), d_range=(-0.0146, 0.0146),
    ),
  ),
  "shin": (
    (r".*_shin_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0508),
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02),
    ),
  ),
  "ankle_pitch": (
    (r".*_ankle_pitch_link",),
    dict(
      alpha_range=(-0.059, 0.0554),
      t1_range=(-0.005, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02),
    ),
  ),
  "foot": (
    (r".*_foot_link",),
    dict(
      alpha_range=(_WALK_ALPHA[0], 0.0647),
      t1_range=(-0.0053, 0.005), t2_range=(-0.005, 0.005),
      t3_range=(-0.005, 0.005), d_range=(-0.02, 0.02),
    ),
  ),
  "crank_rod": (
    (r".*_crank_[AB]", r".*_rod_[AB]"),
    dict(alpha_range=_WALK_ALPHA),
  ),
}


def huphy_flat_tracking_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the 12-action/24-joint closed-loop HUPHY mimic task."""
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": get_huphy_robot_cfg()}

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = HUPHY_ACTION_SCALE

  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.anchor_body_name = "base_link"
  motion.body_names = TRACKED_BODIES
  # Generic MotionCommand perturbs all 24 joints independently. That would
  # break HUPHY's passive 2-RSU closure at reset, so keep the projected
  # reference joint state exact; root pose/velocity DR remains active.
  motion.joint_position_range = (0.0, 0.0)
  motion.pose_range = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
    "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.15, 0.15),
  }

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(L|R)_foot\d+_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.terminations["anchor_pos"].params["threshold"] = 0.4
  cfg.terminations.pop("anchor_ori", None)
  cfg.terminations["ee_body_pos"].params.update(
    {"threshold": 0.4, "body_names": ("L_foot_link", "R_foot_link")}
  )
  cfg.viewer.body_name = "base_link"
  # The detailed capsule model has more potential contacts than G1, including
  # closed-linkage neighbours. The tracking base default (35) is insufficient
  # even for the initial pose.
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 1000

  # Deployable actor: replace the reference-relative 6-D anchor orientation
  # with the IMU-style gravity direction and omit unavailable global state.
  actor = cfg.observations["actor"]
  actor_terms = {}
  for name, term in actor.terms.items():
    if name in ("motion_anchor_pos_b", "base_lin_vel"):
      continue
    if name == "motion_anchor_ori_b":
      actor_terms["projected_gravity"] = ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
      )
    else:
      actor_terms[name] = term
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start"

  return cfg


def huphy_flat_tracking_union_dr_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """HUPHY mimic task with the HuphyChan-dev Union DR envelope."""
  cfg = huphy_flat_tracking_env_cfg(play=play)
  if play:
    return cfg

  # MotionCommand owns the reference-state reset; only physics/sensor DR is
  # replaced here so resuming the existing actor/critic remains shape-safe.
  cfg.events = {}
  for name, (body_names, ranges) in _UNION_INERTIAL_DR.items():
    cfg.events[f"union_inertia_{name}"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=body_names),
        **ranges,
      },
    )

  cfg.events.update(
    {
      "payload": EventTermCfg(
        func=union_dr_events.add_payload_mass,
        mode="reset",
        params={
          "mass_range": (-1.0, 8.0),
          "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        },
      ),
      "pd_gains": EventTermCfg(
        func=dr.pd_gains,
        mode="reset",
        params={
          "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2),
          "operation": "scale", "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "motor_strength": EventTermCfg(
        func=dr.effort_limits,
        mode="reset",
        params={
          "effort_limit_range": (0.8, 1.2), "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "motor_armature": EventTermCfg(
        func=dr.joint_armature,
        mode="reset",
        params={
          "ranges": (0.8, 1.2), "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot", joint_names=MOTOR_JOINTS),
        },
      ),
      "motor_friction": EventTermCfg(
        func=dr.joint_friction,
        mode="reset",
        params={
          "ranges": (0.8, 1.2), "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot", joint_names=MOTOR_JOINTS),
        },
      ),
      "motor_damping": EventTermCfg(
        func=dr.joint_damping,
        mode="reset",
        params={
          "ranges": (0.8, 1.2), "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot", joint_names=MOTOR_JOINTS),
        },
      ),
      "passive_friction": EventTermCfg(
        func=dr.joint_friction,
        mode="reset",
        params={
          "ranges": (0.0, 0.5), "operation": "abs",
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=(r".*_rod_[AB]_u[12]", r".*_ankle_(pitch|roll)_joint"),
          ),
        },
      ),
      "passive_damping": EventTermCfg(
        func=dr.joint_damping,
        mode="reset",
        params={
          "ranges": (0.0, 0.05), "operation": "abs",
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=(r".*_rod_[AB]_u[12]", r".*_ankle_(pitch|roll)_joint"),
          ),
        },
      ),
      "foot_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "ranges": (0.3, 1.7), "operation": "abs", "shared_random": True,
          "asset_cfg": SceneEntityCfg(
            "robot", geom_names=(r"^(L|R)_foot\d+_collision$",)
          ),
        },
      ),
      "encoder_bias": EventTermCfg(
        func=dr.encoder_bias,
        mode="reset",
        params={
          "bias_range": (-0.015, 0.015),
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "imu_mount": EventTermCfg(
        func=dr.site_quat,
        mode="reset",
        params={
          "roll_range": (-math.radians(2.0), math.radians(2.0)),
          "pitch_range": (-math.radians(2.0), math.radians(2.0)),
          "yaw_range": (-math.radians(2.0), math.radians(2.0)),
          "asset_cfg": SceneEntityCfg("robot", site_names=("imu_in_base",)),
        },
      ),
      "reset_hip_spread": EventTermCfg(
        func=union_dr_events.add_hip_roll_reset_spread,
        mode="reset",
        params={
          "spread_range": (-0.10, 0.06), "spread_prob": 0.5,
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=("R_hip_roll_joint", "L_hip_roll_joint"),
            preserve_order=True,
          ),
        },
      ),
      "torso_push": EventTermCfg(
        func=union_dr_events.apply_horizontal_body_push,
        mode="step",
        params={
          "magnitude_range": (0.0, 100.0),
          "duration_s": (0.1, 0.2), "cooldown_s": (2.0, 10.0),
          "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        },
      ),
    }
  )

  for term_name in ("base_ang_vel", "projected_gravity"):
    term = cfg.observations["actor"].terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = 2
    term.delay_per_env = True
    term.delay_update_period = 1_000_000
    term.delay_per_env_phase = False

  robot_cfg = cfg.scene.entities["robot"]
  for actuator in robot_cfg.articulation.actuators:
    actuator.delay_min_lag = 0
    actuator.delay_max_lag = 4
    actuator.delay_update_period = 1_000_000
    actuator.delay_per_env_phase = False

  return cfg
