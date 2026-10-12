"""Pygmalion LegOnly robot (AB ankle: two crank motors drive a 2-RSU ankle linkage).

All values are the resolved values of training run 2026-10-11_23-35-20_rough_box_main_v11a
(reference/env_reference.yaml, section scene.entities.robot). Nothing here reads environment
variables.

Model: assets/pygmalion_legonly_ab.xml (MJCF). 12 actuated joints:
  hip_pitch, hip_roll (RS04), hip_yaw (RS03), knee (RS04), crank_A, crank_B (RS03) per leg.
The ankle pitch/roll hinges and the rod universal joints are passive; four `connect`
equalities close the linkage.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .tn_actuator import TnPdActuatorCfg

ASSETS = Path(__file__).resolve().parent / "assets"
XML = ASSETS / "pygmalion_legonly_ab.xml"

# Joint-name groups used by observations / rewards.
MOTOR_JOINTS = (".*_hip_.*_joint", ".*_knee_joint", ".*_crank_[AB]_joint")  # 12 motor encoders
OBS_JOINTS = MOTOR_JOINTS + (".*_ankle_.*_joint",)  # + passive ankle (critic only)
POSE_JOINTS = (".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint")  # pose reward
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = (r"L_foot\d+_collision", r"R_foot\d+_collision")

ACTION_SCALE = 0.25  # rad per unit action, all 12 joints

# Target clip = centre +- 0.9/2 x joint range (same rule as the 0.9 soft joint limit).
ACTION_CLIP = {
  "R_crank_A_joint": (-1.08, 1.08),
  "R_crank_B_joint": (-1.08, 1.08),
  "L_crank_A_joint": (-1.08, 1.08),
  "L_crank_B_joint": (-1.08, 1.08),
  "R_hip_yaw_joint": (-0.7068582000000001, 0.7068582000000001),
  "L_hip_yaw_joint": (-0.7068582000000001, 0.7068582000000001),
  "R_hip_pitch_joint": (-1.9678634000000002, 0.30979540000000017),
  "R_hip_roll_joint": (-0.3403389, 1.3875369000000002),
  "L_hip_pitch_joint": (-0.30979540000000017, 1.9678634000000002),
  "L_hip_roll_joint": (-1.3875369000000002, 0.3403389),
  "R_knee_joint": (0.10471999999999992, 1.98968),
  "L_knee_joint": (-1.98968, -0.10471999999999992),
}


def tn_curve(family: str) -> tuple[tuple[float, float], ...]:
  """Measured 48 V torque-speed curve (output shaft) + the 200 rpm no-load point, in rad/s."""
  rows = [
    (float(r["speed_rpm"]), float(r["torque_Nm"]))
    for r in csv.DictReader(open(ASSETS / "motor_spec" / f"{family}_TN_curve_48V.csv"))
  ]
  rows.append((200.0, 0.0))
  rpm = 2 * math.pi / 60
  return tuple((w * rpm, t) for w, t in rows)


# Motor dynamics measured on the bench (unloaded output shaft, 2026-09-09/10).
_RS04 = dict(armature=0.0773, frictionloss=0.519, viscous_damping=0.0275)
_RS03 = dict(armature=0.0638, frictionloss=0.251, viscous_damping=0.0252)

ACTUATORS = (
  # Cranks: Unitree G1 ankle gain (28.5 / 1.81) reflected through the 2-RSU lever r = 1.25
  # shared by two motors: K_crank = K_ankle * r^2 / 2.
  TnPdActuatorCfg(
    target_names_expr=(".*_crank_[AB]_joint",),
    stiffness=28.5 * 1.25**2 / 2,
    damping=1.81 * 1.25**2 / 2,
    effort_limit=60.0,
    tn_curve=tn_curve("RS03"),
    **_RS03,
  ),
  # Hip/knee: G1 / K-Bot gains scaled by mass ratio (x1.47).
  TnPdActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    stiffness=150.0,
    damping=6.0,
    effort_limit=60.0,
    tn_curve=tn_curve("RS03"),
    **_RS03,
  ),
  TnPdActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
    stiffness=150.0,
    damping=6.0,
    effort_limit=120.0,
    tn_curve=tn_curve("RS04"),
    **_RS04,
  ),
  TnPdActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=220.0,
    damping=6.0,
    effort_limit=120.0,
    tn_curve=tn_curve("RS04"),
    **_RS04,
  ),
)

# Slight crouch: hip 0.175, knee 0.35 rad; ankle/crank/rod values solve the closed loop.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8975),
  joint_pos={
    "L_hip_pitch_joint": 0.175,
    "R_hip_pitch_joint": -0.175,
    "L_knee_joint": -0.35,
    "R_knee_joint": 0.35,
    "L_crank_A_joint": -0.1438849965517705,
    "L_rod_A_u1": 0.1479036260054202,
    "L_rod_A_u2": -1.6276083554340244e-07,
    "L_crank_B_joint": 0.14687020555452607,
    "L_rod_B_u1": -0.1487249952991747,
    "L_rod_B_u2": 2.412252153998891e-07,
    "L_ankle_pitch_joint": 0.1750072557823706,
    "L_ankle_roll_joint": 4.703560734910088e-06,
    "R_crank_A_joint": 0.14395487369574003,
    "R_rod_A_u1": -0.14797270816558003,
    "R_rod_A_u2": 2.6819833875474064e-06,
    "R_crank_B_joint": -0.14706307530170212,
    "R_rod_B_u1": 0.14892320455004768,
    "R_rod_B_u2": -3.976385353225781e-06,
    "R_ankle_pitch_joint": 0.17516470240179116,
    "R_ankle_roll_joint": -7.751934743886312e-05,
  },
  joint_vel={".*": 0.0},
)

# All collision geoms on; feet condim 3 / friction 0.6 / priority 1, the rest condim 1.
COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(L|R)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(L|R)_foot[1-7]_collision$": 1},
  friction={r"^(L|R)_foot[1-7]_collision$": (0.6,)},
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XML))


def robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9
    ),
  )
