"""HUPHY LegOnly robot (2-RSU "AB" ankle) for the squat task.

Model: assets/huphy_legonly_ab.xml (copy of the Pygmalion MJCF with fixed names and rod
collision capsules; see the comment at the top of the file). 23.63 kg, 12 motors:
hip pitch / hip roll / knee (RS04), hip yaw / crank A / crank B (RS03) per leg. The ankle
pitch / roll hinges and the rod U-joints are passive; four `connect` equalities close the
two ankle loops.

Sign convention: the R leg is the reference. Mirror signs q_L = s * q_R:
hip pitch / roll / yaw, knee, crank A / B -> s = -1; ankle pitch / roll -> s = +1.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .actuator import HuphyMotorCfg

ASSETS = Path(__file__).resolve().parent / "assets"
XML = ASSETS / "huphy_legonly_ab.xml"
LINKAGE_TABLE = ASSETS / "linkage_table.npz"

SIDES = ("R", "L")  # model order: the R leg is defined first in the MJCF
MOTORS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")
MOTOR_JOINT_NAMES = tuple(f"{s}_{m}_joint" for s in SIDES for m in MOTORS)  # = model order
ANKLE_JOINT_NAMES = tuple(f"{s}_ankle_{a}_joint" for s in SIDES for a in ("pitch", "roll"))
ROD_JOINT_NAMES = tuple(f"{s}_rod_{x}_{u}" for s in SIDES for x in "AB" for u in ("u1", "u2"))
POSTURE_JOINT_NAMES = tuple(
  f"{s}_{j}_joint" for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
FOOT_SITES = ("R_foot_site", "L_foot_site")
FOOT_BODIES = ("R_foot_link", "L_foot_link")
FOOT_GEOM_EXPR = r"^(L|R)_foot\d_collision$"
ROD_GEOM_EXPR = r"^(L|R)_rod_[AB]_collision$"
IMU_SITE = "imu_in_base"
BASE_BODY = "base_link"

FAMILY = {
  ".*_hip_pitch_joint": "RS04",
  ".*_hip_roll_joint": "RS04",
  ".*_knee_joint": "RS04",
  ".*_hip_yaw_joint": "RS03",
  ".*_crank_[AB]_joint": "RS03",
}
# Real-robot driver gains (MIT mode, ~/huphy/docs/pd_gains.md).
STIFFNESS = {".*_hip_pitch_joint": 135.0, ".*_hip_roll_joint": 135.0, ".*_hip_yaw_joint": 75.0,
             ".*_knee_joint": 130.0, ".*_crank_[AB]_joint": 60.0}
DAMPING = {".*_hip_pitch_joint": 6.0, ".*_hip_roll_joint": 6.0, ".*_hip_yaw_joint": 4.0,
           ".*_knee_joint": 6.0, ".*_crank_[AB]_joint": 2.7}
# Bench-measured motor dynamics (unloaded output shaft); written into the MJCF joints.
MOTOR_DYNAMICS = {
  "RS04": dict(armature=0.0773, frictionloss=0.519, viscous_damping=0.0275),
  "RS03": dict(armature=0.0638, frictionloss=0.251, viscous_damping=0.0252),
}
# Continuous torque rating used by the thermal penalty [N*m].
RATED_TORQUE = {".*_hip_pitch_joint": 40.0, ".*_hip_roll_joint": 40.0, ".*_hip_yaw_joint": 20.0,
                ".*_knee_joint": 40.0, ".*_crank_[AB]_joint": 20.0}

# Default (action-offset) pose: slight crouch, hip pitch 0.175 / knee 0.35 rad. Crank angles are
# the ones that put the ankle at pitch 0.175 / roll 0 (passive joints: assets/linkage_table.npz).
DEFAULT_MOTOR_POS = {
  "R_hip_pitch_joint": -0.175, "R_hip_roll_joint": 0.0, "R_hip_yaw_joint": 0.0, "R_knee_joint": 0.35,
  "R_crank_A_joint": 0.14395487369574003, "R_crank_B_joint": -0.14706307530170212,
  "L_hip_pitch_joint": 0.175, "L_hip_roll_joint": 0.0, "L_hip_yaw_joint": 0.0, "L_knee_joint": -0.35,
  "L_crank_A_joint": -0.1438849965517705, "L_crank_B_joint": 0.14687020555452607,
}
RESET_CLEARANCE = 0.001  # lowest sole corner this far above the floor at reset [m]

ACTION_SCALE = 0.25  # rad per unit action, all 12 motors
# Target clip in the measured (encoder) frame. Hips / knee: centre +- 0.9/2 x joint range (the
# 0.9 soft limit). Cranks: +-0.95 rad, inside the 0.9 soft limit (+-1.08); the full ankle range
# needs |crank| <= 0.87, the passive-ankle limit penalty handles what lies beyond.
ACTION_CLIP = {
  "R_hip_pitch_joint": (-1.9678634, 0.3097954), "L_hip_pitch_joint": (-0.3097954, 1.9678634),
  "R_hip_roll_joint": (-0.3403389, 1.3875369), "L_hip_roll_joint": (-1.3875369, 0.3403389),
  "R_hip_yaw_joint": (-0.7068582, 0.7068582), "L_hip_yaw_joint": (-0.7068582, 0.7068582),
  "R_knee_joint": (0.10472, 1.98968), "L_knee_joint": (-1.98968, -0.10472),
  "R_crank_A_joint": (-0.95, 0.95), "R_crank_B_joint": (-0.95, 0.95),
  "L_crank_A_joint": (-0.95, 0.95), "L_crank_B_joint": (-0.95, 0.95),
}
SOFT_LIMIT_FACTOR = 0.9

# Squat reference q*(h) for the R leg (pelvis upright, feet flat, whole-body CoM over the sole
# centre), from docs/task_design/height_static.json (reward_design.md section 2.2).
# L leg: hip pitch and knee negated, ankle pitch same sign. Roll / yaw references are 0.
SQUAT_REF_H = (0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.8922, 0.895)
SQUAT_REF_KNEE = (1.479, 1.388, 1.291, 1.186, 1.072, 0.945, 0.795, 0.607, 0.420, 0.378)
SQUAT_REF_HIP_PITCH = (-1.037, -0.969, -0.897, -0.821, -0.739, -0.648, -0.543, -0.412, -0.283, -0.255)
SQUAT_REF_ANKLE_PITCH = (0.442, 0.419, 0.394, 0.366, 0.334, 0.297, 0.252, 0.194, 0.136, 0.123)


def tn_curve(family: str) -> tuple[tuple[float, float], ...]:
  """Measured 48 V torque-speed curve (output shaft) + a 200 rpm no-load point; (rad/s, N*m)."""
  rows = [(float(r["speed_rpm"]), float(r["torque_Nm"]))
          for r in csv.DictReader(open(ASSETS / "motor_spec" / f"{family}_TN_curve_48V.csv"))]
  rows.append((200.0, 0.0))
  k = 2 * math.pi / 60
  return tuple((w * k, t) for w, t in rows)


# Feet: box soles, condim 3, priority 1 (their friction wins over the floor's). Everything else
# (incl. the rod capsules): condim 1.
COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_GEOM_EXPR: 3, ".*_collision": 1},
  priority={FOOT_GEOM_EXPR: 1},
  friction={FOOT_GEOM_EXPR: (0.6,)},
)

ACTUATOR = HuphyMotorCfg(
  target_names_expr=MOTOR_JOINT_NAMES,
  joint_stiffness=STIFFNESS,
  joint_damping=DAMPING,
  joint_family=FAMILY,
  tn_curves={f: tn_curve(f) for f in ("RS03", "RS04")},
  motor_dynamics=MOTOR_DYNAMICS,
  # Command (bus) delay 0 / 5 / 10 ms: one lag per env for all 12 motors, drawn at the first
  # physics step after a reset and held for the episode (update period > episode length).
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=1_000_000,
  delay_per_env_phase=False,
)


def load_default_pose() -> tuple[dict[str, float], float]:
  """Full default joint pose (motors + closed-loop passive joints) and the matching base height."""
  with np.load(LINKAGE_TABLE) as t:
    pose = {str(n): float(v) for n, v in zip(t["default_names"], t["default_values"], strict=True)}
    base_z = float(-t["corner_z0"].min()) + RESET_CLEARANCE
  return pose, base_z


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XML))


def robot_cfg() -> EntityCfg:
  pose, base_z = load_default_pose()
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, base_z), joint_pos=pose, joint_vel={".*": 0.0}),
    collisions=(COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=(ACTUATOR,), soft_joint_pos_limit_factor=SOFT_LIMIT_FACTOR),
  )
