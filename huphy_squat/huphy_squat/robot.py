"""HUPHY LegOnly robot (2-RSU "AB" ankle) for the squat task.

Model: assets/huphy_legonly_ab.xml (the Pygmalion LegOnly MJCF with fixed names and rod
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
# Real-robot driver gains (MIT mode).
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

# Default (action-offset) pose: slight crouch, hip pitch 0.175 / knee 0.35 rad, hips spread
# outward by HIP_SPREAD (feet 33 cm apart at the foot origins; 25 cm without the spread). Hip yaw and the
# crank angles keep each foot level and pointing ahead (ankle pitch 0.175 / roll -0.051);
# passive joints: assets/linkage_table.npz. Solved by scripts/reference/squat_reference.py.
HIP_SPREAD = 0.05  # rad, hip roll outward (R +, L -)
DEFAULT_MOTOR_POS = {
  "R_hip_pitch_joint": -0.175, "R_hip_roll_joint": 0.05, "R_hip_yaw_joint": 0.008837, "R_knee_joint": 0.35,
  "R_crank_A_joint": 0.106983, "R_crank_B_joint": -0.184691,
  "L_hip_pitch_joint": 0.175, "L_hip_roll_joint": -0.05, "L_hip_yaw_joint": -0.008837, "L_knee_joint": -0.35,
  "L_crank_A_joint": -0.106983, "L_crank_B_joint": 0.184691,
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

# Squat reference q*(h) for the R leg with the hips spread by HIP_SPREAD: pelvis upright, feet
# level and pointing ahead. Down to h = 0.71 the whole-body CoM is over the sole centre; below,
# ankle pitch stays at 0.43 (soft limit 0.454) and the CoM moves behind the sole centre
# (SQUAT_REF_COM_DX, -1.9 cm at 0.65 m): the robot sits back. Computed by
# scripts/reference/squat_reference.py (CPU MuJoCo pose solve on the knee grid, linearly interpolated at
# SQUAT_REF_H, 4 decimals). L leg: hip pitch / roll / yaw and knee negated, ankle pitch / roll same sign.
SQUAT_REF_H = (0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.895)
SQUAT_REF_KNEE = (1.6872, 1.6189, 1.5451, 1.4659, 1.3781, 1.2818, 1.178, 1.064, 0.9361, 0.7863, 0.5956, 0.3589)
SQUAT_REF_HIP_PITCH = (-1.2611, -1.192, -1.1177, -1.038, -0.963, -0.8917, -0.8159, -0.7339, -0.643, -0.5378, -0.4052, -0.242)
SQUAT_REF_ANKLE_PITCH = (0.43, 0.43, 0.43, 0.43, 0.4168, 0.3917, 0.3634, 0.3313, 0.2941, 0.2493, 0.1908, 0.1172)
SQUAT_REF_HIP_YAW = (0.1552, 0.125, 0.1023, 0.0846, 0.0717, 0.0618, 0.0531, 0.0451, 0.0374, 0.0298, 0.0214, 0.0123)
SQUAT_REF_ANKLE_ROLL = (-0.163, -0.1346, -0.1139, -0.0983, -0.0874, -0.0795, -0.0729, -0.0673, -0.0625, -0.0582, -0.0544, -0.0515)
# lateral distance of the two foot-link origins in that pose [m]
SQUAT_REF_FOOT_WIDTH = (0.4388, 0.4118, 0.3919, 0.3769, 0.3669, 0.3601, 0.3543, 0.3494, 0.3452, 0.3414, 0.338, 0.3352)
SQUAT_REF_COM_DX = (-0.0249, -0.0192, -0.0123, -0.004, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0)


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
  # Command (bus) delay 0 / 5 / 10 / 15 / 20 ms: one lag per env for all 12 motors, drawn at the first
  # physics step after a reset and held for the episode (update period > episode length).
  delay_min_lag=0,
  delay_max_lag=4,
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
