"""HUPHY/Pygmalion leg-only closed-loop robot for motion tracking."""

import copy

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


HUPHY_XML = Path(
  "/home/jooyoungbae/HuphyChan-dev/pygmalion_locomotion/assets/"
  "pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_"
  "proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml"
)
if not HUPHY_XML.exists():
  raise FileNotFoundError(f"HUPHY MJCF not found: {HUPHY_XML}")

HUPHY_ACTION_SCALE = 0.25

# Only these twelve joints are motorized. Rod universal joints and ankle
# pitch/roll joints remain passive and are closed by the MJCF connect equalities.
HUPHY_ACTUATORS = (
  IdealPdActuatorCfg(
    target_names_expr=(".*_crank_[AB]_joint",),
    stiffness=28.5 * 1.25**2 / 2,
    damping=1.81 * 1.25**2 / 2,
    effort_limit=60.0,
    armature=0.0638,
    frictionloss=0.251,
  ),
  IdealPdActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    stiffness=150.0,
    damping=6.0,
    effort_limit=60.0,
    armature=0.0638,
    frictionloss=0.251,
  ),
  IdealPdActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
    stiffness=150.0,
    damping=6.0,
    effort_limit=120.0,
    armature=0.0773,
    frictionloss=0.519,
  ),
  IdealPdActuatorCfg(
    target_names_expr=(".*_knee_joint",),
    stiffness=220.0,
    damping=6.0,
    effort_limit=120.0,
    armature=0.0773,
    frictionloss=0.519,
  ),
)

HUPHY_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9115),
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

HUPHY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(L|R)_foot\d+_collision$": 3, ".*_collision": 1},
  priority={r"^(L|R)_foot\d+_collision$": 1},
  friction={r"^(L|R)_foot\d+_collision$": (0.6,)},
)


def get_huphy_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HUPHY_XML))


def get_huphy_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HUPHY_INIT_STATE,
    collisions=(HUPHY_COLLISION,),
    spec_fn=get_huphy_spec,
    articulation=EntityArticulationInfoCfg(
      # Each registered task mutates actuator delay settings independently.
      actuators=copy.deepcopy(HUPHY_ACTUATORS),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
