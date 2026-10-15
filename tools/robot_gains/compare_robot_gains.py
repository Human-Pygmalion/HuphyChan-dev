#!/usr/bin/env python3
"""Compare the PD gains (Kp, Kd) that other humanoid RL projects published with ours.

Run (CPU only, always through the mjlab venv, from mujoco-sim/mjlab):
    CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python3 \
        ../../tools/robot_gains/compare_robot_gains.py --cases g1_a g1_b pyg_v7b pyg_trial55

Every robot is loaded in MuJoCo, its leg joints are turned into PD position actuators with
the PUBLISHED Kp / Kd / effort limit, and the following is measured (see docs/170):
  A  initial pose, PD only (targets = default pose, no policy)
     A1 steady torque, steady error, base height drop
     A2 smallest Kp/Kd scale that still stands 5 s (with and without a 0.3 m/s push)
     A3 ankle pitch stiffness / (m g h)
     A4 natural frequency + damping ratio per joint (rotor only; swing; standing)
     A5 torque at action = 1, Kp*half-range vs effort, Kd/Kp, Kp per kg, Kp/(m g L)
  B  walking-load PROXIES (no trained policy):
     B1 quasi-static single-leg support (whole body weight on one leg)
     B2 human-gait peak joint moments scaled by body mass (table in HUMAN_PEAK_NM_PER_KG)
  C  rule inference: correlation / ratio constancy of Kp against candidate drivers

Nothing here touches training code. Results go to tools/robot_gains/results/<case>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MJLAB = REPO / "mujoco-sim" / "mjlab"
G1_XML = MJLAB / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
PYG_MJB = MJLAB / "analysis/out/stand10_v9caps_m8998_s1_model.mjb"
# env.yaml of the v11a rough run: source of the actions clip window and init joint pose
PYG_ENV_YAML = (
    REPO / ".claude/worktrees/vibecoded-audit-20261012/vibecoded-audit-20261012"
    / "reference/rough_main/env.yaml"
)
RESULTS = Path(__file__).resolve().parent / "results"
G = 9.81

# ---------------------------------------------------------------------------------------
# Published configuration (text read from GitHub; commit hashes recorded here)
# ---------------------------------------------------------------------------------------
PUBLISHED = {
    "unitree_rl_gym": {
        "repo": "github.com/unitreerobotics/unitree_rl_gym",
        "commit": "276801e46c5d433564f24658bac64f254b7d2d4b",
        "files": [
            "legged_gym/envs/g1/g1_config.py",
            "legged_gym/envs/h1/h1_config.py",
            "legged_gym/envs/base/legged_robot_config.py",
            "resources/robots/g1_description/g1_12dof.urdf (effort limits, mass sum 32.107 kg)",
            "resources/robots/h1/urdf/h1.urdf (effort limits, mass sum 51.650 kg)",
        ],
        "sim": "dt 0.005, decimation 4 (50 Hz policy), clip_actions 100, armature 0 (Isaac Gym)",
    },
    "booster_gym": {
        "repo": "github.com/BoosterRobotics/booster_gym",
        "commit": "da396a06d6eed99e2de72d7749c48ee8748950f9",
        "files": ["envs/T1.yaml", "resources/T1/T1_locomotion.urdf (effort limits, mass sum 31.614 kg)"],
        "sim": "dt 0.002, decimation 10 (50 Hz policy), action_scale 1.0, clip_actions 1.0, armature 0",
    },
    "isaac_berkeley_humanoid": {
        "repo": "github.com/HybridRobotics/isaac_berkeley_humanoid",
        "commit": "ffc7b2692917fdaf7cd87da53ada778e989dae35",
        "files": [
            "exts/berkeley_humanoid/berkeley_humanoid/assets/berkeley_humanoid.py",
            "exts/.../tasks/locomotion/velocity/velocity_env_cfg.py",
        ],
        "sim": "dt 0.005, decimation 4, action scale 0.5 with use_default_offset",
    },
    "mjlab": {
        "repo": "local mujoco-sim/mjlab (submodule)",
        "commit": "b2d68f9eb4e47bd1f62ff84d149fd6247907aa95",
        "files": ["src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py"],
        "sim": "dt 0.005, decimation 4; Kp = armature*(2*pi*10)^2, Kd = 2*2*armature*(2*pi*10), "
               "action scale = 0.25*effort/Kp",
    },
}

# unitree_rl_gym G1 (12 dof): gains by joint-name keyword, effort from the URDF
G1_URG_GAIN = {  # group -> (Kp, Kd)
    "hip_yaw": (100.0, 2.0), "hip_roll": (100.0, 2.0), "hip_pitch": (100.0, 2.0),
    "knee": (150.0, 4.0), "ankle_pitch": (40.0, 2.0), "ankle_roll": (40.0, 2.0),
}
G1_URG_EFFORT = {  # g1_12dof.urdf <limit effort>
    "hip_yaw": 88.0, "hip_roll": 139.0, "hip_pitch": 88.0, "knee": 139.0,
    "ankle_pitch": 50.0, "ankle_roll": 50.0,
}
G1_URG_DEFAULT = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}  # others 0
G1_URG_ACTION_SCALE = 0.25

# Phase-2 robots (config text read from GitHub; models are NOT on disk, builders are added after the
# user approves the Menagerie download).
PUBLISHED_PHASE2 = {
    "h1": {
        "source": "unitree_rl_gym @276801e4 legged_gym/envs/h1/h1_config.py, resources/robots/h1/urdf/h1.urdf",
        "gains": {"hip_yaw": (150, 2), "hip_roll": (150, 2), "hip_pitch": (150, 2), "knee": (200, 4), "ankle": (40, 2),
                  "torso": (300, 6), "shoulder": (150, 2), "elbow": (100, 2)},
        "effort": {"hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 40},
        "range": {"hip_yaw": (-0.43, 0.43), "hip_roll": (-0.43, 0.43), "hip_pitch": (-1.57, 1.57),
                  "knee": (-0.26, 2.05), "ankle": (-0.87, 0.52)},
        "default": {"hip_pitch": -0.1, "knee": 0.3, "ankle": -0.2},  # everything else 0
        "action_scale": 0.25, "clip_actions": 100.0, "decimation": 4, "dt": 0.005, "urdf_mass_kg": 51.65,
    },
    "t1": {
        "source": "booster_gym @da396a06 envs/T1.yaml, resources/T1/T1_locomotion.urdf (12 leg joints only)",
        "gains": {"hip": (200, 5), "knee": (200, 5), "ankle": (50, 1)},
        "effort": {"hip_pitch": 45, "hip_roll": 30, "hip_yaw": 30, "knee": 60, "ankle_pitch": 24, "ankle_roll": 15},
        "range": {"hip_pitch": (-1.8, 1.57), "hip_roll": (-0.3, 1.57), "hip_yaw": (-1.0, 1.0), "knee": (0.0, 2.34),
                  "ankle_pitch": (-0.87, 0.35), "ankle_roll": (-0.44, 0.44)},
        "default": {"hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.25},  # everything else 0
        "action_scale": 1.0, "clip_actions": 1.0, "decimation": 10, "dt": 0.002, "urdf_mass_kg": 31.61,
    },
    "berkeley": {
        "source": "isaac_berkeley_humanoid @ffc7b269 assets/berkeley_humanoid.py + velocity_env_cfg.py",
        "gains": {"HR": (10, 1.5), "HAA": (10, 1.5), "HFE": (15, 1.5), "KFE": (15, 1.5), "FFE": (1, 0.1), "FAA": (1, 0.1)},
        "effort": {"HR": 20, "HAA": 20, "HFE": 30, "KFE": 30, "FFE": 20, "FAA": 5},
        "armature": {"HR": 6.9e-5 * 81, "HAA": 6.9e-5 * 81, "HFE": 9.4e-5 * 81, "KFE": 1.5e-4 * 81,
                     "FFE": 6.9e-5 * 81, "FAA": 6.1e-6 * 81},
        "default": {"LL_HR": -0.071, "LR_HR": 0.071, "LL_HAA": 0.103, "LR_HAA": -0.103, "LL_HFE": -0.463,
                    "LR_HFE": -0.463, "LL_KFE": 0.983, "LR_KFE": 0.983, "LL_FFE": -0.350, "LR_FFE": -0.350,
                    "LL_FAA": 0.126, "LR_FAA": -0.126},
        "action_scale": 0.5, "clip_actions": None, "decimation": 4, "dt": 0.005, "init_height": 0.515,
        "paper_mass_kg": 16,
    },
}

# Human peak joint moments in normal level walking, N m per kg body mass (docs/170 section 4).
# typical: healthy young adults, Toda 2015 (J Phys Ther Sci 27:1833, men 1.26 m/s: hip extensor 0.72,
#          knee extensor 0.71 / women 0.76, ankle plantarflexor 1.30), hip abductor Molina-Rueda 2014
#          (Braz J Phys Ther 18:252, controls 1.20 m/s: 0.73 / 0.70 first peak, 0.89 / 0.82 second peak).
# high:    knee-osteoarthritis cohort 1.35 m/s, Henriksen 2017 (BMJ Open Sport Exerc Med 2:e000230):
#          hip 1.03, knee extensor 0.80, ankle 1.41, hip abductor 0.89; consistent with the textbook
#          rule of thumb quoted in the task brief (Winter-style hip ~1.0, knee 0.8-1.0, ankle 1.4-1.6).
#          Hip rotation 0.25 and ankle invertor 0.17: Underwood 2004 (Clin Biomech 19:609, sound limb).
# All text excerpts were obtained through a web-page extraction tool -> treat digits as UNVERIFIED
# until checked against the papers.
HUMAN_PEAK_NM_PER_KG = {
    "typical": {"hip_pitch": 0.72, "knee": 0.76, "ankle_pitch": 1.30, "hip_roll": 0.73,
                "hip_yaw": 0.24, "ankle_roll": 0.16},
    "high": {"hip_pitch": 1.03, "knee": 0.80, "ankle_pitch": 1.41, "hip_roll": 0.89,
             "hip_yaw": 0.25, "ankle_roll": 0.17},
}
HUMAN_LEG_LENGTH_M = 0.90   # assumed adult hip-to-ankle length for the optional length scaling (UNVERIFIED)
HUMAN_BODY_MASS_KG = 70.0


# ---------------------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------------------
@dataclass
class Jt:
    name: str                 # full MuJoCo joint name
    group: str                # hip_yaw hip_roll hip_pitch knee ankle_pitch ankle_roll crank
    side: str                 # L / R
    kp: float
    kd: float
    effort: float
    scale: float              # action scale (rad per action unit)
    q0: float                 # default angle
    scored: bool = True       # leg joint that is analysed (False = held arm / waist with PD)
    window: tuple | None = None   # absolute clip bounds on the PD target (None = no clip)
    rng: tuple | None = None      # hard joint range


@dataclass
class Case:
    key: str
    label: str
    model: mujoco.MjModel
    qpos0: np.ndarray                      # full default qpos (base z not yet fitted)
    joints: list
    foot_body: dict                        # side -> body name
    ankle_joint: dict                      # side -> ankle pitch joint name
    hip_joint: dict                        # side -> hip pitch joint name
    ankle_roll_joint: dict
    hip_roll_joint: dict
    knee_joint: dict
    welded: list = field(default_factory=list)
    loop: bool = False
    source: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    action_clip_note: str = ""
    pose_note: str = ""

    # filled by prepare()
    pelvis_tilt_deg: float = 0.0
    qpos_placed: np.ndarray | None = None
    floor_geom: int = -1

    def jt(self, name):
        return next(j for j in self.joints if j.name == name)


# ---------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------
def _set_sim_options_like_mjlab(opt):
    opt.timestep = 0.005
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.impratio = 1.0
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 10
    opt.ls_iterations = 20
    opt.tolerance = 1e-8


def _g1_group(jname):
    for g in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
        if jname.endswith(g + "_joint"):
            return g
    return None


def build_g1(gain: str, pose: str, key: str, label: str, upper: str = "weld") -> Case:
    """gain: 'urg' (unitree_rl_gym) or 'mjlab' (BeyondMimic armature rule).
    pose: 'urg_home' (hip -0.1 knee 0.3 ankle -0.2, arms 0) or 'mjlab_bent' (KNEES_BENT_KEYFRAME).
    upper: 'weld' = waist/arms/wrists locked at their pose (isolates the leg gains; default) or
           'pd' = held by the published mjlab upper-body gains (only for gain == 'mjlab')."""
    sys.path.insert(0, str(MJLAB / "src"))
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as gc  # noqa: E402

    spec = mujoco.MjSpec.from_file(str(G1_XML))
    _set_sim_options_like_mjlab(spec.option)
    fl = spec.worldbody.add_geom()
    fl.name = "floor"
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [0, 0, 0.05]
    fl.contype = 1
    fl.conaffinity = 1
    fl.friction = [1.0, 0.005, 0.0001]
    # mjlab FULL_COLLISION: feet condim 3 priority 1 friction 0.6, everything else condim 1
    for g in spec.geoms:
        if g.name and g.name.endswith("_collision"):
            if re.fullmatch(r"(left|right)_foot[1-7]_collision", g.name):
                g.condim = 3
                g.priority = 1
                g.friction = [0.6, 0.005, 0.0001]
            else:
                g.condim = 1

    # pose
    if pose == "urg_home":
        leg = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}
        arm = {}
    elif pose == "mjlab_bent":
        leg = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363}
        arm = {"elbow": 0.6, "shoulder_pitch": 0.2, "L_shoulder_roll": 0.2, "R_shoulder_roll": -0.2}
    else:
        raise ValueError(pose)

    # actuator table (armature, effort, mjlab Kp/Kd) per mjlab motor family
    fam = {}
    def add_fam(names, arm_, eff, mult=1):
        for n in names:
            fam[n] = (arm_ * mult, eff * mult)
    add_fam(["hip_pitch", "hip_yaw"], gc.ARMATURE_7520_14, gc.ACTUATOR_7520_14.effort_limit)
    add_fam(["hip_roll", "knee"], gc.ARMATURE_7520_22, gc.ACTUATOR_7520_22.effort_limit)
    add_fam(["ankle_pitch", "ankle_roll"], gc.ARMATURE_5020, gc.ACTUATOR_5020.effort_limit, 2)
    waist = {"waist_yaw": (gc.ARMATURE_7520_14, gc.ACTUATOR_7520_14.effort_limit),
             "waist_roll": (gc.ARMATURE_5020 * 2, gc.ACTUATOR_5020.effort_limit * 2),
             "waist_pitch": (gc.ARMATURE_5020 * 2, gc.ACTUATOR_5020.effort_limit * 2)}
    wn = 10 * 2 * math.pi
    zeta = 2.0

    joints = []
    qpos_named = {}
    welded = []
    for jn in [j.name for j in spec.joints]:
        if jn == "floating_base_joint":
            continue
        grp = _g1_group(jn)
        side = "L" if jn.startswith("left") else "R" if jn.startswith("right") else ""
        if grp:  # leg joint
            q0 = leg.get(grp, 0.0)
            arm_, eff_m = fam[grp]
            if gain == "urg":
                kp, kd = G1_URG_GAIN[grp]
                eff = G1_URG_EFFORT[grp]
                scale = G1_URG_ACTION_SCALE
            else:
                kp, kd = arm_ * wn**2, 2 * zeta * arm_ * wn
                eff = eff_m
                scale = 0.25 * eff / kp
            joints.append(Jt(jn, grp, side, kp, kd, eff, scale, q0))
            qpos_named[jn] = q0
            # rotor inertia stays the mjlab (hardware) value in BOTH gain sets
            spec.joint(jn).armature = arm_
        else:  # waist / arm / wrist: held
            base = jn.replace("left_", "").replace("right_", "").replace("_joint", "")
            q0 = 0.0
            if base == "elbow":
                q0 = arm.get("elbow", 0.0)
            elif base == "shoulder_pitch":
                q0 = arm.get("shoulder_pitch", 0.0)
            elif base == "shoulder_roll":
                q0 = arm.get(f"{side}_shoulder_roll", 0.0)
            qpos_named[jn] = q0
            if base in waist:
                arm_, eff_ = waist[base]
            elif base.startswith("wrist_pitch") or base.startswith("wrist_yaw"):
                arm_, eff_ = gc.ARMATURE_4010, gc.ACTUATOR_4010.effort_limit
            else:
                arm_, eff_ = gc.ARMATURE_5020, gc.ACTUATOR_5020.effort_limit
            spec.joint(jn).armature = arm_
            if gain == "mjlab" and upper == "pd":
                joints.append(Jt(jn, "held", side, arm_ * wn**2, 2 * zeta * arm_ * wn, eff_, 0.0,
                                 q0, scored=False))
            else:
                welded.append(jn)  # rl_gym 12-dof config has no upper-body gains; or isolation
                joints.append(Jt(jn, "held", side, 0.0, 0.0, 0.0, 0.0, q0, scored=False))

    # PD position actuators for scored joints and, for the mjlab set, held joints
    for j in joints:
        if j.group == "held" and j.name in welded:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name1 = j.name
            eq.data[:5] = [j.q0, 0, 0, 0, 0]
            eq.solref = [0.01, 1.0]
            eq.active = True
            continue
        a = spec.add_actuator()
        a.name = j.name
        a.target = j.name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = j.kp
        a.biasprm[:3] = [0, -j.kp, -j.kd]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-j.effort, j.effort]
    m = spec.compile()
    for j in joints:
        jid = m.joint(j.name).id
        j.rng = tuple(float(x) for x in m.jnt_range[jid])
        j.window = None  # unitree_rl_gym clip_actions=100 / mjlab no clip: effectively unbounded
    qpos0 = m.qpos0.copy()
    for jn, q in qpos_named.items():
        qpos0[m.jnt_qposadr[m.joint(jn).id]] = q
    qpos0[:7] = [0, 0, 0.8, 1, 0, 0, 0]
    case = Case(
        key=key, label=label, model=m, qpos0=qpos0, joints=joints,
        foot_body={"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"},
        ankle_joint={"L": "left_ankle_pitch_joint", "R": "right_ankle_pitch_joint"},
        ankle_roll_joint={"L": "left_ankle_roll_joint", "R": "right_ankle_roll_joint"},
        hip_joint={"L": "left_hip_pitch_joint", "R": "right_hip_pitch_joint"},
        hip_roll_joint={"L": "left_hip_roll_joint", "R": "right_hip_roll_joint"},
        knee_joint={"L": "left_knee_joint", "R": "right_knee_joint"},
        welded=welded, loop=False,
    )
    if gain == "urg":
        case.source = {"gains": "unitree_rl_gym g1_config.py @276801e4",
                       "effort": "g1_12dof.urdf @276801e4", "action_scale": 0.25,
                       "decimation": 4, "dt": 0.005, "armature": "mjlab hardware table (rl_gym sim uses 0)"}
        case.action_clip_note = "clip_actions=100 -> no effective clip (target may leave joint range)"
        case.notes.append("Upper body (waist, arms, wrists) welded: the 12-dof config publishes no gains for it. "
                          "Model is the mjlab 29-dof g1.xml, so mass differs from the 12-dof URDF (32.1 kg).")
    else:
        case.source = {"gains": "mjlab g1_constants.py (armature*(2 pi 10)^2, damping ratio 2)",
                       "effort": "mjlab ElectricActuator table", "action_scale": "0.25*effort/Kp",
                       "decimation": 4, "dt": 0.005}
        case.action_clip_note = "no clip in the mjlab G1 velocity task"
    case.pose_note = {"urg_home": "unitree_rl_gym default_joint_angles (hip_pitch -0.1, knee 0.3, ankle_pitch -0.2, arms 0)",
                      "mjlab_bent": "mjlab KNEES_BENT_KEYFRAME (hip_pitch -0.312, knee 0.669, ankle_pitch -0.363, elbow 0.6)"}[pose]
    return case


def _parse_env_yaml_clip():
    """Read the actions clip window from the v11a env.yaml (absolute target bounds)."""
    txt = PYG_ENV_YAML.read_text()
    i = txt.index("\nactions:")
    seg = txt[i:i + 4000]
    clip = {}
    for mm in re.finditer(r"^\s{6}(\w+): !!python/tuple\n\s+- (\S+)\n\s+- (\S+)", seg, re.M):
        clip[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
    return clip


def _assert_pose_matches_env_yaml(m, qpos0):
    """The model keyframe must equal the init joint_pos of the v11a run (env.yaml)."""
    txt = PYG_ENV_YAML.read_text()
    i = txt.index("joint_pos:\n          L_hip_pitch_joint")
    seg = txt[i:i + 2500].split("joint_vel:")[0]
    n = 0
    for mm in re.finditer(r"^\s+([LR]_\w+): (\S+)$", seg, re.M):
        jn, val = "robot/" + mm.group(1), float(mm.group(2))
        got = float(qpos0[m.jnt_qposadr[m.joint(jn).id]])
        assert abs(got - val) < 1e-6, (jn, got, val)
        n += 1
    assert n >= 20, n


def build_pyg(crank_kp: float, crank_kd: float, key: str, label: str, hipknee=(150., 6., 220., 6.),
              yaw=None) -> Case:
    """yaw = (Kp, Kd) of the hip yaw joint; None = same as hip pitch/roll (v7b, trial55)."""
    m = mujoco.MjModel.from_binary_path(str(PYG_MJB))
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    clip = _parse_env_yaml_clip()
    hip_kp, hip_kd, knee_kp, knee_kd = hipknee
    qpos0 = m.key_qpos[0].copy()
    joints = []
    for a in range(m.nu):
        jid = m.actuator_trnid[a][0]
        jn = m.joint(jid).name  # 'robot/L_hip_pitch_joint'
        short = jn.split("/")[1]
        side = short[0]
        if "crank" in short:
            grp, kp, kd, eff = "crank", crank_kp, crank_kd, 60.0
        elif "hip_yaw" in short:
            grp, kp, kd, eff = "hip_yaw", (yaw or (hip_kp, hip_kd))[0], (yaw or (hip_kp, hip_kd))[1], 60.0
        elif "hip_pitch" in short:
            grp, kp, kd, eff = "hip_pitch", hip_kp, hip_kd, 120.0
        elif "hip_roll" in short:
            grp, kp, kd, eff = "hip_roll", hip_kp, hip_kd, 120.0
        elif "knee" in short:
            grp, kp, kd, eff = "knee", knee_kp, knee_kd, 120.0
        else:
            raise RuntimeError(short)
        q0 = float(qpos0[m.jnt_qposadr[jid]])
        win = clip.get(short)
        rng = tuple(float(x) for x in m.jnt_range[jid])
        joints.append(Jt(jn, grp, side, kp, kd, eff, 0.25, q0, True, win, rng))
        # PD position actuator, flat effort limit
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, :] = 0
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[a, :] = 0
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kd
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = [-eff, eff]
    _assert_pose_matches_env_yaml(m, qpos0)
    case = Case(
        key=key, label=label, model=m, qpos0=qpos0, joints=joints,
        foot_body={"L": "robot/L_foot_link", "R": "robot/R_foot_link"},
        ankle_joint={"L": "robot/L_ankle_pitch_joint", "R": "robot/R_ankle_pitch_joint"},
        ankle_roll_joint={"L": "robot/L_ankle_roll_joint", "R": "robot/R_ankle_roll_joint"},
        hip_joint={"L": "robot/L_hip_pitch_joint", "R": "robot/R_hip_pitch_joint"},
        hip_roll_joint={"L": "robot/L_hip_roll_joint", "R": "robot/R_hip_roll_joint"},
        knee_joint={"L": "robot/L_knee_joint", "R": "robot/R_knee_joint"},
        welded=[], loop=True,
    )
    case.source = {"gains": "docs/168 + env.yaml of rough_box_main_v11a (train gains)",
                   "effort": "env.yaml (crank/yaw 60, hip pitch-roll/knee 120; flat limit, T-N curve ignored here)",
                   "action_scale": 0.25, "decimation": 4, "dt": m.opt.timestep,
                   "model": "analysis/out/stand10_v9caps_m8998_s1_model.mjb (LegOnly 23.63 kg, AB ankle)"}
    case.action_clip_note = "env.yaml actions.clip = absolute target bounds per joint"
    case.pose_note = "key 0 of the model = env.yaml init_state (PYG_INIT_BENT): hip_pitch 0.175, knee 0.35 (mirrored L/R), ankle pitch 0.175"
    case.notes.append("Model keeps the measured joint friction loss (0.52 N m hip/knee, 0.25 N m crank) and viscous damping.")
    return case


MENAGERIE = REPO / "third_party" / "mujoco_menagerie_subset"   # sparse checkout, commit d7b481ba, not in git
MENAGERIE_COMMIT = "d7b481bac63d3d600ab404ee74f7474c5efe8a67"
_GROUP_ORDER = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")

# per-robot description: how to map Menagerie joint names to our groups, and the published numbers
MENAGERIE_ROBOTS = {
    "h1": dict(
        xml="unitree_h1/h1.xml", pub="h1",
        parse=lambda n: ({"hip_yaw": "hip_yaw", "hip_roll": "hip_roll", "hip_pitch": "hip_pitch", "knee": "knee",
                          "ankle": "ankle_pitch"}.get(n.split("_", 1)[1]), "L" if n.startswith("left") else "R")
        if n.startswith(("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")) else (None, None),
        gain=lambda g: {"hip_yaw": (150, 2), "hip_roll": (150, 2), "hip_pitch": (150, 2), "knee": (200, 4),
                        "ankle_pitch": (40, 2)}[g],
        effort=lambda g: {"hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle_pitch": 40}[g],
        armature=lambda g: 0.1,            # Menagerie class default (rl_gym sim: 0)
        damping=0.0, frictionloss=0.0,
        window=None, scale=0.25, dt=0.005,
        foot={"L": "left_ankle_link", "R": "right_ankle_link"},
        joint=lambda grp, side: f"{'left' if side == 'L' else 'right'}_{'ankle' if grp == 'ankle_pitch' else grp}",
        default={"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2},
        source={"gains": "unitree_rl_gym h1_config.py @276801e4", "effort": "h1.urdf @276801e4",
                "action_scale": 0.25, "decimation": 4, "dt": 0.005,
                "armature": "Menagerie 0.1 (rl_gym sim uses 0; UNVERIFIED for hardware)"},
        clip_note="clip_actions=100 -> no effective clip",
        pose_note="unitree_rl_gym default_joint_angles (hip_pitch -0.1, knee 0.3, ankle -0.2, rest 0)",
        note="Torso, shoulders and elbows welded at 0 (the rl_gym H1 URDF has them fixed; its torso/arm gains are unused).",
    ),
    "t1": dict(
        xml="booster_t1/t1.xml", pub="t1",
        parse=lambda n: (_t1_group(n), "L" if n.startswith("Left") else "R") if n.startswith(("Left_Hip", "Left_Knee", "Left_Ankle", "Right_Hip", "Right_Knee", "Right_Ankle")) else (None, None),
        gain=lambda g: {"hip_yaw": (200, 5), "hip_roll": (200, 5), "hip_pitch": (200, 5), "knee": (200, 5),
                        "ankle_pitch": (50, 1), "ankle_roll": (50, 1)}[g],
        effort=lambda g: {"hip_pitch": 45, "hip_roll": 30, "hip_yaw": 30, "knee": 60, "ankle_pitch": 24,
                          "ankle_roll": 15}[g],
        armature=lambda g: 0.005,          # Menagerie class default (booster_gym sim: 0)
        damping=0.0, frictionloss=0.0,
        window="scale1", scale=1.0, dt=0.002,
        foot={"L": "left_foot_link", "R": "right_foot_link"},
        joint=lambda grp, side: f"{'Left' if side == 'L' else 'Right'}_" + {
            "hip_yaw": "Hip_Yaw", "hip_roll": "Hip_Roll", "hip_pitch": "Hip_Pitch", "knee": "Knee_Pitch",
            "ankle_pitch": "Ankle_Pitch", "ankle_roll": "Ankle_Roll"}[grp],
        default={"hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.25},
        source={"gains": "booster_gym envs/T1.yaml @da396a06", "effort": "T1_locomotion.urdf @da396a06 (Menagerie has ankle pitch 20)",
                "action_scale": 1.0, "decimation": 10, "dt": 0.002,
                "armature": "Menagerie 0.005 (booster_gym sim: 0; UNVERIFIED for hardware)"},
        clip_note="action_scale 1.0 and clip_actions 1.0 -> target limited to default +- 1 rad",
        pose_note="booster_gym default_joint_angles (Hip_Pitch -0.2, Knee 0.4, Ankle_Pitch -0.25, rest 0)",
        note="Waist, arms and head welded at 0 (T1_locomotion.urdf has 12 leg joints only). Joint ranges are Menagerie's.",
    ),
    "berkeley": dict(
        xml="berkeley_humanoid/berkeley_humanoid.xml", pub="berkeley",
        parse=lambda n: ({"HR": "hip_yaw", "HAA": "hip_roll", "HFE": "hip_pitch", "KFE": "knee", "FFE": "ankle_pitch",
                          "FAA": "ankle_roll"}[n[3:]], "L" if n.startswith("LL") else "R"),
        gain=lambda g: {"hip_yaw": (10, 1.5), "hip_roll": (10, 1.5), "hip_pitch": (15, 1.5), "knee": (15, 1.5),
                        "ankle_pitch": (1, 0.1), "ankle_roll": (1, 0.1)}[g],
        effort=lambda g: {"hip_yaw": 20, "hip_roll": 20, "hip_pitch": 30, "knee": 30, "ankle_pitch": 20,
                          "ankle_roll": 5}[g],
        armature=lambda g: {"hip_yaw": 6.9e-5 * 81, "hip_roll": 6.9e-5 * 81, "hip_pitch": 9.4e-5 * 81,
                            "knee": 1.5e-4 * 81, "ankle_pitch": 6.9e-5 * 81, "ankle_roll": 6.1e-6 * 81}[g],
        damping=0.02, frictionloss=None,   # friction_static per group below; dynamic 0.02 as viscous damping
        frictionloss_by=lambda g: {"hip_yaw": 0.3, "hip_roll": 0.3, "hip_pitch": 0.3, "knee": 0.8, "ankle_pitch": 1.0,
                                   "ankle_roll": 0.1}[g],
        window=None, scale=0.5, dt=0.005,
        foot={"L": "ll_faa", "R": "lr_faa"},
        joint=lambda grp, side: f"L{side}_" + {"hip_yaw": "HR", "hip_roll": "HAA", "hip_pitch": "HFE", "knee": "KFE",
                                               "ankle_pitch": "FFE", "ankle_roll": "FAA"}[grp],
        default={"LL_HR": -0.071, "LR_HR": 0.071, "LL_HAA": 0.103, "LR_HAA": -0.103, "LL_HFE": -0.463,
                 "LR_HFE": -0.463, "LL_KFE": 0.983, "LR_KFE": 0.983, "LL_FFE": -0.350, "LR_FFE": -0.350,
                 "LL_FAA": 0.126, "LR_FAA": -0.126},
        source={"gains": "isaac_berkeley_humanoid berkeley_humanoid.py @ffc7b269", "effort": "same file effort_limit",
                "action_scale": 0.5, "decimation": 4, "dt": 0.005,
                "armature": "Isaac config (6.9e-5*81 ...)", "friction": "friction_static 0.3/0.3/0.8/1.0/0.1 as frictionloss"},
        clip_note="no clip in velocity_env_cfg (scale 0.5, use_default_offset)",
        pose_note="Isaac init_state joint_pos (HFE -0.463, KFE 0.983, FFE -0.350, ...)",
        note=("Menagerie's own gains differ from the Isaac config (kfe class kp 1, ffe class kp 1): the published Isaac "
              "numbers are used. HR/HAA/HFE/KFE/FFE/FAA mapped to hip yaw/roll/pitch/knee/ankle pitch/ankle roll "
              "(HR = hip rotation, UNVERIFIED mapping). Motor speed-torque saturation of the Isaac DC-motor model is ignored."),
    ),
}


def _t1_group(n):
    for key, g in (("Hip_Yaw", "hip_yaw"), ("Hip_Roll", "hip_roll"), ("Hip_Pitch", "hip_pitch"),
                   ("Knee", "knee"), ("Ankle_Pitch", "ankle_pitch"), ("Ankle_Roll", "ankle_roll")):
        if key in n:
            return g


def build_menagerie(robot: str, key: str, label: str) -> Case:
    D = MENAGERIE_ROBOTS[robot]
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / D["xml"]))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    _set_sim_options_like_mjlab(spec.option)
    spec.option.timestep = D["dt"]
    fl = spec.worldbody.add_geom()
    fl.name = "floor"
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [0, 0, 0.05]
    fl.contype = 1
    fl.conaffinity = 1
    fl.friction = [1.0, 0.005, 0.0001]
    joints, welded, qnamed = [], [], {}
    for j in spec.joints:
        n = j.name
        if not n or j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        grp, side = D["parse"](n)
        if grp is None:
            welded.append(n)
            qnamed[n] = 0.0
            joints.append(Jt(n, "held", "", 0.0, 0.0, 0.0, 0.0, 0.0, scored=False))
            j.damping = [0.0, 0.0, 0.0]
            j.frictionloss = 0.0
            continue
        kp, kd = D["gain"](grp)
        eff = D["effort"](grp)
        dflt = D["default"]
        q0 = dflt.get(n, dflt.get(grp, 0.0)) if robot != "berkeley" else dflt[n]
        j.armature = D["armature"](grp)
        j.damping = [D["damping"], 0.0, 0.0]
        j.frictionloss = D["frictionloss_by"](grp) if D.get("frictionloss_by") else D["frictionloss"]
        qnamed[n] = q0
        joints.append(Jt(n, grp, side, float(kp), float(kd), float(eff), D["scale"], q0))
    for jt in joints:
        if jt.name in welded:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name1 = jt.name
            eq.data[:5] = [0, 0, 0, 0, 0]
            eq.solref = [0.01, 1.0]
            eq.active = True
            continue
        a = spec.add_actuator()
        a.name = jt.name
        a.target = jt.name
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = jt.kp
        a.biasprm[:3] = [0, -jt.kp, -jt.kd]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-jt.effort, jt.effort]
    m = spec.compile()
    for jt in joints:
        if jt.scored:
            jt.rng = tuple(float(x) for x in m.jnt_range[m.joint(jt.name).id])
            if D["window"] == "scale1":
                jt.window = (jt.q0 - 1.0 * D["scale"], jt.q0 + 1.0 * D["scale"])
    qpos0 = m.qpos0.copy()
    for n, q in qnamed.items():
        qpos0[m.jnt_qposadr[m.joint(n).id]] = q
    qpos0[:7] = [0, 0, 0.8, 1, 0, 0, 0]
    J = D["joint"]
    case = Case(
        key=key, label=label, model=m, qpos0=qpos0, joints=joints, foot_body=dict(D["foot"]),
        ankle_joint={s: J("ankle_pitch", s) for s in "LR"}, ankle_roll_joint={s: J("ankle_roll", s) for s in "LR"}
        if robot != "h1" else {s: None for s in "LR"},
        hip_joint={s: J("hip_pitch", s) for s in "LR"}, hip_roll_joint={s: J("hip_roll", s) for s in "LR"},
        knee_joint={s: J("knee", s) for s in "LR"}, welded=welded, loop=False,
    )
    case.source = D["source"]
    case.action_clip_note = D["clip_note"]
    case.pose_note = D["pose_note"]
    case.notes.append(D["note"])
    case.notes.append(f"Model: Menagerie {D['xml']} @{MENAGERIE_COMMIT[:8]}")
    return case


CASES = {
    "g1_a": lambda: build_g1("urg", "urg_home", "g1_a", "G1-a (rl_gym gains, rl_gym pose)"),
    "g1_b": lambda: build_g1("mjlab", "mjlab_bent", "g1_b", "G1-b (mjlab gains, mjlab bent pose)"),
    "g1_b_updpd": lambda: build_g1("mjlab", "mjlab_bent", "g1_b_updpd",
                                   "G1-b, upper body held by mjlab PD gains", upper="pd"),
    "g1_a_bent": lambda: build_g1("urg", "mjlab_bent", "g1_a_bent", "G1-a gains at mjlab bent pose"),
    "g1_b_home": lambda: build_g1("mjlab", "urg_home", "g1_b_home", "G1-b gains at rl_gym pose"),
    "h1": lambda: build_menagerie("h1", "h1", "H1 (unitree_rl_gym gains)"),
    "t1": lambda: build_menagerie("t1", "t1", "T1 (booster_gym gains)"),
    "berkeley": lambda: build_menagerie("berkeley", "berkeley", "Berkeley Humanoid (Isaac config gains)"),
    "pyg_v7b": lambda: build_pyg(22.265625, 1.4140625, "pyg_v7b", "Pygmalion v7b/v11a (crank 22.3/1.41)"),
    "pyg_trial55": lambda: build_pyg(55.0, 2.5, "pyg_trial55", "Pygmalion trial (crank 55/2.5)"),
    # docs/169 section 7 (flat recommendation) and section 8-7 (mountain recommendation)
    "pyg_rec_flat": lambda: build_pyg(55.0, 2.5, "pyg_rec_flat", "Pygmalion recommended flat (docs/169 s7)",
                                      hipknee=(120., 6., 130., 6.), yaw=(75., 4.)),
    "pyg_rec_mountain": lambda: build_pyg(60.0, 2.7, "pyg_rec_mountain", "Pygmalion recommended mountain (docs/169 s8-7)",
                                          hipknee=(135., 6., 130., 6.), yaw=(75., 4.)),
}


# ---------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------
def floor_id(m):
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            return g
    raise RuntimeError("no floor")


def robot_collision_geoms(m, fid):
    return [g for g in range(m.ngeom)
            if g != fid and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and (m.geom_contype[g] or m.geom_conaffinity[g])]


def min_floor_distance(m, d, geoms, fid):
    fromto = np.zeros(6)
    best = 1e9
    for g in geoms:
        dist = mujoco.mj_geomDistance(m, d, g, fid, 1.0, fromto)
        best = min(best, dist)
    return best


def _sole_tilt(case, d):
    """Mean pitch (rotation about world y) of the flat box soles; 0 when the foot has no box geom
    (capsule feet of G1 / H1 / Pygmalion are flat in the default pose by construction)."""
    m = case.model
    tilts = []
    for s in "LR":
        bid = m.body(case.foot_body[s]).id
        boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        if not boxes:
            continue
        g = max(boxes, key=lambda i: m.geom_size[i][0] * m.geom_size[i][1] * m.geom_size[i][2])
        ax = int(np.argmin(m.geom_size[g]))
        n = d.geom_xmat[g].reshape(3, 3)[:, ax].copy()
        if n[2] < 0:
            n = -n
        tilts.append(math.atan2(n[0], n[2]))
    return float(np.mean(tilts)) if tilts else 0.0


def prepare(case: Case):
    """Make the default pose a flat-foot pose and fit the base height so the lowest collision geom
    just touches the floor. If the published joint angles do not sum to a flat foot (T1, Berkeley),
    the whole robot is pitched about the world y axis until the soles are flat (recorded in
    case.pelvis_tilt_deg) - otherwise the robot would first have to rock onto its toes."""
    m = case.model
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    case.floor_geom = floor_id(m)
    d = mujoco.MjData(m)
    q = case.qpos0.copy()
    q[2] = 1.0
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    t = _sole_tilt(case, d)
    case.pelvis_tilt_deg = 0.0
    if abs(t) > 1e-3:
        # rotate the whole robot by R_y(-t) about the world origin
        quat = np.array([math.cos(-t / 2), 0.0, math.sin(-t / 2), 0.0])
        q[3:7] = quat
        q[0:3] = [0, 0, 1.0]
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        case.pelvis_tilt_deg = math.degrees(-t)
        case.notes.append(f"default pose is not flat-footed: robot pitched {math.degrees(-t):.2f} deg so both soles lie flat")
    geoms = robot_collision_geoms(m, case.floor_geom)
    dist = min_floor_distance(m, d, geoms, case.floor_geom)
    q[2] = q[2] - dist + 5e-4
    case.qpos_placed = q
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return case


def jqadr(m, name):
    return int(m.jnt_qposadr[m.joint(name).id])


def jdadr(m, name):
    return int(m.jnt_dofadr[m.joint(name).id])


def total_mass(m):
    return float(m.body_mass.sum())


def com_height_above_ankles(case):
    m = case.model
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    com = d.subtree_com[int(m.jnt_bodyid[0])]
    ank_z = np.mean([d.xanchor[m.joint(case.ankle_joint[s]).id][2] for s in "LR"])
    return float(com[2] - ank_z), com.copy()


def leg_length(case):
    m = case.model
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    out = []
    for s in "LR":
        h = d.xanchor[m.joint(case.hip_joint[s]).id]
        a = d.xanchor[m.joint(case.ankle_joint[s]).id]
        out.append(float(np.linalg.norm(h - a)))
    return float(np.mean(out))


# ---------------------------------------------------------------------------------------
# PD application and simulation
# ---------------------------------------------------------------------------------------
def apply_gain_scale(case, d_or_none, scale: float, only=None):
    """Scale Kp and Kd of the scored leg joints (arms / welded stay as published).
    only = set of joint groups to scale (default: all scored leg joints)."""
    m = case.model
    for a in range(m.nu):
        jn = m.joint(m.actuator_trnid[a][0]).name
        j = case.jt(jn)
        s = scale if (j.scored and (only is None or j.group in only)) else 1.0
        m.actuator_gainprm[a, 0] = j.kp * s
        m.actuator_biasprm[a, 1] = -j.kp * s
        m.actuator_biasprm[a, 2] = -j.kd * s


def act_index(case):
    m = case.model
    return {m.joint(m.actuator_trnid[a][0]).name: a for a in range(m.nu)}


class contact_mode:
    """'trained' keeps the model as built; 'rigid' = dt 0.001 and contact time constant 0.002 s,
    which makes the standing threshold independent of the soft-contact compliance (G1-a: dt 0.0025 /
    tc 0.005 gave 6.07, dt 0.001 / tc 0.002 gave 6.05) and keeps wn*dt < 1.2 for the stiffest scaled
    joint (position stiffness is integrated explicitly, so a larger dt would add numerical instability)."""

    def __init__(self, case, mode):
        self.m, self.mode = case.model, mode

    def __enter__(self):
        m = self.m
        self.saved = (m.opt.timestep, m.geom_solref.copy())
        if self.mode == "rigid":
            m.opt.timestep = 0.001
            m.geom_solref[:] = [0.002, 1.0]
        return self

    def __exit__(self, *a):
        self.m.opt.timestep, self.m.geom_solref[:] = self.saved


def run_stand(case, scale=1.0, push_vx=0.0, T=5.0, push_t=1.0, collect=False, only=None):
    m = case.model
    apply_gain_scale(case, None, scale, only)
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    ai = act_index(case)
    for jn, a in ai.items():
        d.ctrl[a] = case.jt(jn).q0
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    foot_bodies = {m.body(n).id for n in case.foot_body.values()}
    nsteps = int(round(T / m.opt.timestep))
    push_step = int(round(push_t / m.opt.timestep))
    ok = True
    fail = None
    zs = np.zeros(nsteps)
    forces = np.zeros((nsteps, m.nu))
    qs = np.zeros((nsteps, m.nu))
    adr = [int(m.jnt_qposadr[m.actuator_trnid[a][0]]) for a in range(m.nu)]
    for k in range(nsteps):
        if push_vx and k == push_step:
            d.qvel[0] += push_vx
        mujoco.mj_step(m, d)
        zs[k] = d.qpos[2]
        forces[k] = d.actuator_force
        qs[k] = d.qpos[adr]
        if ok:
            if d.qpos[2] < 0.7 * z0 or not np.isfinite(d.qpos[2]):
                ok, fail = False, ("height", k * m.opt.timestep)
            else:
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if g1 == case.floor_geom or g2 == case.floor_geom:
                        other = g2 if g1 == case.floor_geom else g1
                        if m.geom_bodyid[other] not in foot_bodies:
                            ok, fail = False, ("body_contact", k * m.opt.timestep)
                            break
        if not ok and not collect:
            break
    apply_gain_scale(case, None, 1.0)
    res = {"stood": ok, "fail": fail, "z0": z0, "z_end": float(zs[:k + 1][-int(1.0 / m.opt.timestep):].mean())}
    if collect:
        w = int(1.0 / m.opt.timestep)
        res["force_mean"] = forces[-w:].mean(0)
        res["q_mean"] = qs[-w:].mean(0)
        res["q_std"] = qs[-w:].std(0)
        res["z_mean"] = float(zs[-w:].mean())
    return res


STAND_GRID = (0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def min_standing_scale(case, push_vx=0.0, grid=STAND_GRID, refine=8, only=None, mode="rigid"):
    """Smallest Kp/Kd scale that stands 5 s. The grid goes below AND above 1.0 because the
    published gains may not stand at all (then the scale needed is > 1)."""
    with contact_mode(case, mode):
        return _min_standing_scale(case, push_vx, grid, refine, only)


def _min_standing_scale(case, push_vx, grid, refine, only):
    out = {s: run_stand(case, s, push_vx, only=only)["stood"] for s in grid}
    res = {"grid": {str(k): v for k, v in out.items()}}
    passing = [s for s in grid if out[s]]
    if not passing:
        res.update(min_grid=None, min_refined=None, margin_grid=None, margin_refined=None,
                   note="no grid value stands (max %.0f)" % max(grid))
        return res
    smin = min(passing)
    below = [s for s in grid if s < smin]
    res["min_grid"] = smin
    res["non_monotone"] = any((not out[s]) for s in grid if s > smin)  # a fail above the smallest pass
    if below:
        lo_s, hi_s = below[-1], smin   # lo fails, hi passes
        for _ in range(refine):
            mid = 0.5 * (lo_s + hi_s)
            if run_stand(case, mid, push_vx, only=only)["stood"]:
                hi_s = mid
            else:
                lo_s = mid
        res["min_refined"] = hi_s
    else:
        res["min_refined"] = smin
        res["floor_hit"] = True        # stands even at the lowest scale tried
    res["margin_grid"] = 1.0 / res["min_grid"]
    res["margin_refined"] = 1.0 / res["min_refined"]
    return res


# ---------------------------------------------------------------------------------------
# Linkage ankle: crank -> ankle joint equivalent
# ---------------------------------------------------------------------------------------
def eq_rows(m, d):
    idx = np.where(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
    J = d.efc_J[:d.nefc * m.nv].reshape(d.nefc, m.nv)[idx] if d.nefc else np.zeros((0, m.nv))
    pos = d.efc_pos[idx]
    return idx, J, pos


def ankle_jacobian(case):
    """Return per side {crank_joint_name: [d q_crank / d ankle_pitch, d q_crank / d ankle_roll]}
    from the loop-closure tangent at the default pose (empty for serial ankles)."""
    m = case.model
    if not case.loop:
        return {}
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    _, Jq, _ = eq_rows(m, d)
    out = {}
    for s in "LR":
        ap = jdadr(m, case.ankle_joint[s])
        ar = jdadr(m, case.ankle_roll_joint[s])
        # dofs of this leg's loop: crank A/B + rod u1/u2 joints
        names = [n for n in (m.joint(i).name for i in range(m.njnt))
                 if re.search(rf"/{s}_(crank_[AB]_joint|rod_[AB]_u[12])$", n)]
        cols = [jdadr(m, n) for n in names]
        res = {}
        for tag, drive in (("pitch", ap), ("roll", ar)):
            rhs = -Jq[:, drive]
            sol, *_ = np.linalg.lstsq(Jq[:, cols], rhs, rcond=None)
            resid = float(np.abs(Jq[:, cols] @ sol - rhs).max())
            assert resid < 1e-6, (s, tag, resid)
            for n, v in zip(names, sol):
                if "crank" in n:
                    res.setdefault(n, [0.0, 0.0])[0 if tag == "pitch" else 1] = float(v)
        out[s] = res
    return out


def ankle_equivalent(case):
    """Joint-equivalent ankle stiffness (pitch, roll) per side; for serial ankles read directly."""
    out = {}
    if case.loop:
        J = ankle_jacobian(case)
        for s in "LR":
            Kp = np.zeros((2, 2))
            Kd = np.zeros((2, 2))
            tau_max = np.zeros(2)
            for n, row in J[s].items():
                j = case.jt(n)
                r = np.array(row)
                Kp += j.kp * np.outer(r, r)
                Kd += j.kd * np.outer(r, r)
                tau_max += j.effort * np.abs(r)
            out[s] = {"Kp_pitch": Kp[0, 0], "Kp_roll": Kp[1, 1], "Kp_cross": Kp[0, 1],
                      "Kd_pitch": Kd[0, 0], "Kd_roll": Kd[1, 1],
                      "effort_pitch": tau_max[0], "effort_roll": tau_max[1], "J": J[s]}
    else:
        for s in "LR":
            jp = case.jt(case.ankle_joint[s])
            jr = case.jt(case.ankle_roll_joint[s]) if case.ankle_roll_joint[s] else None
            out[s] = {"Kp_pitch": jp.kp, "Kp_roll": jr.kp if jr else 0.0, "Kd_pitch": jp.kd,
                      "Kd_roll": jr.kd if jr else 0.0, "effort_pitch": jp.effort,
                      "effort_roll": jr.effort if jr else 0.0}
    return out


# ---------------------------------------------------------------------------------------
# A4  natural frequency and damping ratio
# ---------------------------------------------------------------------------------------
def effective_inertias(case, mode):
    """Joint-space effective inertia seen by every actuated joint.
    mode 'swing': base fixed, all OTHER actuated joints locked, passive joints (ankle, rod hinges)
                  and loop constraints active  -> the mass-matrix diagonal for a leg in the air.
    mode 'stand': both feet pinned to the floor (6 rows each), the loops active, every other joint free
                  -> the inertia the joint feels while the body is carried by the legs.
    I_i = 1 / (A_c)_ii with A = M^-1 and A_c = A - A J^T (J A J^T)^+ J A."""
    m = case.model
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    nv = m.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, d, M)
    A = np.linalg.inv(M)
    _, Jeq, _ = eq_rows(m, d)
    ai = act_index(case)
    dof_of = {n: jdadr(m, n) for n in ai}
    out = {}

    def inv_inertia(J, i):
        if J.shape[0] == 0:
            return 1.0 / A[i, i]
        JA = J @ A
        S = JA @ J.T
        Ac = A - JA.T @ np.linalg.pinv(S, rcond=1e-10) @ JA
        return 1.0 / Ac[i, i]

    if mode == "stand":
        rows = [Jeq]
        for s in "LR":
            jp = np.zeros((3, nv))
            jr = np.zeros((3, nv))
            mujoco.mj_jacBody(m, d, jp, jr, m.body(case.foot_body[s]).id)
            rows += [jp, jr]
        J = np.vstack(rows)
        for n, i in dof_of.items():
            out[n] = inv_inertia(J, i)
    else:
        base = np.zeros((6, nv))
        base[np.arange(6), np.arange(6)] = 1.0
        for n, i in dof_of.items():
            locks = []
            for n2, i2 in dof_of.items():
                if n2 != n:
                    r = np.zeros((1, nv))
                    r[0, i2] = 1.0
                    locks.append(r)
            J = np.vstack([Jeq, base] + locks)
            out[n] = inv_inertia(J, i)
    return out


def dynamics_table(case):
    m = case.model
    I_sw = effective_inertias(case, "swing")
    I_st = effective_inertias(case, "stand")
    rows = {}
    for j in case.joints:
        if not j.scored:
            continue
        arm = float(m.dof_armature[jdadr(m, j.name)])
        r = {"armature": arm, "I_swing": I_sw[j.name], "I_stand": I_st[j.name]}
        for tag, I in (("rotor", arm), ("swing", I_sw[j.name]), ("stand", I_st[j.name])):
            r[f"wn_{tag}_hz"] = math.sqrt(j.kp / I) / (2 * math.pi)
            r[f"zeta_{tag}"] = j.kd / (2 * math.sqrt(j.kp * I))
        rows[j.name] = r
    return rows


# ---------------------------------------------------------------------------------------
# A5  rule-of-thumb ratios
# ---------------------------------------------------------------------------------------
def ratio_table(case, m_tot, L):
    rows = {}
    for j in case.joints:
        if not j.scored:
            continue
        half = 0.5 * (j.rng[1] - j.rng[0]) if j.rng and j.rng[1] > j.rng[0] else float("nan")
        rows[j.name] = {
            "kp": j.kp, "kd": j.kd, "effort": j.effort, "action_scale": j.scale,
            "tau_at_action1": j.kp * j.scale,
            "pct_effort_at_action1": 100 * j.kp * j.scale / j.effort,
            "kp_halfrange_over_effort": j.kp * half / j.effort,
            "kd_over_kp": j.kd / j.kp,
            "kp_per_kg": j.kp / m_tot,
            "kp_over_mgL": j.kp / (m_tot * G * L),
            "window_halfwidth_rad": (None if j.window is None else
                                     [j.window[0] - j.q0, j.window[1] - j.q0]),
        }
    return rows


# ---------------------------------------------------------------------------------------
# B1  quasi-static single-leg support
# ---------------------------------------------------------------------------------------
def _loop_dofs(case, side):
    m = case.model
    names = [m.joint(i).name for i in range(m.njnt)
             if re.search(rf"/{side}_(crank_[AB]_joint|rod_[AB]_u[12])$", m.joint(i).name)]
    return names


def close_loop(case, d, side, tol=1e-10, iters=30):
    """Gauss-Newton: move the crank / rod hinges of one leg until the loop equality residual vanishes
    (ankle pitch / roll are given). No-op for serial ankles."""
    if not case.loop:
        return
    m = case.model
    names = _loop_dofs(case, side)
    cols = [jdadr(m, n) for n in names]
    qadr = [jqadr(m, n) for n in names]
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        _, J, r = eq_rows(m, d)
        if np.abs(r).max() < tol:
            return
        dq, *_ = np.linalg.lstsq(J[:, cols], -r, rcond=None)
        for qa, v in zip(qadr, dq):
            d.qpos[qa] += v
    mujoco.mj_forward(m, d)
    _, J, r = eq_rows(m, d)
    assert np.abs(r).max() < 1e-6, f"loop closure failed {np.abs(r).max()}"


def _place_base(case, d, foot_body_id, p_t, R_t):
    """Set the free-joint pose so the stance foot sits exactly at (p_t, R_t)."""
    m = case.model
    d.qpos[0:3] = 0
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    Rf = d.xmat[foot_body_id].reshape(3, 3).copy()
    pf = d.xpos[foot_body_id].copy()
    Rb = R_t @ Rf.T
    pb = p_t - Rb @ pf
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rb.flatten())
    d.qpos[0:3] = pb
    d.qpos[3:7] = quat
    mujoco.mj_forward(m, d)
    return Rb


def _foot_center(case, d, side):
    m = case.model
    bid = m.body(case.foot_body[side]).id
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and (m.geom_contype[i] or m.geom_conaffinity[i])]
    return np.mean([d.geom_xpos[i] for i in g], axis=0)


def single_leg_pose(case, stance, lift_scale=1.0, flip=False):
    """Default pose with the swing leg flexed off the ground and the stance leg leaned (hip roll +
    ankle roll) until the whole-body centre of mass is above the stance foot centre and the pelvis
    is level. Returns (MjData at the pose, info)."""
    m = case.model
    sw = "R" if stance == "L" else "L"
    d0 = mujoco.MjData(m)
    d0.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d0)
    fid = m.body(case.foot_body[stance]).id
    p_t = d0.xpos[fid].copy()
    R_t = d0.xmat[fid].reshape(3, 3).copy()
    centre0 = _foot_center(case, d0, stance)
    q_default = case.qpos_placed.copy()

    def build(ph, pa):
        d = mujoco.MjData(m)
        d.qpos[:] = q_default
        # swing leg flexion
        hj, kj = case.jt(case.hip_joint[sw]), case.jt(case.knee_joint[sw])
        sg = -1.0 if flip else 1.0
        d.qpos[jqadr(m, hj.name)] = hj.q0 + sg * math.copysign(0.35 * lift_scale, hj.q0)
        d.qpos[jqadr(m, kj.name)] = kj.q0 + sg * math.copysign(0.70 * lift_scale, kj.q0)
        # stance lean
        hr, ar = case.hip_roll_joint[stance], case.ankle_roll_joint[stance]
        d.qpos[jqadr(m, hr)] = case.jt(hr).q0 + ph
        if ar is None:
            pass
        elif case.loop:
            d.qpos[jqadr(m, ar)] = d0.qpos[jqadr(m, ar)] + pa
            close_loop(case, d, stance)
        else:
            d.qpos[jqadr(m, ar)] = case.jt(ar).q0 + pa
        Rb = _place_base(case, d, fid, p_t, R_t)
        com = d.subtree_com[int(m.jnt_bodyid[0])]
        roll = math.atan2(Rb[2, 1], Rb[2, 2])
        return d, np.array([com[1] - centre0[1], roll])

    x = np.zeros(2)
    free = [0, 1] if case.ankle_roll_joint[stance] else [0]   # H1 has no ankle roll: pelvis stays tilted
    for _ in range(25):
        d, r = build(*x)
        if np.abs(r[free]).max() < 1e-7:
            break
        Jn = np.zeros((len(free), len(free)))
        eps = 1e-4
        for c, k in enumerate(free):
            xp = x.copy()
            xp[k] += eps
            Jn[:, c] = ((build(*xp)[1] - r))[free] / eps
        x[free] = x[free] - np.linalg.solve(Jn, r[free])
    d, r = build(*x)
    info = {"hip_roll_lean_rad": float(x[0]), "ankle_roll_lean_rad": float(x[1]),
            "com_y_error_m": float(r[0]), "pelvis_roll_rad": float(r[1])}
    # swing clearance
    fl = case.floor_geom
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(case.foot_body[sw]).id
             and (m.geom_contype[g] or m.geom_conaffinity[g])]
    info["swing_clearance_m"] = float(min_floor_distance(m, d, geoms, fl))
    return d, info


def solve_statics(case, d):
    """Joint torques that hold the pose against gravity with the stance foot fixed (a free 6D wrench
    at the foot body), solved from qfrc_bias; loop / weld constraint forces are extra unknowns."""
    m = case.model
    mujoco.mj_forward(m, d)
    nv = m.nv
    ai = act_index(case)
    cols, names = [], []
    for n, a in ai.items():
        c = np.zeros(nv)
        c[jdadr(m, n)] = 1.0
        cols.append(c)
        names.append(n)
    ntau = len(cols)
    _, Jeq, _ = eq_rows(m, d)
    return ntau, names, cols, Jeq


def single_leg_torques(case, stance):
    m = case.model
    best = None
    for lift in (1.0, 1.5, 2.0, 2.5):
        for flip in (False, True):
            d, info = single_leg_pose(case, stance, lift, flip)
            if info["swing_clearance_m"] > 0.02:
                best = (d, info)
                break
        if best:
            break
    if best is None:
        raise RuntimeError("no swing-leg lift gives 2 cm clearance")
    d, info = best
    ntau, names, cols, Jeq = solve_statics(case, d)
    nv = m.nv
    fid = m.body(case.foot_body[stance]).id
    jp = np.zeros((3, nv))
    jr = np.zeros((3, nv))
    mujoco.mj_jacBody(m, d, jp, jr, fid)
    Jf = np.vstack([jp, jr])
    Acols = np.column_stack(cols + [Jf.T] + ([Jeq.T] if Jeq.shape[0] else []))
    b = d.qfrc_bias.copy()
    x, *_ = np.linalg.lstsq(Acols, b, rcond=None)
    resid = float(np.abs(Acols @ x - b).max())
    tau = {n: float(x[i]) for i, n in enumerate(names)}
    # passive dofs must carry no torque: residual check covers it
    wrench = x[ntau:ntau + 6]
    q = {n: float(d.qpos[jqadr(m, n)]) for n in names}
    viol = []
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[jid]:
            v = float(d.qpos[m.jnt_qposadr[jid]])
            lo, hi = m.jnt_range[jid]
            if v < lo - 1e-3 or v > hi + 1e-3:
                viol.append([m.joint(jid).name, v, float(lo), float(hi)])
    info["range_violations"] = viol
    return {"tau": tau, "q": q, "info": info, "resid": resid,
            "foot_force_z": float(wrench[2]), "weight_N": total_mass(m) * G}


def single_leg_table(case):
    out = {}
    for stance in "LR":
        r = single_leg_torques(case, stance)
        rows = {}
        for j in case.joints:
            if not j.scored:
                continue
            tau = r["tau"][j.name]
            e = tau / j.kp
            q = r["q"][j.name]
            a_req = (q + e - j.q0) / j.scale
            row = {"role": "stance" if j.side == stance else "swing", "tau": tau,
                   "pct_effort": 100 * abs(tau) / j.effort, "err_deg": math.degrees(e),
                   "action_units_needed": a_req, "target_abs": q + e}
            if j.window is not None:
                lo, hi = j.window
                row["target_headroom_rad"] = min(q + e - lo, hi - (q + e))
            rows[j.name] = row
        out[stance] = {"rows": rows, "info": r["info"], "resid": r["resid"], "foot_force_z": r["foot_force_z"],
                       "weight_N": r["weight_N"]}
    return out


# ---------------------------------------------------------------------------------------
# A1  steady torque in the default pose (elastic equilibrium with the published gains)
# ---------------------------------------------------------------------------------------
def elastic_hold(case):
    """Both feet pinned, PD targets = default pose, published Kp. The statically indeterminate load
    split is resolved the way the springs resolve it: minimise sum(tau_i^2 / Kp_i) subject to static
    equilibrium (stationarity of the elastic energy). Scale-free: uniform gain scaling does not
    change tau. Returns torque, error (= tau/Kp), the linearised base height drop and the residual."""
    m = case.model
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    nv = m.nv
    ai = act_index(case)
    names = list(ai)
    cols = []
    w = []
    for n in names:
        c = np.zeros(nv)
        c[jdadr(m, n)] = 1.0
        cols.append(c)
        j = case.jt(n)
        w.append(1.0 / j.kp if j.kp > 0 else 1e-6)
    Jf = []
    for s in "LR":
        jp = np.zeros((3, nv))
        jr = np.zeros((3, nv))
        mujoco.mj_jacBody(m, d, jp, jr, m.body(case.foot_body[s]).id)
        Jf.append(np.vstack([jp, jr]))
    _, Jeq, _ = eq_rows(m, d)
    Jfeet = np.vstack(Jf)
    nt = len(names)
    Anu = np.column_stack([Jfeet.T] + ([Jeq.T] if Jeq.shape[0] else []))
    A = np.column_stack(cols + [Anu])
    b = d.qfrc_bias.copy()
    n_all = A.shape[1]
    C = np.zeros(n_all)
    C[:nt] = w
    C[nt:] = 1e-12
    K = np.block([[2 * np.diag(C), A.T], [A, np.zeros((nv, nv))]])
    rhs = np.concatenate([np.zeros(n_all), b])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=1e-14)
    x = sol[:n_all]
    resid = float(np.abs(A @ x - b).max())
    tau = {n: float(x[i]) for i, n in enumerate(names)}
    # linearised base drop: actuated dofs move by -tau/Kp, passive+base dofs follow with feet pinned
    dq = np.zeros(nv)
    act_dofs = []
    for n in names:
        j = case.jt(n)
        dq[jdadr(m, n)] = -tau[n] / j.kp if j.kp > 0 else 0.0
        act_dofs.append(jdadr(m, n))
    free = [i for i in range(nv) if i not in act_dofs]
    Jc = np.vstack([Jfeet] + ([Jeq] if Jeq.shape[0] else []))
    rhs2 = -Jc[:, act_dofs] @ dq[act_dofs]
    sol2, *_ = np.linalg.lstsq(Jc[:, free], rhs2, rcond=None)
    dq[free] = sol2
    comp = float(np.abs(Jc @ dq).max())
    return {"tau": tau, "resid": resid, "base_drop_m": float(-dq[2]), "compat_resid": comp}


def foot_extent_x(case):
    """World x extent (min, max) of the stance-foot collision geoms in the placed default pose and the
    mean ankle-axis x. Used to see where the centre of pressure may sit (heel .. toe)."""
    m = case.model
    d = mujoco.MjData(m)
    d.qpos[:] = case.qpos_placed
    mujoco.mj_forward(m, d)
    xs = []
    for s in "LR":
        bid = m.body(case.foot_body[s]).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and (m.geom_contype[g] or m.geom_conaffinity[g]):
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                R = d.geom_xmat[g].reshape(3, 3)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            xs.append((d.geom_xpos[g] + R @ (c + np.array([sx, sy, sz]) * h))[0])
    ank = float(np.mean([d.xanchor[m.joint(case.ankle_joint[s]).id][0] for s in "LR"]))
    com_x = float(d.subtree_com[int(m.jnt_bodyid[0])][0])
    return {"heel_x": float(min(xs)), "toe_x": float(max(xs)), "ankle_x": ank, "com_x": com_x}


def standing_prediction(case, K_ankle_sum, mgh):
    """Two reasons a PD-held default pose topples, from the pinned-feet statics:
    (1) tipping:  K_ankle < m g h  -> inverted-pendulum instability, scale s_tip = m g h / K
    (2) foot edge: the centre of pressure moves from CoM-x to  e0 / (1 - mgh/K)  (e0 = CoM-x minus ankle-x),
        and must stay under the sole -> K* = mgh / (1 - e0/d_edge)."""
    f = foot_extent_x(case)
    e0 = f["com_x"] - f["ankle_x"]
    d_edge = (f["toe_x"] if e0 >= 0 else f["heel_x"]) - f["ankle_x"]
    out = dict(f, e0=e0, d_edge=d_edge, s_tip=mgh / K_ankle_sum)
    h = mgh / (total_mass(case.model) * G)
    out["capture_offset_push0p3"] = e0 + math.copysign(0.3 * math.sqrt(h / G), e0 if e0 else 1.0)
    out["capture_fits_foot"] = abs(out["capture_offset_push0p3"]) < abs(d_edge)
    if abs(e0) < abs(d_edge):
        out["s_edge"] = mgh / (1 - e0 / d_edge) / K_ankle_sum
    else:
        out["s_edge"] = float("inf")
    return out


# ---------------------------------------------------------------------------------------
# B2  human gait peak moments scaled to the robot
# ---------------------------------------------------------------------------------------
def human_gait_table(case, m_tot, L):
    out = {}
    eq = ankle_equivalent(case)
    for variant, tab in HUMAN_PEAK_NM_PER_KG.items():
        rows = {}
        for mode, factor in (("mass", m_tot), ("mass_x_length", m_tot * (L / HUMAN_LEG_LENGTH_M))):
            # sagittal / frontal peaks as joint torques on the robot (mass scaling: N m/kg * robot kg;
            # mass_x_length: additionally scale by robot hip-ankle length / 0.9 m human)
            for j in case.joints:
                if not j.scored or j.group not in tab:
                    continue
                tau = tab[j.group] * factor
                rows.setdefault(mode, {})[j.name] = {
                    "tau": tau, "err_deg": math.degrees(tau / j.kp),
                    "action_units": tau / j.kp / j.scale,
                    "pct_effort": 100 * tau / j.effort,
                }
            if case.loop:   # ankle moments are applied at the ankle and shared by the two cranks
                tp = tab["ankle_pitch"] * factor
                tr = tab["ankle_roll"] * factor
                for s in "LR":
                    Jm = np.array([eq[s]["J"][n] for n in sorted(eq[s]["J"])])   # rows crank A,B ; cols pitch,roll
                    tc = np.linalg.solve(Jm.T, np.array([tp, tr]))
                    for n, t in zip(sorted(eq[s]["J"]), tc):
                        j = case.jt(n)
                        rows.setdefault(mode, {})[n] = {
                            "tau": float(t), "err_deg": math.degrees(t / j.kp),
                            "action_units": float(t / j.kp / j.scale), "pct_effort": float(100 * abs(t) / j.effort),
                        }
        out[variant] = rows
    return out


# ---------------------------------------------------------------------------------------
# C  which rule explains the gains
# ---------------------------------------------------------------------------------------
def _group_avg(case, per_joint, key=lambda v: v):
    acc = {}
    for j in case.joints:
        if j.scored and j.name in per_joint:
            acc.setdefault(j.group, []).append(key(per_joint[j.name]))
    return {g: float(np.mean(v)) for g, v in acc.items()}


def inference(case, ratios, dyn, hold, single):
    groups = [g for g in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll", "crank")
              if any(j.group == g for j in case.joints)]
    kp = _group_avg(case, ratios, lambda v: v["kp"])
    drv = {
        "effort limit": _group_avg(case, ratios, lambda v: v["effort"]),
        "default-pose gravity torque": _group_avg(case, hold, lambda v: abs(v)),
        "armature (rotor)": _group_avg(case, dyn, lambda v: v["armature"]),
        "link inertia, leg in air": _group_avg(case, dyn, lambda v: v["I_swing"]),
        "link inertia, standing": _group_avg(case, dyn, lambda v: v["I_stand"]),
    }
    stance = {}
    for st in "LR":
        for n, r in single[st]["rows"].items():
            if r["role"] == "stance":
                stance[n] = abs(r["tau"])
    drv["single-leg stance torque"] = _group_avg(case, stance, lambda v: v)
    res = {}
    kv = np.array([kp[g] for g in groups])
    for name, dd in drv.items():
        x = np.array([dd.get(g, np.nan) for g in groups])
        ok = np.isfinite(x) & (x > 0) & (kv > 0)
        entry = {"n": int(ok.sum())}
        if ok.sum() >= 3:
            entry["pearson_r"] = float(np.corrcoef(kv[ok], x[ok])[0, 1])
            entry["pearson_r_loglog"] = float(np.corrcoef(np.log(kv[ok]), np.log(x[ok]))[0, 1])
            ratio = kv[ok] / x[ok]
            entry["ratio_cv"] = float(ratio.std() / ratio.mean())
            entry["ratio_mean"] = float(ratio.mean())
        res[name] = entry
    # constancy of the rule values themselves
    for nm, key in (("action scale x Kp / effort (percent)", "pct_effort_at_action1"),
                    ("Kp x half range / effort", "kp_halfrange_over_effort")):
        gv = _group_avg(case, ratios, lambda v: v[key])
        x = np.array([gv[g] for g in groups if g in gv])
        x = x[np.isfinite(x)]
        res[nm + " (constancy)"] = {"mean": float(x.mean()), "cv": float(x.std() / x.mean()), "n": len(x)}
    res["groups"] = groups
    res["kp_by_group"] = kp
    return res


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------
def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run_case(case, with_trained_contact=True):
    prepare(case)
    m = case.model
    m_tot = total_mass(m)
    h, com = com_height_above_ankles(case)
    L = leg_length(case)
    import hashlib, subprocess
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    res = {"key": case.key, "label": case.label, "source": case.source, "notes": case.notes,
           "repo_head": head, "mujoco": mujoco.__version__,
           "model_sha256": (hashlib.sha256(PYG_MJB.read_bytes()).hexdigest() if case.loop else
                            hashlib.sha256(G1_XML.read_bytes()).hexdigest()),
           "pose_note": case.pose_note, "action_clip_note": case.action_clip_note,
           "pelvis_tilt_deg": case.pelvis_tilt_deg, "mass_kg": m_tot, "com_height_above_ankle_m": h, "leg_length_m": L, "dt": m.opt.timestep}
    # A1
    hold = elastic_hold(case)
    res["A1_elastic_hold"] = hold
    res["A1_rows"] = {j.name: {"group": j.group, "kp": j.kp, "effort": j.effort,
                               "tau": hold["tau"][j.name], "pct_effort": 100 * abs(hold["tau"][j.name]) / j.effort,
                               "err_deg": math.degrees(hold["tau"][j.name] / j.kp)}
                      for j in case.joints if j.scored}
    # A2
    a2 = {"no_push": min_standing_scale(case, 0.0), "push_0p3": min_standing_scale(case, 0.3)}
    a2["ankle_only_no_push"] = min_standing_scale(
        case, 0.0, only={"ankle_pitch", "ankle_roll", "crank"})
    a2["hip_knee_only_no_push"] = min_standing_scale(
        case, 0.0, only={"hip_pitch", "hip_roll", "hip_yaw", "knee"})
    if with_trained_contact:
        a2["no_push_trained_contact"] = min_standing_scale(case, 0.0, mode="trained")
    res["A2"] = a2
    # sim cross-check of the steady torques at a gain scale that stands
    smin = a2["no_push"].get("min_refined")
    if smin:
        s_chk = max(1.0, 1.5 * smin)
        with contact_mode(case, "rigid"):
            r = run_stand(case, s_chk, collect=True)
        ai = act_index(case)
        res["A1_sim_check"] = {"scale": s_chk, "stood": r["stood"],
                               "tau_sim": {n: float(r["force_mean"][a]) for n, a in ai.items() if case.jt(n).scored},
                               "z_drop_m": float(r["z0"] - r["z_mean"]) if r["stood"] else None}
    # A3
    eq = ankle_equivalent(case)
    K_sum = sum(eq[s]["Kp_pitch"] for s in "LR")
    mgh = m_tot * G * h
    res["A3"] = {"K_ankle_pitch_sum": K_sum, "m_g_h": mgh, "ratio": K_sum / mgh,
                 "K_ankle_roll_sum": sum(eq[s]["Kp_roll"] for s in "LR"),
                 "ankle_effort_pitch_sum": sum(eq[s]["effort_pitch"] for s in "LR"),
                 "ankle_equivalent": {s: {k: v for k, v in eq[s].items() if k != "J"} for s in "LR"}}
    res["A3"]["standing_prediction"] = standing_prediction(case, K_sum, mgh)
    # A4 / A5
    dyn = dynamics_table(case)
    res["A4"] = dyn
    ratios = ratio_table(case, m_tot, L)
    res["A5"] = ratios
    # B
    single = single_leg_table(case)
    res["B1"] = single
    res["B2"] = human_gait_table(case, m_tot, L)
    # C
    res["C"] = inference(case, ratios, dyn, hold["tau"], single)
    return res


# ---------------------------------------------------------------------------------------
# Figures (English labels only)  -  python compare_robot_gains.py --plots
# ---------------------------------------------------------------------------------------
PLOT_ORDER = ["g1_a", "g1_b", "h1", "t1", "berkeley", "pyg_v7b", "pyg_trial55"]
PLOT_LABEL = {"g1_a": "G1-a\n(rl_gym)", "g1_b": "G1-b\n(mjlab)", "h1": "H1", "t1": "T1", "berkeley": "Berkeley",
              "pyg_v7b": "Pygmalion\ncrank 22.3", "pyg_trial55": "Pygmalion\ncrank 55"}
GROUPS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll", "crank"]


def _load(keys):
    return {k: json.loads((RESULTS / f"{k}.json").read_text()) for k in keys if (RESULTS / f"{k}.json").exists()}


def _group_vals(r, table, field, groups=GROUPS):
    acc = {}
    for n, v in table.items():
        g = r["A5"][n]["kp"] and next((x for x in groups if f"_{x}" in n.lower() or n.lower().endswith(x)), None)
        acc.setdefault(g, []).append(v[field])
    return {g: float(np.mean(v)) for g, v in acc.items() if g}


def make_plots(outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    R = _load(PLOT_ORDER)
    keys = [k for k in PLOT_ORDER if k in R]
    lab = [PLOT_LABEL[k] for k in keys]
    x = np.arange(len(keys))
    col = {"all": "#1b6ca8", "ankle": "#e08a1e", "tip": "#2a9d4b", "soft": "#7f7f7f", "push": "#b3392f"}

    # ---- fig 1: PD-only standing multiplier
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.16
    cap = 40.0
    def val(r, key, field="min_refined"):
        v = r["A2"][key].get(field)
        return cap if v is None else v
    series = [("all", "all leg joints scaled (rigid contact)", lambda r: val(r, "no_push")),
              ("ankle", "ankle only scaled", lambda r: val(r, "ankle_only_no_push")),
              ("tip", "analytic 1/(K_ankle / m g h)", lambda r: r["A3"]["standing_prediction"]["s_tip"]),
              ("soft", "all scaled, soft default contact", lambda r: val(r, "no_push_trained_contact")),
              ("push", "all scaled, 0.3 m/s push", lambda r: val(r, "push_0p3"))]
    for i, (c, name, f) in enumerate(series):
        vals = [f(R[k]) for k in keys]
        bars = ax.bar(x + (i - 2) * w, vals, w, color=col[c], label=name, hatch=None)
        for b, v in zip(bars, vals):
            if v >= cap:
                b.set_hatch("///")
                b.set_alpha(0.45)
                ax.text(b.get_x() + b.get_width() / 2, cap * 1.03, "no", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(-0.45, 1.06, "published gains (x1)", ha="left", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 60)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("Kp, Kd multiplier needed so PD alone stands 5 s")
    ax.set_title("PD-only standing in the default pose: every published gain set needs a multiplier above 1\n"
                 "(hatched = no multiplier up to 32 stands)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(outdir / "2026-10-14_pd_only_stand_multiplier.png", dpi=140)
    plt.close(fig)

    # ---- fig 2: ankle stiffness / mgh
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = [R[k]["A3"]["ratio"] for k in keys]
    ax.bar(x, vals, 0.55, color=col["all"])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.text(-0.45, 1.03, "toppling stiffness m g h (ratio 1)", fontsize=8)
    ax.axhspan(0.44, 0.93, color="#2a9d4b", alpha=0.18)
    ax.text(-0.45, 0.86, "human passive ankle stiffness 0.44-0.93 of m g h (Loram 2002, Casadio 2005, Vlutters 2015; docs/169)",
            ha="left", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("sum of both ankle pitch stiffnesses / (m g h)")
    ax.set_title("Ankle pitch stiffness relative to the toppling stiffness m g h (default pose)")
    fig.tight_layout()
    fig.savefig(outdir / "2026-10-14_ankle_stiffness_over_mgh.png", dpi=140)
    plt.close(fig)

    # ---- fig 3: required error vs action window
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    gl = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
    for ax, (title, getter) in zip(axs, [
        ("Quasi-static single-leg support (stance leg), proxy",
         lambda r: {n: abs(v["err_deg"]) for n, v in r["B1"]["L"]["rows"].items() if v["role"] == "stance"}),
        ("Human-gait peak moment x body mass (typical cohort), proxy",
         lambda r: {n: abs(v["err_deg"]) for n, v in r["B2"]["typical"]["mass"].items()})]):
        wd = 0.8 / len(keys)
        for i, k in enumerate(keys):
            r = R[k]
            e = getter(r)
            vals = []
            for g in gl:
                cand = []
                for n, v in e.items():
                    nl = n.lower().split("/")[-1]
                    if not (nl.startswith(("l_", "left_", "ll_"))):
                        continue
                    if g == "ankle":
                        if ("ankle_pitch" in nl) or nl.endswith("_ankle") or nl.endswith("_ffe") or ("crank" in nl):
                            cand.append(v)
                    elif g == "hip_yaw" and (nl.endswith("hip_yaw") or nl.endswith("hip_yaw_joint") or nl.endswith("_hr")):
                        cand.append(v)
                    elif g == "hip_roll" and (nl.endswith("hip_roll") or nl.endswith("hip_roll_joint") or nl.endswith("_haa")):
                        cand.append(v)
                    elif g == "hip_pitch" and (nl.endswith("hip_pitch") or nl.endswith("hip_pitch_joint") or nl.endswith("_hfe")):
                        cand.append(v)
                    elif g == "knee" and ("knee" in nl or nl.endswith("_kfe")):
                        cand.append(v)
                vals.append(np.max(cand) if cand else np.nan)
            bars = ax.bar(np.arange(len(gl)) + (i - len(keys) / 2 + 0.5) * wd, vals, wd,
                          label=PLOT_LABEL[k].replace("\n", " "), color=f"C{i}")
            if k == "h1" and "single-leg" in title:   # lean needs 60 deg hip roll: infeasible, not a result
                for b in bars:
                    b.set_hatch("///")
                    b.set_alpha(0.35)
        ax.set_xticks(np.arange(len(gl)))
        ax.set_xticklabels(["hip yaw", "hip roll", "hip pitch", "knee", "ankle pitch\n(crank for Pygmalion)"], fontsize=8)
        ax.set_ylabel("position error needed = torque / Kp  [deg]")
        ax.set_title(title, fontsize=9)
        ax.axhline(57.3, color="k", ls="--", lw=1)
        ax.text(0.02, 62, "57 deg = 1 rad", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3000)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=8, ncol=7, loc="lower center")
    fig.suptitle("Position error each joint needs to carry the walking-load proxies (H1 single-leg bars hatched: needs 60 deg hip roll, infeasible)", fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(outdir / "2026-10-14_required_error_vs_window.png", dpi=140)
    plt.close(fig)

    # ---- fig 4: Kp per kg
    fig, ax = plt.subplots(figsize=(10, 4.8))
    gl2 = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"]
    wd = 0.8 / len(keys)
    for i, k in enumerate(keys):
        r = R[k]
        vals = []
        for g in gl2:
            if g == "ankle_pitch":
                vals.append(r["A3"]["ankle_equivalent"]["L"]["Kp_pitch"] / r["mass_kg"])
            else:
                v = [vv["kp_per_kg"] for n, vv in r["A5"].items()
                     if n.lower().split("/")[-1].startswith(("l_", "left_", "ll_")) and
                     (any(n.lower().endswith(s) for s in (g, g + "_joint")) or (g == "knee" and "knee" in n.lower())) or
                     (g == "hip_yaw" and n.lower().endswith("_hr") and n.startswith("LL")) or
                     (g == "hip_roll" and n.lower().endswith("_haa") and n.startswith("LL")) or
                     (g == "hip_pitch" and n.lower().endswith("_hfe") and n.startswith("LL")) or
                     (g == "knee" and n.lower().endswith("_kfe") and n.startswith("LL"))]
                vals.append(np.mean(v) if v else np.nan)
        ax.bar(np.arange(len(gl2)) + (i - len(keys) / 2 + 0.5) * wd, vals, wd, label=PLOT_LABEL[k].replace("\n", " "))
    ax.set_xticks(np.arange(len(gl2)))
    ax.set_xticklabels(["hip yaw", "hip roll", "hip pitch", "knee", "ankle pitch (joint-equivalent)"], fontsize=8)
    ax.set_ylabel("Kp per kg of robot mass  [N m / rad / kg]")
    ax.set_title("Published Kp divided by whole-robot mass")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "2026-10-14_kp_per_kg.png", dpi=140)
    plt.close(fig)

    # ---- fig 5: which rule is constant across joints (CV of Kp / driver)
    drivers = ["effort limit", "default-pose gravity torque", "single-leg stance torque", "armature (rotor)",
               "link inertia, leg in air", "link inertia, standing"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    M = np.full((len(drivers), len(keys)), np.nan)
    for j, k in enumerate(keys):
        for i, d in enumerate(drivers):
            e = R[k]["C"].get(d, {})
            if "ratio_cv" in e and (k != "h1" or d != "armature (rotor)"):
                M[i, j] = e["ratio_cv"]
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=1.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] > 0.75 else "k", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels(drivers, fontsize=8)
    ax.set_title("Spread of Kp / driver across joint groups (coefficient of variation; small = Kp is proportional to the driver)")
    fig.colorbar(im, ax=ax, label="coefficient of variation")
    fig.tight_layout()
    fig.savefig(outdir / "2026-10-14_rule_constancy.png", dpi=140)
    plt.close(fig)

    # ---- fig 6: damping ratio and natural frequency
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (field, ttl, yl) in zip(axs, [("zeta_rotor", "Damping ratio, rotor inertia only", "damping ratio"),
                                         ("wn_rotor_hz", "Natural frequency, rotor inertia only", "Hz")]):
        wd = 0.8 / len(keys)
        for i, k in enumerate(keys):
            r = R[k]
            vals = []
            for g in gl2:
                v = [vv[field] for n, vv in r["A4"].items()
                     if n.lower().split("/")[-1].startswith(("l_", "left_", "ll_")) and
                     ((g == "ankle_pitch" and ("ankle_pitch" in n.lower() or n.lower().endswith("_ankle") or n.endswith("FFE") or "crank" in n.lower()))
                      or (g == "hip_yaw" and (n.lower().endswith("hip_yaw") or n.lower().endswith("hip_yaw_joint") or n.endswith("_HR")))
                      or (g == "hip_roll" and (n.lower().endswith("hip_roll") or n.lower().endswith("hip_roll_joint") or n.endswith("_HAA")))
                      or (g == "hip_pitch" and (n.lower().endswith("hip_pitch") or n.lower().endswith("hip_pitch_joint") or n.endswith("_HFE")))
                      or (g == "knee" and ("knee" in n.lower() or n.endswith("_KFE"))))]
                vals.append(np.mean(v) if v else np.nan)
            ax.bar(np.arange(len(gl2)) + (i - len(keys) / 2 + 0.5) * wd, vals, wd, label=PLOT_LABEL[k].replace("\n", " "))
        ax.set_xticks(np.arange(len(gl2)))
        ax.set_xticklabels(["hip yaw", "hip roll", "hip pitch", "knee", "ankle pitch\n(crank for Pygmalion)"], fontsize=8)
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
    axs[0].axhline(1.0, color="k", ls="--", lw=1)
    axs[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "2026-10-14_damping_and_frequency.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true", help="only (re)draw figures from results/*.json")
    ap.add_argument("--cases", nargs="+", default=["g1_a", "g1_b", "pyg_v7b", "pyg_trial55"])
    ap.add_argument("--no-trained-contact", action="store_true")
    a = ap.parse_args()
    if a.plots:
        make_plots(REPO / "docs" / "img" / "pd_gain")
        return
    RESULTS.mkdir(exist_ok=True)
    for k in a.cases:
        case = CASES[k]()
        res = run_case(case, with_trained_contact=not a.no_trained_contact)
        (RESULTS / f"{k}.json").write_text(json.dumps(to_py(res), indent=1))
        a2 = res["A2"]["no_push"]
        print(f"{k}: mass {res['mass_kg']:.2f} kg | min stand scale {a2.get('min_refined')} | "
              f"ankle K/mgh {res['A3']['ratio']:.2f}", flush=True)


if __name__ == "__main__":
    main()
