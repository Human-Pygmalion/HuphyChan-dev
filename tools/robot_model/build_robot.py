"""Build the Pygmalion v2 robot description - URDF and MJCF - from one mass-property file.

Inputs
  massprops   ~/pyg_fea/steps/robot_massprops_step.json (massprops_step.py) today; the
              Fusion 360 export with the same schema once the connector is reachable.
              Frame: CAD global mm. Bodies: pelvis, hip_pitch_link, hip_roll_link, thigh,
              shin, foot (left leg).
  meshes      pygmalion_locomotion/assets/pygmalion_v2/meshes/*.stl (meshes_step.py),
              already in each body's link frame, simulator axes, metres.

Frames. CAD -> sim is a +90 deg turn about z: sim = (-y_cad, x_cad, z_cad), so the robot
walks along +x with the left leg at -y, exactly the convention of the existing
pygmalion.xml. Link origins sit on joint points: base at the pelvis centre (CAD 0,70,60);
hip_pitch/hip_roll/thigh at the hip point (-123.7,70,60) where the three hip axes are
concurrent (measured from the motor cylinders, massprops_step.py); shin at the knee
(-123.7,115,-310); ankle_pitch/foot at the ankle (-123.7,145,-800). Joint axes and signs
follow the existing model (hip_pitch +y, hip_roll +x, hip_yaw -z, knee -y, ankle_pitch -y,
ankle_roll -x) so policies, rewards and init keyframes keep their meaning.

The ankle is SERIAL (pitch then roll): the two RS03, the cranks and the clevis fork are on
the shin, the rods half/half, the ankle cross on the foot (massprops_step.py). The upper
body is a placeholder lump on base_link - mass from the catalogue-corrected final-design
table (docs/82), COM and inertia scaled from the old base_link - until the Fusion export
carries the arms.

Checks: joint points must reproduce the CAD leg length; every body inertia must be
positive definite; the MJCF must compile in MuJoCo; total mass is printed against the
final-design table.

Usage: build_robot.py [--massprops=path] [--out=...]   (mjlab .venv python)

Canonical Fusion-variant filenames are always prefixed with exactly one of
``FullDoF_``, ``SemiFullDoF_`` or ``LegOnly_``.  Pass the revision portion to
``--tag``; an already-prefixed tag is normalized instead of double-prefixed.
"""
import json
import os
import sys

import numpy as np

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
MESHDIR = os.environ.get('PYG_MESH_DIR',
                         f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2/meshes')
OUT_URDF = os.environ.get('PYG_OUT_URDF',
                          f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2')
OUT_MJCF = os.environ.get('PYG_OUT_MJCF',
                          f'{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls')
# Give staged/candidate models their own mesh namespace.  This prevents a new Fusion
# snapshot from silently compiling against the production ``assets_v2`` symlink merely
# because both XMLs live in the mjlab asset directory.
MJCF_MESHDIR = os.environ.get('PYG_MJCF_MESHDIR', 'assets_v2')
URDF_MESHDIR = os.environ.get('PYG_URDF_MESHDIR', 'meshes')
R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # cad -> sim
HIP = np.array([-123.7, 70.0, 60.0])
ORIGIN_CAD = {'pelvis': np.array([0.0, 70.0, 60.0]), 'hip_pitch_link': HIP,
              'hip_roll_link': HIP, 'thigh': HIP, 'shin': np.array([-123.7, 115.0, -310.0]),
              'ankle_pitch_link': np.array([-123.7, 145.0, -800.0]),
              'foot': np.array([-123.7, 145.0, -800.0])}
CHAIN = ['hip_pitch_link', 'hip_roll_link', 'thigh', 'shin', 'ankle_pitch_link', 'foot']
LOOP_BODIES = ('crank_A', 'crank_B', 'rod_A', 'rod_B')
# Upper body, articulated on the three joints the CAD actually has. The waist-yaw axis is
# the z line through the pelvis centre; the shoulder pitch (CAD x) and roll (CAD y) axes are
# concurrent at the shoulder point, so both links hang off the same origin.
TORSO_CAD = np.array([0.0, 70.0, 177.5])
SHOULDER_CAD = np.array([-200.0, 85.0, 540.0])
ARM = ['shoulder_pitch_link', 'arm']            # per side, children of the torso
ORIGIN_CAD['torso'] = TORSO_CAD
try:                                    # loop-body frames: crank on its motor axis, rod at its pin
    _lp = json.load(open('/home/syaro/pyg_fea/fusion/ankle_loop_points_v3_printed.json'))
    for _t in 'AB':
        ORIGIN_CAD[f'crank_{_t}'] = np.array(_lp[_t]['motor'])
        ORIGIN_CAD[f'rod_{_t}'] = np.array(_lp[_t]['pin'])
except FileNotFoundError:
    pass
ORIGIN_CAD['shoulder_pitch_link'] = SHOULDER_CAD
ORIGIN_CAD['arm'] = SHOULDER_CAD
# simulator body names: the task configs bind to the old names (L_foot_link etc.)
BNAME = {'hip_pitch_link': 'hip_pitch_link', 'hip_roll_link': 'hip_roll_link', 'thigh': 'thigh_link',
         'shin': 'shin_link', 'ankle_pitch_link': 'ankle_pitch_link', 'foot': 'foot_link',
         'torso': 'torso_link', 'shoulder_pitch_link': 'shoulder_pitch_link', 'arm': 'arm_link'}
JOINT = {   # child body -> (joint name suffix, axis in sim frame, range rad, inherits old model)
    'hip_pitch_link': ('hip_pitch', (0, 1, 0), (-2.18166, 0.523599)),
    # 2026-09-02: hip_roll, hip_yaw, knee and waist_yaw axes flipped (see below) after the
    # user drove every joint by hand in tools/viewer/mjcf_joint_viewer.py and found the
    # positive-qpos direction backwards against the design intent. Flipping the axis alone
    # was tried first on the theory that a range bound is just a physical stop distance and
    # does not care which sign is positive -- the user found that WRONG by direct viewer
    # check: the (lo, hi) bound must be negated and swapped too (new = (-old_hi, -old_lo)),
    # same as flipping any signed axis value elsewhere in this file. See SIGN_FLIPPED below,
    # which is where that renegotiation actually happens (against DESIGN_CAP and against the
    # measured rom_measured.json values feeding the assert, both of which predate the flip
    # and are still written in the OLD sign convention).
    'hip_roll_link': ('hip_roll', (-1, 0, 0), (-0.785398, 0.436332)),
    'thigh': ('hip_yaw', (0, 0, 1), (-0.872665, 0.872665)),
    # knee: the shin knee plates meet the thigh clevis plates at about -120 deg (mesh
    # interference check, red team 2026-08-20) - the old -140 is not reachable on this CAD
    'shin': ('knee', (0, 1, 0), (-2.094395, 0.0)),
    # ankle pitch: the DESIGN cap is -50/+30 (docs/71 s8g, docs/76 s12), not the old +40
    'ankle_pitch_link': ('ankle_pitch', (0, -1, 0), (-0.872665, 0.523599)),
    'foot': ('ankle_roll', (-1, 0, 0), (-0.349066, 0.349066)),
    # Upper body. Signs follow the legs so the two halves read the same way: pitch +q =
    # extension (limb back), roll +q = adduction (limb toward the centreline, axis flipped
    # on the right), yaw +q = turn left. Ranges are the CAD collision sweep (rom_check.py).
    'torso': ('waist_yaw', (0, 0, -1), (-1.047198, 1.047198)),
    # shoulder_pitch: the L axis below is correct as-is (user confirmed); only R was found
    # backwards -- see the explicit R-only negation and R-only range where this JOINT entry
    # is consumed (MJCF and URDF arm-body emission, "== 'shoulder_pitch'").
    'shoulder_pitch_link': ('shoulder_pitch', (0, 1, 0), (-3.141593, 1.047198)),
    # shoulder_roll: flipped 2026-09-02, same story as hip_roll -- L was backwards, R was
    # already correct, so only the base (L) axis is flipped here and R's mirror-negation was
    # removed at the point of use (R now reuses this same, new value unmirrored).
    'arm': ('shoulder_roll', (-1, 0, 0), (-1.570796, 0.261799)),
}
# Joint ranges are MEASURED, not inherited: rom_check.py turns each joint in the assembled
# CAD until two solids that were not already touching push into each other, and writes
# rom_measured.json. The table above is only the fallback for when that file is missing.
# A design cap is applied on top ONLY where something other than part-on-part contact sets
# the limit, and each one carries its reason - a cap must never be a guess dressed as data.
ROM_FILE = '/home/syaro/pyg_fea/fusion/rom_measured.json'
DESIGN_CAP = {
    # the 2-RSU ankle is a CLOSED chain; a serial sweep drives the foot into push rods that
    # would really have followed it, so its range comes from the mechanism studies
    'ankle_pitch': (-50.0, 30.0),      # docs/71 s8g, docs/76 s12
    'ankle_roll': (-20.0, 20.0),       # JS6 clevis swing cone, docs/74 (PYG_ANKLE_ROLL15 -> 15)
    # Geometry allows more than the design wants. Set by the user 2026-08-23; each sits
    # inside the measured stop, which the assert below enforces.
    'hip_pitch': (-120.0, 25.0),       # extension 25 (metal at +26)
    # hip_roll, hip_yaw, knee, waist_yaw: these four bounds are already written in the
    # POST-FLIP convention (2026-09-02) -- e.g. hip_roll used to read (-85.0, 25.0) meaning
    # "abduction 85 (metal at -86) / adduction 25"; with the axis flipped, abduction is now
    # the POSITIVE direction, so the bound became (-25.0, 85.0). SIGN_FLIPPED below applies
    # the same negate-and-swap to the matching rom_measured.json entry before the assert, so
    # both sides of that comparison stay in this same, current convention.
    'hip_roll': (-25.0, 85.0),         # adduction 25 / abduction 85 (metal at -86); the
                                       # inherited -45 had no basis at all - it was the old
                                       # MJCF's number, kept by mistake after a sweep that
                                       # stopped searching at -70 and so never found the
                                       # stop. Adduction 25 is the gen21 verdict (2026-07-13)
    'hip_yaw': (-45.0, 45.0),          # symmetric: unaffected by the negate-and-swap. metal
                                       # at -48 / +52
    'knee': (0.0, 120.0),              # no hyperextension (the sweep finds metal only at +2)
    # Latest walking-model trims (docs/107, user-adjusted 2026-08-28). Geometry permits
    # wider travel, but those wider values are not the policy/model contract.
    'waist_yaw': (-45.0, 45.0),        # symmetric: unaffected by the negate-and-swap
    # shoulder_pitch, shoulder_roll: replaced 2026-09-02 with the user's mechanical-design
    # ROM table (0 deg = arm straight down). The table's own bound (-60, 170) was applied to
    # L verbatim first; the user then found L_shoulder_pitch's ROM direction still backwards
    # against the axis -- the table's sign convention needed the SAME negate-and-swap
    # transform as everywhere else in this dict, so this bound is that transform of the
    # table's number, not the table's number itself. L only for shoulder_pitch -- see
    # SHOULDER_PITCH_R_CAP below for R. shoulder_roll: L's bound is the table's number
    # as-is (confirmed correct); R needed its own separate negate-and-swap, see
    # SHOULDER_ROLL_R_CAP below (this joint's axis is otherwise identical L/R).
    'shoulder_pitch': (-170.0, 60.0),
    'shoulder_roll': (-15.0, 130.0),     # L; see SHOULDER_ROLL_R_CAP for R
}
# shoulder_pitch, hip_pitch, hip_roll, knee: only one side's axis was found backwards, so
# unlike the SIGN_FLIPPED joints above this is not a shared-axis convention flip -- the side
# that needs its axis negated (or, for hip_roll, the side that does NOT -- see the JOINT
# dict comment) ALSO needs its own negated-and-swapped range instead of reusing DESIGN_CAP
# unchanged for both sides. Getting this half of the fix right the first time around is
# exactly what the user had to catch and correct on 2026-09-02, twice: negating (or
# un-negating) the axis without renegotiating the range for the side that changed reproduces
# the same bug on that side.
SHOULDER_PITCH_R_CAP = (-60.0, 170.0)    # negate-swap of DESIGN_CAP['shoulder_pitch']
HIP_PITCH_R_CAP = (-25.0, 120.0)         # negate-swap of DESIGN_CAP['hip_pitch']
HIP_ROLL_R_CAP = (-85.0, 25.0)           # negate-swap of DESIGN_CAP['hip_roll'] -- R's axis
                                         # is NOT flipped (unlike the other three here), but
                                         # it still needs its own range: R was correct all
                                         # along, before any of this joint's other fixes.
KNEE_R_CAP = (-120.0, 0.0)               # negate-swap of DESIGN_CAP['knee']
SHOULDER_ROLL_R_CAP = (-130.0, 15.0)     # negate-swap of DESIGN_CAP['shoulder_roll'] -- same
                                         # "axis unchanged, range needs its own flip" pattern
                                         # as hip_roll: R_shoulder_roll_joint's ROM was found
                                         # backwards while its axis (shared with L) was not.
# shoulder_pitch and shoulder_roll: rom_measured.json's CAD self-collision sweep for these
# two (free_deg up to +90 / +30 respectively) predates the 2026-09 shoulder rework (holder/
# stopper reclassification, mesh changes) and is stale -- the user's mechanical-design table
# above is authoritative even where it exceeds that stale sweep, so these two are exempted
# from the "design cap must fit inside the measured envelope" assert that every other
# DESIGN_CAP entry still has to pass.
DESIGN_CAP_SUPERSEDES_MEASURED = {'shoulder_pitch', 'shoulder_roll'}
CLOSED_CHAIN = {'ankle_pitch', 'ankle_roll'}
# Joints whose axis was flipped 2026-09-02 relative to what rom_measured.json (and
# DESIGN_CAP's own history before this dict was rewritten to match) were measured against.
# The CAD-sweep free_deg pulled from that file for these joints must be negated and swapped
# before comparing against DESIGN_CAP's now-current-convention bound, same transform as the
# bound itself got.
SIGN_FLIPPED = {'hip_roll', 'hip_yaw', 'knee', 'waist_yaw'}
CRANK_RANGE = (-1.2, 1.2)      # rad, wider than the ankle needs; verified by the sweep
EFFORT = {'hip_pitch': 120, 'hip_roll': 120, 'hip_yaw': 60, 'knee': 120,
          'ankle_pitch': 90, 'ankle_roll': 50,
          'waist_yaw': 120, 'shoulder_pitch': 60, 'shoulder_roll': 60}   # RS04 / RS03 / RS03

# 2026-09-15: URDF 의 `<dynamics>` 에 **실측 모터 값**을 같이 낸다.
#
# 왜: 이 값들은 지금까지 mjlab 이 불러올 때만 주입돼서, **URDF 나 MJCF 파일만 다른
# 시뮬레이터에 가져가면 아예 안 들어갔다.** armature 는 100배 작은 자리표시자가, 마찰은
# 아무것도 없는 채로 로봇이 만들어진다. 사용자 지적(2026-09-15): *"실제로 나중에 로봇을
# 다른 플랫폼 시뮬쪽에 달면 문제가 될거같아."*
#
# URDF `friction` = MuJoCo `frictionloss`(움직임을 막는 일정한 힘), `damping` 은 양쪽 다
# 속도 비례 감쇠라 그대로 옮겨진다. **armature 는 URDF 규격에 자리가 없어** 주석과
# 옆 파일(`*.motor_params.json`)로 낸다 — `write_motor_params_xml.py` 와 같은 규약이다.
#
# 2026-09-15 · 사용자 지시: *"모든 설정파일은 같은 값을 가져야 한다고. 혹은 단 하나만
# True 값을 가지도록 하라고."*
#
# 그래서 **여기에는 숫자가 없다.** 하나뿐인 출처
# (`pygmalion_constants.py` 의 `MOTOR_MEAS`)에서 읽어 온다. 예전에는 이 표가 손으로 베낀
# 사본이었고, 바로 그 방식 때문에 2026-09-10 에 발목 값이 안 따라와 한 모형 안에서
# 4.2배 어긋난 채 학습이 돌았다 (`check_motor_params.py` 가 그 사고로 생겼다).
#
# 여기 남는 것은 **숫자가 아니라 배선**이다: 어느 관절에 어느 모터가 달렸는가.
# 그 대응은 `write_motor_params_xml.JOINT_MOTOR` 와 같아야 하고, `check_motor_params.py`
# 가 둘이 같은지 본다.
_wmp_dir = os.path.dirname(os.path.abspath(__file__))
if _wmp_dir not in sys.path:
  sys.path.insert(0, _wmp_dir)
import write_motor_params_xml as _WMP                                    # noqa: E402

JOINT_FAMILY = {                     # 관절 이름 조각 -> 모터 종류
  'hip_pitch': 'RS04', 'hip_roll': 'RS04', 'knee': 'RS04', 'waist_yaw': 'RS04',
  'hip_yaw': 'RS03', 'shoulder_pitch': 'RS03', 'shoulder_roll': 'RS03',
}
MOTOR_DYN = {                        # 관절 이름 조각 -> (damping, frictionloss, armature)
  _frag: (_m['damping'], _m['frictionloss'], _m['armature'])
  for _frag, _fam in JOINT_FAMILY.items()
  for _m in (_WMP.SETS['bench-2026-09-10'][_fam],)
}


def dyn_xml(jn: str, ind: str = '    ') -> str:
  """그 관절의 `<dynamics>` 한 줄 + armature 주석. 구동 관절이 아니면 빈 문자열."""
  v = MOTOR_DYN.get(jn)
  if v is None:
    return ''
  d, f, a = v
  return (f'{ind}<dynamics damping="{d:g}" friction="{f:g}"/>\n'
          f'{ind}<!-- armature (reflected rotor inertia) = {a:g} kg*m^2 : '
          f'no standard URDF field; see the sidecar .motor_params.json -->\n')
# Upper body. When the mass-property file carries an `upper` block (the Fusion export does -
# the current CAD is "wUpper" and models Torso, Neck and the arm), it is used verbatim,
# mirrored to two arms. Otherwise the old placeholder lump is used: the docs/82
# catalogue-corrected table (Torso+Neck+2 arms + WaistYaw2Pitch 0.775, no battery), with the
# old base_link COM re-expressed at the HIP (old base origin = hip + (+0.104, 0, -0.059)).
UPPER_MASS = 15.335 + 0.775
UPPER_COM_SIM = np.array([0.012, 0.0, 0.366])
UPPER_DIAG = np.array([1.62441, 1.27435, 0.55027]) * (UPPER_MASS / 28.0892)
# sole from the CAD: plate bottom 43 mm under the ankle axis, 180 ahead / 80 behind, 100 wide
SOLE_Z, SOLE_X = -0.043, (-0.080, 0.180)


def to_sim_vec(v):
    return R @ np.asarray(v, float)


MOTOR_PROXIES = os.environ.get('PYG_MOTOR_PROXIES', '/home/syaro/pyg_fea/fusion/motor_proxies_fusion.json')

# --- IMU site -----------------------------------------------------------------------------
# Where the E2Box IMU actually is, measured off the live Fusion document on 2026-08-26
# (docs/105_imu_and_shoulder_cad_update.md s1). The old value 0.004 0 0.241 was inherited from
# the pre-v2 model ("old model geometry re-expressed at the hip-level base origin") and had
# never been measured - it sits 310.6 mm away from the real sensor, above the pelvis entirely.
#
# The user gave the position as "-Z 187 mm from the yaw motor surface". Enumerating every
# +-Z face pair 187 +- 1 mm apart found 43 matches and ALL of them are the RS04 Waist_Yaw
# motor - none is the RS03 Hip_Y - with the best pair exact to 0.00 mm:
#   Waist_Yaw face  CAD z = +177.50 mm   ->   IMU face  CAD z = -9.50 mm
# CAD root -> base_link is  (x,y,z) -> (70.0 - y, x, z - 60.0) mm, verified to 0.00 mm against
# both the hip centre and the knee axis.
#   spec face      CAD (7.078, 70.001,  -9.500) -> ( 0.000000, 0.007078, -0.069500) m   <- used
#   EBIMU board CoM CAD (7.078, 70.001, -16.086) -> ( 0.000000, 0.007078, -0.076086) m   (6.6 mm lower)
# The board's occurrence transform is identity and its faces are axis-aligned, so the site
# needs no rotation relative to base_link.
#
# Safe to change: base_lin_vel (the velocimeter on this site, the only position-dependent
# consumer) is a CRITIC-ONLY observation here. The actor sees base_ang_vel - a gyro, uniform
# over a rigid body - and projected_gravity, which is orientation-only. Measured on
# bundleD1_AB/32798, moving the site shifts base_lin_vel by |omega x dr| = 0.021 m/s median /
# 0.047 p95, i.e. 1.3 % of the tracked speed and 9 % of the +-0.5 m/s noise already injected
# into that term (tools/robot_model/loop_tests/imu_site_delta.py).
# NOTE the parse: `bool(os.environ.get(FLAG))` is true for the string "0", so PYG_X=0 turns
# such a flag ON. 21 PYG_* flags in this project read that way (docs/99 s"PYG flag parsing");
# no run has ever tripped it, but this one is written so PYG_IMU_LEGACY=0 means off.
IMU_LEGACY = os.environ.get('PYG_IMU_LEGACY', '').strip().lower() not in ('', '0', 'false', 'no')
IMU_POS = (0.004, 0.0, 0.241) if IMU_LEGACY else (-0.000001, 0.007078, -0.069500)
CENTRELINE_MM = 5.0        # |x_cad| under this = on the centreline, drawn once, not mirrored


def motor_geoms(mp, body, side, collision=False, omit_waist=False):
    """MJCF cylinder visuals for the actuators riding on `body`, in its link frame.

    Placement comes from the LIVE Fusion document (motor_proxies_fusion.py), not from the
    STEP export: the two agree within 4 mm everywhere except the hip-pitch RS04, which moved
    75 mm between revisions, and the STEP has no upper body at all.

    The placeholders are not meshed - gmsh takes minutes on their fine features - and the
    measured centre, axis, radius and length reproduce their envelope exactly.
    """
    out = []
    for key, mo in mp.get('motors', {}).items():
        if omit_waist and mo.get('joint') == 'waist_yaw':
            continue
        if mo['link'] != body:
            continue
        centre = np.array(mo['com'], float)
        if abs(centre[0]) < CENTRELINE_MM and side == 'R':
            continue                       # centreline actuator: drawn once, on the L pass
        c = to_sim_vec((centre - ORIGIN_CAD[body]) / 1000.0)
        ax = to_sim_vec(mo['axis'])
        if side == 'R':
            c = c * np.array([1, -1, 1])
            ax = ax * np.array([1, -1, 1])
        r = mo['r'] / 1000.0
        h = mo['len'] / 2000.0
        nm = mo['joint'] if abs(centre[0]) < CENTRELINE_MM else f'{side}_{mo["joint"]}'
        cls, suffix = ('collision', '_motor_collision') if collision else ('visual', '_motor')
        mat = '' if collision else ' material="black"'
        out.append(f'<geom name="{nm}{suffix}" type="cylinder" size="{r:.4f} {h:.4f}" '
                   f'pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
                   f'zaxis="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}" class="{cls}"{mat}/>')
    return out


def mesh_exists(name):
    """A body can legitimately have no structural mesh: shoulder_pitch_link, after the
    2026-09-02 shoulder reclassification, carries nothing but the shoulder-roll motor's
    stator (a drawn cylinder primitive, not a mesh) -- the same situation ankle_pitch_link
    has always been in. Callers that would otherwise unconditionally reference `{name}.stl`
    must check this first.

    Checks where the COMPILED xml's <mesh file="{name}.stl"> will actually resolve from:
    OUT_MJCF/MJCF_MESHDIR (a symlink, e.g. xmls/assets_v30_armfix -> .../meshes_armfix for
    a staged build) -- NOT the generic MESHDIR that fitted_capsule/mesh_bounds read from
    for collision-shape fitting, which is a separate directory and can legitimately be
    stale or missing a staged build's newest files without affecting compile."""
    return (os.path.exists(f'{OUT_MJCF}/{MJCF_MESHDIR}/{name}.stl')
            if os.path.exists(f'{OUT_MJCF}/{MJCF_MESHDIR}')
            else os.path.exists(f'{MESHDIR}/{name}.stl'))


def mesh_bounds(name):
    """(min, max) of a link-frame STL, metres - collision primitives are sized from the
    real geometry rather than from numbers copied out of the old model."""
    import trimesh
    m = trimesh.load(f'{MESHDIR}/{name}.stl', process=False)
    return np.asarray(m.bounds[0]), np.asarray(m.bounds[1])


def fitted_capsule(name, stl, cls='collision', q=0.90, exclude=None):
    """A capsule that hugs the mesh: axis = the mesh's longest bounding-box direction, ends
    = the extreme projections pulled in by the radius, radius = the q-quantile of the
    vertices' distance from the axis (q < 1 so bolt heads and a flange rim do not fatten the
    whole link). Never degenerates to a sphere - a pill is what was asked for. `exclude` is
    an optional vertex mask to drop first (e.g. the fat shoulder end of the arm rod).
    Non-adjacent overlaps left at the zero pose are trimmed afterwards by
    resolve_zero_pose_overlaps(), which reports every radius it shrinks."""
    import trimesh
    V = np.asarray(trimesh.load(f'{MESHDIR}/{stl}.stl', process=False).vertices)
    if exclude is not None:
        V = V[~exclude(V)]
    c = V.mean(0)
    a = np.zeros(3)
    a[int(np.argmax(V.max(0) - V.min(0)))] = 1.0
    t = (V - c) @ a
    perp = np.linalg.norm((V - c) - np.outer(t, a), axis=1)
    r = float(np.quantile(perp, q))
    half = max((t.max() - t.min()) / 2 - r, 0.005)
    mid = c + a * (t.max() + t.min()) / 2
    p0, p1 = mid - a * half, mid + a * half
    return (f'<geom name="{name}" class="{cls}" type="capsule" fromto="{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f}  '
            f'{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f}" size="{r:.4f}"/>')


def resolve_zero_pose_overlaps(xml_path, margin=0.002, max_iter=40):
    """Shrink collision radii until no NON-ADJACENT pair touches at the zero pose.

    The capsules are fitted to the meshes and only adjacent links are excluded, so wherever
    the CAD nests one link inside another (the roll housing inside the pelvis frame, the
    dummy arm 5 mm into the hip) the fitted shapes overlap. Rather than hide that behind an
    exclusion, both shapes give up half the penetration plus a margin, and every change is
    printed - a radius that had to shrink a lot is a shape the CAD really does overlap.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(xml_path)
    log = {}
    for _ in range(max_iter):
        m = spec.compile()
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        d.qpos[2] = 1.5
        mujoco.mj_forward(m, d)
        hits = [(d.contact[i].geom1, d.contact[i].geom2, d.contact[i].dist) for i in range(d.ncon)
                if m.geom_bodyid[d.contact[i].geom1] and m.geom_bodyid[d.contact[i].geom2]]
        if not hits:
            break
        for g1, g2, dist in hits:
            for g in (g1, g2):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                sg = next(x for x in spec.geoms if x.name == name)
                if sg.type == mujoco.mjtGeom.mjGEOM_BOX:
                    continue                         # the foot box keeps its footprint
                cut = -dist / 2 + margin
                sg.size[0] = max(sg.size[0] - cut, 0.01)
                log[name] = log.get(name, 0.0) + cut
    spec.to_file(xml_path)
    return log


LOOP_PTS = '/home/syaro/pyg_fea/fusion/ankle_loop_points_v3_printed.json'


def loop_xml(s, ind, mp, sign):
    """The 2-RSU ankle as the real mechanism, on one leg (children of the shin; the foot's
    ball sites are emitted by the caller).

    crank_{A,B}   hinge on the shin about the RS03 axis (CAD x -> sim y), origin on that axis
    rod_{A,B}     child of the crank at the crank pin, a UNIVERSAL joint there (two hinges:
                  the crank axis, then the axis perpendicular to it and to the rod) - not a
                  ball joint, so the rod cannot spin about itself and mjlab's joint indexing
                  sees hinges only
    connect       rod far-end site <-> foot ball site, 3 constraints
    DOF: 2 passive ankle hinges + 2 cranks + 2x2 rod hinges - 2x3 connects = 2, exactly the
    two cranks - the ankle pitch/roll hinges are passive and follow the linkage.
    Masses and inertias are the real ones (massprops_fusion with PYG_ANKLE_LOOP=1).
    """
    pts = json.load(open(LOOP_PTS))
    out = []
    for tag in 'AB':
        P = pts[tag]
        motor, pin, ball = (np.array(P[k]) for k in ('motor', 'pin', 'ball'))
        o_c = to_sim_vec((motor - ORIGIN_CAD['shin']) / 1000.0)      # crank origin, shin frame
        pin_rel = to_sim_vec((pin - motor) / 1000.0)                  # pin, crank frame
        vec = to_sim_vec((ball - pin) / 1000.0)                       # rod vector, rod frame
        if sign > 0:                                                  # right leg: mirror y
            o_c, pin_rel, vec = (v * np.array([1, -1, 1]) for v in (o_c, pin_rel, vec))
        axis1 = np.array([0.0, 1.0, 0.0])                             # crank axis (CAD x -> sim y)
        # 2026-09-02: a fourth viewer review found L_crank_A_joint and R_crank_B_joint
        # backwards while L_crank_B and R_crank_A were correct -- a genuine per-(side, tag)
        # asymmetry (crank A and crank B are the two different RS03 mounting orientations on
        # the shin, "faces medially inboard" vs "faces laterally outboard" per
        # motor_sign_convention.json, so there is no reason A and B would share one rule).
        # Both the crank's own joint and its rod's u1 hinge share this axis; axis2 is derived
        # from axis1 by a cross product so it flips sign automatically and needs no separate
        # handling. The range (CRANK_RANGE) is symmetric and untouched by the flip.
        if (s == 'L' and tag == 'A') or (s == 'R' and tag == 'B'):
            axis1 = -axis1
        axis2 = np.cross(axis1, vec / np.linalg.norm(vec))
        axis2 /= np.linalg.norm(axis2)
        mc, cc, Ic = body_inertial(f'crank_{tag}', mp)
        mr, cr, Ir = body_inertial(f'rod_{tag}', mp)
        if sign > 0:
            cc, Ic = mirror(cc, Ic)
            cr, Ir = mirror(cr, Ir)
        pre = 'R_' if sign > 0 else ''
        out.append(f'{ind}<body name="{s}_crank_{tag}" pos="{o_c[0]:.6g} {o_c[1]:.6g} {o_c[2]:.6g}">')
        out.append(f'{ind}  <inertial pos="{cc[0]:.6g} {cc[1]:.6g} {cc[2]:.6g}" mass="{mc:.5g}" fullinertia="{fullinertia(Ic)}"/>')
        out.append(f'{ind}  <joint name="{s}_crank_{tag}_joint" axis="{axis1[0]:.4g} {axis1[1]:.4g} {axis1[2]:.4g}" range="{CRANK_RANGE[0]} {CRANK_RANGE[1]}" armature="0.005" damping="0.2"/>')
        out.append(f'{ind}  <geom mesh="{s}_crank_{tag}" class="visual" material="black"/>')
        out.append(f'{ind}  <body name="{s}_rod_{tag}" pos="{pin_rel[0]:.6g} {pin_rel[1]:.6g} {pin_rel[2]:.6g}">')
        out.append(f'{ind}    <inertial pos="{cr[0]:.6g} {cr[1]:.6g} {cr[2]:.6g}" mass="{mr:.5g}" fullinertia="{fullinertia(Ir)}"/>')
        out.append(f'{ind}    <joint name="{s}_rod_{tag}_u1" axis="{axis1[0]:.4f} {axis1[1]:.4f} {axis1[2]:.4f}" armature="0.0005" damping="0.02"/>')
        out.append(f'{ind}    <joint name="{s}_rod_{tag}_u2" axis="{axis2[0]:.4f} {axis2[1]:.4f} {axis2[2]:.4f}" armature="0.0005" damping="0.02"/>')
        out.append(f'{ind}    <geom mesh="{s}_rod_{tag}" class="visual"/>')
        out.append(f'{ind}    <site name="{s}_rod_{tag}_end" pos="{vec[0]:.6g} {vec[1]:.6g} {vec[2]:.6g}" size="0.004"/>')
        out.append(f'{ind}  </body>')
        out.append(f'{ind}</body>')
    return out


def loop_urdf(s, mp, sign):
    """The same mechanism for the URDF - as a TREE. URDF has no loop closure and no universal
    joint, so: crank (revolute) -> rod_u (1 g dummy link) -> rod (revolute x2 = the universal
    joint); the rod's far end is left open and the closure is stated in a comment. Link masses
    and inertias equal the MJCF's (the dummy links add 2 x 1 g per leg, URDF-only bodies)."""
    pts = json.load(open(LOOP_PTS))
    links, joints = [], []
    for tag in 'AB':
        P = pts[tag]
        motor, pin, ball = (np.array(P[k]) for k in ('motor', 'pin', 'ball'))
        o_c = to_sim_vec((motor - ORIGIN_CAD['shin']) / 1000.0)
        pin_rel = to_sim_vec((pin - motor) / 1000.0)
        vec = to_sim_vec((ball - pin) / 1000.0)
        if sign > 0:
            o_c, pin_rel, vec = (v * np.array([1, -1, 1]) for v in (o_c, pin_rel, vec))
        axis1 = np.array([0.0, 1.0, 0.0])
        if (s == 'L' and tag == 'A') or (s == 'R' and tag == 'B'):
            axis1 = -axis1                          # see loop_xml for why (per-tag asymmetry)
        axis2 = np.cross(axis1, vec / np.linalg.norm(vec)); axis2 /= np.linalg.norm(axis2)
        mc, cc, Ic = body_inertial(f'crank_{tag}', mp)
        mr, cr, Ir = body_inertial(f'rod_{tag}', mp)
        if sign > 0:
            cc, Ic = mirror(cc, Ic)
            cr, Ir = mirror(cr, Ir)
        pre = 'R_' if sign > 0 else ''
        def link(name, m, c, I, mesh=None):
            t = (f'  <link name="{name}">\n    <inertial>\n      <origin xyz="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" rpy="0 0 0"/>\n'
                 f'      <mass value="{m:.5g}"/>\n      {urdf_inertia(I)}\n    </inertial>\n')
            if mesh:
                t += f'    <visual>\n      <geometry><mesh filename="meshes/{mesh}"/></geometry>\n    </visual>\n'
            return t + '  </link>\n'
        def joint(name, parent, child, o, ax, rg=None, effort=0.0):
            lim = (f'    <limit lower="{rg[0]}" upper="{rg[1]}" effort="{effort}" velocity="20"/>\n' if rg
                   else f'    <limit lower="-3.1416" upper="3.1416" effort="0" velocity="50"/>\n')
            return (f'  <joint name="{name}" type="revolute">\n    <origin xyz="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" rpy="0 0 0"/>\n'
                    f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n    <axis xyz="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}"/>\n{lim}  </joint>\n')
        links.append(link(f'{s}_crank_{tag}', mc, cc, Ic, f'{pre}crank_{tag}.stl'))
        links.append(link(f'{s}_rod_{tag}_u', 1e-3, np.zeros(3), np.eye(3) * 1e-7))
        links.append(link(f'{s}_rod_{tag}', mr, cr, Ir, f'{pre}rod_{tag}.stl'))
        joints.append(joint(f'{s}_crank_{tag}_joint', f'{s}_shin_link', f'{s}_crank_{tag}', o_c, axis1, CRANK_RANGE, 60.0))
        joints.append(joint(f'{s}_rod_{tag}_u1', f'{s}_crank_{tag}', f'{s}_rod_{tag}_u', pin_rel, axis1))
        joints.append(joint(f'{s}_rod_{tag}_u2', f'{s}_rod_{tag}_u', f'{s}_rod_{tag}', np.zeros(3), axis2))
        joints.append(f'  <!-- loop closure, not expressible in URDF: the far end of {s}_rod_{tag} at xyz="{vec[0]:.6g} {vec[1]:.6g} {vec[2]:.6g}" '
                      f'(rod frame) is a ball joint on {s}_foot_link - see the <equality><connect> in {s and ""}the MJCF -->\n')
    return links, joints


def foot_ball_sites(s, sign):
    pts = json.load(open(LOOP_PTS))
    out = []
    for tag in 'AB':
        b = to_sim_vec((np.array(pts[tag]['ball']) - ORIGIN_CAD['foot']) / 1000.0)
        if sign > 0:
            b = b * np.array([1, -1, 1])
        out.append(f'<site name="{s}_ball_{tag}" pos="{b[0]:.6g} {b[1]:.6g} {b[2]:.6g}" size="0.004"/>')
    return out


def box_geom(name, lo, hi, cls='collision', shrink=0.0):
    c = (lo + hi) / 2
    h = np.maximum((hi - lo) / 2 - shrink, 0.005)
    return (f'<geom name="{name}" class="{cls}" type="box" '
            f'pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}"/>')


FOOT_SOLE_SEGMENTS = 3   # toe / mid / heel, along the foot's long (fore-aft, sim X) axis


def foot_sole_boxes(name_prefix, lo, hi, n=FOOT_SOLE_SEGMENTS):
    """N box_geoms tiling the foot's full bounding volume along X (fore-aft), keeping the
    same total Y/Z coverage a single box had. Several small pads under a sole give MuJoCo
    real multi-point ground contact (roll/tip about a toe or heel edge) instead of one slab
    that behaves like a rigid skate; still one geom per segment, no per-corner primitives."""
    xs = np.linspace(lo[0], hi[0], n + 1)
    out = []
    for i in range(n):
        seg_lo = np.array([xs[i], lo[1], lo[2]])
        seg_hi = np.array([xs[i + 1], hi[1], hi[2]])
        out.append(box_geom(f'{name_prefix}{i + 1}_collision', seg_lo, seg_hi, cls='foot_box'))
    return out


def body_inertial(b, mp):
    """(com_sim m, I_sim kg m2 about COM) of a massprops body, in its link frame."""
    d = mp['bodies'][b]
    com = to_sim_vec((np.array(d['com']) - ORIGIN_CAD[b]) / 1000.0)
    I = R @ np.array(d['I_com']) @ R.T * 1e-6
    return d['mass'], com, I


def mirror(com, I):
    M = np.diag([1.0, -1.0, 1.0])
    return M @ com, M @ I @ M


def fullinertia(I):
    return f'{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}'


def urdf_inertia(I):
    return (f'<inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" '
            f'iyy="{I[1,1]:.6g}" iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/>')


def assert_left_is_plus_y(text: str, what: str) -> None:
    """`L_` 이 붙은 사슬이 **+y(물리적 왼쪽)** 에 있는지 확인하고, 아니면 멈춘다.

    2026-09-12 에 이것 때문에 실물에서 하루를 썼습니다. 이 빌더는 `sign = -1 if
    s == 'L'` 로 **왼다리를 -y 에** 놓는데(옛 pygmalion.xml 규약), 앞이 +x 이고
    z 가 위인 오른손 좌표계에서 **+y 가 왼쪽**입니다. 그래서 `L_` 사슬이
    물리적으로는 오른다리였고, 실물 왼 무릎에 오른 무릎 한계가 들어가
    **안전 클리핑이 무릎을 하드스톱으로 밀었습니다** (docs/134, docs/136).

    이미 만들어진 모델 파일들은 `tools/robot_model/swap_leg_names.py` 로 이름만
    맞바꿔 고쳤습니다. **이 빌더 자체는 아직 안 고쳤습니다** -- 고치려면 세 군데를
    함께 뒤집어야 하는데, 마지막 하나가 메시 파일 선택이라 STL 내보내기까지
    건드려야 합니다:

        1) `sign = -1.0 if s == 'L' else 1.0`  (746, 829, 966행) -> 부호 반대로
        2) 크랭크 축 예외 `(s=='L' and tag=='A') or (s=='R' and tag=='B')`
           (400, 441행) 과 `-15.0 if s_ == 'L'` (862행) -> 글자 반대로
        3) **메시**: 지금은 s=='R' 일 때만 `R_*.stl` 을 쓰고 왼쪽은 접두사 없는
           파일을 씁니다. 1) 을 뒤집으면 +y 다리가 접두사 없는(=거울 반대) 메시를
           받습니다. STL 내보내기에서 좌우를 같이 바꿔야 맞습니다.

    그때까지는 이 확인이 **조용한 재발을 막습니다.**
    """
    import re as _re
    m = _re.search(r'name="L_hip_pitch_(?:link|joint)"', text)
    if not m:
        return
    for pat in (r'<body name="L_hip_pitch_link"[^>]*pos="[-\d.eE]+ ([-\d.eE]+)',
                r'<joint name="L_hip_pitch_joint".*?xyz="[-\d.eE]+ ([-\d.eE]+)'):
        hit = _re.search(pat, text, _re.S)
        if hit:
            y = float(hit.group(1))
            if y < 0:
                raise SystemExit(
                    f"{what}: L_ 사슬이 y={y:+.4f} (오른쪽)에 있습니다. +y 가 왼쪽입니다.\n"
                    f"  이 빌더의 좌우 배정이 뒤집혀 있습니다 -- 위 함수의 설명을 보고\n"
                    f"  1)2)3) 을 함께 고치거나, 만든 뒤 swap_leg_names.py 를 돌리십시오."
                )
            return


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return
    mpf = next((a.split('=')[1] for a in sys.argv if a.startswith('--massprops=')),
               '/home/syaro/pyg_fea/steps/robot_massprops_step.json')
    # output name: a different mass model must not overwrite pygmalion_v2 - the aluminium
    # build the tasks and keyframes were tuned on
    tag = next((a.split('=')[1] for a in sys.argv if a.startswith('--tag=')), 'pygmalion_v2')
    variant = next((a.split('=')[1] for a in sys.argv if a.startswith('--variant=')), 'full').lower()
    aliases = {'fulldof': 'full', 'semifulldof': 'semi', 'legonly': 'leg'}
    variant = aliases.get(variant, variant)
    assert variant in ('full', 'semi', 'leg'), f'unknown --variant={variant!r}'
    # User filename contract (2026-09-01): the geometry/DOF variant is a filename
    # prefix, not an informal label in a document.  Normalize all callers here so a
    # FullDoF build can no longer silently become the one unprefixed exception.
    variant_prefix = {
        'full': 'FullDoF_',
        'semi': 'SemiFullDoF_',
        'leg': 'LegOnly_',
    }[variant]
    loop_suffix = '_loop' if tag.endswith('_loop') else ''
    revision_tag = tag[:-len(loop_suffix)] if loop_suffix else tag
    for prefix in ('FullDoF_', 'SemiFullDoF_', 'LegOnly_'):
        if revision_tag.startswith(prefix):
            revision_tag = revision_tag[len(prefix):]
            break
    tag = variant_prefix + revision_tag + loop_suffix
    semi_fixed = variant == 'semi'
    leg_only = variant == 'leg'
    # --ankle=loop: the 2-RSU ankle as the closed mechanism (needs a massprops file made with
    # PYG_ANKLE_LOOP=1, and the crank/rod/shin_noloop meshes from meshes_step.py --loop)
    loop = next((a.split('=')[1] for a in sys.argv if a.startswith('--ankle=')), 'serial') == 'loop'
    if loop:
        tag = tag + '_loop' if not tag.endswith('_loop') else tag
    mp = json.load(open(mpf))
    motor_comments = []
    for family in ('RS03', 'RS04'):
        motor_mass = float(mp.get('motor_masses', {}).get(family, 0.0))
        motor_source = mp.get('motor_mass_sources', {}).get(family, 'unspecified')
        motor_comments.append(f'{family} mass {motor_mass:.4f} kg; source: {motor_source}')
    motor_comment = '; '.join(motor_comments).replace('--', '-')
    # ---- resolve joint ranges from the CAD sweep ----
    # measured ranges are not optional: without the sweep file the old MJCF ranges would come
    # back silently, and the ankle design caps with them (red team 2026-08-23)
    assert os.path.exists(ROM_FILE), f'{ROM_FILE} missing - run tools/robot_model/rom_check.py'
    rom = json.load(open(ROM_FILE))
    rom_log = []
    for b, (jn, ax, rg) in list(JOINT.items()):
        lo_d, hi_d = np.degrees(rg)
        src = 'inherited'
        if jn in rom:
            m_lo, m_hi = rom[jn]['free_deg']
            if jn in SIGN_FLIPPED:
                m_lo, m_hi = -m_hi, -m_lo   # rom_measured.json predates the axis flip
            if jn in CLOSED_CHAIN:
                lo_d, hi_d = DESIGN_CAP[jn]
                src = 'mechanism (closed chain)'
            else:
                lo_d, hi_d, src = m_lo, m_hi, 'CAD sweep'
                if jn in DESIGN_CAP:
                    c_lo, c_hi = DESIGN_CAP[jn]
                    if jn not in DESIGN_CAP_SUPERSEDES_MEASURED:
                        assert c_lo >= m_lo - 1e-6 and c_hi <= m_hi + 1e-6, (
                            f'{jn}: design cap {(c_lo, c_hi)} is WIDER than the geometry '
                            f'allows {(m_lo, m_hi)}')
                        src = 'CAD sweep, design-capped'
                    else:
                        src = 'user mechanical-design table, supersedes stale CAD sweep'
                    lo_d, hi_d = c_lo, c_hi
        JOINT[b] = (jn, ax, (float(np.radians(lo_d)), float(np.radians(hi_d))))
        rom_log.append((jn, lo_d, hi_d, src,
                        rom.get(jn, {}).get('blocker_lo'), rom.get(jn, {}).get('blocker_hi')))

    # actuator cylinders are geometry, not mass, and come from the live Fusion document
    assert os.path.exists(MOTOR_PROXIES), (
        f'{MOTOR_PROXIES} missing - run tools/robot_model/motor_proxies_fusion.py')
    mp['motors'] = json.load(open(MOTOR_PROXIES))
    os.makedirs(OUT_URDF, exist_ok=True)
    os.makedirs(OUT_MJCF, exist_ok=True)

    # ---- joint offsets (sim, m) and the leg-length anchor ----
    off = {}
    prev = 'pelvis'
    for b in CHAIN:
        off[b] = to_sim_vec((ORIGIN_CAD[b] - ORIGIN_CAD[prev]) / 1000.0)
        prev = b
    leg = (ORIGIN_CAD['hip_pitch_link'] - ORIGIN_CAD['foot'])[2]
    assert abs(leg - 860.0) < 0.1, f'hip-to-ankle {leg} mm, CAD says 860'

    # ---- base_link: the pelvis. The upper body is ARTICULATED when the mass-property
    # file carries the three CAD upper bodies; only an older file still needs the lump. ----
    def shift(I, m, c, about):
        d = c - about
        return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    m_p, c_p, I_p = body_inertial('pelvis', mp)
    has_upper_data = all(b in mp['bodies'] for b in ('torso',) + tuple(ARM))
    articulated = has_upper_data and not leg_only
    # New Fusion staging exports can merge Torso2ShoulderP into torso.stl.  Older asset
    # sets keep it as a separate visual-only mesh.  Support both without requiring a stale
    # file or drawing the rework twice.
    separate_torso_shpitch = articulated and mesh_exists('torso_shpitch')
    if has_upper_data:
        m_b, c_b, I_b = m_p, c_p, I_p
        upper = {}
        for b in ('torso',) + tuple(ARM):
            m, c, I = body_inertial(b, mp)
            upper[b] = dict(L=(m, c, I), R=(m,) + mirror(c, I))
        m_u = upper['torso']['L'][0] + 2 * sum(upper[b]['L'][0] for b in ARM)
        if leg_only:
            # Its mass-property input has already removed Baselink_toWaistYaw and the
            # Waist_Yaw motor.  Do not reattach any upper-body lump.
            m_u = 0.0
            upper = None
    elif leg_only:
        m_b, c_b, I_b = m_p, c_p, I_p
        upper = None
        m_u = 0.0
    else:                                   # placeholder lump on the pelvis
        upper = None
        m_u, c_u, I_u = UPPER_MASS, UPPER_COM_SIM, np.diag(UPPER_DIAG)
        m_b = m_p + m_u
        c_b = (m_p * c_p + m_u * c_u) / m_b
        I_b = shift(I_p, m_p, c_p, c_b) + shift(I_u, m_u, c_u, c_b)
    # joint offsets for the upper chain (sim, m)
    off['torso'] = to_sim_vec((ORIGIN_CAD['torso'] - ORIGIN_CAD['pelvis']) / 1000.0)
    off['shoulder_pitch_link'] = to_sim_vec(
        (ORIGIN_CAD['shoulder_pitch_link'] - ORIGIN_CAD['torso']) / 1000.0)
    off['arm'] = np.zeros(3)                # pitch and roll axes are concurrent
    legs = {}
    for b in CHAIN:
        m, c, I = body_inertial(b, mp)
        legs[b] = dict(L=(m, c, I), R=(m,) + mirror(c, I))
    loop_mass_per_leg = (sum(float(mp['bodies'][b]['mass']) for b in LOOP_BODIES)
                         if loop else 0.0)
    total = m_b + m_u + 2 * (sum(legs[b]['L'][0] for b in CHAIN) + loop_mass_per_leg)
    UPPER_MASS_USED = m_u
    for b in CHAIN:
        w = np.linalg.eigvalsh(legs[b]['L'][2])
        assert w.min() > 0, f'{b}: non-physical inertia'

    # ---- MJCF ----
    X = []
    X.append(f'<mujoco model="{tag}">\n  <!-- {motor_comment} -->\n'
             f'  <compiler angle="radian" meshdir="{MJCF_MESHDIR}" autolimits="true"/>\n')
    X.append('''  <default>
    <default class="pygmalion">
      <default class="visual">
        <geom group="2" type="mesh" density="0" material="silver" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1"/>
        <default class="foot_capsule">
          <geom type="capsule" size="0.01"/>
        </default>
        <default class="foot_box">
          <geom type="box"/>
        </default>
      </default>
      <default class="hull">
        <geom group="4" type="mesh" density="0" material="hull" contype="0" conaffinity="0"/>
      </default>
      <site group="5" rgba="1 0 0 1"/>
    </default>
  </default>
  <asset>
    <material name="silver" rgba="0.7 0.7 0.7 1"/>
    <material name="hull" rgba="0.2 0.4 0.9 0.25"/>
    <material name="black" rgba="0.2 0.2 0.2 1"/>
    <material name="red" rgba="1.0 0.0 0.0 1.0"/>
''')
    X.append('    <mesh name="pelvis" file="pelvis.stl"/>\n    <mesh name="pelvis_hull" file="pelvis_hull.stl"/>\n')
    for s in 'LR':
        for b in CHAIN:
            if b == 'ankle_pitch_link':
                continue
            f = f'{"R_" if s == "R" else ""}{b}.stl'
            X.append(f'    <mesh name="{s}_{b}" file="{f}"/>\n')
            X.append(f'    <mesh name="{s}_{b}_hull" file="{f.replace(".stl", "_hull.stl")}"/>\n')
    if loop:
        for s_ in 'LR':
            pre = 'R_' if s_ == 'R' else ''
            for b in ('crank_A', 'crank_B', 'rod_A', 'rod_B'):
                X.append(f'    <mesh name="{s_}_{b}" file="{pre}{b}.stl"/>\n')
            X.append(f'    <mesh name="{s_}_shin_noloop" file="{pre}shin_noloop.stl"/>\n')
            X.append(f'    <mesh name="{s_}_foot_noloop" file="{pre}foot_noloop.stl"/>\n')
    if articulated:
        X.append('    <mesh name="torso" file="torso.stl"/>\n'
                 '    <mesh name="torso_hull" file="torso_hull.stl"/>\n')
        for s_ in 'LR':
            pre = 'R_' if s_ == 'R' else ''
            if separate_torso_shpitch:
                X.append(f'    <mesh name="{s_}_torso_shpitch" file="{pre}torso_shpitch.stl"/>\n')
            for b in ARM:
                if not mesh_exists(pre + b):
                    continue                     # e.g. shoulder_pitch_link: motor only, no mesh
                X.append(f'    <mesh name="{s_}_{b}" file="{pre}{b}.stl"/>\n')
                X.append(f'    <mesh name="{s_}_{b}_hull" file="{pre}{b}_hull.stl"/>\n')
    X.append('  </asset>\n  <worldbody>\n    <body name="base_link" childclass="pygmalion">\n      <freejoint name="root"/>\n')
    X.append(f'      <inertial pos="{c_b[0]:.6g} {c_b[1]:.6g} {c_b[2]:.6g}" mass="{m_b:.5g}" fullinertia="{fullinertia(I_b)}"/>\n')
    X.append('      <geom mesh="pelvis" class="visual"/>\n      <geom name="pelvis_hull" mesh="pelvis_hull" class="hull"/>\n')
    # waist-yaw motor stays visible even in LegOnly (2026-09-02): it is real pelvis-mounted
    # mass on the actual hardware regardless of what, if anything, is bolted above it -- see
    # massprops_fusion.LEG_ONLY_CUT for the mass-side half of this same decision.
    for g in motor_geoms(mp, 'pelvis', 'L'):
        X.append('      ' + g + '\n')
    for g in motor_geoms(mp, 'pelvis', 'R'):
        X.append('      ' + g + '\n')

    if not articulated and not leg_only:
        # old model geometry re-expressed at the hip-level base origin (+0.104 x, -0.059 z)
        X.append('      <geom name="base_torso_collision" class="collision" type="capsule" fromto="0.004 0 0.061  0.004 0 0.521" size="0.11"/>\n')
        X.append('      <geom name="base_head_collision" class="collision" type="sphere" pos="0.004 0 0.731" size="0.09"/>\n')
    X.append('      ' + fitted_capsule('base_pelvis_collision', 'pelvis') + '\n')
    X.append(f'      <site name="imu_in_base" size="0.03" pos="{IMU_POS[0]:.6g} {IMU_POS[1]:.6g} {IMU_POS[2]:.6g}"/>\n')
    for s in 'LR':
        sign = -1.0 if s == 'L' else 1.0
        depth = 3
        for b in CHAIN:
            jn, ax, rg = JOINT[b]
            # ankle_roll: unchanged design convention, +q = adduction/inversion on both legs.
            # hip_pitch, knee: NOT a convention choice -- a third viewer review (2026-09-02)
            # found R_hip_pitch_joint and R_knee_joint backwards while their L counterparts
            # were confirmed correct, so R needs the shared JOINT-dict axis negated to get
            # its own, independently-correct sign; L keeps the un-negated value.
            # hip_roll went the OTHER way and was REMOVED from this list: R_hip_roll_joint
            # was already correct before the 2026-09-02 axis flip, so mirroring the newly-
            # flipped L value would have broken R again -- L's base axis was flipped instead
            # (JOINT dict) and R now simply reuses it unmirrored, same as hip_pitch/hip_yaw.
            if s == 'R' and jn in ('ankle_roll', 'hip_pitch', 'knee'):
                ax = tuple(-v for v in ax)
                if jn == 'hip_pitch':
                    rg = tuple(np.radians(HIP_PITCH_R_CAP))
                elif jn == 'knee':
                    rg = tuple(np.radians(KNEE_R_CAP))
            elif s == 'R' and jn == 'hip_roll':
                # axis NOT negated (see the comment above -- R was already correct there),
                # but R still needs its own range: a fourth viewer review (2026-09-02) found
                # R_hip_roll_joint's direction correct but its ROM still the newly-flipped L
                # value, which is backwards for R specifically.
                rg = tuple(np.radians(HIP_ROLL_R_CAP))
            o = off[b].copy()
            if b == 'hip_pitch_link':
                o[1] = sign * abs(o[1])
            ind = '  ' * depth
            m, c, I = legs[b][s]
            X.append(f'{ind}<body name="{s}_{BNAME[b]}" pos="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}">\n')
            X.append(f'{ind}  <inertial pos="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" mass="{m:.5g}" fullinertia="{fullinertia(I)}"/>\n')
            X.append(f'{ind}  <joint name="{s}_{jn}_joint" pos="0 0 0" axis="{ax[0]} {ax[1]} {ax[2]}" range="{rg[0]} {rg[1]}"/>\n')
            if b != 'ankle_pitch_link':
                vis = f'{s}_{b}'
                if loop and b in ('shin', 'foot'):
                    vis = f'{s}_{b}_noloop'          # the cranks and rods are their own bodies now
                X.append(f'{ind}  <geom mesh="{vis}" class="visual"/>\n')
                X.append(f'{ind}  <geom name="{s}_{BNAME[b]}_hull" mesh="{s}_{b}_hull" class="hull"/>\n')
            for g in motor_geoms(mp, b, s):
                X.append(f'{ind}  ' + g + '\n')
            # ---- collision: one capsule that hugs each link's mesh (the hip links and the
            # thigh/shin included), a box for the foot. No hand-typed radii or offsets. ----
            pre = 'R_' if s == 'R' else ''
            if b in ('hip_pitch_link', 'hip_roll_link', 'thigh', 'shin'):
                stl = pre + (f'{b}_noloop' if (loop and b == 'shin') else b)
                X.append(f'{ind}  ' + fitted_capsule(f'{s}_{BNAME[b]}_collision', stl) + '\n')
            if b == 'foot':
                lo_f, hi_f = mesh_bounds(pre + ('foot_noloop' if loop else 'foot'))
                # the sole: FOOT_SOLE_SEGMENTS boxes tiling the same footprint/height a
                # single foot1_collision box had (2026-09-03, replaces the one-box sole) --
                # multi-point ground contact instead of one rigid slab. Names keep the
                # foot[1-7] pattern the task's contact config expects.
                for g in foot_sole_boxes(f'{s}_foot', lo_f, hi_f):
                    X.append(f'{ind}  ' + g + '\n')
                X.append(f'{ind}  <site name="{"left" if s == "L" else "right"}_foot" pos="{(lo_f[0]+hi_f[0])/2:.3f} 0 {lo_f[2]:.3f}" size="0.01"/>\n')
                if loop:
                    for g in foot_ball_sites(s, sign):
                        X.append(f'{ind}  ' + g + '\n')
            if b == 'shin' and loop:
                X.extend(l + '\n' for l in loop_xml(s, ind + '  ', mp, sign))
            depth += 1
        for b in reversed(CHAIN):
            depth -= 1
            X.append('  ' * depth + '</body>\n')
    # ---- upper body: torso on the waist yaw, then an arm on each shoulder ----
    if articulated:
        m, c, I = upper['torso']['L']
        o = off['torso']
        jn, ax, rg = JOINT['torso']
        X.append(f'      <body name="torso_link" pos="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}">\n')
        X.append(f'        <inertial pos="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" mass="{m:.5g}" fullinertia="{fullinertia(I)}"/>\n')
        X.append(f'        <joint name="waist_yaw_joint" pos="0 0 0" axis="{ax[0]} {ax[1]} {ax[2]}" range="{rg[0]} {rg[1]}"/>\n')
        X.append('        <geom mesh="torso" class="visual"/>\n')
        X.append('        <geom name="torso_hull" mesh="torso_hull" class="hull"/>\n')
        if separate_torso_shpitch:
            for s_ in 'LR':
                X.append(f'        <geom mesh="{s_}_torso_shpitch" class="visual" material="black"/>\n')
        for s_ in 'LR':
            for g in motor_geoms(mp, 'torso', s_):
                X.append('        ' + g + '\n')
        X.append('        ' + fitted_capsule('torso_collision', 'torso') + '\n')
        for s_ in 'LR':
            sgn = -1.0 if s_ == 'L' else 1.0
            oa = off['shoulder_pitch_link'].copy()
            oa[1] = sgn * abs(oa[1])
            ind = '        '
            for k, b in enumerate(ARM):
                jn, ax, rg = JOINT[b]
                # shoulder_roll: no longer mirrored for R (2026-09-02). A third viewer
                # review found L_shoulder_roll_joint still backwards while R was already
                # correct -- same "R was fine, don't re-derive it from a newly-flipped L"
                # pattern as hip_roll, so the JOINT-dict base axis was flipped instead (below)
                # and R now simply reuses it unmirrored.
                # shoulder_pitch: L confirmed correct as-is; only R was found backwards
                # (2026-09-02 viewer review). Unlike the roll/adduction joints this is not a
                # shared left/right convention, so R gets its own axis negation AND (per the
                # same user correction applied to SIGN_FLIPPED) its own negated-and-swapped
                # range instead of reusing DESIGN_CAP['shoulder_pitch'] as-is.
                if s_ == 'R' and jn == 'shoulder_pitch':
                    ax = tuple(-v for v in ax)
                    rg = tuple(np.radians(SHOULDER_PITCH_R_CAP))
                elif s_ == 'R' and jn == 'shoulder_roll':
                    # axis NOT negated (matches L, per the comment above), but a fifth
                    # viewer review (2026-09-02) found R_shoulder_roll_joint's ROM backwards
                    # while its axis was confirmed correct -- same pattern as hip_roll: only
                    # the range needs its own R-specific flip.
                    rg = tuple(np.radians(SHOULDER_ROLL_R_CAP))
                oo = oa if k == 0 else off['arm']
                m, c, I = upper[b][s_]
                ind += '  '
                # Both joint coordinates are -15 deg in the verified SemiFull pose.  The
                # right shoulder-roll axis is -X, hence its fixed body rotation is +15 deg
                # about frame X while the left is -15 deg (docs/92, COM match 0.000 mm).
                pose = ''
                if semi_fixed and b == 'arm':
                    physical_x = np.radians(-15.0 if s_ == 'L' else 15.0)
                    pose = f' euler="{physical_x:.9g} 0 0"'
                X.append(f'{ind}<body name="{s_}_{BNAME[b]}" pos="{oo[0]:.6g} {oo[1]:.6g} {oo[2]:.6g}"{pose}>\n')
                X.append(f'{ind}  <inertial pos="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" mass="{m:.5g}" fullinertia="{fullinertia(I)}"/>\n')
                if not semi_fixed:
                    X.append(f'{ind}  <joint name="{s_}_{jn}_joint" pos="0 0 0" axis="{ax[0]} {ax[1]} {ax[2]}" range="{rg[0]} {rg[1]}"/>\n')
                pre_b = ('R_' if s_ == 'R' else '') + b
                has_mesh = mesh_exists(pre_b)
                if has_mesh:
                    X.append(f'{ind}  <geom mesh="{s_}_{b}" class="visual"/>\n')
                    X.append(f'{ind}  <geom name="{s_}_{BNAME[b]}_hull" mesh="{s_}_{b}_hull" class="hull"/>\n')
                for g in motor_geoms(mp, b, s_):
                    X.append(f'{ind}  ' + g + '\n')
                # the arm is a straight rod with a fat shoulder end: fit the capsule to the
                # rod only (drop the top 12 cm) so it does not read as 44 mm all the way down.
                # A meshless body (shoulder_pitch_link: motor only) gets no collision capsule
                # either, same as ankle_pitch_link -- the torso/arm contact excludes already
                # cover both its neighbours, so there is nothing to hit through it.
                if has_mesh:
                    X.append(f'{ind}  ' + fitted_capsule(f'{s_}_{BNAME[b]}_collision', pre_b,
                                                         exclude=(lambda V: V[:, 2] > -0.12) if b == 'arm' else None) + '\n')
            for _ in ARM:
                X.append(ind + '</body>\n')
                ind = ind[:-2]
        X.append('      </body>\n')
    X.append('    </body>\n  </worldbody>\n  <contact>\n')
    # Only ADJACENT bodies are excluded from collision (user, 2026-08-23). The capsules are
    # fitted to the meshes, so non-adjacent links must not overlap at the zero pose on their
    # own; validate_robot.py checks that, and the hip-cluster / arm-hip exclusions the
    # previous model carried are gone.
    for s in 'LR':
        pairs = ['base_link'] + [f'{s}_{BNAME[b]}' for b in CHAIN]
        for a, b in zip(pairs[:-1], pairs[1:]):
            X.append(f'    <exclude body1="{a}" body2="{b}"/>\n')
        # the ankle cross is a 40 g universal-joint body between shin and foot: physically
        # those two are adjacent, so they are excluded across it
        X.append(f'    <exclude body1="{s}_shin_link" body2="{s}_foot_link"/>\n')
        if loop:
            for t in 'AB':          # the loop bodies carry no collision geoms; listed for clarity
                X.append(f'    <exclude body1="{s}_shin_link" body2="{s}_crank_{t}"/>\n')
                X.append(f'    <exclude body1="{s}_crank_{t}" body2="{s}_rod_{t}"/>\n')
                X.append(f'    <exclude body1="{s}_rod_{t}" body2="{s}_foot_link"/>\n')
    if articulated:
        X.append('    <exclude body1="base_link" body2="torso_link"/>\n')
        for s in 'LR':
            X.append(f'    <exclude body1="torso_link" body2="{s}_shoulder_pitch_link"/>\n')
            X.append(f'    <exclude body1="{s}_shoulder_pitch_link" body2="{s}_arm_link"/>\n')
            # The ONE non-adjacent exclusion kept: ArmR_Dummy is a placeholder rod that the CAD
            # itself drives 5.1 mm into the hip roll link at the zero pose (docs/88 s3c). Left
            # to the overlap resolver, clearing it costs the hip roll capsule half its radius
            # and the arm capsule all of it - a real leg link degraded for a dummy. Excluded
            # and reported instead; drop these two lines once the arm CAD is fixed.
            for h in ('hip_pitch_link', 'hip_roll_link'):
                X.append(f'    <exclude body1="{s}_arm_link" body2="{s}_{h}"/>\n')
    X.append('  </contact>\n')
    if loop:
        X.append('  <equality>\n')
        for s in 'LR':
            for t in 'AB':
                X.append(f'    <connect name="{s}_loop_{t}" site1="{s}_rod_{t}_end" site2="{s}_ball_{t}" solref="0.002 1" solimp="0.999 0.9999 0.0001"/>\n')
        X.append('  </equality>\n')
    X.append('''
  <sensor>
    <gyro name="imu_ang_vel" site="imu_in_base"/>
    <velocimeter name="imu_lin_vel" site="imu_in_base"/>
    <accelerometer name="imu_lin_acc" site="imu_in_base"/>
    <framezaxis name="imu_upvector" objtype="body" objname="world" reftype="site" refname="imu_in_base"/>
    <subtreeangmom name="root_angmom" body="base_link"/>
  </sensor>
</mujoco>
''')
    mjcf = ''.join(X)
    assert_left_is_plus_y(mjcf, f'{tag}.xml')
    open(f'{OUT_MJCF}/{tag}.xml', 'w').write(mjcf)
    # The overlap resolver compiles the just-written MJCF, so its mesh link must exist
    # before that compile.  The legacy output directory already had the link and hid this
    # ordering bug; a clean staging directory exposed it immediately.
    link = f'{OUT_MJCF}/{MJCF_MESHDIR}'
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(MESHDIR, link)
    trimmed = resolve_zero_pose_overlaps(f'{OUT_MJCF}/{tag}.xml')
    if trimmed:
        print('collision radii trimmed to clear non-adjacent overlaps at the zero pose:')
        for k, v in sorted(trimmed.items(), key=lambda kv: -kv[1]):
            print(f'   {k:32s} -{v * 1000:5.1f} mm')
    mjcf = open(f'{OUT_MJCF}/{tag}.xml').read()
    open(f'{OUT_URDF}/{tag}.xml', 'w').write(
        mjcf.replace(f'meshdir="{MJCF_MESHDIR}"', f'meshdir="{URDF_MESHDIR}"'))

    # ---- URDF ----
    U = [f'<?xml version="1.0"?>\n<robot name="{tag}">\n'
         f'  <!-- {motor_comment} -->\n']
    def link_xml(name, m, c, I, mesh, hull=None):
        s = f'  <link name="{name}">\n    <inertial>\n      <origin xyz="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" rpy="0 0 0"/>\n      <mass value="{m:.5g}"/>\n      {urdf_inertia(I)}\n    </inertial>\n'
        if mesh:
            s += f'    <visual>\n      <geometry><mesh filename="{URDF_MESHDIR}/{mesh}"/></geometry>\n    </visual>\n'
            s += f'    <collision>\n      <geometry><mesh filename="{URDF_MESHDIR}/{hull or mesh.replace(".stl", "_hull.stl")}"/></geometry>\n    </collision>\n'
        return s + '  </link>\n'
    U.append(link_xml('base_link', m_b, c_b, I_b, 'pelvis.stl'))
    for s in 'LR':
        for b in CHAIN:
            m, c, I = legs[b][s]
            mesh = None if b == 'ankle_pitch_link' else f'{"R_" if s == "R" else ""}{b}.stl'
            U.append(link_xml(f'{s}_{BNAME[b]}', m, c, I, mesh))
    for s in 'LR':
        sign = -1.0 if s == 'L' else 1.0
        parent = 'base_link'
        for b in CHAIN:
            jn, ax, rg = JOINT[b]
            if s == 'R' and jn in ('ankle_roll', 'hip_pitch', 'knee'):
                ax = tuple(-v for v in ax)
                if jn == 'hip_pitch':
                    rg = tuple(np.radians(HIP_PITCH_R_CAP))
                elif jn == 'knee':
                    rg = tuple(np.radians(KNEE_R_CAP))
            elif s == 'R' and jn == 'hip_roll':
                # axis NOT negated (see the comment above -- R was already correct there),
                # but R still needs its own range: a fourth viewer review (2026-09-02) found
                # R_hip_roll_joint's direction correct but its ROM still the newly-flipped L
                # value, which is backwards for R specifically.
                rg = tuple(np.radians(HIP_ROLL_R_CAP))
            o = off[b].copy()
            if b == 'hip_pitch_link':
                o[1] = sign * abs(o[1])
            U.append(f'  <joint name="{s}_{jn}_joint" type="revolute">\n    <origin xyz="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" rpy="0 0 0"/>\n'
                     f'    <parent link="{parent}"/>\n    <child link="{s}_{BNAME[b]}"/>\n    <axis xyz="{ax[0]} {ax[1]} {ax[2]}"/>\n'
                     f'    <limit lower="{rg[0]}" upper="{rg[1]}" effort="{EFFORT[jn]}" velocity="20"/>\n{dyn_xml(jn)}  </joint>\n')
            parent = f'{s}_{BNAME[b]}'
    if loop:
        for s in 'LR':
            links_, joints_ = loop_urdf(s, mp, -1.0 if s == 'L' else 1.0)
            U.extend(links_); U.extend(joints_)
    if articulated:
        m, c, I = upper['torso']['L']
        U.append(link_xml('torso_link', m, c, I, 'torso.stl'))
        for s_ in 'LR':
            for b in ARM:
                m, c, I = upper[b][s_]
                pre_b = ('R_' if s_ == 'R' else '') + b
                U.append(link_xml(f'{s_}_{BNAME[b]}', m, c, I,
                                  f'{pre_b}.stl' if mesh_exists(pre_b) else None))
        jn, ax, rg = JOINT['torso']
        o = off['torso']
        U.append(f'  <joint name="waist_yaw_joint" type="revolute">\n    <origin xyz="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" rpy="0 0 0"/>\n'
                 f'    <parent link="base_link"/>\n    <child link="torso_link"/>\n    <axis xyz="{ax[0]} {ax[1]} {ax[2]}"/>\n'
                 f'    <limit lower="{rg[0]}" upper="{rg[1]}" effort="{EFFORT[jn]}" velocity="20"/>\n{dyn_xml(jn)}  </joint>\n')
        for s_ in 'LR':
            sgn = -1.0 if s_ == 'L' else 1.0
            parent = 'torso_link'
            for k, b in enumerate(ARM):
                jn, ax, rg = JOINT[b]
                # shoulder_roll: no longer mirrored for R (2026-09-02). A third viewer
                # review found L_shoulder_roll_joint still backwards while R was already
                # correct -- same "R was fine, don't re-derive it from a newly-flipped L"
                # pattern as hip_roll, so the JOINT-dict base axis was flipped instead (below)
                # and R now simply reuses it unmirrored.
                # shoulder_pitch: L confirmed correct as-is; only R was found backwards
                # (2026-09-02 viewer review). Unlike the roll/adduction joints this is not a
                # shared left/right convention, so R gets its own axis negation AND (per the
                # same user correction applied to SIGN_FLIPPED) its own negated-and-swapped
                # range instead of reusing DESIGN_CAP['shoulder_pitch'] as-is.
                if s_ == 'R' and jn == 'shoulder_pitch':
                    ax = tuple(-v for v in ax)
                    rg = tuple(np.radians(SHOULDER_PITCH_R_CAP))
                elif s_ == 'R' and jn == 'shoulder_roll':
                    # axis NOT negated (matches L, per the comment above), but a fifth
                    # viewer review (2026-09-02) found R_shoulder_roll_joint's ROM backwards
                    # while its axis was confirmed correct -- same pattern as hip_roll: only
                    # the range needs its own R-specific flip.
                    rg = tuple(np.radians(SHOULDER_ROLL_R_CAP))
                o = off['shoulder_pitch_link'].copy() if k == 0 else off['arm'].copy()
                if k == 0:
                    o[1] = sgn * abs(o[1])
                if semi_fixed:
                    physical_x = np.radians((-15.0 if s_ == 'L' else 15.0) if b == 'arm' else 0.0)
                    U.append(f'  <joint name="{s_}_{jn}_joint" type="fixed">\n'
                             f'    <origin xyz="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" rpy="{physical_x:.9g} 0 0"/>\n'
                             f'    <parent link="{parent}"/>\n    <child link="{s_}_{BNAME[b]}"/>\n  </joint>\n')
                else:
                    U.append(f'  <joint name="{s_}_{jn}_joint" type="revolute">\n    <origin xyz="{o[0]:.6g} {o[1]:.6g} {o[2]:.6g}" rpy="0 0 0"/>\n'
                             f'    <parent link="{parent}"/>\n    <child link="{s_}_{BNAME[b]}"/>\n    <axis xyz="{ax[0]} {ax[1]} {ax[2]}"/>\n'
                             f'    <limit lower="{rg[0]}" upper="{rg[1]}" effort="{EFFORT[jn]}" velocity="20"/>\n{dyn_xml(jn)}  </joint>\n')
                parent = f'{s_}_{BNAME[b]}'
    U.append('</robot>\n')
    assert_left_is_plus_y(''.join(U), f'{tag}.urdf')
    open(f'{OUT_URDF}/{tag}.urdf', 'w').write(''.join(U))

    # ---- compile check ----
    import mujoco
    model = mujoco.MjModel.from_xml_path(f'{OUT_MJCF}/{tag}.xml')
    print(f'MJCF compiled: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, '
          f'{model.nmesh} meshes; total mass {model.body_subtreemass[1]:.3f} kg')
    upper_desc = ("ARTICULATED (5 joints: waist yaw + shoulder pitch/roll x2)" if articulated
                  else "ABSENT (LegOnly flange cut)" if leg_only else "placeholder lump on the pelvis")
    print(f'  base_link (pelvis) = {m_b:.3f} kg · upper body {m_u:.3f} kg {upper_desc}')
    for b in CHAIN:
        m, c, I = legs[b]['L']
        print(f'  {b:16s} {m:6.3f} kg  com {np.round(c, 4)}  I diag {np.round(np.diag(I), 5)}')
    if articulated:
        for b in ('torso',) + tuple(ARM):
            m, c, I = upper[b]['L']
            print(f'  {b:16s} {m:6.3f} kg  com {np.round(c, 4)}  I diag {np.round(np.diag(I), 5)}'
                  + ('' if b == 'torso' else '  (x2)'))
    print(f'  python total {total:.3f} kg')
    print(f'  variant {variant}: '
          f'{"17 movable joints" if variant == "full" else "13 movable joints, arms fixed at -15 deg roll" if variant == "semi" else "12 movable joints, no geometry/mass from waist flange upward"}')
    print(f"\n  {'joint':16s} {'range (deg)':>18s}   source / what stops it")
    for jn, lo_d, hi_d, src, bl, bh in rom_log:
        if leg_only and jn in ('waist_yaw', 'shoulder_pitch', 'shoulder_roll'):
            continue
        stop = f'{bl or "-"} | {bh or "-"}' if src.startswith('CAD') else ''
        print(f'  {jn:16s} [{lo_d:7.1f},{hi_d:7.1f}]   {src:26s} {stop}')
    print(f'-> {OUT_MJCF}/{tag}.xml · {OUT_URDF}/{tag}.urdf')


if __name__ == '__main__':
    main()
