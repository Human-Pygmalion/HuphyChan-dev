#!/usr/bin/env python3
"""loop_bent_keyframe.py - re-express the BENT reset keyframe's ankle-loop joint angles on a
given loop model (2026-09-05, backlog docs/106 "리셋 폐루프 찢김 미수정").

Why: `pygmalion_constants._bent_joint_pos()` copies the crank/rod/ankle angles of the bent
crouch from `pygmalion_v3_printed_loop_bent.json`. On the v30 build the crank hinge axes are
MIRRORED per side (L_crank_A -Y / L_crank_B +Y / R_crank_A +Y / R_crank_B -Y), so the v3
numbers (both cranks -17.12 deg on both sides) put L_crank_A and R_crank_B on the wrong side
of the linkage -> the four `connect` equalities start ~37 mm torn every reset and the crank PD
has to yank them shut in the first ~0.25 s (measured 37.23 mm at reset, 2026-09-05).

What: for each side, weld the shin in space (no gravity), drive the two crank servos and
Newton-iterate the crank pair until the passive ankle hinges read the design crouch
(pitch +0.36 rad, roll 0) with closure < 0.5 mm, then record EVERY loop joint angle
(crank A/B, rod universal hinges u1/u2 x2, ankle pitch/roll). Output has the same schema as
the v3 file so `_bent_joint_pos()` can consume it unchanged.

  .venv/bin/python3 tools/robot_model/loop_bent_keyframe.py \
      --tag LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, mujoco

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The training loader reads the mjlab asset-zoo copy, whose relative meshdir resolves; the
# pygmalion_locomotion/assets copy declares meshdir="assets_v30_armfix/" but the folder there
# is named meshes_v30_armfix/ (compile fails on pelvis.stl). Try the asset zoo first.
XML_DIRS = (REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls",
            REPO / "pygmalion_locomotion/assets/pygmalion_v2")
OUT_DIR = REPO / "pygmalion_locomotion/assets/pygmalion_v2"   # where the v3 bent json lives (constants.py reads here)


def xml_path(tag: str) -> Path:
  for d in XML_DIRS:
    if (d / f"{tag}.xml").exists():
      return d / f"{tag}.xml"
  raise SystemExit(f"{tag}.xml not found in {[str(d) for d in XML_DIRS]}")
KP, KD, DT = 200.0, 5.0, 0.0005          # same servo as tools/robot_model/loop_ankle_verify.py
# 2026-09-14: the ankle target is DERIVED from the crouch, not hard-coded.
#
# Geometry: with the hip flexed h and the knee flexed k, the thigh leans h and the shank
# leans (h - k), so the sole is level only when the ankle takes back (k - h).
#   MID crouch  (PYG_INIT_MID=1, what every LegOnly run uses)  h=0.175 k=0.35 -> 0.175 rad
#   DEEP crouch                                                h=0.32  k=0.67 -> 0.350 rad
#
# The value that stood here, 0.36, is the DEEP figure. Every run since has used the MID
# crouch with the DEEP ankle, which left the soles tilted 10.60 deg toes-up: heel 2.5 mm
# below ground, toe 45.4 mm above, only 4 of the 12 sole corners touching, so the robot
# reset onto its heel line every episode. Flagged as a backlog item on 2026-09-03
# (docs/experiments/2026-09-03_legonly_ab_v2.md) and measured 2026-09-14
# (docs/experiments/2026-09-14_v4_standing_q_torque_gains.md section 2c).
#
# --ankle-pitch overrides it; omit and it follows --hip/--knee.
TARGET_ROLL_RAD = 0.0
LOOP_JOINTS = ("crank_A_joint", "rod_A_u1", "rod_A_u2", "crank_B_joint", "rod_B_u1", "rod_B_u2",
               "ankle_pitch_joint", "ankle_roll_joint")


def build(tag: str, side: str) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(xml_path(tag)))
  spec.body("base_link").pos[2] = 1.5
  spec.option.gravity[:] = [0, 0, 0]
  spec.option.timestep = DT
  w = spec.add_equality(); w.type = mujoco.mjtEq.mjEQ_WELD; w.objtype = mujoco.mjtObj.mjOBJ_BODY
  w.name1 = f"{side}_shin_link"; w.name = "hold_shin"
  for t in "AB":
    a = spec.add_actuator(); a.name = f"{side}_crank_{t}_servo"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT; a.target = f"{side}_crank_{t}_joint"
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0] = KP; a.biasprm[1] = -KP; a.biasprm[2] = -KD
    a.forcerange[:] = [-60, 60]; a.forcelimited = True
  return spec.compile()


def jq(m, d, name):  # joint angle by name
  return d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]]


def closure_mm(m, d, side):
  s = lambda n: d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)]
  return 1e3 * max(np.linalg.norm(s(f"{side}_rod_{t}_end") - s(f"{side}_ball_{t}")) for t in "AB")


def settle(m, d, side, cA, cB, steps=800):
  aA = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_crank_A_servo")
  aB = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_crank_B_servo")
  c0 = np.array([d.ctrl[aA], d.ctrl[aB]])
  for k in range(steps):
    f = min(1.0, k / (steps / 2))
    d.ctrl[aA], d.ctrl[aB] = c0 + f * (np.array([cA, cB]) - c0)
    mujoco.mj_step(m, d)
  return jq(m, d, f"{side}_ankle_pitch_joint"), jq(m, d, f"{side}_ankle_roll_joint")


def solve_side(tag: str, side: str):
  m = build(tag, side); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
  c = np.zeros(2)                                    # crank guess: neutral
  for it in range(25):
    p, r = settle(m, d, side, *c)
    err = np.array([p - TARGET_PITCH_RAD, r - TARGET_ROLL_RAD])
    if np.abs(err).max() < math.radians(0.02) and closure_mm(m, d, side) < 0.5:
      break
    h = math.radians(1.0); J = np.zeros((2, 2))        # finite-difference Jacobian d(pitch,roll)/d(cA,cB)
    for i in range(2):
      dc = c.copy(); dc[i] += h
      pi, ri = settle(m, d, side, *dc, steps=400)
      J[:, i] = (np.array([pi, ri]) - np.array([p, r])) / h
      settle(m, d, side, *c, steps=400)
    c = c - np.linalg.solve(J, err)
  p, r = settle(m, d, side, *c)
  out = {f"{side}_{j}": float(jq(m, d, f"{side}_{j}")) for j in LOOP_JOINTS}
  return out, closure_mm(m, d, side), math.degrees(p), math.degrees(r), it + 1


def joint_sign(m, name: str) -> float:
  """+1 if the joint's long end is positive (same rule as pygmalion_constants.signed_pose)."""
  j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name); lo, hi = m.jnt_range[j]
  return 1.0 if hi >= -lo else -1.0


def bent_base_z(tag: str, joint_pos: dict, hip=0.32, knee=0.67, margin=0.005) -> tuple[float, float]:
  """Base height that puts the lowest sole collision geom exactly `margin` above the floor in
  the bent crouch (whole robot, both legs, no gravity step - pure kinematics)."""
  spec = mujoco.MjSpec.from_file(str(xml_path(tag))); m = spec.compile(); d = mujoco.MjData(m)
  d.qpos[:] = 0; d.qpos[3] = 1.0                       # base at origin, identity orientation
  def setj(n, v): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
  for side in "LR":
    setj(f"{side}_hip_pitch_joint", joint_sign(m, f"{side}_hip_pitch_joint") * hip)
    setj(f"{side}_knee_joint", joint_sign(m, f"{side}_knee_joint") * knee)
  for n, v in joint_pos.items(): setj(n, v)
  mujoco.mj_forward(m, d)
  zmin = 1e9
  for g in range(m.ngeom):
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    if "foot" in b.lower() and (m.geom_contype[g] | m.geom_conaffinity[g]):
      half = m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0]
      zmin = min(zmin, d.geom_xpos[g][2] - half)
  return float(-zmin + margin), float(zmin)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--tag", required=True, help="loop model xml stem in pygmalion_locomotion/assets/pygmalion_v2/")
  ap.add_argument("--out", default=None, help="default: <tag>_bent.json next to the xml")
  ap.add_argument("--hip", type=float, default=0.175, help="hip_pitch flexion magnitude [rad] used for base_z (PYG_INIT_MID crouch = 0.175; deep = 0.32)")
  ap.add_argument("--knee", type=float, default=0.35, help="knee flexion magnitude [rad] used for base_z (PYG_INIT_MID crouch = 0.35; deep = 0.67)")
  ap.add_argument("--ankle-pitch", type=float, default=None,
                  help="ankle_pitch target [rad]; default = knee - hip, the value that puts the sole flat")
  a = ap.parse_args()
  global TARGET_PITCH_RAD
  TARGET_PITCH_RAD = a.ankle_pitch if a.ankle_pitch is not None else (a.knee - a.hip)
  print(f"[bent] ankle_pitch target = {TARGET_PITCH_RAD:.6f} rad = {math.degrees(TARGET_PITCH_RAD):.3f} deg"
        f"  ({'given' if a.ankle_pitch is not None else 'derived from knee - hip'})")
  joint_pos, worst = {}, 0.0
  for side in ("L", "R"):
    jp, cl, p, r, its = solve_side(a.tag, side)
    joint_pos.update(jp); worst = max(worst, cl)
    print(f"[{side}] {its} Newton steps -> ankle pitch {p:+.3f} roll {r:+.3f} deg, closure {cl:.3f} mm")
    for k in LOOP_JOINTS[:6]:
      print(f"      {side}_{k:16s} {math.degrees(jp[f'{side}_{k}']):+8.3f} deg")
  base_z, zmin0 = bent_base_z(a.tag, joint_pos, hip=a.hip, knee=a.knee)
  print(f"bent crouch: sole lowest point at base z=0 is {zmin0*1e3:+.1f} mm -> base_z = {base_z:.4f} m (+5 mm margin)")
  out = Path(a.out) if a.out else OUT_DIR / f"{a.tag}_bent.json"
  json.dump({"base_z": round(base_z, 4), "base_z_crouch": {"hip_pitch": a.hip, "knee": a.knee}, "pose": f"KNEES_BENT (ankle_pitch +{TARGET_PITCH_RAD}, roll 0) re-expressed on {a.tag} - "
                     "crank axes are mirrored per side on this build, so the v3 numbers do not transfer",
             "closure_mm": round(worst, 4), "target_deg": {"ankle_pitch": math.degrees(TARGET_PITCH_RAD), "ankle_roll": 0.0},
             "joint_pos": joint_pos}, open(out, "w"), indent=1)
  print(f"wrote {out}  (worst closure {worst:.3f} mm)")
  return 0 if worst < 1.0 else 1


if __name__ == "__main__":
  sys.exit(main())
