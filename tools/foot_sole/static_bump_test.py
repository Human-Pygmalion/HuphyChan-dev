#!/usr/bin/env python3
"""Static bump test, no policy: a free foot (the variant's simulated collision geometry and inertia) is loaded at the ankle origin
with about half the robot weight, and lowered onto a single bump under the sole. Records the equilibrium tilt, which is the
"see-saw angle" of the survey calculation (research_notes/험지 발바닥 아치 역할과 깊이 기준/arch_terrain_calc.py).

Bump: sphere of radius r = h half buried (top height h, the survey's default) or a flat-top box 20 x 20 mm (height h), at foot-frame
x = 40 / 80 (below the ankle) / 108 (arch apex) / 140 / 200 mm from the heel tip and y = 0, +-40 mm (medial +, right foot).
Two quantities per case: (1) seesaw_med/lat_deg: GEOMETRIC roll angle about the bump (foot flat in pitch, held by the ankle) until another
part of the sole touches the ground, medial side down / lateral side down: the survey's "see-saw angle", computed on the simulated
collision pieces; (2) roll/pitch/tilt of a dynamic run with a PASSIVE ankle (ball joint) under an 11 kg carriage: where the foot settles.
A free foot on a centre-line bump pitches toward the nearer of heel/toe (angle ~ h / distance), smaller than the roll see-saw.
Contact settings = training: foot condim 3, friction 0.6, priority 1, per-geom solref (0.02, 0.08 soft arch); ground solref 0.03.
  PYG_FOOT_XML=<variant xml> mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/static_bump_test.py <assets> <out.csv> <variant> ...
"""
import csv
import itertools
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VARIANTS = sys.argv[3:]
CARRIAGE_KG = 11.0   # half the robot (24.3 kg) minus the foot: a carriage on the ankle origin, free in z only. It gives the contact the
                     # inertia of the robot (a 1.2 kg foot alone would sink 33 mm: the solref constraint is a spring per unit mass)
XS = [40, 80, 108, 140, 200]
YS = [0, 40, -40]
HS = [3, 5, 8, 10, 20]
SHAPES = ["sphere", "box"]
MODES = ["free"]   # roll: ankle holds pitch and yaw (foot flat), roll hinge at the ankle = the survey's see-saw angle;
                           # free: a ball joint at the ankle (passive ankle: the foot also pitches about the bump)


def foot_geoms(variant):
  """compiled foot collision geoms (foot frame) + inertial of the right foot of a variant, with the training collision settings"""
  os.environ["PYG_FOOT_XML"] = "pygmalion_legonly_ab.xml" if variant == "base" else f"pygmalion_legonly_ab_{variant}.xml"
  sys.path[:0] = [str(A.parents[1]), str(os.environ["MJLAB_SRC"])]
  import importlib

  import pygmalion_mjlab.robot as pr
  importlib.reload(pr)
  spec = pr.get_spec()
  pr.COLLISION.edit_spec(spec)
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  b = m.body("R_foot_link").id
  Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
  gl = []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_group[g] != 3:
      continue
    Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    rec = dict(type=int(m.geom_type[g]), condim=int(m.geom_condim[g]), priority=int(m.geom_priority[g]), friction=m.geom_friction[g].copy(),
               solref=m.geom_solref[g].copy(), solimp=m.geom_solimp[g].copy())
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
      rec["verts"] = (v @ Rg.T + pg - pb) @ Rb
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      rec["size"] = m.geom_size[g].copy()
      rec["pos"] = (pg - pb) @ Rb
      Rl = Rb.T @ Rg
      q = np.zeros(4)
      mujoco.mju_mat2Quat(q, Rl.ravel())
      rec["quat"] = q
    else:
      continue
    gl.append(rec)
  inertial = dict(mass=float(m.body_mass[b]), ipos=m.body_ipos[b].copy(), iquat=m.body_iquat[b].copy(), inertia=m.body_inertia[b].copy())
  return gl, inertial


def build(gl, inertial, shape, x_mm, y_mm, h_mm, mode):
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.001
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.noslip_iterations = 0
  wb = spec.worldbody
  fl = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(1.0, 1.0, 0.5), pos=(0, 0, -0.5), solref=(0.03, 1.0),
                   solimp=(0.9, 0.95, 0.003, 0.5, 2.0), friction=(0.6, 0.005, 0.0001))
  x, y, h = (x_mm - 80.0) / 1000.0, y_mm / 1000.0, h_mm / 1000.0   # foot x_f from the heel tip -> world x (ankle at x = 0)
  if shape == "sphere":
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(h, 0, 0), pos=(x, y, 0.0), solref=(0.03, 1.0), solimp=(0.9, 0.95, 0.003, 0.5, 2.0),
                friction=(0.6, 0.005, 0.0001))
  else:
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.010, 0.010, h / 2), pos=(x, y, h / 2), solref=(0.03, 1.0),
                solimp=(0.9, 0.95, 0.003, 0.5, 2.0), friction=(0.6, 0.005, 0.0001))
  cb = wb.add_body(name="carriage", pos=(0, 0, 0.09))
  cb.mass, cb.inertia, cb.explicitinertial = CARRIAGE_KG, (0.05, 0.05, 0.05), True
  cb.add_joint(name="z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1))
  fb = cb.add_body(name="foot")
  if mode == "free":
    fb.add_joint(name="ankle", type=mujoco.mjtJoint.mjJNT_BALL)
  else:
    fb.add_joint(name="roll", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(1, 0, 0))
  fb.mass, fb.ipos, fb.iquat, fb.inertia = inertial["mass"], inertial["ipos"], inertial["iquat"], inertial["inertia"]
  fb.explicitinertial = True
  for k, g in enumerate(gl):
    kw = dict(condim=g["condim"], priority=g["priority"], friction=g["friction"], solref=g["solref"], solimp=g["solimp"],
              contype=1, conaffinity=1)
    if g["type"] == mujoco.mjtGeom.mjGEOM_MESH:
      me = spec.add_mesh(name=f"m{k}")
      me.uservert = g["verts"].ravel().astype(np.float32)
      fb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{k}", **kw)
    else:
      fb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=g["size"], pos=g["pos"], quat=g["quat"], **kw)
  return spec.compile()


def hull_vertices(gl):
  """all convex-piece vertices of the foot (foot frame, m): bottoms of the hulls are the only places a plane can touch"""
  out = []
  for g in gl:
    if g["type"] == mujoco.mjtGeom.mjGEOM_MESH:
      out.append(g["verts"])
    else:
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, g["quat"])
      s = g["size"]
      c = np.array([[a, b, c_] for a in (-s[0], s[0]) for b in (-s[1], s[1]) for c_ in (-s[2], s[2])])
      out.append(c @ R.reshape(3, 3).T + g["pos"])
  return np.vstack(out)


def seesaw(V, zc, x_m, y_m, h_m, shape):
  """Geometric see-saw angle [deg]: the foot stands flat (pitch 0) on the bump top; it rolls about the bump (point for the sphere, the
  box edge on the side it tips to) until another part of the sole touches the ground. Returns (medial side down, lateral side down).
  V: foot vertices, zc: sole height under the bump relative to the ankle origin."""
  A = h_m - zc                               # ankle height when the sole rests on the bump top
  out = []
  for sgn in (-1.0, +1.0):                   # sgn +1: lateral side down (phi > 0 lifts +y, the medial side), -1: medial side down
    yp = y_m + (0.0 if shape == "sphere" else -sgn * 0.010)
    ang = np.radians(np.arange(0.0, 45.0, 0.05))
    res = 45.0
    for a in ang:
      phi = sgn * a
      dy, dz = V[:, 1] - yp, V[:, 2] + A - h_m
      zr = h_m + dy * math.sin(phi) + dz * math.cos(phi)
      if zr.min() <= 0.0 and a > 0:
        res = math.degrees(a)
        break
    out.append(res)
  return out[0], out[1]                      # medial down, lateral down


def run_case(args):   # carriage starts 70 mm above the origin height: the foot is dropped on the bump
  gl, inertial, shape, x_mm, y_mm, h_mm, side, mode = args
  m = build(gl, inertial, shape, x_mm, y_mm, h_mm, mode)
  m.dof_damping[:] = 0.2
  m.dof_damping[0] = 30.0   # the carriage slide
  d = mujoco.MjData(m)
  b = m.body("foot").id
  mujoco.mj_forward(m, d)
  gid = np.zeros(1, np.int32)
  dist = mujoco.mj_ray(m, d, np.array([(x_mm - 80.0) / 1000.0, y_mm / 1000.0, -0.3]), np.array([0.0, 0.0, 1.0]),
                       None, 0, -1, gid)       # flg_static = 0: skip the floor and the bump, hit the foot
  zc = (-0.3 + dist) - d.xpos[b][2] if dist >= 0 else float("nan")
  ss_med, ss_lat = seesaw(hull_vertices(gl), zc, 0.0, y_mm / 1000.0, h_mm / 1000.0, shape)
  tilt0 = math.radians(0.5) * side                      # initial roll about the foot x axis (side +1 = medial side down on a right foot)
  d.qpos[0] = 0.0
  if mode == "free":
    d.qpos[1:5] = np.array([math.cos(tilt0 / 2), math.sin(tilt0 / 2), 0.0, 0.0])
  else:
    d.qpos[1] = tilt0
  mujoco.mj_forward(m, d)
  for _ in range(2000):
    mujoco.mj_step(m, d)
  R = d.xmat[b].reshape(3, 3)
  roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))      # rotation about world x: + lifts the +y (medial) side, the lateral edge goes down
  pitch = math.degrees(-math.asin(np.clip(R[2, 0], -1, 1)))
  tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
  # contacts carrying load: bump share and the support extent
  fb = fs = 0.0
  pts = []
  for i in range(d.ncon):
    c = d.contact[i]
    f6 = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f6)
    g1, g2 = c.geom1, c.geom2
    if g1 == 0 or g2 == 0:     # floor
      fs += f6[0]
    elif g1 == 1 or g2 == 1:   # bump
      fb += f6[0]
    if f6[0] > 0.5:
      pts.append(c.pos)
  vel = float(np.linalg.norm(d.qvel))
  return dict(seesaw_med_deg=ss_med, seesaw_lat_deg=ss_lat, sole_under_bump_mm=zc * 1000, roll_deg=roll, pitch_deg=pitch, tilt_deg=tilt, bump_force_share=fb / max(fb + fs, 1e-9), n_loaded_contacts=len(pts), settle_vel=vel,
              z_ankle_mm=float(d.xpos[b][2] * 1000))


def init_worker(src):
  os.environ["MJLAB_SRC"] = src


if __name__ == "__main__":
  src = str(Path(os.environ.get("MJLAB_SRC", A.parents[3] / "stock_mjlab/src")))
  os.environ["MJLAB_SRC"] = src
  rows = []
  for v in VARIANTS:
    gl, inertial = foot_geoms(v)
    jobs = [(gl, inertial, sh, x, y, h, side, mode) for mode, sh, x, y, h, side in itertools.product(MODES, SHAPES, XS, YS, HS, (1,))]
    with Pool(6) as pool:
      res = pool.map(run_case, jobs, chunksize=8)
    for (g_, i_, sh, x, y, h, side, mode), r in zip(jobs, res):
      rows.append(dict(variant=v, mode=mode, shape=sh, x_mm=x, y_mm=y, h_mm=h, start_side=side, **r))
    print(v, len(jobs), "cases done", flush=True)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print("wrote", OUT, len(rows))
