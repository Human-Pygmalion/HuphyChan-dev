#!/usr/bin/env python3
"""Interactive 3D page of the WHOLE robot model, training model next to a foot variant, in the training start pose
(INIT_STATE joint angles, base height + PYG_INIT_DZ for the variant), visual and collision layers.
Usage (mjlab venv, stock mjlab on the path):
  PYTHONPATH=<stock_mjlab/src>:<branch>/mjlab_policy mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/build_robot_viewer.py \
      <assets_dir> <out.html> [variant=v5full] [dz=0.014]
"""
import base64
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

import pygmalion_mjlab.robot as pr

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "v5full"
DZ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.014


def b64(a, dt):
  return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def geom_world(m, d, g):
  t = m.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    if m.geom_group[g] == 3:
      h = ConvexHull(v)
      f = h.simplices.copy()
      c = v.mean(0)
      n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
      flip = np.einsum("ij,ij->i", n, v[f[:, 0]] - c) < 0
      f[flip] = f[flip][:, ::-1]
    else:
      f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
  elif t == mujoco.mjtGeom.mjGEOM_BOX:
    bx = trimesh.creation.box(extents=2 * m.geom_size[g])
    v, f = bx.vertices, bx.faces
  elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
    cp = trimesh.creation.capsule(height=2 * m.geom_size[g][1], radius=m.geom_size[g][0], count=[12, 12])
    v, f = cp.vertices, cp.faces
  elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
    sp = trimesh.creation.icosphere(subdivisions=2, radius=m.geom_size[g][0])
    v, f = sp.vertices, sp.faces
  elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
    cy = trimesh.creation.cylinder(radius=m.geom_size[g][0], height=2 * m.geom_size[g][1], sections=16)
    v, f = cy.vertices, cy.faces
  else:
    return None
  R = d.geom_xmat[g].reshape(3, 3)
  return v @ R.T + d.geom_xpos[g], np.asarray(f)


def model_layers(xml, base_z):
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  d.qpos[:] = m.qpos0
  d.qpos[0:3] = (0.0, 0.0, base_z)
  d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for jn, q in pr.INIT_STATE.joint_pos.items():
    d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q
  mujoco.mj_kinematics(m, d)
  layers = {"visual": [], "collision": [], "foot": []}
  for g in range(m.ngeom):
    if m.geom_group[g] not in (2, 3):
      continue
    r = geom_world(m, d, g)
    if r is None:
      continue
    v, f = r
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    key = "visual" if m.geom_group[g] == 2 else ("foot" if "_foot" in name else "collision")
    layers[key].append((v, f))
  out = {}
  for k, items in layers.items():
    if not items:
      out[k] = None
      continue
    V, F, off = [], [], 0
    for v, f in items:
      V.append(v)
      F.append(f + off)
      off += len(v)
    V, F = np.vstack(V) * 1000.0, np.vstack(F)
    out[k] = {"v": b64(V, "<f4"), "f": b64(F, "<u4"), "nf": int(len(F))}
  low = min(np.min(d.geom_xpos[g][2]) for g in range(m.ngeom))
  return out, float(m.body_subtreemass[1])


base, mb = model_layers(A / "pygmalion_legonly_ab.xml", pr.INIT_STATE.pos[2])
var, mv = model_layers(A / f"pygmalion_legonly_ab_{VARIANT}.xml", pr.INIT_STATE.pos[2] + DZ)
data = {"models": [{"name": "training model (3 foot boxes)", "dy": 0.0, "layers": base, "mass": mb},
                   {"name": f"{VARIANT} (sole collision pieces, +{DZ * 1000:.0f} mm start height)", "dy": 700.0, "layers": var, "mass": mv}]}
tpl = Path(__file__).with_name("robot_viewer_template.html").read_text()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tpl.replace("__DATA__", json.dumps(data)))
print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB", {m["name"]: {k: (v["nf"] if v else 0) for k, v in m["layers"].items()} for m in data["models"]})
