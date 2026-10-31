#!/usr/bin/env python3
"""Build a self-contained interactive 3D page (three.js) of the right foot in the foot_link frame [mm]:
robot visual mesh, CAD sole (MedialPlusX STL), v3full collision (sole strips + plate slab + upper boxes), training-model boxes.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/build_foot_viewer.py <assets_dir> <out.html> [variant] [cad_stl] [report_json]
       defaults: v3full, sole_v3/cad/R_sole_v3_MedialPlusX.stl, sole_v3/v3_collision_report.json
"""
import base64
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "v3full"
CAD = A / (sys.argv[4] if len(sys.argv) > 4 else "sole_v3/cad/R_sole_v3_MedialPlusX.stl")
REP = A / (sys.argv[5] if len(sys.argv) > 5 else "sole_v3/v3_collision_report.json")


def b64(a, dt):
  return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def pack(v, f):
  return {"v": b64(v, "<f4"), "f": b64(f, "<u4"), "nv": int(len(v)), "nf": int(len(f))}


def geom_mesh(m, g):
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, m.geom_quat[g])
  R, p = mat.reshape(3, 3), m.geom_pos[g]
  t = m.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    if m.geom_group[g] == 3:
      h = ConvexHull(v)
      v, f = h.points, h.simplices
      # outward winding for three.js
      c = v.mean(0)
      n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
      flip = np.einsum("ij,ij->i", n, v[f[:, 0]] - c) < 0
      f[flip] = f[flip][:, ::-1]
    else:
      f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
  elif t == mujoco.mjtGeom.mjGEOM_BOX:
    bx = trimesh.creation.box(extents=2 * m.geom_size[g])
    v, f = bx.vertices, bx.faces
  else:
    return None
  return (v @ R.T + p) * 1000.0, np.asarray(f)


def foot(xml):
  m = mujoco.MjModel.from_xml_path(str(xml))
  b = m.body("R_foot_link").id
  vis, col = [], []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b:
      continue
    r = geom_mesh(m, g)
    if r is None:
      continue
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "visual"
    (vis if m.geom_group[g] == 2 else col if m.geom_group[g] == 3 else []).append((name, *r))
  return vis, col


vis, col = foot(A / f"pygmalion_legonly_ab_{VARIANT}.xml")
_, base = foot(A / "pygmalion_legonly_ab.xml")
cad = trimesh.load(CAD)
data = {"visual": [pack(v, f) for _, v, f in vis], "cad": pack(cad.vertices * 1000.0, cad.faces), "collision": [], "base": []}
for name, v, f in col:
  zmin, zmax = v[:, 2].min(), v[:, 2].max()
  kind = "box" if zmax > -30 else ("plate" if zmin > -43.2 else "strip")  # boxes reach above the plate; plate slab z -43..-35
  data["collision"].append(dict(pack(v, f), name=name, kind=kind, x0=float(v[:, 0].min()), x1=float(v[:, 0].max())))
data["collision"].sort(key=lambda c: (c["kind"] != "strip", c["x0"]))
for name, v, f in base:
  data["base"].append(pack(v, f))
rep = json.loads(REP.read_text())
probe_f = A / "sole_v3/cad/fusion_probe_vs_mjcf.json"
probe = json.loads(probe_f.read_text()) if probe_f.exists() else {}
pv = np.array([abs(r[4] - r[2]) for r in probe[VARIANT]]) if VARIANT in probe else np.array([np.nan])
data["stats"] = dict(variant=VARIANT, n_strips=sum(c["kind"] == "strip" for c in data["collision"]), fill_max=rep["fill_max_mm"],
                     contact_model=rep["coverage"]["contact_model_cm2"], contact_true=rep["coverage"]["contact_true_cm2"],
                     probe_mean=float(pv.mean()), probe_max=float(pv.max()), probe_n=int(len(pv)))
tpl = (Path(__file__).with_name("foot_viewer_template.html")).read_text()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tpl.replace("__DATA__", json.dumps(data)))
print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB", data["stats"])
