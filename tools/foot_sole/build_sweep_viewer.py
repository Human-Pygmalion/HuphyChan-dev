#!/usr/bin/env python3
"""One interactive page for the stage-1 sole sweep: pick a variant, see its right-foot collision pieces in 3D (soft-arch wedge
pieces highlighted) next to v5, plus the ground-contact side profiles (lowest collision surface along the foot) of both.
Profiles come from mj_ray straight down through the compiled MJCF, so they show exactly what the simulator collides with.
  mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/build_sweep_viewer.py <assets_dir> <out.html> <name>=<label> ...
"""
import base64
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ITEMS = [a.split("=", 1) for a in sys.argv[3:]]


def b64(a, dt):
  return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def foot(name):
  xml = A / ("pygmalion_legonly_ab.xml" if name == "base" else f"pygmalion_legonly_ab_{name}.xml")
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  b = m.body("R_foot_link").id
  Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
  layers = {"sole": [], "soft": [], "bracket": []}
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_group[g] != 3:
      continue
    Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
      s = m.geom_size[g]
      v = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
    else:
      continue
    w = (v @ Rg.T + pg - pb) @ Rb  # foot frame
    h = ConvexHull(w)
    f = h.simplices.copy()
    c = w.mean(0)
    n = np.cross(w[f[:, 1]] - w[f[:, 0]], w[f[:, 2]] - w[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, w[f[:, 0]] - c) < 0
    f[flip] = f[flip][:, ::-1]
    soft = m.geom_solref[g][0] > 0.03
    low = w[:, 2].min() < -0.044
    key = "soft" if soft else ("sole" if low else "bracket")
    layers[key].append((w * 1000, f))
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
    out[k] = {"v": b64(np.vstack(V), "<f4"), "f": b64(np.vstack(F), "<u4")}
  # lowest collision surface along x at the midline and 45 mm toward the medial side (right foot: +y), by upward rays from below the sole
  geomgroup = np.array([0, 0, 0, 1, 0, 0], np.uint8)
  prof = {}
  for lab, y in (("mid", 0.0), ("medial", 0.045)):
    zs = []
    for x in np.arange(-0.082, 0.182, 0.002):
      p0 = pb + Rb @ np.array([x, y, -0.10])  # from below the sole, upward: the first hit is the lowest collision surface
      gid = np.zeros(1, np.int32)
      dist = mujoco.mj_ray(m, d, p0, Rb[:, 2], geomgroup, 1, -1, gid)
      zs.append(round(float((-0.10 + dist) * 1000), 2) if dist >= 0 and m.geom_bodyid[gid[0]] == b else None)
    prof[lab] = zs
  return out, prof


data = {"x": [round(float(x) * 1000, 1) for x in np.arange(-0.082, 0.182, 0.002)], "variants": []}
for name, label in ITEMS:
  lay, prof = foot(name)
  data["variants"].append({"name": name, "label": label, "layers": lay, "prof": prof})
  print(name, {k: (v is not None) for k, v in lay.items()}, flush=True)
tpl = Path(__file__).with_name("sweep_viewer_template.html").read_text()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tpl.replace("__DATA__", json.dumps(data)))
print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")
