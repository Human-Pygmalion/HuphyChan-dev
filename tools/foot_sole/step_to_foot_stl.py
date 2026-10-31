#!/usr/bin/env python3
"""Tessellate the v3 sole STEPs (FEET document frame, mm) into foot_link-frame STLs (metres) for the right and left foot.

The STEPs are the same files the CAD assembly uses (`Sole_v3_byClaude(Mirror)` = MedialPlusX on the right foot, checked on
2026-10-31 by 50 ray probes in Fusion: |difference to the v3full MJCF| <= 0.085 mm).
Frame map (right foot, see compare_fusion_vs_mjcf.py): x = 145 - y_d, y = x_d + 123.7, z = z_d + 800 [mm]; left foot = mirror y.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/step_to_foot_stl.py <step_zip> <out_dir> [MedialPlusX]
"""
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

ZIP, OUT = Path(sys.argv[1]), Path(sys.argv[2])  # ZIP: the v3 STEP zip, or a folder of STEP files (v5: foot_cad/v5/out)
VAR = sys.argv[3] if len(sys.argv) > 3 else "MedialPlusX"
if ZIP.is_dir() and (ZIP / "Feet_Up.step").exists():  # v5 parts exported from the live CAD (upload_receiver.py), user-edited carrier
  PARTS = {"carrier": "Sole_Carrier_v5_ABS.step", "tape": "DS_Tape_1mm.step", "sheet": "Rubber_Sole_4mm.step", "feet_up": "Feet_Up.step"}
  VAR = "v5cad_" + VAR
elif ZIP.is_dir():  # v5 right foot
  PARTS = {"carrier": "Pygmalion_v5_Sole_Carrier_v5_ABS_RightFoot_MedialPlusX.step", "tape": "Pygmalion_v5_DS_Tape_1mm_RightFoot_MedialPlusX.step",
           "sheet": "Pygmalion_v5_Rubber_Sole_4mm_RightFoot_MedialPlusX.step"}
  VAR = "v5_" + VAR
else:
  PARTS = {"carrier": f"Pygmalion_v3_Sole_Carrier_v3_{VAR}.step", "tape": f"Pygmalion_v3_DS_Tape_1mm_{VAR}.step",
           "sheet": f"Pygmalion_v3_Rubber_Sheet_4mm_{VAR}.step"}


def tessellate(path, lin=0.05, ang=0.2):
  r = STEPControl_Reader()
  r.ReadFile(str(path))
  r.TransferRoots()
  shape = r.OneShape()
  BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
  V, F = [], []
  ex = TopExp_Explorer(shape, TopAbs_FACE)
  while ex.More():
    face = TopoDS.Face_s(ex.Current())
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is not None:
      tr = loc.Transformation()
      base = sum(len(v) for v in V)
      pts = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i + 1).Transformed(tr) for i in range(tri.NbNodes()))])
      tris = np.array([tri.Triangle(i + 1).Get() for i in range(tri.NbTriangles())]) - 1
      if face.Orientation() == TopAbs_REVERSED:
        tris = tris[:, ::-1]
      V.append(pts)
      F.append(tris + base)
    ex.Next()
  return np.vstack(V), np.vstack(F)


OUT.mkdir(parents=True, exist_ok=True)
meshes = []
with tempfile.TemporaryDirectory() as td:
  for name, fn in PARTS.items():
    if ZIP.is_dir():
      src = ZIP / fn
    else:
      with zipfile.ZipFile(ZIP) as z:
        z.extract(fn, td)
      src = Path(td) / fn
    v, f = tessellate(src)
    vf = np.c_[145.0 - v[:, 1], v[:, 0] + 123.7, v[:, 2] + 800.0]  # flip of two axes keeps the orientation (det = +1)
    m = trimesh.Trimesh(vf, f, process=True)
    meshes.append(m)
    if ZIP.is_dir():  # v5: per-part STLs too (foot_link frame, metres; left = mirror y) for per-material mass properties
      for side, sg in (("R", 1.0), ("L", -1.0)):
        pm = m.copy()
        pm.vertices = pm.vertices * np.array([1e-3, sg * 1e-3, 1e-3])
        if sg < 0:
          pm.invert()
        pm.export(OUT / f"{side}_{name}_{VAR}.stl")
    print(f"{name}: {len(m.vertices)} verts, {len(m.faces)} faces, volume {abs(m.volume) / 1000:.3f} cm3, "
          f"z {m.bounds[0][2]:.2f}..{m.bounds[1][2]:.2f} mm", flush=True)
sole = trimesh.util.concatenate(meshes[:3])  # the sole layers only (Feet_Up sits on top of the plate)
for side, s in (("R", 1.0), ("L", -1.0)):
  mm = sole.copy()
  mm.vertices = mm.vertices * np.array([1e-3, s * 1e-3, 1e-3])
  if s < 0:
    mm.invert()
  mm.export(OUT / (f"{side}_sole_{VAR}.stl" if VAR.startswith("v5") else f"{side}_sole_v3_{VAR}.stl"))
print("saved", OUT)
