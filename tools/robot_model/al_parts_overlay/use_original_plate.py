#!/usr/bin/env python3
"""User 2026-11-04: the Fusion edit of Yaw2KneePlateA is wrong, the ORIGINAL geometry is right -> use the original mesh and volume.
Writes into the round-2 snapshot (committed, so the build does not depend on the archive):
  original_geometry.json   Yaw2KneePlateA mass properties of the original geometry at Al 6061 (volume 119.2408 cm3 x 2.70 g/cm3; the tensor of the dump the shipped
                           model was built from, scaled by Al mass / printed mass: the dump scaled the CAD aluminium tensor uniformly)
  cad_mm/Yaw2KneePlateA.stl  the original tessellation (Fusion v5 archive /home/syaro/pyg_fea/fusion/alu_parts_v5, key 016, CAD mm; low quality, 1902 vertices)
Cross-check against the archive index: mass 321.95 g, volume 119.2408 cm3, COM (-93.618, 92.531, -235.528) mm."""
import json
from pathlib import Path
import numpy as np, trimesh
REPO = Path(__file__).resolve().parents[3]
SNAP = REPO / "tools/robot_model/fusion_snapshots/alparts_20261104"
d0 = json.load(open(REPO / "tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json"))
k = [k for k in d0 if k.endswith("::Yaw2KneePlateA")]; assert len(k) == 1
o = d0[k[0]]
m_al = o["v"] * 2.70 / 1000.0
e = dict(o, m=m_al, I=[x * m_al / o["m"] for x in o["I"]], mat="Aluminum 6061 (original geometry)", live=True, source="original: Fusion v5 archive / shipped model dump; user 2026-11-04")
assert abs(m_al * 1000 - 321.95) < 0.02 and np.allclose(np.array(o["c"]) * 10, [-93.618, 92.531, -235.528], atol=0.01)
json.dump({"Yaw2KneePlateA": e}, open(SNAP / "original_geometry.json", "w"), indent=1)
z = np.load("/home/syaro/pyg_fea/fusion/alu_parts_v5/all_meshes.npz", allow_pickle=True)
trimesh.Trimesh(z["016|v"].reshape(-1, 3), z["016|f"].reshape(-1, 3), process=False).export(SNAP / "cad_mm/Yaw2KneePlateA.stl")
print(f"Yaw2KneePlateA original: {m_al*1000:.2f} g, {o['v']} cm3; mesh from the v5 archive")
