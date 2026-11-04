#!/usr/bin/env python3
"""Where does the current Yaw2KneePlateA differ from the original one the model was built from (Fusion v5 archive, volume 119.2408 cm3)?
Surface samples of each mesh that are > 1.2 mm from the other mesh, projected on the plate (y, z) plane. usage: plate_diff.py <out.png>"""
import sys
from pathlib import Path
import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]
z = np.load("/home/syaro/pyg_fea/fusion/alu_parts_v5/all_meshes.npz", allow_pickle=True)
old = trimesh.Trimesh(z["016|v"].reshape(-1, 3), z["016|f"].reshape(-1, 3), process=False)
new = trimesh.load(REPO / "tools/robot_model/fusion_snapshots/alparts_20261104/cad_mm/Yaw2KneePlateA.stl", process=False)
def far(a, b, thr=1.2):
    p, _ = trimesh.sample.sample_surface(a, 80000, seed=0)
    d = trimesh.proximity.closest_point(b, p)[1]
    return p[d > thr]
po, pn = far(old, new), far(new, old)
allp, _ = trimesh.sample.sample_surface(new, 40000, seed=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1, 1]})
ax[0].scatter(allp[:, 1], allp[:, 2], s=0.3, c="#c8ccd2")
ax[0].scatter(po[:, 1], po[:, 2], s=3, c="#1f5fd6", label=f"only in the ORIGINAL ({len(po)} samples)")
ax[0].scatter(pn[:, 1], pn[:, 2], s=3, c="#e61919", label=f"only in the CURRENT ({len(pn)} samples)")
ax[0].set_aspect("equal"); ax[0].set_xlabel("CAD y [mm]"); ax[0].set_ylabel("CAD z [mm]"); ax[0].legend(loc="lower left", fontsize=9)
ax[0].set_title("Yaw2KneePlateA: where the geometry changed (plate seen along x)")
# side view (x vs z) of the two pockets: original surface vs current surface
sel = lambda p: p[(p[:, 1] > 110) & (p[:, 1] < 150)]
ax[1].scatter(sel(po)[:, 0], sel(po)[:, 2], s=3, c="#1f5fd6", label="original surface")
ax[1].scatter(sel(pn)[:, 0], sel(pn)[:, 2], s=3, c="#e61919", label="current surface")
ax[1].set_xlabel("CAD x [mm] (plate thickness direction, plate spans -97.45 .. -89.45)"); ax[1].set_ylabel("CAD z [mm]")
ax[1].set_title("slice y 110-150 mm: the pocket floor moved in x"); ax[1].legend(fontsize=9)
fig.suptitle(f"original 119.24 cm3 -> current {new.volume/1000:.2f} cm3 (tessellated)  |  Fusion value 115.76 cm3", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=105); print("wrote", sys.argv[1])
