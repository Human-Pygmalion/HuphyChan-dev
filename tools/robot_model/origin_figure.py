"""Draw where the URDF/MJCF origin (base_link) sits, in CAD (Fusion) coordinates.

Reads the shell MJCF at zero pose (root at the origin), turns every point back into CAD
millimetres with  cad = BASE_CAD + (sim_y, -sim_x, sim_z)  (inverse of sim = (-cad_y, cad_x, cad_z),
build_robot.py), and marks the origin on front / side / top views.
usage: origin_figure.py [--xml PATH] [--out PNG]   (mjlab .venv python)
"""
import argparse
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/syaro/MikuchanRemote/Human-Pygmalion"
XML = f"{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_shell.xml"
BASE_CAD = np.array([0.0, 70.0, 60.0])  # build_robot.py ORIGIN_CAD['pelvis'], mm
HIP_X = 123.7                           # hip axis points sit at CAD x = +/-123.7, same y,z


def to_cad(p_sim_m):
    p = np.atleast_2d(p_sim_m) * 1000.0
    return BASE_CAD + np.stack([p[:, 1], -p[:, 0], p[:, 2]], axis=1)


ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=XML)
ap.add_argument("--out", default=f"{REPO}/docs/img/d455/base_link_origin_cad.png")
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
assert np.allclose(d.xpos[1], 0), "root body is not at the world origin"

pts = []
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    i = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
    pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
cloud = to_cad(np.vstack(pts))[::6]

site = lambda n: to_cad(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)])[0]
cam = site("d435_link")

# (title, horizontal axis, vertical axis, labels)
views = [("FRONT  (look along CAD +Y)", 0, 2, "CAD X [mm]", "CAD Z [mm]"),
         ("SIDE  (look along CAD +X)", 1, 2, "CAD Y [mm]  (robot forward = -Y)", "CAD Z [mm]"),
         ("TOP  (look down CAD -Z)", 0, 1, "CAD X [mm]", "CAD Y [mm]  (robot forward = -Y)")]
fig, axs = plt.subplots(1, 3, figsize=(15, 9), gridspec_kw={"width_ratios": [560, 300, 560]})
for ax, (t, h, v, xl, yl) in zip(axs, views):
    ax.scatter(cloud[:, h], cloud[:, v], s=0.4, c="0.75")
    ax.scatter(*[[BASE_CAD[k]] for k in (h, v)], s=220, c="red", marker="o", zorder=5, label="BASE_LINK ORIGIN = (0, 70, 60)")
    ax.scatter([0], [0], s=90, c="blue", marker="s", zorder=5, label="CAD global origin (0, 0, 0)")
    hips = np.array([[-HIP_X, 70, 60], [HIP_X, 70, 60]])
    ax.scatter(hips[:, h], hips[:, v], s=70, c="orange", marker="D", zorder=5, label="hip joints (x = +/-123.7)")
    ax.scatter([cam[h]], [cam[v]], s=70, c="green", marker="^", zorder=5, label="d435_link")
    ax.axhline(BASE_CAD[v], c="red", lw=0.5, ls=":")
    ax.axvline(BASE_CAD[h], c="red", lw=0.5, ls=":")
    ax.set_aspect("equal")
    ax.set_title(t)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
axs[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Where the URDF origin (base_link) is, in Fusion CAD coordinates [mm] - midpoint of the two hip joints", fontsize=13)
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print("saved", a.out, "| d435 CAD =", np.round(cam, 1))
