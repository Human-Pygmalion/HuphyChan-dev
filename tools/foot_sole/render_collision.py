#!/usr/bin/env python3
"""Draw the right-foot collision geometry of the base model and the v3 variants from the COMPILED MjModel (no OpenGL needed).
Row per variant: (left) bottom view = ground clearance of the lowest collision surface [mm] above the lowest point of the foot,
(right) side profile of that surface along x at the medial edge / centre line / lateral edge.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/render_collision.py <assets_dir> <out.png>"""
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = [("base: 3 boxes = bare plate, 100 mm wide", "pygmalion_legonly_ab.xml"),
            ("v3full: 47 convex pieces, real height (-57 mm)", "pygmalion_legonly_ab_v3full.xml"),
            ("v3same: same pieces, base height (-43 mm)", "pygmalion_legonly_ab_v3same.xml"),
            ("v3sph: 12 spheres r 6 mm (design-note model)", "pygmalion_legonly_ab_v3sph.xml")]
GX = np.arange(-0.085, 0.185, 0.001)
GY = np.arange(-0.062, 0.0625, 0.001)
XX, YY = np.meshgrid(GX, GY, indexing="ij")


def lowest_z(m, b):
  """Lowest collision z [m] in foot_link frame at each (x, y) grid point (inf where no geom)."""
  Z = np.full(XX.shape, np.inf)
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b or m.geom_group[g] != 3:
      continue
    R = m.geom_quat[g]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, R)
    mat = mat.reshape(3, 3)
    p = m.geom_pos[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
      sx, sy, sz = m.geom_size[g]
      mk = (np.abs(XX - p[0]) <= sx) & (np.abs(YY - p[1]) <= sy)
      Z = np.where(mk, np.minimum(Z, p[2] - sz), Z)
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
      r = m.geom_size[g][0]
      d2 = (XX - p[0]) ** 2 + (YY - p[1]) ** 2
      mk = d2 <= r * r
      Z = np.where(mk, np.minimum(Z, p[2] - np.sqrt(np.clip(r * r - d2, 0, None))), Z)
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ mat.T + p
      hull = ConvexHull(v)
      eq = hull.equations[hull.equations[:, 2] < -1e-9]
      inside2d = ConvexHull(v[:, :2])
      e2 = inside2d.equations
      pts = np.c_[XX.ravel(), YY.ravel()]
      inn = np.all(pts @ e2[:, :2].T + e2[:, 2] <= 1e-9, axis=1).reshape(XX.shape)
      zl = np.max((eq[:, 0][None] * pts[:, 0:1] + eq[:, 1][None] * pts[:, 1:2] + eq[:, 3][None]) / (-eq[:, 2][None]), axis=1).reshape(XX.shape)
      Z = np.where(inn, np.minimum(Z, zl), Z)
  return Z


def main():
  A, OUT = Path(sys.argv[1]), sys.argv[2]
  fig, ax = plt.subplots(len(VARIANTS), 2, figsize=(14, 3.4 * len(VARIANTS)), gridspec_kw=dict(width_ratios=[1.35, 1]))
  for r, (title, fn) in enumerate(VARIANTS):
    m = mujoco.MjModel.from_xml_path(str(A / fn))
    b = m.body("R_foot_link").id
    Z = lowest_z(m, b)
    z0 = np.min(Z[np.isfinite(Z)])
    C = np.where(np.isfinite(Z), (Z - z0) * 1000, np.nan)
    im = ax[r, 0].pcolormesh(GX * 1000, GY * 1000, np.clip(C, 0, 8).T, cmap="viridis", vmin=0, vmax=8, shading="auto")
    ax[r, 0].contour(GX * 1000, GY * 1000, np.nan_to_num(C, nan=99).T, levels=[0.2], colors="r", linewidths=0.8)
    ax[r, 0].set_aspect("equal")
    ax[r, 0].set_title(f"{title}\nbottom view: clearance above lowest point [mm], red = within 0.2 mm (flat-ground contact)", fontsize=8.5)
    ax[r, 0].set_xlabel("x from ankle [mm] (toe +)")
    ax[r, 0].set_ylabel("y [mm] (medial +)")
    ax[r, 0].axvline(0, color="w", lw=0.6, ls=":")
    fig.colorbar(im, ax=ax[r, 0], fraction=0.025)
    area = np.nansum(C <= 0.2) / 100.0
    ax[r, 0].text(0.01, 0.02, f"contact area within 0.2 mm: {area:.0f} cm2   lowest point z = {z0*1000:.1f} mm", transform=ax[r, 0].transAxes,
                  color="w", fontsize=8, va="bottom")
    for yv, lab, col in ((0.045, "medial y=+45", "tab:red"), (0.0, "centre y=0", "k"), (-0.045, "lateral y=-45", "tab:blue")):
      j = np.argmin(np.abs(GY - yv))
      ax[r, 1].plot(GX * 1000, Z[:, j] * 1000, color=col, lw=1.2, label=lab)
    ax[r, 1].axhline(z0 * 1000, color="gray", lw=0.6, ls="--")
    ax[r, 1].set_ylim(z0 * 1000 - 1, z0 * 1000 + 16)
    ax[r, 1].set_title("side profile of the lowest collision surface (ankle at x = 0)", fontsize=8.5)
    ax[r, 1].set_xlabel("x from ankle [mm]")
    ax[r, 1].set_ylabel("z in foot frame [mm]")
    ax[r, 1].legend(fontsize=7, loc="upper center")
    ax[r, 1].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(OUT, dpi=105)
  print("saved", OUT)


if __name__ == "__main__":
  main()
