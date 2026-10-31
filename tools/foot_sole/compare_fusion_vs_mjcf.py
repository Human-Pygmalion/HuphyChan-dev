#!/usr/bin/env python3
"""Compare the real sole solid (Fusion `Sole_v3_byClaude`, fetched by fusion_sole_fetch.py) with the MJCF collision model.

Fusion document frame (mm) -> right foot_link frame (mm): x = 145 - y_d, y = x_d + 123.7, z = z_d + 800
(FEET plate: heel y_d = 225 -> x = -80, bottom z_d = -843 -> z = -43; plate centre x_d = -123.7 -> y = 0; +x_d = robot left =
medial for this RIGHT foot, whose centre sits at x_d = -123.7 mm like the sim right foot at y = -124 mm).

Outputs in <out>: compare.json, fig_fusion_vs_mjcf.png, and right/left foot STLs of the real sole (foot_link frame, metres)
for use as visual reference meshes.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/compare_fusion_vs_mjcf.py <fusion_mesh_dir> <v3full.xml> <out>
"""
import json
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_collision import GX, GY, XX, YY, lowest_z  # noqa: E402

MDIR, XML, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
idx = json.loads((MDIR / "index.json").read_text())


def to_foot(v):
  return np.c_[145.0 - v[:, 1], v[:, 0] + 123.7, v[:, 2] + 800.0]


meshes = {}
for b in idx["bodies"]:
  d = np.load(MDIR / b["file"])
  meshes[b["name"].split(":")[0]] = trimesh.Trimesh(to_foot(d["v"]), d["f"], process=True)
union = trimesh.util.concatenate(list(meshes.values()))

# lowest real surface z at each grid point (vertical ray from below)
pts = np.c_[XX.ravel() * 1000, YY.ravel() * 1000]
orig = np.c_[pts, np.full(len(pts), -100.0)]
loc, ray_id, _ = union.ray.intersects_location(orig, np.tile([0, 0, 1.0], (len(pts), 1)), multiple_hits=True)
zr = np.full(len(pts), np.inf)
np.minimum.at(zr, ray_id, loc[:, 2])
ZR = zr.reshape(XX.shape)

m = mujoco.MjModel.from_xml_path(str(XML))
ZM = lowest_z(m, m.body("R_foot_link").id) * 1000
# compare only where the real sole exists (exclude the plate overhang, which the real solid list does not contain)
sole = np.isfinite(ZR)
dz = np.where(sole & np.isfinite(ZM), ZM - ZR, np.nan)
cont_r = sole & (ZR <= np.nanmin(ZR) + 0.2)
cont_m = np.isfinite(ZM) & (ZM <= np.nanmin(ZM) + 0.2)
xs = GX * 1000
med = {}
for x0 in (-20, 0, 25, 35, 50):  # arch section x from the ankle [mm]
  i = np.argmin(np.abs(xs - x0))
  col = ZR[i]
  ok = np.isfinite(col)
  ys = GY[ok] * 1000
  med[f"x{x0}"] = dict(y_min=float(ys.min()), y_max=float(ys.max()), lift_at_ymax=float(col[ok][-1] - np.nanmin(ZR)),
                        lift_at_ymin=float(col[ok][0] - np.nanmin(ZR)))
res = dict(
  real_lowest_z_mm=float(np.nanmin(ZR)), mjcf_lowest_z_mm=float(np.nanmin(ZM)),
  dz_mm=dict(mean=float(np.nanmean(dz)), p1=float(np.nanpercentile(dz, 1)), p99=float(np.nanpercentile(dz, 99)),
             abs_p99=float(np.nanpercentile(np.abs(dz), 99)), abs_max=float(np.nanmax(np.abs(dz)))),
  footprint_cm2=dict(real=float(sole.sum()) / 100, mjcf_sole_region=float((sole & np.isfinite(ZM)).sum()) / 100),
  contact_cm2=dict(real=float(cont_r.sum()) / 100, mjcf=float((cont_m & sole).sum()) / 100,
                   both=float((cont_r & cont_m).sum()) / 100, real_only=float((cont_r & ~cont_m).sum()) / 100,
                   mjcf_only=float((cont_m & sole & ~cont_r).sum()) / 100),
  arch_sections=med,
  body_volumes_cm3={k: round(float(abs(v.volume)) / 1000, 3) for k, v in meshes.items()},
  arch_side="medial (+y, toward the robot centre)" if med["x25"]["lift_at_ymax"] > med["x25"]["lift_at_ymin"] else "LATERAL (-y): mirrored",
)
# same comparison against the MJCF mirrored left-right (= the MedialMinusX STEP): separates a shape error from a side error
ZMm = ZM[:, ::-1]
dzm = np.where(sole & np.isfinite(ZMm), ZMm - ZR, np.nan)
cont_mm = np.isfinite(ZMm) & (ZMm <= np.nanmin(ZMm) + 0.2)
res["vs_mirrored_mjcf"] = dict(
  dz_mm=dict(mean=float(np.nanmean(dzm)), abs_p99=float(np.nanpercentile(np.abs(dzm), 99)), abs_max=float(np.nanmax(np.abs(dzm)))),
  contact_cm2=dict(mjcf=float((cont_mm & sole).sum()) / 100, both=float((cont_r & cont_mm).sum()) / 100,
                   real_only=float((cont_r & ~cont_mm).sum()) / 100, mjcf_only=float((cont_mm & sole & ~cont_r).sum()) / 100))
(OUT / "compare.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

# reference STLs of the real sole (all visible bodies) in foot_link frame, metres; left foot = mirror y
for side, s in (("R", 1.0), ("L", -1.0)):
  mm = union.copy()
  mm.vertices = mm.vertices * np.array([1e-3, s * 1e-3, 1e-3])
  if s < 0:
    mm.invert()
  mm.export(OUT / f"{side}_sole_v3_real.stl")

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
z0 = np.nanmin(ZR)
for a, Z, t in ((ax[0, 0], ZR, "REAL sole (Fusion Sole_v3_byClaude, visible bodies)"), (ax[0, 1], ZM, "MJCF collision v3full (47 convex pieces)")):
  C = np.where(np.isfinite(Z), Z - z0, np.nan)
  im = a.pcolormesh(xs, GY * 1000, np.clip(C, 0, 8).T, cmap="viridis", vmin=0, vmax=8, shading="auto")
  a.contour(xs, GY * 1000, np.nan_to_num(C, nan=99).T, levels=[0.2], colors="r", linewidths=0.8)
  a.set_aspect("equal")
  a.set_title(t + "\nclearance above the lowest point [mm], red = within 0.2 mm", fontsize=9)
  a.set_xlabel("x from ankle [mm]")
  a.set_ylabel("y [mm] (medial +)")
  fig.colorbar(im, ax=a, fraction=0.025)
im = ax[1, 0].pcolormesh(xs, GY * 1000, dz.T, cmap="coolwarm", vmin=-0.3, vmax=0.3, shading="auto")
ax[1, 0].set_aspect("equal")
ax[1, 0].set_title(f"MJCF minus REAL bottom height [mm] (|dz| P99 {res['dz_mm']['abs_p99']:.3f}, max {res['dz_mm']['abs_max']:.3f})", fontsize=9)
ax[1, 0].set_xlabel("x from ankle [mm]")
fig.colorbar(im, ax=ax[1, 0], fraction=0.025)
for yv, col in ((40, "tab:red"), (0, "k"), (-40, "tab:blue")):
  j = np.argmin(np.abs(GY * 1000 - yv))
  ax[1, 1].plot(xs, ZR[:, j], color=col, lw=2.2, alpha=0.45, label=f"real y={yv}")
  ax[1, 1].plot(xs, ZM[:, j], color=col, lw=0.9, ls="--", label=f"MJCF y={yv}")
ax[1, 1].set_ylim(z0 - 1, z0 + 14)
ax[1, 1].set_title("side profiles: real (thick) vs MJCF collision (dashed)", fontsize=9)
ax[1, 1].set_xlabel("x from ankle [mm]")
ax[1, 1].set_ylabel("z in foot frame [mm]")
ax[1, 1].legend(fontsize=7, ncol=3)
ax[1, 1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_fusion_vs_mjcf.png", dpi=105)
print("saved", OUT)
