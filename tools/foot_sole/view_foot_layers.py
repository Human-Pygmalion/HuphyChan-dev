#!/usr/bin/env python3
"""Interactive 3D view (plotly HTML) + static PNG of the right foot in the foot_link frame:
  robot visual mesh (R_foot_noloop: plate + ankle bracket), CAD sole (cad/R_sole_v3_MedialPlusX.stl),
  v3full collision (47 convex sole pieces + plate slab + cut boxes), and the training model's 3 collision boxes.
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/view_foot_layers.py <assets_dir> <out_dir>
"""
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def geom_tris(m, g):
  """Triangles (n,3,3) of a geom in its body frame, mm."""
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, m.geom_quat[g])
  R, p = mat.reshape(3, 3), m.geom_pos[g]
  t = m.geom_type[g]
  if t == mujoco.mjtGeom.mjGEOM_MESH:
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    nf = m.mesh_facenum[mid]
    f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + nf] if nf else ConvexHull(v).simplices
    if m.geom_group[g] == 3:  # collision = convex hull of the vertices
      f = ConvexHull(v).simplices
    tri = v[f]
  elif t == mujoco.mjtGeom.mjGEOM_BOX:
    s = m.geom_size[g]
    bx = trimesh.creation.box(extents=2 * s)
    tri = bx.vertices[bx.faces]
  elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
    sp = trimesh.creation.icosphere(subdivisions=2, radius=m.geom_size[g][0])
    tri = sp.vertices[sp.faces]
  else:
    return None
  return (tri @ R.T + p) * 1000


def body_layers(xml, body="R_foot_link"):
  m = mujoco.MjModel.from_xml_path(str(xml))
  b = m.body(body).id
  vis, col = [], []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != b:
      continue
    tri = geom_tris(m, g)
    if tri is None:
      continue
    if m.geom_group[g] == 2:
      vis.append(tri)
    elif m.geom_group[g] == 3:
      col.append(tri)
  return (np.concatenate(vis) if vis else None), col


vis, col_v3 = body_layers(A / "pygmalion_legonly_ab_v3full.xml")
_, col_base = body_layers(A / "pygmalion_legonly_ab.xml")
cad = trimesh.load(A / "sole_v3/cad/R_sole_v3_MedialPlusX.stl")
cad_tri = cad.vertices[cad.faces] * 1000


def mesh3d(tri, name, color, opacity, visible=True):
  v = tri.reshape(-1, 3)
  n = len(tri)
  return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=np.arange(n) * 3, j=np.arange(n) * 3 + 1, k=np.arange(n) * 3 + 2,
                   color=color, opacity=opacity, name=name, showlegend=True, flatshading=True, visible=visible)


def edges3d(tris, name, color, visible=True):
  xs, ys, zs = [], [], []
  for tri in tris:
    h = ConvexHull(tri.reshape(-1, 3))
    pts = h.points
    # draw only feature edges (between non-coplanar facets) so the 47 pieces read as blocks
    eq = h.equations
    edge_f = {}
    for fi, s in enumerate(h.simplices):
      for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
        edge_f.setdefault(tuple(sorted((a, b))), []).append(fi)
    for (a, b), fs in edge_f.items():
      if len(fs) == 2 and np.dot(eq[fs[0], :3], eq[fs[1], :3]) > 0.999:
        continue
      xs += [pts[a, 0], pts[b, 0], None]
      ys += [pts[a, 1], pts[b, 1], None]
      zs += [pts[a, 2], pts[b, 2], None]
  return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=2), name=name, visible=visible)


fig = go.Figure()
fig.add_trace(mesh3d(vis, "robot visual mesh (R_foot_noloop: plate + bracket)", "lightgray", 0.35))
fig.add_trace(mesh3d(cad_tri, "CAD sole (MedialPlusX: carrier + tape + sheet)", "darkorange", 0.55))
fig.add_trace(mesh3d(np.concatenate(col_v3), "v3full collision (solid)", "limegreen", 0.25, visible="legendonly"))
fig.add_trace(edges3d(col_v3, "v3full collision edges (47 sole pieces + plate + boxes)", "green"))
fig.add_trace(edges3d(col_base, "training model collision (3 boxes, bare plate)", "royalblue", visible="legendonly"))
fig.update_layout(
  title="Right foot, foot_link frame [mm] (x toe +, y medial/left +, z up; ankle origin at 0). Click legend items to toggle.",
  scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
             camera=dict(eye=dict(x=-1.3, y=-1.6, z=-0.7))),
  legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=40, b=0))
html = OUT / "foot_layers_v3full.html"
fig.write_html(html, include_plotlyjs=True)

# static PNG: bottom projection + three side sections + one front section, all layers in the same frame
def section(tris_or_mesh, origin, normal):
  m = tris_or_mesh if isinstance(tris_or_mesh, trimesh.Trimesh) else trimesh.Trimesh(*_hull(tris_or_mesh))
  sec = m.section(plane_origin=origin, plane_normal=normal)
  return [] if sec is None else [e for e in sec.discrete]


def outline(segs, axes):
  """Outer boundary of a planar section given as 3D polylines -> list of (n,2) arrays in the chosen 2 axes."""
  from shapely.geometry import LineString
  from shapely.ops import polygonize, unary_union
  lines = [LineString(e[:, axes]) for e in segs if len(e) > 1]
  polys = list(polygonize(unary_union(lines)))
  if not polys:
    return [e[:, axes] for e in segs]
  u = unary_union(polys).buffer(0.05).buffer(-0.05)
  geoms = getattr(u, "geoms", [u])
  return [np.array(g.exterior.coords) for g in geoms]


def _hull(tri):
  h = ConvexHull(tri.reshape(-1, 3))
  return h.points, h.simplices


vis_m = trimesh.Trimesh(vis.reshape(-1, 3), np.arange(len(vis) * 3).reshape(-1, 3), process=True)
cad_m = trimesh.Trimesh(cad_tri.reshape(-1, 3), np.arange(len(cad_tri) * 3).reshape(-1, 3), process=True)
sole_pieces = [t for t in col_v3 if t.reshape(-1, 3)[:, 2].min() < -50]  # pieces below the plate
cmap = plt.get_cmap("tab20")
f2 = plt.figure(figsize=(18, 11))
gs = f2.add_gridspec(3, 3, height_ratios=[1.6, 0.55, 0.75])
ax = f2.add_subplot(gs[0, :2])
for k, tri in enumerate(sole_pieces):
  h = ConvexHull(tri.reshape(-1, 3)[:, :2])
  ax.fill(*h.points[h.vertices].T, color=cmap(k % 20), alpha=0.75, lw=0.4, ec="k")
for e in outline(section(cad_m, (0, 0, -48.5), (0, 0, 1)), [0, 1]):
  ax.plot(e[:, 0], e[:, 1], color="darkorange", lw=2.2)
for e in outline(section(cad_m, (0, 0, -56.85), (0, 0, 1)), [0, 1]):
  ax.plot(e[:, 0], e[:, 1], color="darkorange", lw=1.4, ls="--")
for e in section(vis_m, (0, 0, -42.0), (0, 0, 1)):
  ax.plot(e[:, 0], e[:, 1], color="gray", lw=1.0)
for tri in col_base:
  pts = tri.reshape(-1, 3)
  x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
  ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", color="royalblue", lw=1.0)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_aspect("equal")
ax.set_xlabel("x from ankle [mm] (toe +)")
ax.set_ylabel("y [mm] (medial +)")
ax.set_title(f"bottom projection: {len(sole_pieces)} convex collision pieces (colours); CAD sole outline (orange) and its flat-ground "
             "contact zone within 0.15 mm (orange dashed); robot visual plate (grey); training-model boxes (blue dashed)", fontsize=8.5)
ax3 = f2.add_subplot(gs[0, 2], projection="3d")
for k, tri in enumerate(sole_pieces):
  h = ConvexHull(tri.reshape(-1, 3))
  ax3.add_collection3d(Poly3DCollection(h.points[h.simplices], facecolor=cmap(k % 20), edgecolor="none", alpha=0.9))
allv = np.concatenate([t.reshape(-1, 3) for t in sole_pieces])
lo, hi = allv.min(0), allv.max(0)
ax3.set_box_aspect((hi - lo) * np.array([1, 1, 3]))
ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
ax3.view_init(elev=-30, azim=-60)
ax3.set_title("collision pieces from below (z x3)", fontsize=9)
for k, (yv, lab) in enumerate(((40.0, "medial y=+40"), (0.0, "centre y=0"), (-40.0, "lateral y=-40"))):
  a = f2.add_subplot(gs[1 + k // 3 if k < 3 else 2, k % 3]) if False else f2.add_subplot(gs[1, k])
  for e in outline(section(vis_m, (0, yv, 0), (0, 1, 0)), [0, 2]):
    a.fill(e[:, 0], e[:, 1], color="0.85", ec="gray", lw=0.6)
  for e in outline(section(cad_m, (0, yv, 0), (0, 1, 0)), [0, 2]):
    a.fill(e[:, 0], e[:, 1], color=(1.0, 0.6, 0.1, 0.45), ec="darkorange", lw=1.6)
  for tri in col_v3:
    for e in section(tri, (0, yv, 0), (0, 1, 0)):
      a.plot(e[:, 0], e[:, 2], color="green", lw=0.7)
  a.set_aspect(2.0)
  a.set_ylim(-59, -33)
  a.set_title(f"side section {lab} (z x2): visual (grey), CAD (orange), collision (green)", fontsize=8.5)
  a.set_xlabel("x [mm]")
  a.set_ylabel("z [mm]")
for k, xv in enumerate((-70.0, 30.0, 170.0)):
  a = f2.add_subplot(gs[2, k])
  for e in outline(section(vis_m, (xv, 0, 0), (1, 0, 0)), [1, 2]):
    a.fill(e[:, 0], e[:, 1], color="0.85", ec="gray", lw=0.6)
  for e in outline(section(cad_m, (xv, 0, 0), (1, 0, 0)), [1, 2]):
    a.fill(e[:, 0], e[:, 1], color=(1.0, 0.6, 0.1, 0.45), ec="darkorange", lw=1.6)
  for tri in col_v3:
    for e in section(tri, (xv, 0, 0), (1, 0, 0)):
      a.plot(e[:, 1], e[:, 2], color="green", lw=0.7)
  a.set_aspect(2.0)
  a.set_ylim(-59, -33)
  a.set_xlim(-65, 65)
  a.set_title(f"front section x={xv:+.0f} mm ({['heel rocker', 'arch', 'toe spring'][k]}), medial right (z x2)", fontsize=8.5)
  a.set_xlabel("y [mm] (medial +)")
  a.set_ylabel("z [mm]")
f2.tight_layout()
f2.savefig(OUT / "foot_layers_v3full.png", dpi=100)
print("saved", html, OUT / "foot_layers_v3full.png")
