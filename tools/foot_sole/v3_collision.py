#!/usr/bin/env python3
"""Pygmalion sole v3 -> MuJoCo collision geometry for the LegOnly AB model (foot_link frame).

The v3 sole (Feet_design/Feet_Design_Workspace/foot_cad/v3, parameters fp3.py) is a non-convex solid: flat contact zones, two
convex rockers and a concave (lifted) medial arch. MuJoCo collides meshes as convex hulls, so the solid between the sheet bottom
and the plate bottom is cut into full-width transverse strips (cuts along x only, starting at the rocker tangents and the arch
span, halved until every strip meets two tolerances):
  fill  = max over the piece of (true bottom z - hull bottom z): material the hull adds below the real surface  (<= --tol-fill)
  bulge = max distance from the hull footprint to the true outline                                                (<= --tol-bulge)
The foot frame of fp3 (x heel->toe 0..260, y +medial, z up, ground 0, plate bottom z = S) maps to foot_link as
  x_l = x_f - 80,  y_l = s*y_f (s = +1 right foot, -1 left foot: medial = toward the robot centre line),  z_l = z_f - S - 43
(sim plate bottom = -43 mm, ankle origin 80 mm from the heel: both read from the visual plate mesh of pygmalion_legonly_ab.xml).

Variants written next to the source XML:
  v3full    sole strips at the real height (sole bottom -57 mm) + rounded plate slab + the three original boxes cut to the volume
            above the plate (z -35..+15 mm) + 0.258 kg sole mass
  v3same    as v3full, everything of the sole shifted up by S = 14 mm (sole bottom -43 mm = the base model): shape change only
  v3lat     as v3full with the sole mirrored left-right: arch on the LATERAL side (= the MedialMinusX STEP on a right foot,
            which is how Sole_v3_byClaude sits in the Fusion assembly on 2026-10-31)
  v3sph     the 12-sphere approximation of the v3 design notes (sole_v3_collision_mjcf_snippet.xml) at the real height + plate + boxes + mass
All new geoms are named <side>_foot<k>_collision so FOOT_GEOMS (r"L_foot\\d+_collision") still matches; the robot's CollisionCfg
regex [1-7] must be widened to \\d+ by the caller (foot_variant_eval.py does it).

Run with the foot-design venv (shapely, scipy):  ~/.venvs/pyg_foot/bin/python tools/foot_sole/v3_collision.py --xml <legonly_ab.xml>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon, box

REPO = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--xml", required=True, help="pygmalion_legonly_ab.xml of the policy branch")
ap.add_argument("--design", default="v3", choices=["v3", "v5"], help="v3: fp3, rubber covers the whole sole; "
                "v5: fp5, rubber_plan_outline.npy + exposed carrier underside (carrier_z) where there is no rubber")
ap.add_argument("--tol-fill", type=float, default=0.15, help="mm")
ap.add_argument("--tol-bulge", type=float, default=0.5, help="mm")
ap.add_argument("--min-cell", type=float, default=3.0, help="mm")
ap.add_argument("--report", default="", help="JSON quality report path (default: next to the XML)")
ap.add_argument("--override", default="", help='v5 only: JSON of fp5.P overrides for a parameter sweep, e.g. {"heel_flat": 10}')
ap.add_argument("--soft-arch", type=float, default=0.0,
                help="v5 only: solref time constant for the arch-wedge pieces (soft TPU carrier wedge, where the carrier is lifted "
                     "> 0.5 mm by the arch cone); the wedge is cut out of the strips as its own pieces. 0 = off")
ap.add_argument("--name", default="", help="v5 only: variant name written as <xml stem>_<name>.xml (default v5full)")
args = ap.parse_args()
DESIGN = args.design
sys.path.insert(0, str(REPO / f"Feet_design/Feet_Design_Workspace/foot_cad/{DESIGN}"))
fp = __import__("fp3" if DESIGN == "v3" else "fp5")
if args.override:
  # re-run fp5 with the overridden parameters so every derived quantity (rocker radii, arch cone) follows them
  import types  # noqa: E402
  _src = Path(fp.__file__).read_text()
  _ovr = json.loads(args.override)
  assert set(_ovr) <= set(fp.P), f"unknown fp5 parameters: {set(_ovr) - set(fp.P)}"
  _src = _src.replace("Z0 = P[", f"P.update({_ovr!r})\nZ0 = P[", 1)
  fp = types.ModuleType("fp5_override")
  fp.__file__ = __import__("fp5").__file__
  exec(compile(_src, fp.__file__, "exec"), fp.__dict__)

P = fp.P
S = P["S"]
if DESIGN == "v3":
  SOLE = fp.SOLE
  EXPOSED = []
else:
  SOLE = Polygon(np.load(REPO / "Feet_design/Feet_Design_Workspace/foot_cad/v5/rubber_plan_outline.npy")).buffer(0)
  _ex = fp.CARRIER.difference(SOLE)
  EXPOSED = [g for g in getattr(_ex, "geoms", [_ex]) if g.area >= 20.0]  # drop boundary slivers (< 0.2 cm2)
PLATE_Z = -43.0  # sim plate bottom in foot_link [mm]
ANKLE_X = 80.0  # ankle origin from the heel [mm]
SOLE_MASS = 0.258 if DESIGN == "v3" else 0.252  # kg: v3 PA12 119 + tape 19 + sheet 120 g; v5 ABS 40 % infill ~110 + 19 + 123 g

# ---- true sheet-bottom height field zb(x, y) on a 0.5 mm grid (normal offset d = tape + sheet from the carrier underside)
h = 0.5
gx = np.arange(-2.0, P["L"] + 2.0 + h, h)
gy = np.arange(-60.0, 60.0 + h, h)
U, V = np.meshgrid(gx, gy, indexing="ij")
Xo, Yo, Zo = fp.offset_points(U.ravel(), V.ravel(), fp.Z0)
ZB = griddata((Xo, Yo), Zo, (U, V), method="linear")
ZB = np.where(np.isnan(ZB), griddata((Xo, Yo), Zo, (U, V), method="nearest"), ZB)


def zb(x, y):
  i = np.clip(np.rint((np.asarray(x) - gx[0]) / h).astype(int), 0, len(gx) - 1)
  j = np.clip(np.rint((np.asarray(y) - gy[0]) / h).astype(int), 0, len(gy) - 1)
  return ZB[i, j]


def boundary_pts(poly, step=1.0):
  out = []
  for ring in [poly.exterior]:
    L = ring.length
    for s in np.arange(0.0, L, step):
      p = ring.interpolate(s)
      out.append((p.x, p.y))
  return np.array(out)


def piece(region, zfun=None):
  """Convex hull of the sole solid inside `region` (shapely polygon): bottom = true surface, top = plate bottom (z = S)."""
  zfun = zfun or zb
  x0, y0, x1, y1 = region.bounds
  xs = np.arange(np.floor(x0), x1 + 1.0, 1.0)
  ys = np.arange(np.floor(y0), y1 + 1.0, 1.0)
  G = np.array([(a, b) for a in xs for b in ys if region.contains(Point(a, b))]).reshape(-1, 2)
  B = boundary_pts(region, 1.0)
  XY = np.vstack([G, B]) if len(G) else B
  bot = np.c_[XY, zfun(XY[:, 0], XY[:, 1])]
  top = np.c_[B, np.full(len(B), S)]
  pts = np.vstack([bot, top])
  hull = ConvexHull(pts)
  # lowest hull z at each sample (x, y): max over downward facets of (n_x x + n_y y + off) / (-n_z)
  eq = hull.equations[hull.equations[:, 2] < -1e-9]
  zl = np.max((eq[:, 0][None] * XY[:, 0:1] + eq[:, 1][None] * XY[:, 1:2] + eq[:, 3][None]) / (-eq[:, 2][None]), axis=1)
  fill = float(np.max(bot[:, 2] - zl))
  foot2d = Polygon(pts[hull.vertices][:, :2]).convex_hull
  bulge = float(max((region.distance(Point(c)) for c in foot2d.exterior.coords), default=0.0))
  return pts[hull.vertices], fill, bulge


def split(x0, x1):
  """Full-width transverse strip [x0, x1]; halve along x only until the strip meets both tolerances."""
  region = SOLE.intersection(box(x0, -70, x1, 70))
  if region.is_empty or region.area < 1.0:
    return []
  verts, fill, bulge = piece(region)
  if (fill <= args.tol_fill and bulge <= args.tol_bulge) or x1 - x0 < 2 * args.min_cell:
    return [dict(verts=verts, fill=fill, bulge=bulge, bounds=region.bounds, area=region.area)]
  xm = 0.5 * (x0 + x1)
  return split(x0, xm) + split(xm, x1)


def split_free(region, zfun):
  """Exposed-carrier pieces (thin, irregular): halve along the longer side until both tolerances hold."""
  if region.is_empty or region.area < 1.0:
    return []
  if region.geom_type != "Polygon":
    return [q for g in getattr(region, "geoms", []) for q in split_free(g, zfun)]
  verts, fill, bulge = piece(region, zfun)
  x0, y0, x1, y1 = region.bounds
  if (fill <= args.tol_fill and bulge <= args.tol_bulge) or max(x1 - x0, y1 - y0) < 2 * args.min_cell:
    return [dict(verts=verts, fill=fill, bulge=bulge, bounds=region.bounds, area=region.area, kind="carrier")]
  if x1 - x0 >= y1 - y0:
    xm = 0.5 * (x0 + x1)
    halves = (box(x0 - 1, y0 - 1, xm, y1 + 1), box(xm, y0 - 1, x1 + 1, y1 + 1))
  else:
    ym = 0.5 * (y0 + y1)
    halves = (box(x0 - 1, y0 - 1, x1 + 1, ym), box(x0 - 1, ym, x1 + 1, y1 + 1))
  return [q for hv in halves for q in split_free(region.intersection(hv), zfun)]


# start from natural cuts (rocker tangents, arch span) so the recursion does not split convex zones needlessly
cuts = [0.0, P["heel_flat"], P["arch_med_x"][0] - 5, P["arch_med_x"][1] + 5, P["toe_flat"], P["L"] + 1]
pieces = []
for a, b in zip(cuts[:-1], cuts[1:]):
  pieces += split(a, b)
for g in EXPOSED:
  pieces += split_free(g, lambda x, y: fp.carrier_z(np.asarray(x, float), np.asarray(y, float)))
if args.soft_arch > 0:
  # wedge = carrier lifted > 0.5 mm by the arch cone (the part a TPU wedge would replace), as a polygon from a 1 mm grid
  from shapely.ops import unary_union as _uu  # noqa: E402
  _cells = [box(a - 0.5, b - 0.5, a + 0.5, b + 0.5) for a in np.arange(0.5, P["L"], 1.0) for b in np.arange(-57.5, 58.0, 1.0)
            if fp.cone_lift(a, b) > 0.5]
  WEDGE = _uu(_cells).buffer(0.01).buffer(-0.01).intersection(SOLE)
  keep, soft = [], []
  for p in pieces:
    if p.get("kind") == "carrier" or not Polygon(p["verts"][:, :2]).convex_hull.intersects(WEDGE):
      keep.append(p)
      continue
    strip = SOLE.intersection(box(p["bounds"][0], -70, p["bounds"][2], 70))
    keep += [dict(q, kind="rubber") for q in split_free(strip.difference(WEDGE), zb)]
    soft += [dict(q, kind="arch") for q in split_free(strip.intersection(WEDGE), zb)]
  pieces = keep + soft
  print(f"soft arch: wedge {WEDGE.area / 100:.1f} cm2, {len(soft)} wedge pieces (solref {args.soft_arch})", flush=True)
fills = [p["fill"] for p in pieces]
bulges = [p["bulge"] for p in pieces]
print(f"{DESIGN} pieces: {len(pieces)} ({sum(p.get('kind') == 'carrier' for p in pieces)} exposed carrier)   fill max {max(fills):.3f} mm   bulge max {max(bulges):.3f} mm", flush=True)

# coverage check: true sole footprint vs union of piece footprints
from shapely.ops import unary_union  # noqa: E402

fp_union = unary_union([Polygon(p["verts"][:, :2]).convex_hull for p in pieces])
cov = dict(true_area_cm2=SOLE.area / 100, pieces_area_cm2=fp_union.area / 100, outside_cm2=fp_union.difference(SOLE).area / 100,
           missing_cm2=SOLE.difference(fp_union).area / 100)

# flat-ground contact zone check: true surface within 0.2 mm of the ground vs hull bottom within 0.2 mm
xs = np.arange(0.5, P["L"], 1.0)
ys = np.arange(-57.0, 57.5, 1.0)
inside = [(a, b) for a in xs for b in ys if SOLE.contains(Point(a, b))]
IN = np.array(inside)
true_contact = zb(IN[:, 0], IN[:, 1]) <= 0.2
hz = np.full(len(IN), np.inf)
for p in pieces:
  hull = ConvexHull(p["verts"])
  eq = hull.equations[hull.equations[:, 2] < -1e-9]
  x0, y0, x1, y1 = p["bounds"]
  m = (IN[:, 0] >= x0 - 1e-6) & (IN[:, 0] <= x1 + 1e-6) & (IN[:, 1] >= y0 - 1e-6) & (IN[:, 1] <= y1 + 1e-6)
  if not m.any():
    continue
  zl = np.max((eq[:, 0][None] * IN[m, 0:1] + eq[:, 1][None] * IN[m, 1:2] + eq[:, 3][None]) / (-eq[:, 2][None]), axis=1)
  # only count where the sample lies inside this hull's footprint
  fpoly = Polygon(p["verts"][:, :2]).convex_hull
  inm = np.array([fpoly.buffer(1e-6).contains(Point(a, b)) for a, b in IN[m]])
  idx = np.flatnonzero(m)[inm]
  hz[idx] = np.minimum(hz[idx], zl[inm])
model_contact = hz <= 0.2
cov.update(contact_true_cm2=float(true_contact.sum()) / 100, contact_model_cm2=float(model_contact.sum()) / 100,
           contact_agree=float((true_contact == model_contact).mean()))
print("coverage", json.dumps({k: round(v, 3) for k, v in cov.items()}), flush=True)


# ---- MJCF text
def fmt(a):
  return " ".join(f"{v:.5f}" for v in np.asarray(a).ravel())


def plate_slab():
  r = 30.0
  pts = []
  for cx, cy, a0 in ((P["L"] - r, 57.5 - r, 0), (r, 57.5 - r, 90), (r, -57.5 + r, 180), (P["L"] - r, -57.5 + r, 270)):
    for t in np.linspace(a0, a0 + 90, 7):
      pts.append((cx + r * np.cos(np.radians(t)), cy + r * np.sin(np.radians(t))))
  pts = np.array(pts)
  return np.vstack([np.c_[pts, np.full(len(pts), S)], np.c_[pts, np.full(len(pts), S + 8.0)]])


def to_link(v, side, dz, mirror=False):
  s = (1.0 if side == "R" else -1.0) * (-1.0 if mirror else 1.0)
  return np.c_[(v[:, 0] - ANKLE_X) / 1000.0, s * v[:, 1] / 1000.0, (v[:, 2] + dz) / 1000.0]


SPHERES = [(11.1, 25.4, 7.5), (11.1, -17.0, 7.5), (34.0, 45.9, 6.0), (34.0, -32.8, 6.0), (58.0, 42.5, 6.0), (84.0, -38.9, 6.0),
           (168.0, 48.2, 6.0), (134.0, -48.9, 6.0), (208.0, 50.2, 6.0), (208.0, -50.2, 6.0), (245.1, 34.1, 8.3), (245.1, -33.4, 8.3)]


def foot_block(side, variant, meshes):
  dz = PLATE_Z - S if variant != "v3same" else PLATE_Z  # sole bottom -57 (real) or -43 (same height as base)
  lines, k = [], 0

  def add_mesh(name, v, extra=""):
    nonlocal k
    k += 1
    meshes.append(f'    <mesh name="{name}" vertex="{fmt(v)}"/>')
    lines.append(f'                  <geom name="{side}_foot{k}_collision" class="collision" type="mesh" mesh="{name}"{extra}/>')

  # the three original boxes, cut to the volume above the plate (z -35 .. +15 mm)
  for x in (-0.0367, 0.05, 0.1367):
    k += 1
    lines.append(f'                  <geom name="{side}_foot{k}_collision" class="foot_box" size="0.0433 0.05 0.025" pos="{x} 0 -0.010"/>')
  add_mesh(f"{side}_{variant}_plate", to_link(plate_slab(), side, dz))
  if variant == "v3sph":
    for (x, y, z) in SPHERES:
      k += 1
      c = to_link(np.array([[x, y, z]]), side, dz)[0]
      lines.append(f'                  <geom name="{side}_foot{k}_collision" class="collision" type="sphere" size="0.006" pos="{fmt(c)}"/>')
  else:
    for i, p in enumerate(pieces):
      add_mesh(f"{side}_{variant}_s{i:02d}", to_link(p["verts"], side, dz, mirror=(variant == "v3lat")),
               f' solref="{args.soft_arch} 1"' if p.get("kind") == "arch" else "")
  return lines


src = Path(args.xml)
txt = src.read_text()
report = dict(override=json.loads(args.override or "{}"), tol_fill_mm=args.tol_fill, tol_bulge_mm=args.tol_bulge, n_pieces=len(pieces), fill_max_mm=max(fills),
              bulge_max_mm=max(bulges), coverage=cov, sole_mass_kg=SOLE_MASS, variants={})
for variant in (("v3full", "v3same", "v3sph", "v3lat") if DESIGN == "v3" else (args.name or "v5full",)):
  meshes, out = [], txt
  for side in "RL":
    pat = re.compile(rf'(\s*<geom name="{side}_foot\d_collision"[^\n]*\n)+')
    m = pat.search(out)
    assert m, f"no foot boxes for {side}"
    out = out[: m.start()] + "\n" + "\n".join(foot_block(side, variant, meshes)) + "\n" + out[m.end():]
    # sole mass: combine with the foot_link inertial (parallel axis), sole as a 260 x 114 x 14 mm slab at its volume centroid
    im = re.search(rf'(<body name="{side}_foot_link">\s*<inertial )pos="([^"]+)" quat="([^"]+)" mass="([^"]+)" diaginertia="([^"]+)"/>', out)
    assert im, f"no inertial for {side}_foot_link"
    p0 = np.array(list(map(float, im.group(2).split())))
    q = np.array(list(map(float, im.group(3).split())))
    m0 = float(im.group(4))
    d0 = np.array(list(map(float, im.group(5).split())))
    w, x, y, z = q / np.linalg.norm(q)
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    I0 = Rq @ np.diag(d0) @ Rq.T
    dz = (PLATE_Z - S) if variant != "v3same" else PLATE_Z
    c1 = np.array([(130.0 - ANKLE_X) / 1000, 0.0, (S / 2 + dz) / 1000])
    a, b, c = 0.260, 0.114, S / 1000
    I1 = SOLE_MASS / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    M = m0 + SOLE_MASS
    pc = (m0 * p0 + SOLE_MASS * c1) / M

    def shift(I, mm, r):
      return I + mm * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    Itot = shift(I0, m0, p0 - pc) + shift(I1, SOLE_MASS, c1 - pc)
    full = f'pos="{fmt(pc)}" mass="{M:.5f}" fullinertia="{Itot[0,0]:.6e} {Itot[1,1]:.6e} {Itot[2,2]:.6e} {Itot[0,1]:.6e} {Itot[0,2]:.6e} {Itot[1,2]:.6e}"/>'
    out = out[: im.start()] + im.group(1) + full + out[im.end():]
  out = out.replace("<asset>", "<asset>\n" + "\n".join(meshes), 1)
  # default pose translated up (qpos0 only; resets use the task's init_state): with the base at the origin the mesh sole sits deep
  # in the terrain and the simulator's start-up contact count (~200) exceeds the training nconmax of 150. Translation only.
  out = out.replace('<body name="base_link" childclass', '<body name="base_link" pos="0 0 0.92" childclass', 1)
  dst = src.with_name(src.stem + f"_{variant}.xml")
  dst.write_text(out)
  report["variants"][variant] = str(dst)
  print("wrote", dst, flush=True)

report["pieces"] = [dict(bounds=[round(v, 2) for v in p["bounds"]], fill=round(p["fill"], 4), bulge=round(p["bulge"], 4), nverts=len(p["verts"]))
                    for p in pieces]
rp = Path(args.report) if args.report else src.with_name(f"{DESIGN}_collision_report.json")
rp.write_text(json.dumps(report, indent=1))
print("report", rp)
