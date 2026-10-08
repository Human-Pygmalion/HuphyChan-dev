#!/usr/bin/env python3
"""Ground surface of the mountain-trail map as (1) a collider and (2) a display mesh, from the same heights.

  hfield   10 cm height grid for MuJoCo (rows = y, cols = x). Cells the scan never saw are filled from their
           neighbours when close to seen ground, and raised into a 2 m wall when far from it (no walking into the void).
  trees    one vertical cylinder per patch of vegetation cells (prep_terrain.py class VEG): centre, radius, base height.
  mesh     25 cm triangle surface (seen ground only) with per-vertex colour = the map's own ground colour,
           tinted by terrain class; also written as terrain_mesh.ply (binary, vertex colours) for other simulators.
usage: prep_terrain_mesh.py <map.ply> <terrain.npz from prep_terrain.py> <out dir>
"""
import os
import sys

import numpy as np
from scipy import ndimage

ply, terr, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
T = np.load(terr)
cls25, org25, c25 = T["cls"], T["origin"].astype(float), float(T["cell"])

f = open(ply, "rb")
h = b""
while not h.endswith(b"end_header\n"):
    h += f.readline()
raw = np.fromfile(f, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")]))
P = np.stack([raw["x"], raw["y"], raw["z"]], 1)
C = np.stack([raw["r"], raw["g"], raw["b"]], 1).astype(np.float32)


def ground_grid(cell, x0, y0, nx, ny, pct=0.05):
    ix = ((P[:, 0] - x0) / cell).astype(np.int64)
    iy = ((P[:, 1] - y0) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    lin = (iy * nx + ix)[ok]
    z = P[ok, 2]
    order = np.lexsort((z, lin))
    ls, zs = lin[order], z[order]
    st = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
    en = np.r_[st[1:], len(ls)]
    n = en - st
    g = np.full(nx * ny, np.nan, np.float32)
    k = n >= 2
    g[ls[st[k]]] = zs[st[k] + (pct * (n[k] - 1)).astype(int)]
    return g.reshape(ny, nx)


def clean(g, size):
    seen = ~np.isnan(g)
    base = np.where(seen, g, np.nanmedian(g))
    med = ndimage.median_filter(base, size=size)
    seen &= ~(np.abs(base - med) > 0.4)  # spikes (below-ground noise, stray returns)
    return np.where(seen, med * 0.5 + base * 0.5, np.nan).astype(np.float32), seen


# ---------- 10 cm collider
c10 = 0.10
x0, y0 = org25
nx10, ny10 = int(cls25.shape[1] * c25 / c10), int(cls25.shape[0] * c25 / c10)
g10, seen10 = clean(ground_grid(c10, x0, y0, nx10, ny10), 5)
dist, (ri, ci) = ndimage.distance_transform_edt(~seen10, return_indices=True)
filled = g10[ri, ci]
filled = ndimage.gaussian_filter(filled, 0.8)  # soft, keeps 10 cm steps but removes per-cell jitter
wall = dist * c10 > 1.0
hf = np.where(wall, filled + 2.0, filled).astype(np.float32)
print(f"hfield {nx10} x {ny10} @ {c10} m, seen {seen10.mean() * 100:.1f} %, wall {wall.mean() * 100:.1f} %, z {hf.min():.2f}..{hf.max():.2f}")

# ---------- trees from vegetation patches (25 cm classes)
lab, n = ndimage.label(cls25 == 3)
trees = []
for k, sl in enumerate(ndimage.find_objects(lab), start=1):
    rr, cc = np.nonzero(lab[sl] == k)
    area = len(rr) * c25 * c25
    if area < 0.06:
        continue
    yc = org25[1] + (rr.mean() + sl[0].start + 0.5) * c25
    xc = org25[0] + (cc.mean() + sl[1].start + 0.5) * c25
    r = float(np.clip(np.sqrt(area / np.pi) * 0.6, 0.08, 0.6))  # trunk thinner than the canopy footprint
    zi, zj = int((yc - y0) / c10), int((xc - x0) / c10)
    trees.append((xc, yc, float(hf[np.clip(zi, 0, ny10 - 1), np.clip(zj, 0, nx10 - 1)]), r))
trees = np.array(trees, np.float32)
print(f"trees {len(trees)} cylinders, radius {trees[:, 3].min():.2f}..{trees[:, 3].max():.2f} m")

# ---------- 25 cm display mesh with colour
c = c25
nx, ny = cls25.shape[1], cls25.shape[0]
g25, seen25 = clean(ground_grid(c, x0, y0, nx, ny), 3)
g25 = ndimage.gaussian_filter(np.where(seen25, g25, 0), 0.7) / np.maximum(ndimage.gaussian_filter(seen25.astype(np.float32), 0.7), 1e-6)
# ground colour per cell: mean colour of points within 0.15 m of the ground surface
ix = ((P[:, 0] - x0) / c).astype(np.int64).clip(0, nx - 1)
iy = ((P[:, 1] - y0) / c).astype(np.int64).clip(0, ny - 1)
near = np.abs(P[:, 2] - g25[iy, ix]) < 0.15
lin = iy[near] * nx + ix[near]
cnt = np.bincount(lin, minlength=nx * ny)
col = np.stack([np.bincount(lin, C[near, i], minlength=nx * ny) for i in range(3)], 1) / np.maximum(cnt, 1)[:, None]
col = col.reshape(ny, nx, 3)
TINT = np.array([[1.00, 0.86, 0.45], [0.92, 0.62, 0.35], [0.95, 0.35, 0.30], [0.35, 0.85, 0.40], [0.6, 0.6, 0.6]])
vcol = (0.55 * col / 255.0 + 0.45 * TINT[cls25]).clip(0, 1)
# vertices at cell centres; two triangles per 2x2 block where all four corners were seen
V = np.stack(np.meshgrid(x0 + (np.arange(nx) + 0.5) * c, y0 + (np.arange(ny) + 0.5) * c), -1)
V = np.concatenate([V, g25[..., None]], -1).reshape(-1, 3).astype(np.float32)
idx = np.arange(nx * ny).reshape(ny, nx)
q = seen25[:-1, :-1] & seen25[1:, :-1] & seen25[:-1, 1:] & seen25[1:, 1:]
a_, b_, c_, d_ = idx[:-1, :-1][q], idx[:-1, 1:][q], idx[1:, 1:][q], idx[1:, :-1][q]
F = np.concatenate([np.stack([a_, b_, c_], 1), np.stack([a_, c_, d_], 1)]).astype(np.int32)
# display only: drop near-vertical triangles (> 0.5 m rise over one 25 cm cell, about 63 deg) - they are mostly
# ground estimates pulled up under trees and show as thin fins; the 10 cm collider above is not changed
zf = g25.reshape(-1)[F]
F = F[(zf.max(1) - zf.min(1)) < 0.5]
used = np.unique(F)
remap = np.full(nx * ny, -1, np.int64)
remap[used] = np.arange(len(used))
V, VC, F = V[used], (vcol.reshape(-1, 3)[used] * 255).astype(np.uint8), remap[F].astype(np.int32)
print(f"mesh {len(V):,} vertices, {len(F):,} triangles")

# binary PLY with vertex colours
with open(f"{out}/terrain_mesh.ply", "wb") as fo:
    fo.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"property uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
    vd = np.zeros(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vd["p"], vd["c"] = V, VC
    fo.write(vd.tobytes())
    fd = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
    fd["n"], fd["i"] = 3, F
    fo.write(fd.tobytes())
np.savez_compressed(f"{out}/terrain_mesh.npz", hfield=hf, hf_origin=np.array([x0, y0], np.float32), hf_cell=np.float32(c10),
                    hf_seen=seen10, trees=trees, mesh_v=V, mesh_c=VC, mesh_f=F)
print("saved", out)
