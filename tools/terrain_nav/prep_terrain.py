#!/usr/bin/env python3
"""Mountain-trail map (Map_Data/*.ply, coloured lidar map) -> what RViz needs for the navigation picture:
  * display cloud   : voxel-downsampled points with their RGB
  * ground model    : 2.5D height map (robust low percentile per cell, filled + smoothed)
  * semantic classes per cell, from geometry only:
        TRAIL  smooth, gentle ground       ROUGH  uneven or 10-20 deg      STEEP  > 20 deg
        VEG    vegetation / trunks 0.3-2.2 m above ground                  UNKNOWN no ground seen
  * traversal cost per class and an A* path from start to goal over that cost (8-connected), with height
Everything is computed - no hand-drawn path.
usage: prep_terrain.py <map.ply> <out.npz> [--start x y] [--goal x y] [--voxel 0.15] [--cell 0.25]
"""
import argparse
import heapq

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("ply")
ap.add_argument("out")
ap.add_argument("--start", type=float, nargs=2, default=[3.0, 0.5])
ap.add_argument("--goal", type=float, nargs=2, default=[73.0, 133.0])
ap.add_argument("--voxel", type=float, default=0.15)
ap.add_argument("--cell", type=float, default=0.25)
a = ap.parse_args()

TRAIL, ROUGH, STEEP, VEG, UNKNOWN = 0, 1, 2, 3, 4
NAMES = ["TRAIL", "ROUGH", "STEEP", "VEG", "UNKNOWN"]
COST = np.array([1.0, 4.0, 40.0, 300.0, np.inf])  # VEG: very expensive but not a wall (thin false positives must not cut the trail); UNKNOWN: never

# ---------- load
f = open(a.ply, "rb")
head = b""
while not head.endswith(b"end_header\n"):
    head += f.readline()
dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
raw = np.fromfile(f, dtype=dt)
P = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
C = np.stack([raw["r"], raw["g"], raw["b"]], 1)
print(f"points {len(P):,}")

# ---------- display cloud: one point per voxel, mean colour
vk = np.floor(P / a.voxel).astype(np.int64)
key = (vk[:, 0] - vk[:, 0].min()) * 10**10 + (vk[:, 1] - vk[:, 1].min()) * 10**5 + (vk[:, 2] - vk[:, 2].min())
uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
cloud = np.zeros((len(uk), 3), np.float64)
np.add.at(cloud, inv, P)
cloud = (cloud / cnt[:, None]).astype(np.float32)
crgb = np.zeros((len(uk), 3), np.float64)
np.add.at(crgb, inv, C)
crgb = (crgb / cnt[:, None]).astype(np.uint8)
keep = cnt >= 2  # drop single stray points
cloud, crgb = cloud[keep], crgb[keep]
print(f"display cloud {len(cloud):,} voxels ({a.voxel} m)")

# ---------- ground height map
x0, y0 = np.floor(P[:, 0].min()) - 1, np.floor(P[:, 1].min()) - 1
nx = int(np.ceil((P[:, 0].max() - x0) / a.cell)) + 2
ny = int(np.ceil((P[:, 1].max() - y0) / a.cell)) + 2
ix = ((P[:, 0] - x0) / a.cell).astype(int)
iy = ((P[:, 1] - y0) / a.cell).astype(int)
lin = iy * nx + ix
order = np.lexsort((P[:, 2], lin))  # by cell, then height
ls, zs = lin[order], P[order, 2]
starts = np.r_[0, np.flatnonzero(np.diff(ls)) + 1]
ends = np.r_[starts[1:], len(ls)]
n_in = ends - starts
ground = np.full(nx * ny, np.nan, np.float32)
ok = n_in >= 3
pick = starts[ok] + (0.05 * (n_in[ok] - 1)).astype(int)  # 5th percentile: below-ground noise is rare and sparse
ground[ls[starts[ok]]] = zs[pick]
ground = ground.reshape(ny, nx)
seen = ~np.isnan(ground)
# remove spikes, fill small holes, smooth
g = np.where(seen, ground, np.nanmedian(ground))
med = ndimage.median_filter(g, size=5)
spike = seen & (np.abs(g - med) > 0.6)
seen &= ~spike
g = np.where(seen, g, 0.0)
w = ndimage.uniform_filter(seen.astype(np.float32), 5)
gs = ndimage.uniform_filter(g, 5)
fill = (~seen) & (w > 0.2)
dem = np.where(seen, g, np.where(fill, gs / np.maximum(w, 1e-6), np.nan)).astype(np.float32)
known = ~np.isnan(dem)
dem_s = ndimage.gaussian_filter(np.where(known, dem, 0), 1.0) / np.maximum(ndimage.gaussian_filter(known.astype(np.float32), 1.0), 1e-6)
dem_s = np.where(known, dem_s, np.nan).astype(np.float32)

# ---------- geometry features
gy, gx = np.gradient(np.where(known, dem_s, 0), a.cell)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
rough = np.abs(np.where(known, dem, 0) - np.where(known, dem_s, 0))
# vegetation: points 0.3-2.2 m above local ground, counted per cell
cz = dem_s.reshape(-1)[lin]
h = P[:, 2] - cz
veg_pts = np.bincount(lin[(h > 0.3) & (h < 2.2)], minlength=nx * ny).reshape(ny, nx)
all_pts = np.bincount(lin, minlength=nx * ny).reshape(ny, nx)

cls = np.full((ny, nx), UNKNOWN, np.uint8)
cls[known] = TRAIL
cls[known & ((slope > 10) | (rough > 0.12))] = ROUGH
cls[known & (slope > 20)] = STEEP
vegm = known & (veg_pts >= 12) & (veg_pts > 0.2 * all_pts)
vegm = ndimage.binary_opening(vegm, structure=np.ones((3, 3)))  # drop 1-2 cell lines (mapping carrier / stray points), keep bushes and trunks
cls[vegm] = VEG
for k in range(5):
    print(f"  {NAMES[k]:8s} {np.mean(cls == k) * 100:5.1f} % of cells")

# ---------- A* over cost
cost = COST[cls]
# soft preference for staying away from vegetation edges
dist = ndimage.distance_transform_edt(cls != VEG) * a.cell
cost = cost * (1.0 + 2.0 * np.exp(-dist / 0.6))
# keep to the middle of the trail: cheaper the farther a trail cell is from any non-trail cell
dtrail = ndimage.distance_transform_edt(cls == TRAIL) * a.cell
cost = np.where(cls == TRAIL, cost * (1.0 + 1.5 * np.exp(-dtrail / 0.5)), cost)


def cell_of(xy):
    return int((xy[1] - y0) / a.cell), int((xy[0] - x0) / a.cell)


def nearest_free(rc):
    free = np.argwhere(np.isfinite(cost))
    d = np.hypot(free[:, 0] - rc[0], free[:, 1] - rc[1])
    return tuple(free[np.argmin(d)])


s, t = nearest_free(cell_of(a.start)), nearest_free(cell_of(a.goal))
nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
gscore = np.full(cost.shape, np.inf)
gscore[s] = 0
parent = {}
pq = [(0.0, s)]
while pq:
    _, u = heapq.heappop(pq)
    if u == t:
        break
    for dr, dc in nb:
        v = (u[0] + dr, u[1] + dc)
        if not (0 <= v[0] < ny and 0 <= v[1] < nx) or not np.isfinite(cost[v]):
            continue
        step = np.hypot(dr, dc) * 0.5 * (cost[u] + cost[v])
        if gscore[u] + step < gscore[v]:
            gscore[v] = gscore[u] + step
            parent[v] = u
            heapq.heappush(pq, (gscore[v] + np.hypot(v[0] - t[0], v[1] - t[1]), v))
if t not in parent:
    np.savez_compressed(a.out + ".nopath.npz", dem=dem_s, cls=cls, slope=slope.astype(np.float32), veg=veg_pts, allp=all_pts, rough=rough.astype(np.float32), origin=np.array([x0, y0], np.float32)); raise SystemExit("no path found")
cells = [t]
while cells[-1] != s:
    cells.append(parent[cells[-1]])
cells = np.array(cells[::-1])
xy = np.stack([x0 + (cells[:, 1] + 0.5) * a.cell, y0 + (cells[:, 0] + 0.5) * a.cell], 1)
# smooth the grid staircase (moving average, ends fixed)
k = 9
pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
xy = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], 1)
pz = dem_s[((xy[:, 1] - y0) / a.cell).astype(int), ((xy[:, 0] - x0) / a.cell).astype(int)]
path = np.column_stack([xy, pz]).astype(np.float32)
seg = np.diff(path, axis=0)
length = float(np.sum(np.linalg.norm(seg[:, :2], axis=1)))
pc = cls[cells[:, 0], cells[:, 1]]
print(f"path {len(path)} points, {length:.1f} m, climb {path[:, 2].max() - path[:, 2].min():.1f} m, "
      + ", ".join(f"{NAMES[k]} {np.mean(pc == k) * 100:.0f}%" for k in range(3)))

# semantic labels for the picture: walk along the path, look up to 6 m to each side, name what is there
labels = []
sl = np.where(known, slope, 0)
for i in range(40, len(path) - 40, 110):
    p0, d = path[i], path[min(i + 5, len(path) - 1), :2] - path[max(i - 5, 0), :2]
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    for side in (1, -1):
        for r in np.arange(1.5, 6.0, 0.25):
            q = p0[:2] + side * r * nrm
            ri, ci = int((q[1] - y0) / a.cell), int((q[0] - x0) / a.cell)
            if not (0 <= ri < ny and 0 <= ci < nx):
                break
            k = int(cls[ri, ci])
            if k == VEG:
                labels.append(("TREES", float(q[0]), float(q[1]), float(dem_s[ri, ci] + 2.5)))
                break
            if k == STEEP:
                w0 = sl[max(ri - 4, 0):ri + 5, max(ci - 4, 0):ci + 5]
                labels.append((f"STEEP SLOPE {np.median(w0):.0f} deg", float(q[0]), float(q[1]), float(dem_s[ri, ci] + 1.5)))
                break
    labels.append(("TRAIL", float(p0[0]), float(p0[1]), float(p0[2] + 0.6)))
print(f"{len(labels)} semantic labels")

np.savez_compressed(
    a.out, labels=np.array([l[0] for l in labels]), label_xyz=np.array([l[1:] for l in labels], np.float32), cloud=cloud, cloud_rgb=crgb, dem=dem_s, cls=cls, slope=slope.astype(np.float32),
    origin=np.array([x0, y0], np.float32), cell=np.float32(a.cell), path=path, path_cls=pc,
    start=np.array(a.start, np.float32), goal=np.array(a.goal, np.float32), names=np.array(NAMES), cost=COST)
print("saved", a.out)
