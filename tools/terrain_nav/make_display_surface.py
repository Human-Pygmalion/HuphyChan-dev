#!/usr/bin/env python3
"""Opaque ground surface for pictures and videos, in MAP coordinates: the 10 cm collider heights (prep_terrain_mesh.py)
every <stride> cells, coloured with the survey's own ground colour (mean colour of map points within 0.3 m of the ground,
nearest-filled where a cell has none).  No holes: unsurveyed cells (the collider's 2 m walls) are drawn at the filled
ground height in dark grey, and only within 3 m of surveyed ground.
Display only, two optional clean-ups (the physics collider is not changed):
  --open-m 1.5   grey-scale morphological opening: removes bumps narrower than 1.5 m. The 10 cm ground estimate has tall
                 thin spikes where the scan saw no ground under a canopy (the "ground" came from leaves); wide slopes and
                 the trail itself are untouched.
  --corridor-m 8 draw only within this distance of the planned path (needs --terrain): for pictures where the rest
                 stays as scan points with the sky behind them.
usage: make_display_surface.py <terrain_mesh.npz> <map_010.npy> <out surface.npz> [--stride 3] [--open-m 1.5]
                               [--corridor-m 8 --terrain terrain.npz]
out: V (n,3) float32 map frame, C (n,3) uint8, F (m,3) uint32, stride_m
"""
import argparse
import time

import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("cloud")
ap.add_argument("out")
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--open-m", type=float, default=0.0)
ap.add_argument("--corridor-m", type=float, default=0.0)
ap.add_argument("--terrain", default="")
ap.add_argument("--wall-rim-m", type=float, default=3.0, help="how far into the unsurveyed (dark grey) area to draw; 0.3 for videos")
a = ap.parse_args()
t0 = time.time()
M = np.load(a.mesh)
stride = a.stride
HF = M["hfield"].astype(np.float32)
x0, y0 = (float(v) for v in M["hf_origin"])
c = float(M["hf_cell"])
NR, NC = HF.shape
WALL = ndimage.distance_transform_edt(~M["hf_seen"]) * c > 1.0
HF = np.where(WALL, HF - 2.0, HF)  # the walls are physics only
if a.open_m > 0:
    HF = ndimage.grey_opening(HF, size=(int(round(a.open_m / c)),) * 2).astype(np.float32)

cl = np.load(a.cloud)
P = cl[:, :3]
rgb = cl[:, 3].view(np.uint32)
jj = ((P[:, 0] - x0) / c).astype(np.int64)
ii = ((P[:, 1] - y0) / c).astype(np.int64)
ok = (jj >= 0) & (jj < NC) & (ii >= 0) & (ii < NR)
ok[ok] &= np.abs(P[ok, 2] - HF[ii[ok], jj[ok]]) < 0.3
lin = ii[ok] * NC + jj[ok]
cnt = np.bincount(lin, minlength=NR * NC)
CG = np.stack([np.bincount(lin, ((rgb[ok] >> s) & 255).astype(np.float64), minlength=NR * NC) for s in (16, 8, 0)], 1)
CG = (CG / np.maximum(cnt, 1)[:, None]).reshape(NR, NC, 3)
fi = ndimage.distance_transform_edt((cnt == 0).reshape(NR, NC), return_distances=False, return_indices=True)
CG = CG[fi[0], fi[1]]
CG[WALL] = (92, 94, 100)
CG = CG.astype(np.uint8)

show = ndimage.distance_transform_edt(WALL) * c < a.wall_rim_m
if a.corridor_m > 0:
    path = np.load(a.terrain)["path"][:, :2]
    onpath = np.zeros((NR, NC), bool)
    pi = np.clip(((path[:, 1] - y0) / c).astype(int), 0, NR - 1)
    pj = np.clip(((path[:, 0] - x0) / c).astype(int), 0, NC - 1)
    onpath[pi, pj] = True
    show &= ndimage.distance_transform_edt(~onpath) * c < a.corridor_m
ri, ci = np.arange(0, NR, stride), np.arange(0, NC, stride)
sub = show[np.ix_(ri, ci)]
xs = (x0 + (ci + 0.5) * c).astype(np.float32)
ys = (y0 + (ri + 0.5) * c).astype(np.float32)
V = np.stack(np.broadcast_arrays(xs[None, :], ys[:, None], HF[np.ix_(ri, ci)]), -1).reshape(-1, 3)
idx = np.arange(sub.size).reshape(sub.shape)
q = sub[:-1, :-1] & sub[1:, :-1] & sub[:-1, 1:] & sub[1:, 1:]
A, B, C, D = idx[:-1, :-1][q], idx[:-1, 1:][q], idx[1:, 1:][q], idx[1:, :-1][q]
F = np.concatenate([np.stack([A, B, C], 1), np.stack([A, C, D], 1)])
used = np.unique(F)
remap = np.full(sub.size, -1, np.int64)
remap[used] = np.arange(len(used))
np.savez_compressed(a.out, V=V[used].astype(np.float32), C=CG[np.ix_(ri, ci)].reshape(-1, 3)[used],
                    F=remap[F].astype(np.uint32), stride_m=np.float32(stride * c))
print(f"surface {len(used):,} vertices, {len(F):,} triangles at {stride * c:.2f} m; {time.time() - t0:.0f} s -> {a.out}")
