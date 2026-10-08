#!/usr/bin/env python3
"""Terrain for the MuJoCo teleop: the 10 cm collider + tree cylinders from prep_terrain_mesh.py, shifted so the trail
start is the world origin with ground height 0 (the robot spawns at the origin on z = 0).
map coordinates = sim coordinates + offset.   usage: make_sim_terrain.py <terrain.npz> <terrain_mesh.npz> <out.npz>"""
import sys

import numpy as np

T, M = np.load(sys.argv[1]), np.load(sys.argv[2])
hf, (x0, y0), c = M["hfield"].astype(np.float64), M["hf_origin"].astype(float), float(M["hf_cell"])
sx, sy = (float(v) for v in T["path"][0, :2])  # first point of the planned path = on the trail
z0 = float(hf[int((sy - y0) / c), int((sx - x0) / c)])
ny, nx = hf.shape
trees = M["trees"].astype(np.float64).copy()
# trunk spans from the LOWEST ground under its footprint (minus 0.3 m, buried) to the HIGHEST plus 3 m: trees stand on
# hillsides, and a cylinder starting at the centre-cell height floats above the downhill side (a robot walked under one)
lo, hi = np.empty(len(trees)), np.empty(len(trees))
for n, (tx, ty, _, r) in enumerate(trees):
    j0, j1 = int((tx - r - 0.2 - x0) / c), int((tx + r + 0.2 - x0) / c) + 1
    i0, i1 = int((ty - r - 0.2 - y0) / c), int((ty + r + 0.2 - y0) / c) + 1
    patch = hf[max(i0, 0):max(i1, 1), max(j0, 0):max(j1, 1)]
    lo[n], hi[n] = patch.min() - 0.3, patch.max() + 3.0
trees = np.column_stack([trees[:, 0] - sx, trees[:, 1] - sy, lo - z0, trees[:, 3], hi - z0])  # x, y, bottom, radius, top
np.savez_compressed(sys.argv[3], hfield=(hf - z0).astype(np.float32), center=np.array([x0 + nx * c / 2 - sx, y0 + ny * c / 2 - sy]),
                    half=np.array([nx * c / 2, ny * c / 2]), cell=c, trees=trees.astype(np.float32), offset=np.array([sx, sy, z0]))
print(f"sim terrain {nx} x {ny}, offset map = sim + ({sx:.2f}, {sy:.2f}, {z0:.2f}), z {hf.min() - z0:.2f}..{hf.max() - z0:.2f}, trees {len(trees)}")
