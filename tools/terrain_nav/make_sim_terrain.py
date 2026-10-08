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
trees[:, 0] -= sx
trees[:, 1] -= sy
trees[:, 2] -= z0
np.savez_compressed(sys.argv[3], hfield=(hf - z0).astype(np.float32), center=np.array([x0 + nx * c / 2 - sx, y0 + ny * c / 2 - sy]),
                    half=np.array([nx * c / 2, ny * c / 2]), cell=c, trees=trees.astype(np.float32), offset=np.array([sx, sy, z0]))
print(f"sim terrain {nx} x {ny}, offset map = sim + ({sx:.2f}, {sy:.2f}, {z0:.2f}), z {hf.min() - z0:.2f}..{hf.max() - z0:.2f}, trees {len(trees)}")
