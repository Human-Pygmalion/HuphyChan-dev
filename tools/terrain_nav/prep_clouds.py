#!/usr/bin/env python3
"""Whole map for interactive RViz: the PLY as packed xyz+rgb float32 arrays at several resolutions.
  map_full.npy (every point), map_005.npy (5 cm voxels), map_010.npy (10 cm voxels); each N x 4 (x, y, z, rgb-as-float).
usage: prep_clouds.py <map.ply> <out dir>"""
import os
import sys

import numpy as np

ply, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
f = open(ply, "rb")
h = b""
while not h.endswith(b"end_header\n"):
    h += f.readline()
a = np.fromfile(f, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")]))
P = np.stack([a["x"], a["y"], a["z"]], 1)
C = np.stack([a["r"], a["g"], a["b"]], 1).astype(np.float64)


def pack(p, c):
    rgb = (c[:, 0].astype(np.uint32) << 16) | (c[:, 1].astype(np.uint32) << 8) | c[:, 2].astype(np.uint32)
    return np.column_stack([p.astype(np.float32), rgb.view(np.float32)])


np.save(f"{out}/map_full.npy", pack(P, C))
for v, tag in [(0.05, "005"), (0.10, "010")]:
    k = np.floor(P / v).astype(np.int64)
    k -= k.min(0)
    _, inv, cnt = np.unique(k[:, 0] * 10**12 + k[:, 1] * 10**6 + k[:, 2], return_inverse=True, return_counts=True)
    ps, cs = np.zeros((len(cnt), 3)), np.zeros((len(cnt), 3))
    np.add.at(ps, inv, P)
    np.add.at(cs, inv, C)
    np.save(f"{out}/map_{tag}.npy", pack(ps / cnt[:, None], cs / cnt[:, None]))
    print(tag, len(cnt))
print("full", len(P))
