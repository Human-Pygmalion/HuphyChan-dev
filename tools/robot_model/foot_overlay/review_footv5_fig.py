#!/usr/bin/env python3
"""Side view (x forward, z up, ankle frame, mm) of the R foot parts of the footv5 URDF with each part's centre of mass. usage (mjlab venv): review_footv5_fig.py <out.png>"""
import sys, trimesh, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
U = '/home/syaro/MikuchanRemote/Human-Pygmalion/pygmalion_locomotion/assets/pygmalion_v2/meshes_footv5/'
P = {'carrier': ('#d9cc40', 'Sole carrier (ABS, 119 g)'), 'tape': ('#bbbbbb', 'Tape 1 mm (19 g)'), 'sheet': ('#2b8cbe', 'Rubber 4 mm (123 g)'), 'feet_up': ('#59c047', 'Feet_Up cover (PLA, 80 g, user)')}
fig, ax = plt.subplots(figsize=(10, 4.2))
for p, (col, lab) in P.items():
    m = trimesh.load(U + f'R_v5_{p}.stl'); v = m.vertices * 1000
    ax.scatter(v[::6, 0], v[::6, 2], s=0.6, c=col, label=lab, alpha=.5)
    if not m.is_watertight: m.merge_vertices(); trimesh.repair.fill_holes(m)
    c = m.center_mass * 1000; ax.plot(c[0], c[2], 'k+', ms=10)
ax.plot(41.4, -34.6, 'r*', ms=14, label='foot total COM (1.186 kg)'); ax.plot(0, 0, 'ko', label='ankle origin')
ax.axhline(-57, c='r', ls=':', lw=.8); ax.text(-80, -56, 'sole bottom z=-57 mm', color='r', fontsize=8, va='bottom')
ax.set_aspect('equal'); ax.set_xlabel('x forward (mm)'); ax.set_ylabel('z up (mm)'); ax.legend(fontsize=7, loc='upper right'); ax.set_title('footv5 right foot, side view (black + = part COM)')
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=130)
