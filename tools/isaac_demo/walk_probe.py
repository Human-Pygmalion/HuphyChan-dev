#!/usr/bin/env python3
"""Does the baked LegOnly-AB model + v11a policy walk straight in plain MuJoCo (no mjlab at run time)?

usage: walk_probe.py --cache <dir> [--vx 0.8] [--seconds 15] [--out traj.npz]
Writes the 50 Hz trajectory (base pose + 12 leg joints) so the Isaac side can be tested without the physics.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pygviewer'))
import pygviewer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--policy', default='v11a')
ap.add_argument('--vx', type=float, default=0.8)
ap.add_argument('--seconds', type=float, default=15.0)
ap.add_argument('--warm', type=float, default=2.0)
ap.add_argument('--out', default='')
a = ap.parse_args()
pygviewer.CACHE_DIR = a.cache
from pygviewer.contract import load_contract  # noqa: E402
from pygviewer.sim_core import SimCore  # noqa: E402

pc = json.load(open(f'{a.cache}/{a.policy}.policy_contract.json'))
core = SimCore(load_contract(a.cache, pc['variant']), realtime=False)
core.load_policy(onnx=f'{a.cache}/{a.policy}.onnx', policy_contract=pc)
core.set_base(mode='free', ground=True)
core.reset('knees_bent')
core.submit({'op': 'mode', 'value': 'policy_sim'})
n = int(a.seconds / core.dt)
ramp = int(a.warm / core.dt)
t, pos, quat, q = [], [], [], []
fell = False
for i in range(n):
    if i % core.decimation == 0:
        core.cmd = np.array([a.vx * min(1.0, i / max(ramp, 1)), 0.0, 0.0])
    core.step_n(1)
    if i % core.decimation == 0:
        qp = core.d.qpos
        t.append(core.d.time); pos.append(qp[core.free_adr:core.free_adr + 3].copy())
        quat.append(qp[core.free_adr + 3:core.free_adr + 7].copy()); q.append(qp[core.all_q].copy())
        if qp[core.free_adr + 2] < 0.45:
            fell = True
            break
pos = np.array(pos); t = np.array(t)
k = t > a.warm
v = (pos[-1, 0] - pos[k][0, 0]) / (t[-1] - t[k][0]) if k.any() else 0
print(f'steps {len(t)} fell={fell} x {pos[0, 0]:.2f}->{pos[-1, 0]:.2f} y {pos[-1, 1]:.2f} z mean {pos[:, 2].mean():.3f} mean vx after warm {v:.3f}')
print('joint order', core.names)
if a.out:
    np.savez(a.out, t=t, pos=pos, quat=np.array(quat), q=np.array(q), names=np.array(core.names))
