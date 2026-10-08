#!/usr/bin/env python3
"""Does the walking policy actually get along the real trail?  Closed loop on the baked trail terrain:
planned A* path -> pure-pursuit path follower -> velocity command (vx, wz) -> v11a policy -> MuJoCo.

Fall = base less than 0.45 m above the ground under it, or body tilt over 60 deg (absolute height is meaningless on a
trail that descends 9 m).  Prints distance along the path, falls, cross-track error, and saves the trajectory.
usage (3090): trail_walk_check.py --cache ~/pyg_demo/cache_trail --nav ~/pyg_demo/nav [--seconds 60] [--vx 0.5] [--out x.npz]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pygviewer"))
import pygviewer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--nav", required=True)
ap.add_argument("--policy", default="v11a")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--vx", type=float, default=0.5)
ap.add_argument("--lookahead", type=float, default=1.0)
ap.add_argument("--out", default="")
ap.add_argument("--reverse", action="store_true", help="walk the planned path backwards (goal -> start; the trail climbs 9 m that way)")
a = ap.parse_args()
pygviewer.CACHE_DIR = a.cache
import mujoco  # noqa: E402
from pygviewer.contract import load_contract  # noqa: E402
from pygviewer.sim_core import SimCore  # noqa: E402

ST = np.load(f"{a.nav}/trail_sim_terrain.npz")
OFF = ST["offset"].astype(float)
HF, HC, HH, CELL = ST["hfield"], ST["center"].astype(float), ST["half"].astype(float), float(ST["cell"])
PATH = (np.load(f"{a.nav}/terrain.npz")["path"] - OFF)[:, :2]
if a.reverse:
    PATH = PATH[::-1].copy()
SEG = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))]


def ground_z(x, y):
    j = int(np.clip((x - (HC[0] - HH[0])) / CELL, 0, HF.shape[1] - 1))
    i = int(np.clip((y - (HC[1] - HH[1])) / CELL, 0, HF.shape[0] - 1))
    return float(HF[i, j])


pc = json.load(open(f"{a.cache}/{a.policy}.policy_contract.json"))
core = SimCore(load_contract(a.cache, pc["variant"]), realtime=False)
core.load_policy(onnx=f"{a.cache}/{a.policy}.onnx", policy_contract=pc)
core.set_base(mode="free", ground=True)
m, d = core.m, core.d
assert m.nhfield == 1, "model has no terrain heightfield - bake with PYG_BAKE_TERRAIN"
core.reset("knees_bent")
fa = core.free_adr
d.qpos[fa:fa + 2] += PATH[0]  # spawn on the first path point (the origin unless --reverse)
d.qpos[fa + 2] += ground_z(*PATH[0]) - ground_z(0.0, 0.0)
yaw0 = math.atan2(*(PATH[8] - PATH[0])[::-1])
d.qpos[fa + 3:fa + 7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
mujoco.mj_forward(m, d)
core.submit({"op": "mode", "value": "policy_sim"})

log = dict(t=[], xyz=[], above=[], tilt=[], s=[], xte=[], cmd=[], qpos=[])  # qpos: full state every policy step (for replay/render)
fell, k_near = "", 0
for it in range(int(a.seconds / (core.dt * core.decimation))):
    p = d.qpos[fa:fa + 3]
    w, x, y, z = d.qpos[fa + 3:fa + 7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    up_z = 1 - 2 * (x * x + y * y)  # body z-axis . world z
    # nearest path point (search forward only) and the lookahead target
    win = slice(k_near, min(k_near + 80, len(PATH)))
    k_near = win.start + int(np.argmin(np.linalg.norm(PATH[win] - p[:2], axis=1)))
    k_t = min(int(np.searchsorted(SEG, SEG[k_near] + a.lookahead)), len(PATH) - 1)
    err = math.atan2(PATH[k_t, 1] - p[1], PATH[k_t, 0] - p[0]) - yaw
    err = (err + math.pi) % (2 * math.pi) - math.pi
    wz = float(np.clip(1.5 * err, -0.6, 0.6))
    vx = a.vx * max(0.2, math.cos(err)) if abs(err) < 1.2 else 0.0  # slow down in sharp turns
    core.cmd = np.array([vx, 0.0, wz])
    core.step_n(core.decimation)
    above = float(p[2] - ground_z(p[0], p[1]))
    tilt = math.degrees(math.acos(np.clip(up_z, -1, 1)))
    log["t"].append(d.time); log["xyz"].append(p.copy()); log["above"].append(above); log["tilt"].append(tilt)
    log["qpos"].append(d.qpos.copy())
    log["s"].append(SEG[k_near]); log["xte"].append(float(np.linalg.norm(PATH[k_near] - p[:2]))); log["cmd"].append((vx, wz))
    if above < 0.45 or tilt > 60:
        fell = f"t={d.time:.1f}s above={above:.2f} m tilt={tilt:.0f} deg"
        break
    if k_near >= len(PATH) - 3:
        break

L = {k: np.array(v) for k, v in log.items()}
xyz = L["xyz"]
print(f"time {L['t'][-1]:.1f} s | along path {L['s'][-1]:.1f} m of {SEG[-1]:.1f} | walked {np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum():.1f} m"
      f" | height change {xyz[-1, 2] - xyz[0, 2]:+.2f} m | cross-track mean {L['xte'].mean():.2f} max {L['xte'].max():.2f} m"
      f" | base above ground {L['above'].min():.2f}..{L['above'].max():.2f} m | tilt max {L['tilt'].max():.0f} deg"
      f" | {'FELL ' + fell if fell else 'no fall'}")
if a.out:
    np.savez(a.out, **L, path=PATH, offset=OFF, joint_names=np.array([m.joint(j).name for j in range(m.njnt)]),
             jnt_qposadr=m.jnt_qposadr, free_adr=fa, dt=core.dt * core.decimation)
