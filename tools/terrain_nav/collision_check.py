#!/usr/bin/env python3
"""Is the trail terrain really a collider?  Two measurements on the baked trail model.

  1. ground: walk the planned path for 12 s; every policy step sum the contact normal force per world geom kind
     (trail heightfield / tree cylinders / the old flat floor plane).  If the terrain is real, the heightfield carries
     the robot's weight and the flat floor carries nothing.
  2. tree: put the robot 1.2 m in front of the nearest tree to the trail start, facing it, walk 0.4 m/s for 8 s while
     steering at the trunk centre (the policy alone drifts off heading on slopes);
     a collider stops the body at the trunk (closest approach >= radius) and records tree contacts.
usage (3090): collision_check.py --cache ~/pyg_demo/cache_trail --nav ~/pyg_demo/nav
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
a = ap.parse_args()
pygviewer.CACHE_DIR = a.cache
import mujoco  # noqa: E402
from pygviewer.contract import load_contract  # noqa: E402
from pygviewer.sim_core import SimCore  # noqa: E402

ST = np.load(f"{a.nav}/trail_sim_terrain.npz")
OFF = ST["offset"].astype(float)
PATH = (np.load(f"{a.nav}/terrain.npz")["path"] - OFF)[:, :2]
pc = json.load(open(f"{a.cache}/{a.policy}.policy_contract.json"))
core = SimCore(load_contract(a.cache, pc["variant"]), realtime=False)
core.load_policy(onnx=f"{a.cache}/{a.policy}.onnx", policy_contract=pc)
core.set_base(mode="free", ground=True)
m, d = core.m, core.d
fa = core.free_adr
KIND = {}
for g in range(m.ngeom):
    n = m.geom(g).name
    KIND[g] = "heightfield" if n == "trail_ground" else "tree" if n.startswith("trail_tree") else \
        "flat floor" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE else None
weight = float(m.body_subtreemass[m.body("robot/base_link").id] * 9.81)
f6 = np.zeros(6)


def contact_forces():
    out = {"heightfield": 0.0, "tree": 0.0, "flat floor": 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        k = KIND.get(int(c.geom1)) or KIND.get(int(c.geom2))
        if k:
            mujoco.mj_contactForce(m, d, i, f6)
            out[k] += abs(f6[0])
    return out


def place(x, y, yaw):
    core.reset("knees_bent")
    hf_z = ST["hfield"]
    i = int((y - (ST["center"][1] - ST["half"][1])) / ST["cell"])
    j = int((x - (ST["center"][0] - ST["half"][0])) / ST["cell"])
    d.qpos[fa:fa + 3] += [x, y, float(hf_z[i, j]) - float(hf_z[int(ST["half"][1] / ST["cell"] - ST["center"][1] / ST["cell"]), int(ST["half"][0] / ST["cell"] - ST["center"][0] / ST["cell"])])]
    d.qpos[fa + 3:fa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    core.submit({"op": "mode", "value": "policy_sim"})


# ---- 1. ground
yaw0 = math.atan2(*(PATH[8] - PATH[0])[::-1])
place(0.0, 0.0, yaw0)
F = []
z0 = d.qpos[fa + 2]
for it in range(int(12 / (core.dt * core.decimation))):
    core.cmd = np.array([0.4 if it > 50 else 0.0, 0.0, 0.0])
    core.step_n(core.decimation)
    F.append(contact_forces())
mean = {k: np.mean([f[k] for f in F[50:]]) for k in F[0]}
print(f"[ground] robot weight {weight:.0f} N; mean contact normal force while walking 10 s: "
      + ", ".join(f"{k} {v:.0f} N" for k, v in mean.items())
      + f"; steps with heightfield contact {np.mean([f['heightfield'] > 0 for f in F[50:]]) * 100:.0f} %"
      + f"; base height change {d.qpos[fa + 2] - z0:+.2f} m")

# ---- 2. tree: independent of the policy (its steering is weak and it resists pushes) - stand 0.3 m from a trunk on the
# most level spot found, then give the whole robot 1.5 m/s straight at the trunk.  A collider stops it with tree contact
# force; without one it would cross the trunk axis within 0.4 s.
T = ST["trees"]
HF_, X0_, Y0_, C_ = ST["hfield"], ST["center"][0] - ST["half"][0], ST["center"][1] - ST["half"][1], float(ST["cell"])


def flatness(x, y):  # height range of the ground in a 1.2 m square (the robot's footprint) - a level spot to stand on
    j, i = int((x - X0_) / C_), int((y - Y0_) / C_)
    return float(np.ptp(HF_[i - 6:i + 7, j - 6:j + 7]))


best = None  # trees within 40 m of the start; approach from 16 directions; most level standing spot wins
for k_, (x_, y_, _, r_, _) in enumerate(T):
    if np.hypot(x_, y_) > 40:
        continue
    for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        sx_, sy_ = x_ - (r_ + 0.3) * math.cos(a_), y_ - (r_ + 0.3) * math.sin(a_)
        fl_ = flatness(sx_, sy_)
        if best is None or fl_ < best[0]:
            best = (fl_, k_, a_, sx_, sy_)
fl, k, ang, sx, sy = best
tx, ty, tz, tr = (float(v) for v in T[k][:4])
print(f"[tree] chosen: tree #{k} at ({tx:.2f},{ty:.2f}) radius {tr:.2f} m, standing spot ground range {fl:.2f} m over 1.2 m")
place(sx, sy, ang)
closest, tree_steps, tree_f = 9.0, 0, []
for it in range(int(2 / (core.dt * core.decimation))):
    core.cmd = np.zeros(3)
    if it == 50:  # after 1 s of standing: throw the whole robot at the trunk at 1.5 m/s
        fd = int(m.jnt_dofadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]])
        d.qvel[fd:fd + 3] = [1.5 * math.cos(ang), 1.5 * math.sin(ang), 0.0]
    core.step_n(core.decimation)
    f = contact_forces()
    tree_steps += f["tree"] > 0
    tree_f.append(f["tree"])
    if it % 25 == 0:
        print(f"[trace] t={it * core.dt * core.decimation:.1f} s surface distance {math.hypot(d.qpos[fa] - tx, d.qpos[fa + 1] - ty) - tr:+.2f} m, tree force {f['tree']:.0f} N")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] > 1 and m.geom_contype[g]:
            closest = min(closest, math.hypot(d.geom_xpos[g, 0] - tx, d.geom_xpos[g, 1] - ty) - tr)
d.xfrc_applied[:] = 0
p = d.qpos[fa:fa + 3]
print(f"[tree] tree #{k} radius {tr:.2f} m; robot 0.30 m from the trunk surface, thrown at it at 1.5 m/s after 1 s standing, watched 1 s: "
      f"steps with tree contact {tree_steps} of {len(tree_f) - 50}, peak tree contact force {max(tree_f):.0f} N, "
      f"closest robot collision-geom centre to trunk surface {closest:+.2f} m, base ended {math.hypot(p[0] - tx, p[1] - ty) - tr:+.2f} m from the surface "
      f"(no collider: it would reach the trunk axis, {-tr:.2f} m)")

# ---- 3. tree, static: put the pelvis inside the same trunk and run one collision pass (no policy, no dynamics).
# The throw above may not reach the trunk (the policy recovers and steps back); this shows the contact pairs exist.
d.qpos[fa:fa + 3] = [tx - 0.1, ty, float(T[k][2]) + 1.0]
d.qpos[fa + 3:fa + 7] = [1, 0, 0, 0]
d.qvel[:] = 0
mujoco.mj_forward(m, d)
hits = sorted({m.geom(int(c.geom1 if KIND.get(int(c.geom2)) == "tree" else c.geom2)).name.replace("robot/", "")
               for c in d.contact[:d.ncon] if "tree" in (KIND.get(int(c.geom1)), KIND.get(int(c.geom2)))})
print(f"[tree static] pelvis placed inside trunk #{k}: {sum('tree' in (KIND.get(int(c.geom1)), KIND.get(int(c.geom2))) for c in d.contact[:d.ncon])} "
      f"robot-tree contacts, robot collision geoms touching: {', '.join(hits)}")
