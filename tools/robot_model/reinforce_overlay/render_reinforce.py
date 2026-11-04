#!/usr/bin/env python3
"""Offscreen (EGL) renders of the reinforcement model: whole body from the front and the side, close-ups of the hip plate, the knee plate and the
camera bracket. Plates are orange. Runs on the 3090 host (see docs/185 §10): MUJOCO_GL=egl python3 render_reinforce.py <xml> <out.png>.
Labels are English only."""
import sys
import mujoco, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xml, out = sys.argv[1], sys.argv[2]
m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.offwidth = m.vis.global_.offheight = 720
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 720, 720)
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[2] = 1                                  # visual meshes only (hulls and collision shapes hidden)
base = d.xpos[m.body("base_link").id]


def pos(body, off=(0, 0, 0)):
    b = m.body(body).id
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array(off)


def shot(lookat, dist, az, el):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam, opt)
    return r.render().copy()


views = [("front (from +x)", shot(base + [0, 0, -0.30], 1.9, 180, -5)),
         ("side (from +y, left leg side)", shot(base + [0, 0, -0.30], 1.9, 90, -5)),
         ("hip plates (orange), front-left", shot(pos("R_hip_roll_link", (0, 0.1, 0)), 0.75, 140, -15)),
         ("knee plates (orange), front-left", shot(pos("R_shin_link", (0, 0.12, -0.10)), 0.75, 140, -15)),
         ("pelvis camera + bracket, from front-right", shot(base + [0.10, 0, -0.02], 0.8, 215, 10)),
         ("pelvis camera + bracket, from above", shot(base + [0.10, 0, -0.02], 0.8, 180, -70))]
fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
for a, (t, im) in zip(ax.ravel(), views):
    a.imshow(im); a.set_title(t, fontsize=12); a.axis("off")
mass = m.body_mass.sum()
fig.suptitle(f"reinf_loop (footv5 + D455 + reinforcement)\nplates 160 g hip + 158.2 g knee per leg, Pelvis_Camera 166 g; total {mass:.3f} kg", fontsize=12)
fig.tight_layout(h_pad=2.5)
fig.savefig(out, dpi=110)
print("wrote", out)
