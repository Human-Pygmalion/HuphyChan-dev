#!/usr/bin/env python3
"""Offscreen render of the aluminium visualization. Run on the 3090 host (MUJOCO_GL=egl) on a self-contained package made by `package_vis.py`:
    python3 render_al_parts.py <alvis.xml> <out.png>
red = aluminium converted in rounds 1-2, dark red = already aluminium, grey translucent = everything else. English labels only."""
import sys
import mujoco, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

xml, out = sys.argv[1], sys.argv[2]
m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.offwidth = m.vis.global_.offheight = 760
m.vis.headlight.ambient[:] = 0.55; m.vis.headlight.diffuse[:] = 0.85; m.vis.headlight.specular[:] = 0.2     # the red parts read as red, not near-black
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 760, 760)
opt_grey = mujoco.MjvOption(); opt_grey.geomgroup[:] = 0; opt_grey.geomgroup[2] = 1     # translucent rest
opt_red = mujoco.MjvOption(); opt_red.geomgroup[:] = 0; opt_red.geomgroup[1] = 1         # aluminium parts, opaque, own pass
base = d.xpos[m.body("base_link").id]
pos = lambda b, off=(0, 0, 0): d.xpos[m.body(b).id] + d.xmat[m.body(b).id].reshape(3, 3) @ np.array(off)


def shot(lookat, dist, az, el):
    """two passes: the translucent rest, then the aluminium parts on top (opaque, they may sit inside a translucent shell, so they are not dimmed by it)"""
    cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam, opt_grey); grey = r.render().copy()
    r.update_scene(d, cam, opt_red); red = r.render().copy()
    mask = red.max(axis=2) > 10
    grey[mask] = red[mask]
    return grey


views = [("front (from +x)", shot(base + [0, 0, -0.42], 1.6, 180, -4)),
         ("left side (from +y)", shot(base + [0, 0, -0.42], 1.6, 90, -4)),
         ("hip: HipRoll2Yaw A / B / Mount", shot(pos("L_hip_roll_link", (0, 0.04, -0.05)), 0.62, 125, -12)),
         ("thigh: Yaw2KneePlateA", shot(pos("L_thigh_link", (-0.03, 0.05, -0.22)), 0.7, 125, -10)),
         ("knee-shin: Knee2Ankle Idle / Rotor sides", shot(pos("L_shin_link", (-0.03, 0.05, -0.12)), 0.75, 125, -10)),
         ("ankle: cranks, revised core, rods", shot(pos("L_shin_link", (-0.03, 0.06, -0.42)), 0.6, 125, -8))]
fig, ax = plt.subplots(2, 3, figsize=(15, 11.6))
for a, (t, im) in zip(ax.ravel(), views):
    a.imshow(im); a.set_title(t, fontsize=12, pad=6); a.axis("off")
fig.legend(handles=[Patch(color="#f21414", label="aluminium (converted now: 11 parts per leg; already aluminium: AnkleA2B braces, Arm A/B rods)"),
                    Patch(facecolor="#b8c2d1", alpha=0.35, label="not aluminium (printed parts, Knee2Ankle _2 plates and Connector A/B, motors, bearings, screws, foot, camera): translucent")],
           loc="lower center", ncol=1, fontsize=11, frameon=False)
fig.suptitle("Aluminium parts in LegOnly ..._reinf_alparts2_loop (both legs)", fontsize=13)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.1, hspace=0.12, wspace=0.03)
fig.savefig(out, dpi=105)
print("wrote", out)
