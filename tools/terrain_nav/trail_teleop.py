#!/usr/bin/env python3
"""Drive Huphy (LegOnly-AB + v11a walking policy) yourself on the real mountain-trail terrain - MuJoCo physics with the
10 cm heightfield + tree cylinders baked into the model (bake.py PYG_BAKE_TERRAIN), shown in the browser with viser.

  3D view + sliders/buttons :  http://<host>:8095/      (viser, bound to 0.0.0.0)
  keyboard page            :  http://<host>:8096/      W/S or arrows = faster/slower, A/D = turn, Space = stop, R = reset

The policy only gets a velocity command (vx, wz) - exactly what a planner would send; legs, balance and footholds on the
uneven ground are the trained policy's own. Physics 200 Hz, policy 50 Hz, real time.
run (3090, mjlab venv):  python trail_teleop.py --cache ~/pyg_demo/cache_trail --nav ~/pyg_demo/nav
"""
import argparse
import http.server
import json
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pygviewer"))
import pygviewer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--nav", required=True, help="dir with trail_sim_terrain.npz, terrain.npz, terrain_mesh.npz, map_010.npy")
ap.add_argument("--policy", default="v11a")
ap.add_argument("--port", type=int, default=8095)
ap.add_argument("--key-port", type=int, default=8096)
ap.add_argument("--cloud", default="010", help="map cloud resolution: 010 (2.2 M points) or 005 (5.3 M)")
a = ap.parse_args()
pygviewer.CACHE_DIR = a.cache
import mujoco  # noqa: E402
import trimesh  # noqa: E402
import viser  # noqa: E402
from pygviewer.contract import load_contract  # noqa: E402
from pygviewer.sim_core import SimCore  # noqa: E402

ST = np.load(f"{a.nav}/trail_sim_terrain.npz")
OFF = ST["offset"].astype(float)  # map = sim + OFF
HF, HC, HH = ST["hfield"], ST["center"].astype(float), ST["half"].astype(float)
CELL = float(ST["cell"])


def ground_z(x, y):
    j = int(np.clip((x - (HC[0] - HH[0])) / CELL, 0, HF.shape[1] - 1))
    i = int(np.clip((y - (HC[1] - HH[1])) / CELL, 0, HF.shape[0] - 1))
    return float(HF[i, j])


# ---------------- physics
pc = json.load(open(f"{a.cache}/{a.policy}.policy_contract.json"))
core = SimCore(load_contract(a.cache, pc["variant"]), realtime=False)
core.load_policy(onnx=f"{a.cache}/{a.policy}.onnx", policy_contract=pc)
core.set_base(mode="free", ground=True)
m, d = core.m, core.d
lock = threading.Lock()
state = dict(vx=0.0, wz=0.0, fallen=False, follow=True, reset=None, t_cmd=time.time())


def do_reset(x=0.0, y=0.0, yaw=0.0):
    core.reset("knees_bent")
    fa = core.free_adr
    d.qpos[fa + 0] += x
    d.qpos[fa + 1] += y
    d.qpos[fa + 2] += ground_z(x, y) - ground_z(0.0, 0.0)
    d.qpos[fa + 3:fa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(m, d)
    core.submit({"op": "mode", "value": "policy_sim"})
    state["fallen"] = False


def base_pose():
    fa = core.free_adr
    p = d.qpos[fa:fa + 3].copy()
    w, x, y, z = d.qpos[fa + 3:fa + 7]
    return p, math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def physics():
    do_reset()
    nxt = time.perf_counter()
    while True:
        with lock:
            r = state["reset"]
            state["reset"] = None
            if r is not None:
                do_reset(*r)
            core.cmd = np.array([state["vx"], 0.0, state["wz"]])
            core.step_n(core.decimation)
            p, _ = base_pose()
            if p[2] - ground_z(p[0], p[1]) < 0.45:
                state["fallen"] = True
        nxt += core.decimation * core.dt
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            nxt = time.perf_counter()  # behind: do not try to catch up


# ---------------- viser scene
srv = viser.ViserServer(host="0.0.0.0", port=a.port)
srv.scene.set_up_direction("+z")
cl = np.load(f"{a.nav}/map_{a.cloud}.npy")
rgb = cl[:, 3].view(np.uint32)
cols = np.stack([(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], 1).astype(np.uint8)
h_cloud = srv.scene.add_point_cloud("/map", points=(cl[:, :3] - OFF).astype(np.float32), colors=cols, point_size=0.02)
TM = np.load(f"{a.nav}/terrain_mesh.npz")
tmesh = trimesh.Trimesh(TM["mesh_v"] - OFF, TM["mesh_f"], vertex_colors=np.column_stack([TM["mesh_c"], np.full(len(TM["mesh_c"]), 255)]), process=False)
h_mesh = srv.scene.add_mesh_trimesh("/terrain_mesh", tmesh, visible=False)
T = np.load(f"{a.nav}/terrain.npz")
srv.scene.add_spline_catmull_rom("/planned_path", positions=(T["path"] - OFF + [0, 0, 0.15]).astype(np.float32)[::3],
                                 color=(60, 235, 130), line_width=4.0)

# robot: visual geoms (group 2) of every robot body, posed per body each frame
bodies = {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if b == 0 or int(m.geom_group[g]) != 2:
        continue
    if b not in bodies:
        bodies[b] = srv.scene.add_frame(f"/robot/b{b}", show_axes=False)
    rgba = (m.geom_rgba[g][:3] * 255).astype(int)
    kw = dict(position=tuple(m.geom_pos[g]), wxyz=tuple(m.geom_quat[g]))
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        srv.scene.add_mesh_simple(f"/robot/b{b}/g{g}", vertices=v, faces=f, color=tuple(int(c) for c in rgba), **kw)
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cyl = trimesh.creation.cylinder(radius=float(m.geom_size[g][0]), height=2 * float(m.geom_size[g][1]), sections=20)
        srv.scene.add_mesh_simple(f"/robot/b{b}/g{g}", vertices=cyl.vertices, faces=cyl.faces, color=(30, 30, 34), **kw)

# ---------------- GUI
with srv.gui.add_folder("Drive"):
    s_vx = srv.gui.add_slider("forward speed [m/s]", min=-0.4, max=1.2, step=0.05, initial_value=0.0)
    s_wz = srv.gui.add_slider("turn rate [rad/s]", min=-0.6, max=0.6, step=0.05, initial_value=0.0)
    b_f, b_b = srv.gui.add_button("faster"), srv.gui.add_button("slower")
    b_l, b_r = srv.gui.add_button("turn left"), srv.gui.add_button("turn right")
    b_s = srv.gui.add_button("STOP")
with srv.gui.add_folder("View / reset"):
    c_follow = srv.gui.add_checkbox("camera follows robot", True)
    c_mesh = srv.gui.add_checkbox("terrain mesh (= collider)", False)
    c_cloud = srv.gui.add_checkbox("point cloud", True)
    b_r0 = srv.gui.add_button("reset at trail start")
    b_rh = srv.gui.add_button("reset here (get up)")
status = srv.gui.add_markdown("starting ...")
srv.gui.add_markdown(f"Keyboard: open **http://10.8.0.14:{a.key_port}/** in another tab (W A S D, Space, R).")


def set_cmd(vx=None, wz=None):
    with lock:
        if vx is not None:
            state["vx"] = float(np.clip(vx, -0.4, 1.2))
        if wz is not None:
            state["wz"] = float(np.clip(wz, -0.6, 0.6))
    s_vx.value, s_wz.value = state["vx"], state["wz"]


s_vx.on_update(lambda _: set_cmd(vx=s_vx.value))
s_wz.on_update(lambda _: set_cmd(wz=s_wz.value))
b_f.on_click(lambda _: set_cmd(vx=state["vx"] + 0.1))
b_b.on_click(lambda _: set_cmd(vx=state["vx"] - 0.1))
b_l.on_click(lambda _: set_cmd(wz=state["wz"] + 0.1))
b_r.on_click(lambda _: set_cmd(wz=state["wz"] - 0.1))
b_s.on_click(lambda _: set_cmd(0.0, 0.0))
c_follow.on_update(lambda _: state.update(follow=c_follow.value))
c_mesh.on_update(lambda _: setattr(h_mesh, "visible", c_mesh.value))
c_cloud.on_update(lambda _: setattr(h_cloud, "visible", c_cloud.value))


def reset_here(_=None):
    p, yaw = base_pose()
    with lock:
        state["reset"] = (float(p[0]), float(p[1]), yaw)
    set_cmd(0.0, 0.0)


b_r0.on_click(lambda _: (state.update(reset=(0.0, 0.0, 0.0)), set_cmd(0.0, 0.0)))
b_rh.on_click(reset_here)


# ---------------- keyboard page
KEY_HTML = """<!doctype html><html><body style="background:#111;color:#ddd;font:20px sans-serif;text-align:center">
<h2>Huphy trail teleop - keyboard</h2><p>W / Up: faster &nbsp; S / Down: slower &nbsp; A / Left, D / Right: turn &nbsp; Space: stop &nbsp; R: get up here</p>
<p id=s>-</p><script>
document.addEventListener('keydown', e => { const k = e.key.toLowerCase(); let q = null;
 if (k=='w'||k=='arrowup') q='dvx=0.1'; else if (k=='s'||k=='arrowdown') q='dvx=-0.1'; else if (k=='a'||k=='arrowleft') q='dwz=0.1';
 else if (k=='d'||k=='arrowright') q='dwz=-0.1'; else if (k==' ') q='stop=1'; else if (k=='r') q='reset=1';
 if (q) { e.preventDefault(); fetch('/cmd?'+q).then(r=>r.text()).then(t=>document.getElementById('s').textContent=t); } });
</script></body></html>"""


class Keys(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/cmd"):
            from urllib.parse import parse_qs, urlparse
            q = {k: float(v[0]) for k, v in parse_qs(urlparse(self.path).query).items()}
            if "stop" in q:
                set_cmd(0.0, 0.0)
            if "reset" in q:
                reset_here()
            set_cmd(vx=state["vx"] + q.get("dvx", 0.0), wz=state["wz"] + q.get("dwz", 0.0))
            body = f"vx {state['vx']:+.2f} m/s   wz {state['wz']:+.2f} rad/s".encode()
            ctype = "text/plain"
        else:
            body, ctype = KEY_HTML.encode(), "text/html"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


threading.Thread(target=lambda: http.server.ThreadingHTTPServer(("0.0.0.0", a.key_port), Keys).serve_forever(), daemon=True).start()
threading.Thread(target=physics, daemon=True).start()
print(f"teleop: http://10.8.0.14:{a.port}/  keyboard: http://10.8.0.14:{a.key_port}/", flush=True)

# ---------------- render loop (30 Hz)
dist0 = 0.0
last_p = None
while True:
    with lock:
        xpos, xquat = d.xpos.copy(), d.xquat.copy()
        p, yaw = base_pose()
        fallen = state["fallen"]
    for b, fr in bodies.items():
        fr.position, fr.wxyz = tuple(xpos[b]), tuple(xquat[b])
    if last_p is not None:
        dist0 += float(np.linalg.norm(p[:2] - last_p[:2]))
    last_p = p
    mp = p + OFF
    status.content = (f"**{'FALLEN - press reset' if fallen else 'walking (RL policy v11a)'}**  \n"
                      f"command vx {state['vx']:+.2f} m/s, wz {state['wz']:+.2f} rad/s  \n"
                      f"map position ({mp[0]:.1f}, {mp[1]:.1f}), height {mp[2]:.2f} m, walked {dist0:.1f} m")
    if state["follow"]:
        cam = p + np.array([-3.2 * math.cos(yaw), -3.2 * math.sin(yaw), 1.4])
        for c in srv.get_clients().values():
            c.camera.position, c.camera.look_at = tuple(cam), tuple(p + [0, 0, 0.2])
    time.sleep(1 / 30)
