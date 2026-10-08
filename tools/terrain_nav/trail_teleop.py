#!/usr/bin/env python3
"""Drive Huphy (LegOnly-AB + v11a walking policy) yourself on the real mountain-trail terrain, in one web page.

  http://<host>:8095/   (the same page is also served on :8096)
  keyboard on that page: W/Up faster, S/Down slower, A/D turn, Space stop, R get up here

Physics: MuJoCo with the 10 cm heightfield + tree trunks baked into the model (bake.py PYG_BAKE_TERRAIN), 200 Hz physics,
50 Hz policy, real time.  The policy only gets a velocity command (vx, wz) - exactly what a planner would send; legs,
balance and footholds on the uneven ground are the trained policy's own.

The page draws the terrain as a surface (no point clouds): the whole map from the collider heights every 30 cm, and the
exact 10 cm collider around the robot; tree trunks are the collider cylinders.  The contact panel shows, live, which
world geometry each foot touches and how much force the heightfield / trees / old flat floor carry.

LiDAR: a generic 32-beam spinning LiDAR on the pelvis casts rays into the same terrain (heightfield + trunks; the robot is
not in the ray scene) at ~10 Hz; returns are filtered (range 0.3-15 m, artificial walls, 10 cm voxel, grid-minimum ground
segmentation, 0.30 m per 0.5 m column).  Shown as this scan's points and as a terrain map built from all scans (10 cm elevation grid, a surface).
run (3090, mjlab venv):  python trail_teleop.py --cache ~/pyg_demo/cache_trail --nav ~/pyg_demo/nav
"""
import argparse
import gzip
import http.server
import json
import math
import os
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pygviewer"))
import pygviewer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--nav", required=True, help="dir with trail_sim_terrain.npz, terrain.npz, terrain_mesh.npz, map_010.npy")
ap.add_argument("--policy", default="v11a")
ap.add_argument("--port", type=int, default=8095)
ap.add_argument("--port2", type=int, default=8096, help="second port serving the same page (0 = off)")
ap.add_argument("--three", default=os.path.expanduser("~/pyg_demo/nav/web/three"), help="local three.js (build/, examples/jsm/)")
ap.add_argument("--lidar-hz", type=float, default=10.0)
ap.add_argument("--mesh-stride", type=int, default=3, help="whole-map surface: every n-th 10 cm collider vertex")
a = ap.parse_args()
pygviewer.CACHE_DIR = a.cache
import mujoco  # noqa: E402
from scipy import ndimage  # noqa: E402
from pygviewer.contract import load_contract  # noqa: E402
from pygviewer.sim_core import SimCore  # noqa: E402

WEB = os.path.join(HERE, "web")
ST = np.load(f"{a.nav}/trail_sim_terrain.npz")
OFF = ST["offset"].astype(float)  # map = sim + OFF
HF, HC, HH = ST["hfield"].astype(np.float32), ST["center"].astype(float), ST["half"].astype(float)
CELL = float(ST["cell"])
NR, NC = HF.shape
X0, Y0 = HC[0] - HH[0], HC[1] - HH[1]
# the MuJoCo heightfield's vertex lattice (what the feet actually touch): nrow x ncol vertices spanning the full size
VX = (X0 + np.arange(NC) * (2 * HH[0] / (NC - 1))).astype(np.float32)
VY = (Y0 + np.arange(NR) * (2 * HH[1] / (NR - 1))).astype(np.float32)
TREES = ST["trees"].astype(np.float64)  # x, y, bottom, radius, top
PATH = (np.load(f"{a.nav}/terrain.npz")["path"] - OFF).astype(np.float32)
YAW0 = math.atan2(PATH[8, 1] - PATH[0, 1], PATH[8, 0] - PATH[0, 0])  # first metre of the planned path
# cells more than 1 m from any surveyed ground are the artificial 2 m walls of prep_terrain_mesh.py - not real terrain
WALL = ndimage.distance_transform_edt(~np.load(f"{a.nav}/terrain_mesh.npz")["hf_seen"]) * CELL > 1.0


def ground_z(x, y):
    j = int(np.clip((x - X0) / CELL, 0, NC - 1))
    i = int(np.clip((y - Y0) / CELL, 0, NR - 1))
    return float(HF[i, j])


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


# ---------------- surface colours: the survey's own colour of the ground, per 10 cm cell
t0 = time.time()
cl = np.load(f"{a.nav}/map_010.npy")
P = cl[:, :3].astype(np.float64) - OFF
rgb = cl[:, 3].view(np.uint32)
jj = ((P[:, 0] - X0) / CELL).astype(np.int64)
ii = ((P[:, 1] - Y0) / CELL).astype(np.int64)
ok = (jj >= 0) & (jj < NC) & (ii >= 0) & (ii < NR)
ok[ok] &= np.abs(P[ok, 2] - HF[ii[ok], jj[ok]]) < 0.3  # points on the ground, not leaves above it
lin = ii[ok] * NC + jj[ok]
cnt = np.bincount(lin, minlength=NR * NC)
CG = np.stack([np.bincount(lin, ((rgb[ok] >> sh) & 255).astype(np.float64), minlength=NR * NC) for sh in (16, 8, 0)], 1)
CG = (CG / np.maximum(cnt, 1)[:, None]).reshape(NR, NC, 3)
fi = ndimage.distance_transform_edt((cnt == 0).reshape(NR, NC), return_distances=False, return_indices=True)
CG = CG[fi[0], fi[1]]
CG[WALL] = (92, 94, 100)  # unsurveyed: dark grey
CG = CG.astype(np.uint8)
del cl, P, rgb, jj, ii, ok, lin, cnt, fi

# ---------------- whole-map surface from the collider heights (every n-th vertex), plus a 3 m rim of the walls
s = a.mesh_stride
show = ndimage.distance_transform_edt(WALL) * CELL < 3.0
ri, ci = np.arange(0, NR, s), np.arange(0, NC, s)
sub = show[np.ix_(ri, ci)]
# display only: the artificial 2 m walls (no survey data) are drawn at the filled ground height in dark grey, not as
# 2 m spikes; physics and the exact 10 cm patch keep them
HF_SHOW = np.where(WALL, HF - 2.0, HF)
V = np.stack(np.broadcast_arrays(VX[ci][None, :], VY[ri][:, None], HF_SHOW[np.ix_(ri, ci)]), -1).reshape(-1, 3)
idx = np.arange(sub.size).reshape(sub.shape)
q = sub[:-1, :-1] & sub[1:, :-1] & sub[:-1, 1:] & sub[1:, 1:]
A_, B_, C_, D_ = idx[:-1, :-1][q], idx[:-1, 1:][q], idx[1:, 1:][q], idx[1:, :-1][q]
F = np.concatenate([np.stack([A_, B_, C_], 1), np.stack([A_, C_, D_], 1)])
used = np.unique(F)
remap = np.full(sub.size, -1, np.int64)
remap[used] = np.arange(len(used))
TV, TC, TF = V[used].astype(np.float32), CG[np.ix_(ri, ci)].reshape(-1, 3)[used], remap[F].astype(np.uint32)
TERRAIN_GZ = gzip.compress(TV.tobytes() + pad4(TC.tobytes()) + TF.tobytes(), 5)
print(f"surface {len(TV):,} vertices, {len(TF):,} triangles, {len(TERRAIN_GZ) / 1e6:.1f} MB gzip; prep {time.time() - t0:.0f} s", flush=True)

# ---------------- physics
pc = json.load(open(f"{a.cache}/{a.policy}.policy_contract.json"))
core = SimCore(load_contract(a.cache, pc["variant"]), realtime=False)
core.load_policy(onnx=f"{a.cache}/{a.policy}.onnx", policy_contract=pc)
core.set_base(mode="free", ground=True)
m, d = core.m, core.d
assert m.nhfield == 1, "model has no terrain heightfield - bake with PYG_BAKE_TERRAIN (docs/161 §5-2)"
lock = threading.Lock()
state = dict(vx=0.0, wz=0.0, fallen=False, reset=None, walked=0.0, contact={}, feet={"L": [], "R": []})
KIND = {}
for g in range(m.ngeom):
    n = m.geom(g).name
    KIND[g] = "heightfield" if n == "trail_ground" else "tree" if n.startswith("trail_tree") else \
        "flat floor" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE else None
SIDE = {g: ("L" if "/L_" in m.body(int(m.geom_bodyid[g])).name else "R" if "/R_" in m.body(int(m.geom_bodyid[g])).name else None)
        for g in range(m.ngeom)}
WEIGHT = float(m.body_subtreemass[m.body("robot/base_link").id] * 9.81)
_f6 = np.zeros(6)


def contacts():
    force = {"heightfield": 0.0, "tree": 0.0, "flat floor": 0.0}
    feet = {"L": set(), "R": set()}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        k = KIND.get(g1) or KIND.get(g2)
        if not k:
            continue
        mujoco.mj_contactForce(m, d, i, _f6)
        force[k] += abs(_f6[0])
        side = SIDE.get(g2 if KIND.get(g1) else g1)
        if side:
            feet[side].add(k)
    return force, feet


def do_reset(x=0.0, y=0.0, yaw=YAW0):
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
    n, last = 0, None
    ema = {"heightfield": 0.0, "tree": 0.0, "flat floor": 0.0}
    while True:
        with lock:
            r = state["reset"]
            state["reset"] = None
            if r is not None:
                do_reset(*r)
                last = None
            core.cmd = np.array([state["vx"], 0.0, state["wz"]])
            core.step_n(core.decimation)
            p, _ = base_pose()
            if p[2] - ground_z(p[0], p[1]) < 0.45:
                state["fallen"] = True
            force, feet = contacts()
            for k in ema:
                ema[k] = 0.9 * ema[k] + 0.1 * force[k]
            state["contact"] = dict(ema)
            state["feet"] = {k: sorted(v) for k, v in feet.items()}
            n += 1
            if n % 25 == 0:  # walked distance sampled every 0.5 s (not every step: the body sways side to side)
                if last is not None:
                    state["walked"] += float(np.linalg.norm(p[:2] - last))
                last = p[:2].copy()
        nxt += core.decimation * core.dt
        dt = nxt - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        else:
            nxt = time.perf_counter()  # behind: do not try to catch up


# ---------------- LiDAR: rays into the same terrain the physics uses (heightfield + tree trunks), robot excluded.
# mj_multiRay against the 1232 x 2022 heightfield took ~22 s per scan, so the rays are cast here in numpy instead:
# march each ray over the heightfield in 5 cm steps, refine the crossing linearly, and intersect the trunks exactly.
LIDAR = dict(beams=32, el=(-35.0, 15.0), az_step=1.0, rmin=0.3, rmax=15.0, mount=(0.0, 0.0, 0.35), noise=0.02,
             voxel=0.10, ground_cell=0.5, ground_tol=0.30, march=0.05)
_el = np.deg2rad(np.linspace(*LIDAR["el"], LIDAR["beams"]))
_az = np.deg2rad(np.arange(0.0, 360.0, LIDAR["az_step"]))
_E, _A = np.meshgrid(_el, _az, indexing="ij")
DIRS = np.stack([np.cos(_E) * np.cos(_A), np.cos(_E) * np.sin(_A), np.sin(_E)], -1).reshape(-1, 3)
BEAM_SUB = ((np.arange(LIDAR["beams"])[:, None] % 8 == 3) & (np.arange(len(_az))[None, :] % 12 == 0)).reshape(-1)  # 4 rings x 30
TS = np.arange(LIDAR["rmin"], LIDAR["rmax"] + 1e-6, LIDAR["march"], dtype=np.float32)
rng = np.random.default_rng(0)
lidar = dict(n=len(DIRS), ret=0, wall=0, kept=0, ground=0, obst=0, ms=0.0, hz=0.0, seq=0, cells=0)
lidar_bin = [b""]
# terrain map built from scans, on the heightfield's 10 cm cells: mean height of ground returns + 'obstacle seen here'
E_sum = np.zeros((NR, NC), np.float32)
E_cnt = np.zeros((NR, NC), np.int32)
E_obst = np.zeros((NR, NC), bool)


def ray_hits(o, V):
    """Distance along each unit ray V from o to the first heightfield / trunk hit (inf: none within rmax)."""
    Pr = o[None, None, :].astype(np.float32) + V[:, None, :].astype(np.float32) * TS[None, :, None]  # rays x steps x 3
    j = np.clip(((Pr[..., 0] - X0) / CELL).astype(np.int32), 0, NC - 1)
    i = np.clip(((Pr[..., 1] - Y0) / CELL).astype(np.int32), 0, NR - 1)
    dz = Pr[..., 2] - HF[i, j]  # >0 above ground
    below = dz <= 0
    k = np.argmax(below, axis=1)
    rows = np.arange(len(V))
    hit = below[rows, k]
    t = np.full(len(V), np.inf)
    kk = np.maximum(k - 1, 0)
    a0, a1 = dz[rows, kk], dz[rows, k]
    frac = np.where(k > 0, a0 / np.maximum(a0 - a1, 1e-6), 0.0)
    t[hit] = (TS[kk] + frac * LIDAR["march"])[hit]
    on_wall = np.zeros(len(V), bool)
    on_wall[hit] = WALL[i[rows, k], j[rows, k]][hit]
    near = np.hypot(TREES[:, 0] - o[0], TREES[:, 1] - o[1]) < LIDAR["rmax"] + TREES[:, 3]
    A2 = V[:, 0] ** 2 + V[:, 1] ** 2
    for cx, cy, cz, r, ctop in TREES[near]:
        dx, dy = o[0] - cx, o[1] - cy
        B2 = 2 * (dx * V[:, 0] + dy * V[:, 1])
        disc = B2 * B2 - 4 * A2 * (dx * dx + dy * dy - r * r)
        okc = (disc > 0) & (A2 > 1e-9)
        tc = np.where(okc, (-B2 - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(A2, 1e-9)), np.inf)
        z = o[2] + V[:, 2] * tc
        closer = okc & (tc > LIDAR["rmin"]) & (z > cz) & (z < ctop) & (tc < t)
        t = np.where(closer, tc, t)
        on_wall &= ~closer
    return t, on_wall


def voxel_keys(Pk, v):
    k = np.floor(Pk / v).astype(np.int64) + 2 ** 20
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def lidar_scan(origin, R):
    t0 = time.perf_counter()
    vec = DIRS @ R.T
    dist, on_wall = ray_hits(origin, vec)
    ok = np.isfinite(dist) & (dist > LIDAR["rmin"]) & (dist < LIDAR["rmax"])  # 1. range
    n_wall = int((ok & on_wall).sum())
    ok &= ~on_wall  # 2. returns on the artificial walls (no survey data there) are not real observations
    r = dist[ok] + rng.normal(0.0, LIDAR["noise"], ok.sum())  # 2 cm range noise, as a real sensor
    Pp = origin + vec[ok] * r[:, None]
    sb = ok & BEAM_SUB
    beams = np.concatenate([np.broadcast_to(origin, (int(sb.sum()), 3)), origin + vec[sb] * dist[sb][:, None]], 1)
    _, inv = np.unique(voxel_keys(Pp, LIDAR["voxel"]), return_inverse=True)  # 3. 10 cm voxel: one point (mean) each
    c = np.bincount(inv)
    Pv = np.stack([np.bincount(inv, Pp[:, i]) for i in range(3)], 1) / c[:, None]
    # 4. ground segmentation from the scan itself: per 0.5 m column, within 0.30 m of the lowest point = ground
    #    (0.30 m over 0.5 m = slopes up to about 31 deg count as ground; 0.15 m marked every hillside as obstacle)
    _, ginv = np.unique(voxel_keys(np.c_[Pv[:, :2], np.zeros(len(Pv))], LIDAR["ground_cell"]), return_inverse=True)
    zmin = np.full(ginv.max() + 1, np.inf)
    np.minimum.at(zmin, ginv, Pv[:, 2])
    ground = Pv[:, 2] - zmin[ginv] < LIDAR["ground_tol"]
    lidar.update(ret=int(ok.sum()), wall=n_wall, kept=len(Pv), ground=int(ground.sum()), obst=int((~ground).sum()),
                 ms=(time.perf_counter() - t0) * 1000)
    return Pv, ground, beams


def lidar_loop():
    t_prev = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        with lock:
            fa = core.free_adr
            p = d.qpos[fa:fa + 3].copy()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, d.qpos[fa + 3:fa + 7].copy())
        R = R.reshape(3, 3)
        Pv, ground, beams = lidar_scan(p + R @ np.array(LIDAR["mount"]), R)
        i = np.clip(((Pv[:, 1] - Y0) / CELL).astype(np.int64), 0, NR - 1)
        j = np.clip(((Pv[:, 0] - X0) / CELL).astype(np.int64), 0, NC - 1)
        np.add.at(E_sum, (i[ground], j[ground]), Pv[ground, 2].astype(np.float32))
        np.add.at(E_cnt, (i[ground], j[ground]), 1)
        E_obst[i[~ground], j[~ground]] = True
        head = np.array([len(Pv), len(beams)], np.uint32).tobytes()
        lidar_bin[0] = head + Pv.astype(np.float32).tobytes() + beams.astype(np.float32).tobytes() + ground.astype(np.uint8).tobytes()
        lidar["hz"] = 1.0 / max(time.perf_counter() - t_prev, 1e-3)
        t_prev = time.perf_counter()
        lidar["seq"] += 1
        if lidar["seq"] % 10 == 0:
            lidar["cells"] = int(((E_cnt > 0) | E_obst).sum())
        time.sleep(max(0.0, 1.0 / a.lidar_hz - (time.perf_counter() - t0)))


def window(x, y, half):
    j0, j1 = max(int((x - half - X0) / CELL), 0), min(int((x + half - X0) / CELL), NC - 1)
    i0, i1 = max(int((y - half - Y0) / CELL), 0), min(int((y + half - Y0) / CELL), NR - 1)
    return i0, i1, j0, j1


def patch_bin(x, y, half=15.0):
    """Exact collider around (x, y): heightfield vertices at full 10 cm resolution + survey colour."""
    i0, i1, j0, j1 = window(x, y, half)
    head = np.array([VX[j0], VX[1] - VX[0], VY[i0], VY[1] - VY[0], i1 - i0 + 1, j1 - j0 + 1], np.float32)
    return head.tobytes() + HF[i0:i1 + 1, j0:j1 + 1].tobytes() + CG[i0:i1 + 1, j0:j1 + 1].tobytes()


def elev_bin(x, y, half=15.0):
    """LiDAR terrain map around (x, y): ground height per 10 cm cell (NaN = no ground return yet) + obstacle flag."""
    i0, i1, j0, j1 = window(x, y, half)
    c = E_cnt[i0:i1 + 1, j0:j1 + 1]
    ob = E_obst[i0:i1 + 1, j0:j1 + 1]
    # surface height from ground returns only; cells where something stands above the ground are flagged (orange),
    # not raised to the obstacle top (that drew trunk tops as tall sheets)
    z = np.where(c > 0, E_sum[i0:i1 + 1, j0:j1 + 1] / np.maximum(c, 1), np.nan).astype(np.float32)
    head = np.array([X0 + (j0 + 0.5) * CELL, CELL, Y0 + (i0 + 0.5) * CELL, CELL, i1 - i0 + 1, j1 - j0 + 1], np.float32)
    return head.tobytes() + z.astype(np.float32).tobytes() + ob.astype(np.uint8).tobytes()


# ---------------- robot meshes (visual geoms, group 2) for the page
bodies = sorted({int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_bodyid[g] > 1 and m.geom_group[g] == 2})
robot_geoms, vbuf, fbuf, nv_, nf_ = [], [], [], 0, 0
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if b not in bodies or m.geom_group[g] != 2:
        continue
    e = dict(b=bodies.index(b), pos=m.geom_pos[g].tolist(), quat=m.geom_quat[g].tolist(), rgba=m.geom_rgba[g].tolist())
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        e.update(type="mesh", vo=nv_, vn=len(v), fo=nf_, fn=len(f))
        vbuf.append(v.astype(np.float32))
        fbuf.append(f.astype(np.uint32))
        nv_, nf_ = nv_ + len(v), nf_ + len(f)
    elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        e.update(type="cylinder", size=m.geom_size[g][:2].tolist())
    else:
        continue
    robot_geoms.append(e)
ROBOT_GZ = gzip.compress(np.concatenate(vbuf).tobytes() + np.concatenate(fbuf).tobytes(), 5)
META = json.dumps(dict(
    terrain=dict(nv=len(TV), nf=len(TF), stride_m=round(s * CELL, 2)),
    robot=dict(nbody=len(bodies), nv=nv_, nf=nf_, geoms=robot_geoms),
    trees=TREES.round(3).tolist(),
    path=np.c_[PATH[:, :2], [ground_z(x, y) + 0.08 for x, y in PATH[:, :2]]][::2].round(3).tolist(),
    lidar=dict(beams=LIDAR["beams"], rays=len(DIRS), rmin=LIDAR["rmin"], rmax=LIDAR["rmax"]),
    weight=round(WEIGHT), policy=a.policy)).encode()


def set_cmd(vx=None, wz=None):
    with lock:
        if vx is not None:
            state["vx"] = float(np.clip(round(vx, 2), -0.4, 1.2))
        if wz is not None:
            state["wz"] = float(np.clip(round(wz, 2), -0.6, 0.6))


def state_json():
    with lock:
        p, yaw = base_pose()
        xb = np.c_[d.xpos[bodies], d.xquat[bodies]].round(4).ravel().tolist()
        st = dict(t=round(float(d.time), 2), vx=state["vx"], wz=state["wz"], fallen=state["fallen"], walked=round(state["walked"], 1),
                  p=p.round(3).tolist(), yaw=round(yaw, 3), above=round(p[2] - ground_z(p[0], p[1]), 2),
                  map=(p + OFF).round(1).tolist(), bodies=xb, contact={k: round(v) for k, v in state["contact"].items()},
                  feet=state["feet"])
    st["lidar"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in lidar.items()}
    return json.dumps(st).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, body, ctype="application/octet-stream", gz=None, cache=False):
        """gz: the body already gzip-compressed (sent as is to browsers that accept gzip)."""
        accept = "gzip" in self.headers.get("Accept-Encoding", "")
        if gz is not None:
            body, zipped = (gz, True) if accept else (gzip.decompress(gz), False)
        elif accept and len(body) > 4000:
            body, zipped = gzip.compress(body, 1), True
        else:
            zipped = False
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        if zipped:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self.send(open(os.path.join(WEB, "index.html"), "rb").read(), "text/html; charset=utf-8")
            elif u.path == "/app.js":
                self.send(open(os.path.join(WEB, "app.js"), "rb").read(), "text/javascript")
            elif u.path.startswith("/three/"):
                root = os.path.realpath(a.three)
                fn = os.path.realpath(os.path.join(root, u.path[len("/three/"):]))
                if not fn.startswith(root + os.sep) or not os.path.isfile(fn):
                    self.send_error(404)
                    return
                self.send(open(fn, "rb").read(), "text/javascript", cache=True)
            elif u.path == "/meta.json":
                self.send(META, "application/json")
            elif u.path == "/terrain.bin":
                self.send(b"", gz=TERRAIN_GZ)
            elif u.path == "/robot.bin":
                self.send(b"", gz=ROBOT_GZ)
            elif u.path == "/state":
                self.send(state_json(), "application/json")
            elif u.path == "/lidar.bin":
                self.send(lidar_bin[0])
            elif u.path == "/patch.bin":
                self.send(patch_bin(float(q["x"]), float(q["y"])))
            elif u.path == "/elev.bin":
                self.send(elev_bin(float(q["x"]), float(q["y"])))
            elif u.path == "/cmd":
                if "stop" in q:
                    set_cmd(0.0, 0.0)
                if "reset" in q:
                    with lock:
                        p, yaw = base_pose()
                        state["reset"] = (float(p[0]), float(p[1]), yaw)
                    set_cmd(0.0, 0.0)
                if "start" in q:
                    with lock:
                        state["reset"] = (0.0, 0.0, YAW0)
                        state["walked"] = 0.0
                    set_cmd(0.0, 0.0)
                if "clear" in q:
                    E_cnt[:] = 0
                    E_sum[:] = 0
                    E_obst[:] = False
                set_cmd(vx=float(q["vx"]) if "vx" in q else state["vx"] + float(q.get("dvx", 0.0)),
                        wz=float(q["wz"]) if "wz" in q else state["wz"] + float(q.get("dwz", 0.0)))
                self.send(json.dumps(dict(vx=state["vx"], wz=state["wz"])).encode(), "application/json")
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass


threading.Thread(target=physics, daemon=True).start()
threading.Thread(target=lidar_loop, daemon=True).start()
for port in [a.port] + ([a.port2] if a.port2 else []):
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"teleop page: http://10.8.0.14:{a.port}/" + (f"  (also :{a.port2})" if a.port2 else ""), flush=True)
while True:
    time.sleep(3600)
