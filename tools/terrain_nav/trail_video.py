#!/usr/bin/env python3
"""Render a recorded trail walk (trail_walk_check.py --out run.npz) in Isaac Sim RTX: full-body Huphy, the real terrain,
the scanned trees, an HDRI sky.  Isaac only draws - the walk itself was simulated in MuJoCo (v11a policy, collider
terrain) and is replayed frame by frame; PhysX is never started.

  ground   surface.npz (make_display_surface.py, map frame): the collider heights, survey ground colour
  trees    map points more than 1 m above the ground within 30 m of the path (the scan itself, as small discs)
  robot    huphy_full_shell_loop (URDF -> USD by the Isaac importer, as in tools/isaac_demo/warehouse_demo.py);
           every frame, the recorded MuJoCo leg state goes through forward kinematics of the full shell model
           (upper body held at its default pose - the walking model has no upper body)
  sky      HDRI dome light (texture = the sky, also the ambient light) + a distant light at the HDRI sun direction
  camera   follows behind and beside the robot, heading smoothed over ~2 s so the body sway does not shake the view

Real time: the run is 50 Hz; every 2nd state is a frame at 25 fps, so 1 s of video = 1 s of walking.
usage (3090, Isaac Sim python):
  python trail_video.py --run ~/pyg_demo/video/uphill_run.npz --hdri ~/pyg_demo/hdri/x_4k.hdr --out walk.mp4 [--t0 0 --t1 600]
  python trail_video.py ... --stills 5,60 --out still     (PNG test frames, then exit)
"""
import argparse
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--nav", default=os.path.expanduser("~/pyg_demo/nav"), help="trail_sim_terrain.npz, terrain_mesh.npz, map_010.npy, surface_s2.npz")
ap.add_argument("--surface", default="surface_video.npz", help="make_display_surface.py --stride 2 --open-m 1.5 (thin canopy spikes removed)")
ap.add_argument("--leaf-width", type=float, default=0.04, help="drawn size of a tree scan point [m]")
ap.add_argument("--leaf-gain", type=float, default=1.4, help="brighten the scan colours of the trees (they were scanned in shade)")
ap.add_argument("--hdri", required=True)
ap.add_argument("--dome-intensity", type=float, default=450.0)
ap.add_argument("--sun-intensity", type=float, default=1100.0)
ap.add_argument("--dome-rot", type=float, default=0.0, help="extra rotation of the sky about the vertical axis [deg]")
ap.add_argument("--assets", default=os.path.expanduser("~/pyg_demo"))
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=25.0)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--t1", type=float, default=1e9)
ap.add_argument("--stills", default="", help="comma list of run seconds: write PNGs <out>_<t>.png and exit")
ap.add_argument("--updates", type=int, default=2, help="renderer updates per frame (more = cleaner lighting, slower)")
ap.add_argument("--label", default="uphill", help="word for the overlay: uphill / downhill")
ap.add_argument("--policy-label", default="RL walking policy (v11a)", help="policy name for the overlay")
a = ap.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402

W, H = a.res
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})
import mujoco  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

URDF = f"{a.assets}/assets/huphy_full_shell_loop.urdf"
XML = f"{a.assets}/assets/xmls/huphy_full_shell_loop.xml"
USD = f"{a.assets}/usd/huphy_full_shell_loop.usd"


# ----------------------------------------------------------------------------- small math (as in warehouse_demo.py)
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mat2quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = [0.0] * 4
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def hom(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def look_at_c2w(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return hom(np.stack([r, u, -f], axis=1), eye)


def parse_urdf(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    links = [lk.get("name") for lk in root.findall("link")]
    fixed = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        fixed[j.find("child").get("link")] = (j.find("parent").get("link"), hom(rpy_to_R(*rpy), xyz))
    return links, fixed


# ----------------------------------------------------------------------------- data
R_ = np.load(a.run)
QP, DT = R_["qpos"], float(R_["dt"])
FELL = bool(len(R_["above"]) and (R_["above"][-1] < 0.45 or R_["tilt"][-1] > 60))  # trail_walk_check's fall test
OFF = R_["offset"].astype(float)
FA = int(R_["free_adr"])
run_adr = {str(n).split("/")[-1]: int(ad) for n, ad in zip(R_["joint_names"], R_["jnt_qposadr"])}  # "robot/R_knee_joint" -> "R_knee_joint"
ST = np.load(f"{a.nav}/trail_sim_terrain.npz")
HF, HC, HH, CELL = ST["hfield"], ST["center"].astype(float), ST["half"].astype(float), float(ST["cell"])


def ground_z(x, y):
    j = int(np.clip((x - (HC[0] - HH[0])) / CELL, 0, HF.shape[1] - 1))
    i = int(np.clip((y - (HC[1] - HH[1])) / CELL, 0, HF.shape[0] - 1))
    return float(HF[i, j])


viz = mujoco.MjModel.from_xml_path(XML)
vd = mujoco.MjData(viz)
vfree = int(viz.jnt_qposadr[[j for j in range(viz.njnt) if viz.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]])
jmap = [(int(viz.jnt_qposadr[j]), run_adr[viz.joint(j).name.split("/")[-1]]) for j in range(viz.njnt)
        if viz.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and viz.joint(j).name.split("/")[-1] in run_adr]
print(f"full-body model: {viz.njnt} joints, {len(jmap)} driven by the recorded walk, rest at default", flush=True)
bodies = {viz.body(b).name: b for b in range(viz.nbody)}


def link_poses(q):
    vd.qpos[:] = viz.qpos0
    vd.qpos[vfree:vfree + 7] = q[FA:FA + 7]
    for va, ra in jmap:
        vd.qpos[va] = q[ra]
    mujoco.mj_kinematics(viz, vd)
    return {nm: hom(vd.xmat[b].reshape(3, 3), vd.xpos[b]) for nm, b in bodies.items()}


# ----------------------------------------------------------------------------- stage
omni.usd.get_context().new_stage()
st = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
st.DefinePrim("/World", "Xform")


def vertex_color_material(path, rough=0.85):
    """UsdPreviewSurface whose diffuse colour is the prim's displayColor primvar."""
    mat = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(st, f"{path}/ColorReader")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


lin = lambda c: (np.asarray(c, np.float32) / 255.0) ** 2.2  # noqa: E731  sRGB -> linear for displayColor

# ground
S = np.load(f"{a.nav}/{a.surface}")
V = S["V"].astype(np.float64) - OFF
F = S["F"].astype(np.int64)
n = np.zeros_like(V)
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
for k in range(3):
    np.add.at(n, F[:, k], fn)
n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
g = UsdGeom.Mesh.Define(st, "/World/Ground")
g.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
g.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
g.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
g.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
g.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
pv = g.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
gc = lin(S["C"])
gm = gc.mean(1, keepdims=True)
gc = np.clip((gm + (gc - gm) * 1.4) * 0.8, 0, 1).astype(np.float32)  # the scan colours are pale: a little more saturation, a little darker
pv.Set(Vt.Vec3fArray.FromNumpy(gc))
UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(vertex_color_material("/World/Looks/Ground"))
print(f"ground {len(V):,} vertices, {len(F):,} triangles", flush=True)

# trees: the scan's own points more than 1 m above the ground, near the path
cl = np.load(f"{a.nav}/map_010.npy")
P = cl[:, :3].astype(np.float64) - OFF
rgb = cl[:, 3].view(np.uint32)
path = R_["path"]
near = np.zeros(len(P), bool)
for c in path[::20]:
    near |= np.abs(P[:, 0] - c[0]) + np.abs(P[:, 1] - c[1]) < 30.0
gz = HF[np.clip(((P[:, 1] - (HC[1] - HH[1])) / CELL).astype(int), 0, HF.shape[0] - 1),
        np.clip(((P[:, 0] - (HC[0] - HH[0])) / CELL).astype(int), 0, HF.shape[1] - 1)]
veg = near & (P[:, 2] - gz > 1.0) & (P[:, 2] - gz < 25.0)  # above 1 m: trunks and canopy, not low bushes / ground noise
pts = UsdGeom.Points.Define(st, "/World/Trees")
pts.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P[veg].astype(np.float32)))
pts.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(int(veg.sum()), a.leaf_width, np.float32)))
pts.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
cols = np.stack([(rgb[veg] >> s) & 255 for s in (16, 8, 0)], 1)
pts.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.clip(lin(cols) * a.leaf_gain, 0, 1).astype(np.float32)))
UsdShade.MaterialBindingAPI.Apply(pts.GetPrim()).Bind(vertex_color_material("/World/Looks/Trees", 0.95))
# the canopy points stand in for leaves; as shadow casters 1.3 M discs turn the trail almost black, so they cast none
pts.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
print(f"trees: {int(veg.sum()):,} scan points", flush=True)

# sky + sun
E = cv2.imread(a.hdri, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
lum = cv2.GaussianBlur(E.mean(-1), (0, 0), 5)
r_, c_ = np.unravel_index(np.argmax(lum), lum.shape)
sun_az = 360.0 * c_ / E.shape[1]
sun_el = 90.0 - 180.0 * r_ / E.shape[0]
dome = UsdLux.DomeLight.Define(st, "/World/Sky")
dome.CreateTextureFileAttr().Set(a.hdri)
dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
dome.CreateIntensityAttr().Set(a.dome_intensity)
UsdGeom.Xformable(dome).AddRotateZOp().Set(a.dome_rot)
sun = UsdLux.DistantLight.Define(st, "/World/Sun")
sun.CreateIntensityAttr().Set(a.sun_intensity)
sun.CreateAngleAttr().Set(0.6)
sx = UsdGeom.Xformable(sun)
# a distant light shines along its -Z; point it from the sun direction (azimuth measured like the latlong texture)
az_world = math.radians(sun_az + a.dome_rot)
d_sun = np.array([math.cos(math.radians(sun_el)) * math.cos(az_world), math.cos(math.radians(sun_el)) * math.sin(az_world), math.sin(math.radians(sun_el))])
sx.AddTransformOp().Set(Gf.Matrix4d(look_at_c2w(d_sun * 10, (0, 0, 0)).T.tolist()))
print(f"HDRI {os.path.basename(a.hdri)}: sun azimuth {sun_az:.0f} deg, elevation {sun_el:.0f} deg", flush=True)

# robot
robot = st.DefinePrim("/World/Huphy", "Xform")
robot.GetReferences().AddReference(USD)
app.update()
for p in Usd.PrimRange(robot):
    for s_ in list(p.GetAppliedSchemas()):
        if s_.startswith(("Physics", "Physx", "PhysX")):
            p.RemoveAppliedSchema(s_)
    if (p.GetName() == "joints" and p.GetTypeName() == "Scope") or p.GetName() == "collisions":
        p.SetActive(False)
links, fixed = parse_urdf(URDF)
prims = {lk: st.GetPrimAtPath(f"/World/Huphy/{lk}") for lk in links}
prims = {k: v for k, v in prims.items() if v.IsValid()}
ops = {lk: (p.GetAttribute("xformOp:translate"), p.GetAttribute("xformOp:orient")) for lk, p in prims.items()}
print(f"robot links {len(links)}, drawn {len(prims)}", flush=True)


def set_op(o, T):
    tq, oq = o
    q = mat2quat(T[:3, :3])
    t = T[:3, 3]
    ct, co = tq.Get(), oq.Get()
    tq.Set(type(ct)(*map(float, t)) if ct is not None else Gf.Vec3d(*map(float, t)))
    oq.Set(type(co)(*map(float, q)) if co is not None else Gf.Quatd(*map(float, q)))


def pose_robot(q):
    poses = link_poses(q)
    world = {}

    def get(lk):
        if lk not in world:
            if lk in poses:
                world[lk] = poses[lk]
            elif lk in fixed:
                world[lk] = get(fixed[lk][0]) @ fixed[lk][1]
            else:
                world[lk] = poses.get("base_link", np.eye(4))
        return world[lk]
    with Sdf.ChangeBlock():
        for lk in prims:
            set_op(ops[lk], get(lk))


# camera
cam = UsdGeom.Camera.Define(st, "/World/Cam")
hap = 20.955
cam.GetHorizontalApertureAttr().Set(hap)
cam.GetVerticalApertureAttr().Set(hap * H / W)
cam.GetFocalLengthAttr().Set(hap / (2 * math.tan(math.radians(55.0) / 2)))
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 2000.0))
cam_xf = UsdGeom.Xformable(cam).AddTransformOp()
app.update()
rp = rep.create.render_product("/World/Cam", (W, H))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])
omni.timeline.get_timeline_interface().play()
for _ in range(30):
    app.update()

# ----------------------------------------------------------------------------- camera path: smoothed heading
xyz = QP[:, FA:FA + 3]
step = max(1, round(1.0 / (a.fps * DT)))  # 50 Hz states -> 25 fps frames = every 2nd state
k0, k1 = int(a.t0 / DT), min(int(a.t1 / DT), len(QP) - 1)
heading = np.zeros(len(QP))
hv = None
for k in range(len(QP)):  # direction of travel over the last ~1 s, low-pass ~2 s
    d = xyz[k, :2] - xyz[max(k - 50, 0), :2]
    v = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.05 else (hv if hv is not None else np.array([1.0, 0.0]))
    hv = v if hv is None else hv * 0.985 + v * 0.015
    hv /= np.linalg.norm(hv)
    heading[k] = math.atan2(hv[1], hv[0])
cam_s = None


def camera_at(k):
    global cam_s
    p = xyz[k]
    h = heading[k]
    fwd, left = np.array([math.cos(h), math.sin(h), 0.0]), np.array([-math.sin(h), math.cos(h), 0.0])
    eye = p - 3.2 * fwd + 1.2 * left
    eye[2] = max(p[2] + 0.45, ground_z(eye[0], eye[1]) + 1.0)
    cam_s = eye if cam_s is None else cam_s * 0.9 + eye * 0.1
    return look_at_c2w(cam_s, p + 2.0 * fwd + np.array([0, 0, 0.05]))  # aim ahead of the robot: trail, trees and sky in view


FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay(img, k):
    t = k * DT
    climbed = xyz[k, 2] - xyz[0, 2]
    s_along = float(R_["s"][min(k, len(R_["s"]) - 1)])
    lines = [f"Huphy  |  {a.policy_label}, MuJoCo physics on the scanned trail  |  real time 1x",
             f"t {t:5.1f} s   {a.label} trail {s_along:5.1f} / 182 m   height {climbed:+.1f} m   command {R_['cmd'][min(k, len(R_['cmd']) - 1)][0]:.2f} m/s"]
    if FELL and k >= len(QP) - 1 - 2 * step:  # the recording stops at the fall: say so on its last frames
        cv2.rectangle(img, (W // 2 - 260, H // 2 - 70), (W // 2 + 260, H // 2 + 70), (20, 20, 160), -1)
        cv2.putText(img, "FELL", (W // 2 - 110, H // 2 + 35), FONT, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    band = img[0:92].astype(np.float32) * 0.45  # dark band so the text reads on any sky or ground
    img[0:92] = band.astype(np.uint8)
    for i, s_ in enumerate(lines):
        cv2.putText(img, s_, (24, 36 + i * 36), FONT, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return img


def render(k):
    pose_robot(QP[k])
    cam_xf.Set(Gf.Matrix4d(camera_at(k).T.tolist()))
    for _ in range(a.updates):
        app.update()
    im = np.asarray(ann.get_data())
    return cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2BGR)


if a.stills:
    for ts in [float(v) for v in a.stills.split(",")]:
        k = min(int(ts / DT), len(QP) - 1)
        for kk in range(max(k - 100, 0), k, step):  # settle the smoothed camera on the way there
            camera_at(kk)
        for _ in range(8):
            render(k)
        cv2.imwrite(f"{a.out}_{int(ts)}.png", overlay(render(k), k))
        print("still", f"{a.out}_{int(ts)}.png", flush=True)
else:
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for kk in range(max(k0 - 100, 0), k0, step):
        camera_at(kk)
    for _ in range(8):
        render(k0)
    t0 = time.time()
    frames = range(k0, k1, step)
    for i, k in enumerate(frames):
        wr.write(overlay(render(k), k))
        if i % 250 == 0:
            el = time.time() - t0
            print(f"frame {i}/{len(frames)}  run t {k * DT:.0f} s  {el / max(i, 1):.2f} s/frame", flush=True)
    wr.release()
    print("wrote", a.out, len(frames), "frames", flush=True)
app.close()
