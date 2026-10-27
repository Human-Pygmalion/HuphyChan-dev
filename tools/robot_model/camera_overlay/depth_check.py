#!/usr/bin/env python3
"""Check that the D455 camera of the MJCF produces the depth image geometry says it must.

Scene: the LegOnly AB D455 MJCF standing with straight legs (base_link at z = 0.903 m), a floor plane at z = 0, a wall
plane ahead at x = WALL_X and a wall on the robot's left at y = SIDE_Y (the side wall makes the image depend on the camera's
left-right position, which floor and front wall alone cannot see). All three are planes, so the correct depth of every pixel is known in closed form:
    ray through pixel (u, v):  d_cam = ((u + .5 - cx) / fx, -(v + .5 - cy) / fy, -1)   (MuJoCo camera: +x right, +y up, looks -z)
    world:                     P = C + t * R d_cam,   depth = t  (distance along the optical axis)
The camera pose C, R is NOT taken from the MJCF: it is rebuilt from the URDF chain (base_link -> d455_mount -> d455_link ->
d455_depth_optical_frame) in the generated URDF. So one comparison covers URDF pose, MJCF pose, intrinsics and renderer.

Three depth sources are compared with the closed form:
  R) MuJoCo ray casting (mj_multiRay, no OpenGL): checks pose, axes and the pinhole mapping used by the closed form
  A) MuJoCo's OpenGL renderer (mujoco.Renderer, --gl; needs a GL/EGL driver, runs on the GPU VM): checks sensorsize/focal handling
  B) mjlab's CameraSensor (mujoco_warp) in a mjlab Scene - the path a training environment would use: same check, other renderer
Only pixels that hit the floor or the wall count; pixels that hit the robot's own collision capsules are reported separately.

usage (mjlab venv):  depth_check.py [--gl] [--no-mjlab] [--device cpu|cuda:0] [--out PNG] [--json JSON]   (--gl needs MUJOCO_GL=egl)
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
XMLS = f'{REPO}/mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls'
URDF = f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_d455_loop.urdf'
MJCF = f'{XMLS}/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_d455_loop.xml'
BASE_Z, WALL_X, SIDE_Y = 0.903, 0.45, 0.35     # the wall is close enough to fill the top of the image, so the floor/wall edge is in view
W, H, HFOV = 848, 480, 87.0


def urdf_camera_pose():
    """Camera centre and rotation (columns = MuJoCo camera x right, y up, z backward) in the base_link frame, from the URDF."""
    r = ET.parse(URDF).getroot()
    J = {j.find('child').get('link'): (j.find('parent').get('link'), np.array(j.find('origin').get('xyz').split(), float),
                                       np.array(j.find('origin').get('rpy').split(), float)) for j in r.findall('joint')}
    chain, link = [], 'd455_depth_optical_frame'
    while link != 'base_link':
        p, x, y = J[link]
        chain.append((x, y))
        link = p
    R, t = np.eye(3), np.zeros(3)
    for x, y in reversed(chain):
        t = t + R @ x
        R = R @ Rot.from_euler('xyz', y).as_matrix()
    # R is the ROS optical frame: x right, y down, z forward. MuJoCo camera: x right, y up, z backward.
    return t, R @ np.diag([1.0, -1.0, -1.0])


def closed_form(t_base, R_cam):
    fx = (W / 2) / math.tan(math.radians(HFOV / 2))
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - W / 2) / fx, -(v - H / 2) / fx, -np.ones_like(u)], axis=-1)
    d_w = d_cam @ R_cam.T
    C = np.array([0.0, 0.0, BASE_Z]) + t_base
    with np.errstate(divide='ignore', invalid='ignore'):
        t_floor = np.where(d_w[..., 2] < 0, -C[2] / d_w[..., 2], np.inf)
        t_wall = np.where(d_w[..., 0] > 0, (WALL_X - C[0]) / d_w[..., 0], np.inf)
        t_side = np.where(d_w[..., 1] > 0, (SIDE_Y - C[1]) / d_w[..., 1], np.inf)
    t = np.minimum(np.minimum(t_floor, t_wall), t_side)
    return t, t_wall < t_floor, C


def scene_spec():
    spec = mujoco.MjSpec.from_file(MJCF)
    spec.body('base_link').pos = [0.0, 0.0, BASE_Z]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720     # MuJoCo's default offscreen buffer is 640 x 480
    wb = spec.worldbody
    wb.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], pos=[0, 0, 0], rgba=[0.6, 0.6, 0.6, 1])
    wb.add_geom(name='wall', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 3, 3], pos=[WALL_X + 0.02, 0, 3], rgba=[0.8, 0.3, 0.3, 1])
    wb.add_geom(name='left_wall', type=mujoco.mjtGeom.mjGEOM_BOX, size=[3, 0.02, 3], pos=[0, SIDE_Y + 0.02, 3], rgba=[0.3, 0.3, 0.8, 1])
    return spec


def stats(name, rendered, expected, valid):
    e = np.abs(rendered - expected)[valid] * 1000.0     # mm
    return dict(renderer=name, pixels_compared=int(valid.sum()), max_abs_mm=float(e.max()), mean_abs_mm=float(e.mean()),
                p99_abs_mm=float(np.percentile(e, 99)), rms_mm=float(np.sqrt((e ** 2).mean())))


def build(spec):
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def env_ids(model):
    return {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('floor', 'wall', 'left_wall')}


def run_raycast(model, data):
    fx = (W / 2) / math.tan(math.radians(HFOV / 2))
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - W / 2) / fx, -(v - H / 2) / fx, -np.ones_like(u)], axis=-1).reshape(-1, 3)
    R = data.cam_xmat[0].reshape(3, 3)
    d_w = d_cam @ R.T
    norm = np.linalg.norm(d_w, axis=1)
    unit = np.ascontiguousarray((d_w / norm[:, None]).reshape(-1))
    geomid = np.zeros(len(d_cam), np.int32)
    dist = np.zeros(len(d_cam))
    mujoco.mj_multiRay(model, data, data.cam_xpos[0].copy(), unit, np.array([1, 0, 0, 1, 0, 0], np.uint8), True, -1,
                       geomid, dist, None, len(d_cam), 50.0)
    depth = np.where(geomid >= 0, dist / norm, 0.0).reshape(H, W)       # distance along the optical axis
    hit_env = np.isin(geomid, list(env_ids(model).values())).reshape(H, W)
    return depth, hit_env


def run_gl(model, data):
    r = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 0, 0, 1, 0, 0]                # groups 0 and 3, like mjlab's i2rt YAM camera (visual meshes of group 2 are not drawn)
    r.enable_depth_rendering()
    r.update_scene(data, camera='d455_depth', scene_option=opt)
    depth = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(data, camera='d455_depth', scene_option=opt)
    seg = r.render().copy()
    return depth, np.isin(seg[..., 0], list(env_ids(model).values()))


def run_mjlab(spec, device):
    import torch  # noqa: PLC0415
    from mjlab.entity import EntityCfg  # noqa: PLC0415
    from mjlab.scene import Scene, SceneCfg  # noqa: PLC0415
    from mjlab.sensor import CameraSensorCfg  # noqa: PLC0415
    from mjlab.sim.sim import Simulation, SimulationCfg  # noqa: PLC0415
    cam = CameraSensorCfg(name='d455', camera_name='robot/d455_depth', width=W, height=H, data_types=('depth',),
                          enabled_geom_groups=(0, 3), use_textures=False, use_shadows=False)
    scene = Scene(SceneCfg(num_envs=1, env_spacing=5.0, entities={'robot': EntityCfg(spec_fn=lambda: spec)}, sensors=(cam,)), device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(scene.sensor_context)
    sim.forward()
    sim.sense()
    d = scene['d455'].data.depth[0, ..., 0].detach().cpu().numpy().astype(float)
    return d, str(torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gl', action='store_true')
    ap.add_argument('--no-mjlab', action='store_true')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=f'{REPO}/docs/img/d455/depth_check.png')
    ap.add_argument('--json', default=f'{REPO}/docs/img/d455/depth_check.json')
    a = ap.parse_args()
    t_base, R_cam = urdf_camera_pose()
    expected, is_wall, C = closed_form(t_base, R_cam)
    spec = scene_spec()
    model, data = build(spec)
    cxp, cxm = data.cam_xpos[0].copy(), data.cam_xmat[0].reshape(3, 3).copy()
    # the camera pose MuJoCo built from the MJCF must equal the one rebuilt from the URDF
    pose_err_mm = float(np.linalg.norm(cxp - C) * 1000)
    rot_err_deg = float(np.degrees(np.arccos(np.clip((np.trace(cxm.T @ R_cam) - 1) / 2, -1, 1))))
    ray, hit_env = run_raycast(model, data)
    valid = hit_env & np.isfinite(expected)
    out = dict(camera_pose_error_mm=pose_err_mm, camera_rotation_error_deg=rot_err_deg,
               camera_world_m=np.round(C, 6).tolist(), image=f'{W}x{H}', hfov_deg=HFOV,
               mjcf_fovy_deg=float(model.cam_fovy[0]),
               robot_pixels=int((~hit_env).sum()), robot_pixel_fraction=float((~hit_env).mean()),
               centre_pixel_expected_m=float(expected[H // 2, W // 2]),
               results=[stats('R: MuJoCo ray casting (mj_multiRay)', ray, expected, valid)])
    res = {'R: MuJoCo ray casting': (ray, valid)}
    # sensitivity: how large would the disagreement be if the camera pose were wrong by a little? (the check must be able to fail)
    def rx(deg):
        c, sn = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
    sens = []
    for label, tb, Rc in [('camera pitched +0.1 deg', t_base, R_cam @ rx(0.1)), ('camera pitched +1 deg', t_base, R_cam @ rx(1.0)),
                          ('camera moved 5 mm forward', t_base + np.array([0.005, 0, 0]), R_cam),
                          ('camera moved 1 mm up', t_base + np.array([0, 0, 0.001]), R_cam),
                          ('camera turned upside down (180 deg about its axis)', t_base, R_cam @ np.diag([-1.0, -1.0, 1.0])),
                          ('Left IR on the wrong side (mirrored left-right)', t_base * np.array([1, -1, 1]), np.diag([1, -1, 1]) @ R_cam @ np.diag([-1.0, 1.0, 1.0]))]:
        e2, _, _ = closed_form(tb, Rc)
        ok = valid & np.isfinite(e2)
        sens.append(dict(case=label, max_abs_mm=float(np.abs(ray - e2)[ok].max() * 1000), mean_abs_mm=float(np.abs(ray - e2)[ok].mean() * 1000)))
    out['sensitivity_if_pose_were_wrong'] = sens
    if a.gl:
        d1, env1 = run_gl(model, data)
        v1 = env1 & np.isfinite(expected) & (d1 > 0)
        out['results'].append(stats('A: MuJoCo OpenGL renderer (mujoco.Renderer)', d1, expected, v1))
        res['A: MuJoCo OpenGL renderer'] = (d1, v1)
    if not a.no_mjlab:
        try:
            d2, _ = run_mjlab(spec, a.device)
            v2 = valid & (d2 > 0)
            out['results'].append(stats(f'B: mjlab CameraSensor (mujoco_warp, {a.device})', d2, expected, v2))
            res[f'B: mjlab CameraSensor ({a.device})'] = (d2, v2)
        except Exception as e:  # noqa: BLE001
            out['mjlab_error'] = repr(e)[:600]
            print('mjlab path failed:', repr(e)[:600])
    json.dump(out, open(a.json, 'w'), indent=1)
    print(json.dumps(out, indent=1))

    import matplotlib  # noqa: PLC0415
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: PLC0415
    n = len(res)
    fig, axs = plt.subplots(n, 3, figsize=(16, 3.9 * n), squeeze=False)
    vmax = float(np.nanpercentile(expected[np.isfinite(expected)], 99))
    valid0 = valid
    for row, (nm, (d, valid)) in enumerate(res.items()):
        im = axs[row, 0].imshow(np.where(valid, d, np.nan), cmap='viridis', vmin=0, vmax=vmax)
        axs[row, 0].set_title(f'{nm}: rendered depth [m]'); fig.colorbar(im, ax=axs[row, 0], fraction=0.03)
        im = axs[row, 1].imshow(np.where(valid, expected, np.nan), cmap='viridis', vmin=0, vmax=vmax)
        axs[row, 1].set_title('closed form (floor + wall planes) [m]'); fig.colorbar(im, ax=axs[row, 1], fraction=0.03)
        err = np.where(valid, (d - expected) * 1000, np.nan)
        lim = max(0.01, float(np.nanpercentile(np.abs(err), 99.9)))
        im = axs[row, 2].imshow(err, cmap='coolwarm', vmin=-lim, vmax=lim)
        axs[row, 2].set_title('rendered - closed form [mm]'); fig.colorbar(im, ax=axs[row, 2], fraction=0.03)
        for ax in axs[row]:
            ax.set_xlabel('u [px]'); ax.set_ylabel('v [px]')
    fig.suptitle(f'D455 depth check (848x480, HFOV 87 deg): camera {np.round(C, 3).tolist()} m, looks 60 deg down; floor z=0, front wall x={WALL_X} m, left wall y={SIDE_Y} m')
    fig.tight_layout(); fig.savefig(a.out, dpi=100)
    print('saved', a.out)


if __name__ == '__main__':
    main()
