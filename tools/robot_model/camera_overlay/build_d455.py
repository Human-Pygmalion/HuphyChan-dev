#!/usr/bin/env python3
"""Add the Realsense D455 (CAD component `Pelvis_Camera`) to the LegOnly AB (_loop) URDF.

The shipped models are NOT touched. For the base URDF this writes a sibling
`<name>_d455_loop.urdf` (+ the unchanged motor sidecar under the new name) and one new mesh
`meshes_d455/d455.stl`, already in the d455_link frame, metres. The generated files are never
hand-edited; change this script or the CAD snapshot and rebuild.

Input  : the snapshot written by extract_d455.py (CAD document name/version, D455 tessellation in the
         CAD root frame, the occurrence transform of `Realsense D455`).
Frames : follow docs/External "Intel RealSense D455 - URDF 센서 통합" section 2:
         base_link -> d455_mount (design origin: back plate, left-right and height centre)
                   -> d455_link  (depth frame = Left IR centre, ROS body axes x fwd, y left, z up)
                   -> d455_depth_optical_frame, d455_color_*, d455_infra2_*, d455_imu_frame
How the camera pose is read from the CAD (all checks are asserted below):
  * the occurrence transform of `Realsense D455` is the vendor mesh frame; realsense-ros maps that mesh
    frame to ROS body axes with rpy(pi/2, 0, pi/2), i.e. vendor x = left, y = up, z = forward;
  * the tessellation measures 26.0 x 124.0 x 29.0 mm along (forward, left, up) = datasheet depth x width x height;
  * the USB-C receptacle sits 37 mm to the camera's right and on the underside (checked once by hand).
Also placed: `Pelvis_Camera_Mount`, the PLA-printed bracket, rigidly fixed to base_link (fixed joint, user 2026-10-28). Its mass is the
CAD volume times the printed structural density of the shell/elbow overlay (shell_overlay/manifest.json `structural_printed`, 0.8874 g/cm3,
docs/89); COM and inertia are the CAD tensor scaled by the same ratio and moved into the base_link frame. CAD material (steel) is not used.

usage (mjlab venv): build_d455.py --snapshot <dir written by extract_d455.py>
"""
import argparse
import json
import math
import os
import re
import shutil

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
ASSETS = f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2'
BASES = ['LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop']   # AB (2-RSU) only; RP is not built (user, 2026-10-27)
BASE_CAD = np.array([0.0, 70.0, 60.0])           # base_link origin in CAD mm (build_robot.py ORIGIN_CAD['pelvis'])
SIZE = (0.026, 0.124, 0.029)                     # datasheet depth x width x height, m
MASS = 0.116                                     # datasheet 116 g (+-10 %); the xacro's 0.072 is marked unreliable there
LINK_IN_MOUNT = np.array([0.02135, 0.0475, 0.0])   # Left IR centre from the back-plate midpoint (21.35 mm fwd, 47.5 mm left)
COM_IN_LINK = np.array([-0.00835, -0.0475, 0.0])   # box centre seen from the Left IR centre
OPT = (-math.pi / 2, 0.0, -math.pi / 2)            # optical frame rpy of realsense-ros
FACES = 30000
RHO_PRINTED = json.load(open(f'{REPO}/tools/robot_model/shell_overlay/manifest.json', encoding='utf-8'))['density_g_cm3']['structural_printed']   # g/cm3
R_SIM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # CAD -> sim (same as build_robot.R)
MOUNT_RGBA = '0.86 0.86 0.80 1'


def sim(v):                                       # CAD -> sim axes: sim = (-y, x, z)
    v = np.asarray(v, float)
    return np.array([-v[..., 1], v[..., 0], v[..., 2]]).T if v.ndim > 1 else np.array([-v[1], v[0], v[2]])


def f6(x):
    return f'{x:.6g}'


def vec(v):
    return ' '.join(f6(x) for x in v)


def joint(name, parent, child, xyz, rpy):
    return (f'  <joint name="{name}" type="fixed">\n    <origin xyz="{vec(xyz)}" rpy="{vec(rpy)}"/>\n'
            f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n  </joint>\n')


def mount_props(J):
    """Pelvis_Camera_Mount as a PLA part in the base_link frame: (mass kg, com m, inertia about the com kg m^2, details)."""
    r = J['mount_props']
    v_cm3, m_cad = r['v'], r['m']
    m = v_cm3 * RHO_PRINTED / 1000.0                                  # same rule as shell_overlay.overlay_core.part_item('estimate_density')
    k = m / m_cad
    com_cad = np.array(r['c']) * 10.0                                 # cm -> mm
    xx, yy, zz, xy, yz, xz = r['I']
    Io = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * 100.0 * k           # kg cm^2 -> kg mm^2, scaled to the printed mass
    Ic = Io - m * (com_cad @ com_cad * np.eye(3) - np.outer(com_cad, com_cad))      # about the COM
    com = R_SIM @ (com_cad - BASE_CAD) / 1000.0
    I = R_SIM @ Ic @ R_SIM.T * 1e-6
    w = np.linalg.eigvalsh(I)
    assert w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-6), f'inertia is not physical: {w}'
    return m, com, I, dict(volume_cm3=v_cm3, cad_material=r['mat'], cad_mass_kg=m_cad, density_g_cm3=RHO_PRINTED, mass_scale_vs_cad=k)


def mount_block(m, com, I):
    s = ('  <link name="pelvis_camera_mount">\n    <inertial>\n      <origin xyz="%s" rpy="0 0 0"/>\n      <mass value="%s"/>\n'
         '      <inertia ixx="%s" ixy="%s" ixz="%s" iyy="%s" iyz="%s" izz="%s"/>\n    </inertial>\n'
         '    <visual>\n      <geometry><mesh filename="meshes_d455/pelvis_camera_mount.stl"/></geometry>\n'
         '      <material name="pla_mount"><color rgba="%s"/></material>\n    </visual>\n'
         '    <collision>\n      <geometry><mesh filename="meshes_d455/pelvis_camera_mount_hull.stl"/></geometry>\n    </collision>\n  </link>\n'
         % (vec(com), f6(m), f6(I[0, 0]), f6(I[0, 1]), f6(I[0, 2]), f6(I[1, 1]), f6(I[1, 2]), f6(I[2, 2]), MOUNT_RGBA))
    # rigid: a fixed joint on base_link at the base_link origin; the mesh and COM are already in the base_link frame
    return s + joint('pelvis_camera_mount_joint', 'base_link', 'pelvis_camera_mount', (0, 0, 0), (0, 0, 0))


def build_block(p_mount_sim, rpy_mount):
    w, d, h = SIZE[1], SIZE[0], SIZE[2]
    ixx, iyy, izz = MASS * (w * w + h * h) / 12, MASS * (d * d + h * h) / 12, MASS * (d * d + w * w) / 12
    s = ('  <!-- D455 OVERLAY, generated by tools/robot_model/camera_overlay/build_d455.py from the CAD component '
         'Pelvis_Camera / Realsense D455 (docs/External D455 notes). Do not hand-edit; rebuild. -->\n')
    s += '  <link name="d455_mount"/>\n'
    s += joint('d455_mount_joint', 'base_link', 'd455_mount', p_mount_sim, rpy_mount)
    s += (f'  <link name="d455_link">\n    <inertial>\n      <origin xyz="{vec(COM_IN_LINK)}" rpy="0 0 0"/>\n'
          f'      <mass value="{MASS}"/>\n      <inertia ixx="{f6(ixx)}" ixy="0" ixz="0" iyy="{f6(iyy)}" iyz="0" izz="{f6(izz)}"/>\n'
          f'    </inertial>\n    <visual>\n      <geometry><mesh filename="meshes_d455/d455.stl"/></geometry>\n      <material name="d455_blue"><color rgba="0.10 0.35 0.85 1"/></material>\n    </visual>\n'
          f'    <collision>\n      <origin xyz="{vec(COM_IN_LINK)}" rpy="0 0 0"/>\n'
          f'      <geometry><box size="{vec(SIZE)}"/></geometry>\n    </collision>\n  </link>\n')
    s += joint('d455_link_joint', 'd455_mount', 'd455_link', LINK_IN_MOUNT, (0, 0, 0))
    frames = [('d455_depth_optical_frame', 'd455_link', (0, 0, 0), OPT),
              ('d455_color_frame', 'd455_link', (0, -0.059, 0), (0, 0, 0)),
              ('d455_color_optical_frame', 'd455_color_frame', (0, 0, 0), OPT),
              ('d455_infra2_frame', 'd455_link', (0, -0.095, 0), (0, 0, 0)),
              ('d455_infra2_optical_frame', 'd455_infra2_frame', (0, 0, 0), OPT),
              ('d455_imu_frame', 'd455_link', (-0.01602, -0.03022, 0.0074), (0, 0, 0))]
    for name, parent, xyz, rpy in frames:
        s += f'  <link name="{name}"/>\n' + joint(name.replace('_frame', '_joint'), parent, name, xyz, rpy)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    a = ap.parse_args()
    J = json.load(open(f'{a.snapshot}/d455_bodies.json'))
    mesh = trimesh.load(f'{a.snapshot}/cad_mm/d455.stl', process=False)
    V = np.asarray(mesh.vertices)

    R = np.array(J['T_d455_in_pelvis_camera_cm']).reshape(4, 4)[:3, :3]     # vendor mesh frame in the CAD root
    left, up, fwd = R[:, 0], R[:, 1], R[:, 2]
    assert np.allclose(np.cross(left, up), fwd, atol=1e-6) and np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    A, B, C = V @ fwd, V @ left, V @ up
    dims = (np.ptp(A), np.ptp(B), np.ptp(C))
    assert all(abs(x - y) < 0.4 for x, y in zip(dims, (26.0, 124.0, 29.0))), f'mesh is not a D455 box: {dims}'
    a_back, b_mid, c_mid = A.min(), (B.min() + B.max()) / 2, (C.min() + C.max()) / 2
    p_mount = a_back * fwd + b_mid * left + c_mid * up                      # CAD mm
    p_link = p_mount + 1000 * LINK_IN_MOUNT[0] * fwd + 1000 * LINK_IN_MOUNT[1] * left

    Rs = np.stack([sim(fwd), sim(left), sim(up)], axis=1)                   # camera axes in sim axes
    rpy = Rot.from_matrix(Rs).as_euler('xyz')
    assert abs(rpy[0]) < 1e-6 and abs(rpy[2]) < 1e-6, f'camera is rolled/yawed: {rpy}'
    p_mount_sim = sim(p_mount - BASE_CAD) / 1000.0

    # mesh into the d455_link frame (forward, left, up from the Left IR centre), metres, decimated for a light visual
    lp = np.array([p_link @ fwd, p_link @ left, p_link @ up])
    m = trimesh.Trimesh(np.stack([A - lp[0], B - lp[1], C - lp[2]], axis=1) / 1000.0, mesh.faces, process=False)
    full = len(m.faces)
    m = m.simplify_quadric_decimation(face_count=FACES)
    os.makedirs(f'{ASSETS}/meshes_d455', exist_ok=True)
    m.export(f'{ASSETS}/meshes_d455/d455.stl')
    print(f'mesh {full} -> {len(m.faces)} faces; extent {np.round(m.extents * 1000, 1)} mm')

    mm, mcom, mI, minfo = mount_props(J)
    mt = trimesh.load(f'{a.snapshot}/cad_mm/mount.stl', process=False)
    mv = (np.asarray(mt.vertices) - BASE_CAD) @ R_SIM.T / 1000.0                    # CAD mm -> base_link frame, metres
    mmesh = trimesh.Trimesh(mv, mt.faces, process=False)
    mmesh.export(f'{ASSETS}/meshes_d455/pelvis_camera_mount.stl')
    mmesh.convex_hull.export(f'{ASSETS}/meshes_d455/pelvis_camera_mount_hull.stl')
    print(f'mount bracket: {mm * 1000:.2f} g (CAD {minfo["cad_mass_kg"] * 1000:.0f} g {minfo["cad_material"]}, {minfo["volume_cm3"]:.2f} cm3 x {RHO_PRINTED} g/cm3); '
          f'COM in base_link {np.round(mcom, 4)} m; extent {np.round(mmesh.extents * 1000, 1)} mm')
    block = build_block(p_mount_sim, rpy) + mount_block(mm, mcom, mI)
    for base in BASES:
        txt = open(f'{ASSETS}/{base}.urdf', encoding='utf-8').read()
        name = base + '_d455' if not base.endswith('_loop') else base[:-5] + '_d455_loop'
        txt = re.sub(r'<robot name="[^"]*">', f'<robot name="{name}">', txt, count=1)
        assert txt.rstrip().endswith('</robot>')
        txt = txt.rstrip()[:-len('</robot>')] + block + '</robot>\n'
        open(f'{ASSETS}/{name}.urdf', 'w', encoding='utf-8').write(txt)
        shutil.copy(f'{ASSETS}/{base}.motor_params.json', f'{ASSETS}/{name}.motor_params.json')
        print('wrote', f'{name}.urdf')

    out = dict(cad_document=J['doc'], cad_version=J['version'], cad_unsaved_edits=J['modified_unsaved'],
               base_link_origin_cad_mm=BASE_CAD.tolist(),
               camera_axes_in_cad=dict(forward=np.round(fwd, 6).tolist(), left=np.round(left, 6).tolist(), up=np.round(up, 6).tolist()),
               mesh_extent_mm_fwd_left_up=np.round(dims, 2).tolist(),
               d455_mount_in_cad_mm=np.round(p_mount, 3).tolist(),
               d455_link_in_cad_mm=np.round(p_link, 3).tolist(),
               d455_mount_in_base_link_m=np.round(p_mount_sim, 6).tolist(),
               d455_mount_rpy_rad=np.round(rpy, 6).tolist(), pitch_deg=round(math.degrees(rpy[1]), 3),
               d455_link_in_base_link_m=np.round(p_mount_sim + Rs @ LINK_IN_MOUNT, 6).tolist(),
               mass_kg=MASS, mass_source='datasheet 116 g +-10 %',
               mount=dict(mass_kg=mm, com_base_link_m=np.round(mcom, 7).tolist(), inertia_base_link_kg_m2=np.round(mI, 9).tolist(), **minfo,
                          attached='fixed joint on base_link (rigid)', density_source='shell_overlay/manifest.json structural_printed = docs/89 printed ratio 0.329 x 2.70'),
               assumptions=['design origin = back plate, left-right and height centre (M4 hole midpoint not measured in the CAD)',
                            'mass is the datasheet value, not the CAD body'],
               not_placed=[])
    json.dump(out, open(f'{ASSETS}/d455_overlay_report.json', 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('d455_mount_in_base_link_m', 'pitch_deg', 'd455_link_in_base_link_m')}))


if __name__ == '__main__':
    main()
