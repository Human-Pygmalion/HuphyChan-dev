"""Rotate the base_link (pelvis) freely and watch the IMU's RPY + gyro respond live.

Companion to tools/imu/web_viewer.py (the same kind of check, but on the real
EBIMU sensor instead of the simulated one). Drag the roll/pitch/yaw sliders --
each is one MuJoCo mju_euler2Quat("XYZ") extrinsic rotation of the base free
joint, matching the convention mjlab/utils/lab_api/math.py::euler_xyz_from_quat
uses to read RPY back out -- and the panel shows what the sim's own `imu_ang_vel`
(gyro) and `imu_upvector` (projected gravity) sensors report for that motion.
The coordinate triad (red=X, green=Y, blue=Z) is drawn at the `imu_in_base` site's
own live pose, not the pelvis body origin -- the site has a position offset from
the body but no orientation override, so its axes track the body's rotation.

The gyro reading is not free: dragging a slider is a KINEMATIC pose change (no
dynamics stepping), so the angular velocity fed to the sensor is derived by
finite-differencing the quaternion between consecutive updates with MuJoCo's
own mju_subQuat/mju_quatIntegrate pair (the same pair mj_step itself uses for
free joints) -- not hand-derived, so it can't introduce a new sign bug.

Usage:  .venv/bin/python3 tools/robot_model/imu_convention_viewer.py [--port=8891] [--xml=...]
"""
import argparse
import time
from collections import deque

import mujoco
import numpy as np
import trimesh
import viser
import viser.uplot as vup

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
DEFAULT_XML = f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2/pygmalion_v4_printed.xml'
STAND_Z = 0.903


def local_primitive(m, g):
    t = m.geom_type[g]
    r = float(m.geom_size[g][0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        h = float(m.geom_size[g][1])
        cap = trimesh.creation.capsule(height=2 * h, radius=r, count=[12, 12])
        cap.apply_translation([0, 0, -cap.bounds.mean(0)[2]])
        return cap
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=r)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=r, height=2 * float(m.geom_size[g][1]), sections=24)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * np.array(m.geom_size[g][:3]))
    return None


def sensor_slice(m, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise SystemExit(f"sensor '{name}' not found in this XML -- pick one that defines it (pygmalion_v4_printed.xml does)")
    adr = m.sensor_adr[sid]
    dim = m.sensor_dim[sid]
    return adr, dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8891)
    p.add_argument('--xml', default=DEFAULT_XML)
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)

    root_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root')
    if root_jid < 0 or m.jnt_type[root_jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise SystemExit("this XML has no free 'root' joint on the base -- can't rotate the base freely")
    qadr = m.jnt_qposadr[root_jid]   # -> (x, y, z, qw, qx, qy, qz)
    dadr = m.jnt_dofadr[root_jid]    # -> (vx, vy, vz, wx, wy, wz)

    gyro_adr, gyro_dim = sensor_slice(m, 'imu_ang_vel')
    grav_adr, grav_dim = sensor_slice(m, 'imu_upvector')
    imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'imu_in_base')

    server = viser.ViserServer(host='0.0.0.0', port=args.port, label='Pygmalion IMU convention check')
    server.scene.add_grid('/ground', width=4, height=4, cell_size=0.25, plane='xy')

    # --- robot meshes, same rendering pattern as tools/robot_model/web_viewer.py ---
    handles = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f'geom{g}'
        body = m.geom_bodyid[g]
        if body == 0:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            V, F = m.mesh_vert[a:a + n], m.mesh_face[fa:fa + fn]
            h = server.scene.add_mesh_simple(f'/robot/visual/{name}', V, F, color=(150, 160, 180))
        else:
            tm = local_primitive(m, g)
            if tm is None:
                continue
            h = server.scene.add_mesh_simple(f'/robot/collision/{name}', tm.vertices, tm.faces,
                                              color=(60, 170, 90), opacity=0.35, visible=False)
        handles.append((g, h))

    imu_frame = server.scene.add_frame('/robot/imu_axes', axes_length=0.15, axes_radius=0.005)

    state = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'prev_quat': np.array([1.0, 0.0, 0.0, 0.0]),
             'prev_t': time.monotonic()}
    t0 = time.monotonic()
    HIST = 300  # ~ a few seconds of dragging at typical slider event rates
    hist = {k: deque(maxlen=HIST) for k in ('t', 'roll', 'pitch', 'yaw', 'gx', 'gy', 'gz')}

    def apply():
        rpy = np.radians([state['roll'], state['pitch'], state['yaw']])
        quat = np.zeros(4)
        # "XYZ" (uppercase) = extrinsic, matching euler_xyz_from_quat's own
        # docstring ("XYZ extrinsic convention") so slider-set RPY round-trips.
        mujoco.mju_euler2Quat(quat, rpy, 'XYZ')

        now = time.monotonic()
        dt = max(now - state['prev_t'], 1e-4)
        angvel = np.zeros(3)
        mujoco.mju_subQuat(angvel, quat, state['prev_quat'])  # qb*quat(res) = qa, qb=prev, qa=new
        angvel /= dt
        state['prev_quat'] = quat.copy()
        state['prev_t'] = now

        d.qpos[qadr:qadr + 3] = [0.0, 0.0, STAND_Z]
        d.qpos[qadr + 3:qadr + 7] = quat
        d.qvel[dadr:dadr + 3] = 0.0
        d.qvel[dadr + 3:dadr + 6] = angvel
        mujoco.mj_forward(m, d)

        for g, h in handles:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            h.position = tuple(d.geom_xpos[g])
            h.wxyz = tuple(q)
        imu_quat = np.zeros(4)
        mujoco.mju_mat2Quat(imu_quat, d.site_xmat[imu_site])
        imu_frame.position = tuple(d.site_xpos[imu_site])
        imu_frame.wxyz = tuple(imu_quat)

        gyro = d.sensordata[gyro_adr:gyro_adr + gyro_dim]
        grav = d.sensordata[grav_adr:grav_adr + grav_dim]
        gdeg = np.degrees(gyro)

        # Fixed-width fields inside a code fence: every number keeps the same
        # on-screen column no matter its sign or digit count, so a moving
        # value reads as a changing NUMBER, not a jumping BLOB of text.
        readout.content = (
            "```\n"
            f"{'':>10}{'X / roll':>12}{'Y / pitch':>12}{'Z / yaw':>12}\n"
            f"{'set (deg)':>10}{state['roll']:>12.2f}{state['pitch']:>12.2f}{state['yaw']:>12.2f}\n"
            f"{'gyro rad/s':>10}{gyro[0]:>12.3f}{gyro[1]:>12.3f}{gyro[2]:>12.3f}\n"
            f"{'gyro deg/s':>10}{gdeg[0]:>12.2f}{gdeg[1]:>12.2f}{gdeg[2]:>12.2f}\n"
            f"{'up_vec':>10}{grav[0]:>12.3f}{grav[1]:>12.3f}{grav[2]:>12.3f}\n"
            f"{'proj_grav':>10}{-grav[0]:>12.3f}{-grav[1]:>12.3f}{-grav[2]:>12.3f}\n"
            "```\n"
            "up_vec = raw `imu_upvector` sensor ({0,0,1} when level). "
            "proj_grav = -up_vec = what the policy observation contains."
        )

        now_t = time.monotonic() - t0
        hist['t'].append(now_t)
        hist['roll'].append(state['roll'])
        hist['pitch'].append(state['pitch'])
        hist['yaw'].append(state['yaw'])
        hist['gx'].append(gdeg[0])
        hist['gy'].append(gdeg[1])
        hist['gz'].append(gdeg[2])
        rpy_plot.data = (np.array(hist['t']), np.array(hist['roll']), np.array(hist['pitch']), np.array(hist['yaw']))
        gyro_plot.data = (np.array(hist['t']), np.array(hist['gx']), np.array(hist['gy']), np.array(hist['gz']))

    with server.gui.add_folder('Base (pelvis) orientation'):
        s_roll = server.gui.add_slider('roll (X)', min=-180.0, max=180.0, step=0.5, initial_value=0.0)
        s_pitch = server.gui.add_slider('pitch (Y)', min=-180.0, max=180.0, step=0.5, initial_value=0.0)
        s_yaw = server.gui.add_slider('yaw (Z)', min=-180.0, max=180.0, step=0.5, initial_value=0.0)

        def make_cb(key, s):
            def cb(_):
                state[key] = s.value
                apply()
            return cb
        s_roll.on_update(make_cb('roll', s_roll))
        s_pitch.on_update(make_cb('pitch', s_pitch))
        s_yaw.on_update(make_cb('yaw', s_yaw))

        def reset(_):
            s_roll.value = s_pitch.value = s_yaw.value = 0.0
            state['roll'] = state['pitch'] = state['yaw'] = 0.0
            apply()
        server.gui.add_button('reset to level').on_click(reset)

    readout = server.gui.add_markdown('move a slider')

    axis_series = lambda label, color: vup.Series(label=label, stroke=color, width=2)  # noqa: E731
    with server.gui.add_folder('RPY (deg) vs time'):
        rpy_plot = server.gui.add_uplot(
            data=(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
            series=(vup.Series(label='t'),
                    axis_series('roll', 'red'), axis_series('pitch', 'green'), axis_series('yaw', 'blue')),
            height=160,
        )
    with server.gui.add_folder('gyro (deg/s) vs time'):
        gyro_plot = server.gui.add_uplot(
            data=(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)),
            series=(vup.Series(label='t'),
                    axis_series('gx', 'red'), axis_series('gy', 'green'), axis_series('gz', 'blue')),
            height=160,
        )

    server.gui.add_markdown(
        '**Reading this**: roll/pitch/yaw are extrinsic-XYZ Euler angles of the pelvis free '
        'joint -- the same convention `euler_xyz_from_quat` (mjlab/utils/lab_api/math.py) uses '
        'to decode a policy observation quaternion, so "set pitch +30" here is the same motion '
        'as pitch=+30deg in that function\'s output. `imu_ang_vel`/`imu_upvector` are read '
        'straight off the MJCF sensors, not recomputed. Gyro is nonzero only while a slider is '
        'actively moving (kinematic viewer, no physics stepping -- angular velocity is the '
        'finite difference between consecutive poses via MuJoCo\'s own mju_subQuat).'
    )

    apply()
    print(f'viser up on 0.0.0.0:{args.port}', flush=True)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
