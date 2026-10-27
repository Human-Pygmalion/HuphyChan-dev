#!/usr/bin/env python3
"""Show the D455 URDF in RViz on a virtual display and save pictures (side / oblique / front).

Runs where ROS 2 Humble + Xvfb + ImageMagick `import` exist (the 3090 VM; `source ~/ros_env_rviz.sh` first).
Publishes, besides the robot (robot_state_publisher, all joints at zero):
  world -> base_link at z = 0.903 m (straight legs, soles on the floor)
  /d455_markers   black cylinders = the MJCF's motors (only with --mjcf; the URDF itself has no motor shapes),
                  yellow depth frustum (87 x 58 deg, 0.8 m) in d455_depth_optical_frame,
                  green wire box = datasheet 26 x 124 x 29 mm around the camera body (d455_link),
                  red dot = where the optical axis meets the floor
usage: ROS_DOMAIN_ID=71 rviz_check.py --urdf <...d455_loop.urdf> --out-dir <dir> [--display :96] [--mjcf <...d455_loop.xml>]
(use a ROS domain of your own: other demos on the same machine publish /robot_description and TF too)
"""
import argparse
import math
import os
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

BASE_Z = 0.903                       # base_link height above the floor with straight legs
HFOV, VFOV, RANGE = 87.0, 58.0, 0.8  # datasheet depth FOV, drawing range
BOX, COM = (0.026, 0.124, 0.029), (-0.00835, -0.0475, 0.0)

CFG = """Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Floor
      Enabled: true
      Cell Size: 0.25
      Plane Cell Count: 16
      Color: 150; 150; 160
      Alpha: 0.7
      Reference Frame: world
    - Class: rviz_default_plugins/RobotModel
      Name: Robot
      Enabled: true
      Description Source: Topic
      Description Topic:
        Value: /robot_description
        Depth: 1
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
      Visual Enabled: true
      Collision Enabled: false
      Alpha: 1
    - Class: rviz_default_plugins/TF
      Name: Frames
      Enabled: true
      Frames:
        All Enabled: false
        base_link:
          Value: true
        d455_link:
          Value: true
        d455_depth_optical_frame:
          Value: true
      Marker Scale: 0.12
      Show Arrows: false
      Show Axes: true
      Show Names: true
    - Class: rviz_default_plugins/MarkerArray
      Name: D455 helpers
      Enabled: true
      Topic:
        Value: /d455_markers
        Depth: 5
        Reliability Policy: Reliable
  Global Options:
    Background Color: 238; 240; 244
    Fixed Frame: world
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: %(dist)s
      Focal Point:
        X: %(fx)s
        Y: %(fy)s
        Z: %(fz)s
      Pitch: %(pitch)s
      Yaw: %(yaw)s
      Near Clip Distance: 0.01
      Target Frame: <Fixed Frame>
Window Geometry:
  Displays:
    collapsed: true
  Views:
    collapsed: true
  Height: 1000
  Width: 1600
  X: 0
  Y: 0
"""
# Orbit view in RViz: yaw 0 looks from -x towards +x (the robot's back); pi looks at the robot's face.
VIEWS = {'side': dict(dist=1.3, fx=0.12, fy=0.0, fz=0.8, pitch=0.12, yaw=-1.5708),
         'oblique': dict(dist=1.2, fx=0.12, fy=0.0, fz=0.75, pitch=0.45, yaw=2.4),
         'front': dict(dist=1.4, fx=0.12, fy=0.0, fz=0.85, pitch=0.2, yaw=3.1416),
         'close_side': dict(dist=0.40, fx=0.12, fy=0.0475, fz=0.93, pitch=0.0, yaw=-1.5708),
         'close_oblique': dict(dist=0.40, fx=0.12, fy=0.0475, fz=0.93, pitch=0.55, yaw=2.5),
         'close_front': dict(dist=0.45, fx=0.12, fy=0.0, fz=0.93, pitch=0.15, yaw=3.1416)}


def prep_urdf(path):
    t = open(path, encoding='utf-8').read()
    d = os.path.dirname(os.path.abspath(path))
    t = re.sub(r'filename="(meshes[^"]*)"', lambda m: f'filename="file://{d}/{m.group(1)}"', t)
    grey = '<material name="rv_grey"><color rgba="0.72 0.73 0.76 1"/></material>'
    t = re.sub(r'<visual>.*?</visual>', lambda m: m.group(0) if '<material' in m.group(0)
               else m.group(0)[:-len('</visual>')] + grey + '</visual>', t, flags=re.S)
    return t


def chain_pose(root, link):
    J = {j.find('child').get('link'): (j.find('parent').get('link'), np.array(j.find('origin').get('xyz').split(), float),
                                       np.array(j.find('origin').get('rpy').split(), float)) for j in root.findall('joint')}
    from scipy.spatial.transform import Rotation as Rot  # noqa: PLC0415
    ch = []
    while link != 'base_link':
        p, x, y = J[link]
        ch.append((x, y))
        link = p
    R, t = np.eye(3), np.zeros(3)
    for x, y in reversed(ch):
        t = t + R @ x
        R = R @ Rot.from_euler('xyz', y).as_matrix()
    return t, R


def motor_cylinders(mjcf):
    """The `*_motor` cylinders of the MJCF (the URDF has none): [(parent link, pos, quat wxyz, radius, half length)]."""
    out = []

    def walk(el, body):
        for ch in el:
            if ch.tag == 'body':
                walk(ch, ch.get('name'))
            elif ch.tag == 'geom' and (ch.get('name') or '').endswith('_motor') and ch.get('type') == 'cylinder':
                size = [float(x) for x in ch.get('size').split()]
                out.append((body, [float(x) for x in ch.get('pos', '0 0 0').split()],
                            [float(x) for x in ch.get('quat', '1 0 0 0').split()], size[0], size[1]))
    walk(ET.parse(mjcf).getroot().find('worldbody'), 'world')
    return out


def pt(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


class Pub(Node):
    def __init__(self, root, motors=()):
        super().__init__('d455_check')
        self.motors = motors
        self.names = [j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed']
        self.js = self.create_publisher(JointState, '/joint_states', 10)
        self.mk = self.create_publisher(MarkerArray, '/d455_markers', 5)
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'world', 'base_link'
        tf.transform.translation.z = BASE_Z
        tf.transform.rotation.w = 1.0
        self.stf = StaticTransformBroadcaster(self)
        tf.header.stamp = self.get_clock().now().to_msg()
        self.stf.sendTransform(tf)
        t, R = chain_pose(root, 'd455_link')
        t[2] += BASE_Z
        view = R[:, 0]                                    # optical axis = d455_link +x
        self.hit = t + view * (-t[2] / view[2])           # where it meets the floor (world frame)
        print(f'optical axis meets the floor at x={self.hit[0]:.3f} m, y={self.hit[1]:.3f} m (world, base_link at x=0)', flush=True)
        self.create_timer(0.2, self.tick)

    def markers(self):
        out = []
        hx, hy = math.tan(math.radians(HFOV / 2)) * RANGE, math.tan(math.radians(VFOV / 2)) * RANGE
        corners = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]       # optical frame: x right, y down, z forward
        m = Marker()
        m.header.frame_id, m.ns, m.id, m.type = 'd455_depth_optical_frame', 'frustum', 0, Marker.LINE_LIST
        m.scale.x, m.color.r, m.color.g, m.color.b, m.color.a = 0.004, 1.0, 0.8, 0.0, 1.0
        for i, c in enumerate(corners):
            n = corners[(i + 1) % 4]
            m.points += [pt(0, 0, 0), pt(c[0], c[1], RANGE), pt(c[0], c[1], RANGE), pt(n[0], n[1], RANGE)]
        out.append(m)
        b = Marker()
        b.header.frame_id, b.ns, b.id, b.type = 'd455_link', 'datasheet_box', 0, Marker.LINE_LIST
        b.scale.x, b.color.g, b.color.a = 0.003, 0.7, 1.0
        c0 = np.array(COM)
        h = np.array(BOX) / 2
        v = [c0 + h * np.array(s) for s in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            b.points += [pt(*v[i]), pt(*v[j])]
        out.append(b)
        s = Marker()
        s.header.frame_id, s.ns, s.id, s.type = 'world', 'floor_hit', 0, Marker.SPHERE
        s.pose.position = pt(*self.hit)
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.05
        s.color.r, s.color.a = 1.0, 1.0
        out.append(s)
        for i, (link, pos, q, r, hl) in enumerate(self.motors):           # display only: same cylinders the MJCF draws
            c = Marker()
            c.header.frame_id, c.ns, c.id, c.type = link, 'motors', i, Marker.CYLINDER
            c.pose.position = pt(*pos)
            c.pose.orientation.w, c.pose.orientation.x, c.pose.orientation.y, c.pose.orientation.z = q
            c.scale.x = c.scale.y = 2 * r
            c.scale.z = 2 * hl
            c.color.r = c.color.g = c.color.b = 0.12
            c.color.a = 1.0
            out.append(c)
        return out

    def tick(self):
        now = self.get_clock().now().to_msg()
        j = JointState()
        j.header.stamp, j.name, j.position = now, self.names, [0.0] * len(self.names)
        self.js.publish(j)
        arr = MarkerArray()
        for mk in self.markers():
            mk.header.stamp = now
            arr.markers.append(mk)
        self.mk.publish(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--display', default=':96')
    ap.add_argument('--mjcf', default=None, help='MJCF whose *_motor cylinders are drawn as markers (display only)')
    ap.add_argument('--wait', type=float, default=14.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    root = ET.parse(a.urdf).getroot()
    env = dict(os.environ, DISPLAY=a.display)
    procs = [subprocess.Popen(['Xvfb', a.display, '-screen', '0', '1600x1000x24'], stderr=subprocess.DEVNULL)]
    time.sleep(2)
    params = f'{a.out_dir}/rsp_params.yaml'            # a parameter file: the URDF text cannot go through `-p name:=value`
    body = '\n'.join('      ' + line for line in prep_urdf(a.urdf).splitlines())
    open(params, 'w', encoding='utf-8').write('robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n' + body + '\n')
    procs.append(subprocess.Popen(['/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher', '--ros-args',
                                   '--params-file', params], env=env, stdout=subprocess.DEVNULL))
    rclpy.init()
    node = Pub(root, motor_cylinders(a.mjcf) if a.mjcf else ())
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(3)
    try:
        for name, v in VIEWS.items():
            cfg = f'{a.out_dir}/{name}.rviz'
            open(cfg, 'w').write(CFG % v)
            rv = subprocess.Popen(['rviz2', '-d', cfg], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(a.wait)
            subprocess.run(['import', '-window', 'root', f'{a.out_dir}/rviz_{name}.png'], env=env, check=True)
            rv.terminate()
            rv.wait(timeout=10)
            print('saved', f'{a.out_dir}/rviz_{name}.png', flush=True)
    finally:
        for p in procs[::-1]:
            p.terminate()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
