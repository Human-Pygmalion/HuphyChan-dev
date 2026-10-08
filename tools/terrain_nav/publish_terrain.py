#!/usr/bin/env python3
"""Publish the mountain-trail navigation scene for RViz (system python3 + ROS 2 Humble, 3090 VM).

Topics: /terrain_cloud (coloured map, rgb), /scan_cloud (map points inside the lidar range around the robot),
/semantic (trail / rough / steep / vegetation cells near the path), /labels (semantic text), /plan (A* path),
/plan_line (thick path: done = dim, ahead = bright), /goal_markers, /robot_markers, /joint_states, TF map->base_link.
Computed (prep_terrain.py): classes, costs, path. The robot pose is placed ON the planned path at --at; its joint
angles come from a real v11a walking snapshot (world_dump.npz). No trajectory is hand-drawn.
usage: publish_terrain.py --terrain terrain.npz --urdf huphy_rviz.urdf --dump world_dump.npz [--at 0.42] [--no-text]
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

ap = argparse.ArgumentParser()
ap.add_argument("--terrain", required=True)
ap.add_argument("--urdf", required=True)
ap.add_argument("--dump", required=True, help="world_dump.npz from the Isaac walk: joint angles of a real walking step")
ap.add_argument("--at", type=float, default=0.42, help="robot position as a fraction of the path length")
ap.add_argument("--scan-radius", type=float, default=7.0)
ap.add_argument("--corridor", type=float, default=4.5, help="semantic cells drawn within this distance of the path")
ap.add_argument("--pelvis", type=float, default=0.86, help="base_link height above ground while walking [m]")
ap.add_argument("--no-text", action="store_true")
ap.add_argument("--label-min", type=float, default=6.0)
ap.add_argument("--label-max", type=float, default=45.0)
ap.add_argument("--dots", action="store_true", help="big disc at the robot (for the top-down overview)")
a = ap.parse_args()

T = np.load(a.terrain)
D = np.load(a.dump)
path = T["path"].astype(np.float64)
cls, dem, org, cell = T["cls"], T["dem"], T["origin"], float(T["cell"])
CLS_RGBA = [(0.95, 0.80, 0.35, 0.22), (0.85, 0.55, 0.25, 0.30), (0.95, 0.25, 0.20, 0.38), (0.20, 0.85, 0.35, 0.45)]

# robot on the path
seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
s = np.r_[0, np.cumsum(seg)]
i_at = int(np.searchsorted(s, a.at * s[-1]))
j0, j1 = max(i_at - 6, 0), min(i_at + 6, len(path) - 1)
yaw = math.atan2(path[j1, 1] - path[j0, 1], path[j1, 0] - path[j0, 0])
pw = np.array([path[i_at, 0], path[i_at, 1], path[i_at, 2] + a.pelvis])
qw = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
print(f"robot at {a.at:.2f} of {s[-1]:.1f} m: {pw.round(2)}, yaw {math.degrees(yaw):.0f} deg", flush=True)

root = ET.parse(a.urdf).getroot()
joint_names = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
qmap = dict(zip([str(n) for n in D["joint_names"]], D["qpos"].tolist()))
joint_pos = [float(qmap.get(n, 0.0)) for n in joint_names]


def mat2quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        k = math.sqrt(t + 1.0) * 2
        return (0.25 * k, (R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k_ = (i + 1) % 3, (i + 2) % 3
    k = math.sqrt(R[i, i] - R[j, j] - R[k_, k_] + 1.0) * 2
    q = [0.0] * 4
    q[1 + i] = 0.25 * k
    q[0] = (R[k_, j] - R[j, k_]) / k
    q[1 + j] = (R[j, i] + R[i, j]) / k
    q[1 + k_] = (R[k_, i] + R[i, k_]) / k
    return tuple(q)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def cloud_msg(pts, rgb=None):
    pts = np.asarray(pts, np.float32)
    m = PointCloud2()
    m.header.frame_id = "map"
    m.height, m.width = 1, len(pts)
    names = "xyz" if rgb is None else ["x", "y", "z", "rgb"]
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]
    if rgb is None:
        data = pts
    else:
        packed = (rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2].astype(np.uint32))
        data = np.column_stack([pts, packed.view(np.float32)])
    m.point_step = 4 * len(names)
    m.row_step = m.point_step * len(pts)
    m.is_bigendian, m.is_dense = False, True
    m.data = np.ascontiguousarray(data, np.float32).tobytes()
    return m


def robot_markers():
    mk = MarkerArray()
    cols = {m_.get("name"): [float(v) for v in m_.find("color").get("rgba").split()]
            for m_ in root.findall("material") if m_.find("color") is not None}
    i = 0
    for link in root.findall("link"):
        for v in link.findall("visual"):
            geom, org_ = v.find("geometry"), v.find("origin")
            xyz = [float(t) for t in (org_.get("xyz", "0 0 0") if org_ is not None else "0 0 0").split()]
            rpy = [float(t) for t in (org_.get("rpy", "0 0 0") if org_ is not None else "0 0 0").split()]
            mat = v.find("material")
            rgba = [0.72, 0.73, 0.76, 1.0]
            if mat is not None:
                c = mat.find("color")
                rgba = [float(t) for t in c.get("rgba").split()] if c is not None else cols.get(mat.get("name"), rgba)
            m = Marker()
            m.header.frame_id = link.get("name")
            m.ns, m.id, m.action = "robot", i, Marker.ADD
            i += 1
            m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
            m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = mat2quat(rpy_to_R(*rpy))
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            me, cy = geom.find("mesh"), geom.find("cylinder")
            if me is not None:
                m.type, m.mesh_resource, m.mesh_use_embedded_materials = Marker.MESH_RESOURCE, me.get("filename"), False
                m.scale.x = m.scale.y = m.scale.z = 1.0
            elif cy is not None:
                m.type = Marker.CYLINDER
                m.scale.x = m.scale.y = 2 * float(cy.get("radius"))
                m.scale.z = float(cy.get("length"))
            else:
                continue
            mk.markers.append(m)
    return mk


class Scene(Node):
    def __init__(self):
        super().__init__("huphy_terrain_scene")
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = {k: self.create_publisher(t, "/" + k, lat) for k, t in [
            ("terrain_cloud", PointCloud2), ("scan_cloud", PointCloud2), ("semantic", MarkerArray), ("labels", MarkerArray),
            ("plan", Path), ("plan_line", MarkerArray), ("goal_markers", MarkerArray), ("robot_markers", MarkerArray)]}
        self.p_js = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)
        cl, rgb = T["cloud"], T["cloud_rgb"]
        near = np.linalg.norm(cl[:, :2] - pw[:2], axis=1) < a.scan_radius
        self.msgs = {"terrain_cloud": cloud_msg(cl, rgb), "scan_cloud": cloud_msg(cl[near & (cl[:, 2] > self.ground_at(cl) + 0.3) & (cl[:, 2] < self.ground_at(cl) + 3.0)][::2]),
                     "robot_markers": robot_markers(), "semantic": self.semantic(), "labels": self.labels(),
                     "plan": self.plan(), "plan_line": self.plan_line(), "goal_markers": self.goal()}
        self.create_timer(0.1, self.fast)
        self.create_timer(1.0, self.slow)
        self.slow()

    def ground_at(self, pts):
        r = np.clip(((pts[:, 1] - org[1]) / cell).astype(int), 0, dem.shape[0] - 1)
        c = np.clip(((pts[:, 0] - org[0]) / cell).astype(int), 0, dem.shape[1] - 1)
        g = dem[r, c]
        return np.where(np.isnan(g), np.inf, g)

    def mk(self, ns, i, typ, rgba=(1, 1, 1, 1), scale=(1, 1, 1), xyz=(0, 0, 0)):
        m = Marker()
        m.header.frame_id = "map"
        m.ns, m.id, m.type, m.action = ns, i, typ, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = map(float, scale)
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
        return m

    def semantic(self):
        rr, cc = np.nonzero(cls < 4)
        cx, cy = org[0] + (cc + 0.5) * cell, org[1] + (rr + 0.5) * cell
        # distance to the path (coarse): keep cells within the corridor
        tree_pts = path[::4, :2]
        d = np.concatenate([np.min(np.hypot(cx[i:i + 20000, None] - tree_pts[None, :, 0], cy[i:i + 20000, None] - tree_pts[None, :, 1]), axis=1)
                            for i in range(0, len(cx), 20000)])
        keep = d < a.corridor
        out = MarkerArray()
        for k in range(4):
            sel = keep & (cls[rr, cc] == k)
            m = self.mk("semantic", k, Marker.CUBE_LIST, CLS_RGBA[k], (cell * 0.92, cell * 0.92, 0.005))
            m.points = [Point(x=float(x), y=float(y), z=float(dem[r, c] + 0.04)) for x, y, r, c in zip(cx[sel], cy[sel], rr[sel], cc[sel])]
            out.markers.append(m)
        return out

    def labels(self):
        out = MarkerArray()
        if a.no_text:
            return out
        placed = []
        for i, (txt, xyz) in enumerate(zip(T["labels"], T["label_xyz"])):
            if any(np.linalg.norm(xyz[:2] - q[:2]) < 12.0 for q in placed):  # one label per 12 m
                continue
            dr = np.linalg.norm(xyz[:2] - pw[:2])
            if not (a.label_min < dr < a.label_max):  # too close to the camera looks huge, too far is unreadable
                continue
            placed.append(xyz)
            col = {"TRAIL": (1.0, 0.85, 0.4, 1.0), "TREES": (0.35, 0.95, 0.45, 1.0)}.get(str(txt), (1.0, 0.4, 0.35, 1.0))
            m = self.mk("labels", i, Marker.TEXT_VIEW_FACING, col, (0, 0, 0.55), xyz)
            m.text = str(txt)
            out.markers.append(m)
        return out

    def plan(self):
        p = Path()
        p.header.frame_id = "map"
        for x, y, z in path:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), float(z + 0.12)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        return p

    def plan_line(self):
        out = MarkerArray()
        for k, (sl, rgba, w) in enumerate([(slice(0, i_at + 1), (0.35, 0.65, 0.45, 0.8), 0.18),
                                            (slice(i_at, None), (0.2, 1.0, 0.5, 1.0), 0.28)]):
            m = self.mk("plan_line", k, Marker.LINE_STRIP, rgba, (w, 0, 0))
            m.points = [Point(x=float(x), y=float(y), z=float(z + 0.15)) for x, y, z in path[sl]]
            out.markers.append(m)
        return out

    def goal(self):
        g = path[-1]
        out = MarkerArray()
        out.markers.append(self.mk("goal", 0, Marker.CYLINDER, (1, 1, 1, 1), (0.08, 0.08, 2.4), (g[0], g[1], g[2] + 1.2)))
        out.markers.append(self.mk("goal", 1, Marker.CYLINDER, (1.0, 0.72, 0.15, 0.6), (2.0, 2.0, 0.05), (g[0], g[1], g[2] + 0.05)))
        flag = self.mk("goal", 2, Marker.TRIANGLE_LIST, (1.0, 0.72, 0.15, 1.0))
        flag.points = [Point(x=float(g[0]), y=float(g[1]), z=float(g[2] + 2.4)), Point(x=float(g[0] + 1.0), y=float(g[1]), z=float(g[2] + 2.1)),
                       Point(x=float(g[0]), y=float(g[1]), z=float(g[2] + 1.8))]
        out.markers.append(flag)
        if a.dots:
            out.markers.append(self.mk("goal", 5, Marker.CYLINDER, (0.3, 0.85, 1.0, 1.0), (3.0, 3.0, 0.3), (pw[0], pw[1], path[i_at, 2] + 0.3)))
        st = path[0]
        out.markers.append(self.mk("goal", 3, Marker.CYLINDER, (0.3, 0.85, 1.0, 0.7), (1.6, 1.6, 0.05), (st[0], st[1], st[2] + 0.05)))
        if not a.no_text:
            t = self.mk("goal", 4, Marker.TEXT_VIEW_FACING, (1.0, 0.72, 0.15, 1.0), (0, 0, 1.0), (g[0], g[1], g[2] + 3.1))
            t.text = "GOAL"
            out.markers.append(t)
        return out

    def fast(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, pw)
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = qw
        self.tf.sendTransform(t)
        js = JointState()
        js.header.stamp = t.header.stamp
        js.name, js.position = joint_names, joint_pos
        self.p_js.publish(js)

    def slow(self):
        now = self.get_clock().now().to_msg()
        for k, m in self.msgs.items():
            if hasattr(m, "header"):
                m.header.stamp = now
            self.pub[k].publish(m)


rclpy.init()
rclpy.spin(Scene())
