#!/usr/bin/env python3
"""Interactive navigation on the whole mountain-trail map (system python3 + ROS 2 Humble, 3090 VM).

In RViz:  "2D Pose Estimate" (/initialpose) = where the robot starts,  "2D Goal Pose" (/goal_pose) = where to go.
Every new goal (or start) re-plans with A* over the terrain-class cost grid from prep_terrain.py and publishes the path,
its statistics and the robot standing at the start facing the path.
Clouds (latched, switch them in RViz): /map_cloud_010 (10 cm), /map_cloud_005 (5 cm), /map_cloud_full (every point);
/semantic_cloud (terrain class per 25 cm cell, rgb), /plan, /plan_line, /nav_markers (start, goal, stats), /robot_markers,
/joint_states and TF map->base_link (the robot model comes from robot_state_publisher).
usage: nav_server.py --data <dir with terrain.npz, map_*.npy> --urdf huphy_rviz.urdf --dump world_dump.npz
"""
import argparse
import heapq
import math
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--urdf", required=True)
ap.add_argument("--dump", required=True, help="world_dump.npz: joint angles of a real v11a walking step (Isaac walk)")
ap.add_argument("--pelvis", type=float, default=0.86)
a = ap.parse_args()

T = np.load(f"{a.data}/terrain.npz")
cls, dem, org, cell = T["cls"], T["dem"], T["origin"].astype(float), float(T["cell"])
COST = T["cost_grid"].astype(np.float64)
COST[COST < 0] = np.inf
NAMES = [str(n) for n in T["names"]]
ny, nx = cls.shape
CLS_RGB = np.array([[235, 200, 90], [205, 135, 65], [230, 60, 50], [50, 200, 90]], np.uint32)

D = np.load(a.dump)
root = ET.parse(a.urdf).getroot()
joint_names = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
qmap = dict(zip([str(n) for n in D["joint_names"]], D["qpos"].tolist()))
joint_pos = [float(qmap.get(n, 0.0)) for n in joint_names]


def to_cell(x, y):
    return int((y - org[1]) / cell), int((x - org[0]) / cell)


def nearest_free(rc, r_max=12):
    r0, c0 = rc
    for r in range(r_max + 1):
        win = COST[max(r0 - r, 0):r0 + r + 1, max(c0 - r, 0):c0 + r + 1]
        ok = np.argwhere(np.isfinite(win))
        if len(ok):
            k = ok[np.argmin(np.hypot(ok[:, 0] + max(r0 - r, 0) - r0, ok[:, 1] + max(c0 - r, 0) - c0))]
            return int(k[0] + max(r0 - r, 0)), int(k[1] + max(c0 - r, 0))
    return None


def astar(s, t):
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g = np.full(COST.shape, np.inf)
    g[s] = 0.0
    par = {}
    pq = [(0.0, s)]
    while pq:
        _, u = heapq.heappop(pq)
        if u == t:
            break
        gu = g[u]
        for dr, dc in nb:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < ny and 0 <= v[1] < nx):
                continue
            cv = COST[v]
            if not np.isfinite(cv):
                continue
            nv = gu + math.hypot(dr, dc) * 0.5 * (COST[u] + cv)
            if nv < g[v]:
                g[v] = nv
                par[v] = u
                heapq.heappush(pq, (nv + math.hypot(v[0] - t[0], v[1] - t[1]), v))
    if t != s and t not in par:
        return None
    cells = [t]
    while cells[-1] != s:
        cells.append(par[cells[-1]])
    return np.array(cells[::-1])


def cells_to_path(cells):
    xy = np.stack([org[0] + (cells[:, 1] + 0.5) * cell, org[1] + (cells[:, 0] + 0.5) * cell], 1)
    k = min(9, len(xy) // 2 * 2 + 1)
    if k >= 3:
        pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
        xy = np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(2)], 1)
    r = np.clip(((xy[:, 1] - org[1]) / cell).astype(int), 0, ny - 1)
    c = np.clip(((xy[:, 0] - org[0]) / cell).astype(int), 0, nx - 1)
    z = np.nan_to_num(dem[r, c], nan=0.0)
    return np.column_stack([xy, z])


def cloud(arr4, name):
    m = PointCloud2()
    m.header.frame_id = "map"
    m.height, m.width = 1, len(arr4)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(["x", "y", "z", "rgb"])]
    m.point_step, m.row_step = 16, 16 * len(arr4)
    m.is_bigendian, m.is_dense = False, True
    m.data = np.ascontiguousarray(arr4, np.float32).tobytes()
    print(f"{name}: {len(arr4):,} points", flush=True)
    return m


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


def robot_markers():
    mk = MarkerArray()
    cols = {m_.get("name"): [float(v) for v in m_.find("color").get("rgba").split()]
            for m_ in root.findall("material") if m_.find("color") is not None}
    i = 0
    for link in root.findall("link"):
        for v in link.findall("visual"):
            geom, o = v.find("geometry"), v.find("origin")
            xyz = [float(t) for t in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(t) for t in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
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


class Nav(Node):
    def __init__(self):
        super().__init__("huphy_trail_nav")
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = {k: self.create_publisher(t, "/" + k, lat) for k, t in [
            ("map_cloud_010", PointCloud2), ("map_cloud_005", PointCloud2), ("map_cloud_full", PointCloud2),
            ("semantic_cloud", PointCloud2), ("plan", Path), ("plan_line", MarkerArray), ("nav_markers", MarkerArray),
            ("robot_markers", MarkerArray)]}
        self.p_js = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_start, 10)
        # latched clouds: published once (they are large)
        for tag in ("010", "005", "full"):
            self.pub["map_cloud_" + tag].publish(cloud(np.load(f"{a.data}/map_{tag}.npy"), "map_cloud_" + tag))
        rr, cc = np.nonzero(cls < 4)
        sem = np.column_stack([org[0] + (cc + 0.5) * cell, org[1] + (rr + 0.5) * cell, dem[rr, cc] + 0.05])
        rgb = CLS_RGB[cls[rr, cc]]
        packed = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype(np.uint32).view(np.float32)
        self.pub["semantic_cloud"].publish(cloud(np.column_stack([sem.astype(np.float32), packed]), "semantic_cloud"))
        self.rmk = robot_markers()
        # re-sent every 2 s: RViz keeps a marker's "no transform" error until the marker arrives again after TF exists
        self.create_timer(2.0, lambda: self.pub["robot_markers"].publish(self.rmk))
        self.start = tuple(T["start"].astype(float))
        self.goal = tuple(T["goal"].astype(float))
        self.pw, self.qw = np.array([self.start[0], self.start[1], 0.0]), (1.0, 0.0, 0.0, 0.0)
        self.plan()
        self.create_timer(0.1, self.tick)

    def on_start(self, msg):
        self.start = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.plan()

    def on_goal(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.plan()

    def mk(self, ns, i, typ, rgba, scale, xyz=(0, 0, 0)):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, i, typ, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, xyz)
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = map(float, scale)
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
        return m

    def plan(self):
        t0 = time.time()
        s, g = nearest_free(to_cell(*self.start)), nearest_free(to_cell(*self.goal))
        out = MarkerArray()
        if s is None or g is None:
            self.get_logger().warn("start or goal is not on mapped ground")
            return
        cells = astar(s, g)
        if cells is None:
            self.get_logger().warn("no path")
            return
        path = cells_to_path(cells)
        L = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        pc = cls[cells[:, 0], cells[:, 1]]
        share = {NAMES[k]: float(np.mean(pc == k)) for k in range(4)}
        self.get_logger().info(f"plan {L:.1f} m in {time.time() - t0:.1f} s, " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in share.items()))
        # robot at the start, facing along the path
        j1 = min(8, len(path) - 1)
        yaw = math.atan2(path[j1, 1] - path[0, 1], path[j1, 0] - path[0, 0])
        self.pw = np.array([path[0, 0], path[0, 1], path[0, 2] + a.pelvis])
        self.qw = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        p = Path()
        p.header.frame_id = "map"
        for x, y, z in path:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), float(z + 0.12)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self.pub["plan"].publish(p)
        line = self.mk("plan_line", 0, Marker.LINE_STRIP, (0.2, 1.0, 0.5, 1.0), (0.35, 0, 0))
        line.points = [Point(x=float(x), y=float(y), z=float(z + 0.2)) for x, y, z in path]
        self.pub["plan_line"].publish(MarkerArray(markers=[line]))
        ge = path[-1]
        out.markers += [self.mk("nav", 0, Marker.CYLINDER, (1, 1, 1, 1), (0.1, 0.1, 3.0), (ge[0], ge[1], ge[2] + 1.5)),
                        self.mk("nav", 1, Marker.CYLINDER, (1.0, 0.72, 0.15, 0.7), (2.5, 2.5, 0.06), (ge[0], ge[1], ge[2] + 0.05)),
                        self.mk("nav", 2, Marker.CYLINDER, (0.3, 0.85, 1.0, 0.7), (2.0, 2.0, 0.06), (path[0, 0], path[0, 1], path[0, 2] + 0.05))]
        txt = self.mk("nav", 3, Marker.TEXT_VIEW_FACING, (1.0, 0.85, 0.4, 1.0), (0, 0, 1.6), (ge[0], ge[1], ge[2] + 4.2))
        txt.text = f"GOAL  {L:.0f} m  |  climb {path[:, 2].max() - path[:, 2].min():.1f} m  |  trail {share['TRAIL'] * 100:.0f}%"
        out.markers.append(txt)
        self.pub["nav_markers"].publish(out)

    def tick(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, self.pw)
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = self.qw
        self.tf.sendTransform(t)
        js = JointState()
        js.header.stamp = t.header.stamp
        js.name, js.position = joint_names, joint_pos
        self.p_js.publish(js)


rclpy.init()
rclpy.spin(Nav())
