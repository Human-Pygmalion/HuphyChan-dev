#!/usr/bin/env python3
"""Does the PLA mount bracket sit where the camera sits? The bracket and the D455 are placed by two different transforms
(bracket: CAD -> base_link; camera: CAD -> d455_link -> URDF chain -> base_link). Their gap must not depend on which way it is measured.
Also: bracket mass check (volume x printed density) and which part of the bracket touches the camera.  usage (mjlab venv): mount_check.py"""
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
A = f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2'
SNAP = f'{REPO}/tools/robot_model/camera_overlay/snapshot_20261028'
URDF = f'{A}/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_d455_loop.urdf'
BASE_CAD = np.array([0.0, 70.0, 60.0])
R_SIM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def gap(a, b):
    d = cKDTree(b).query(a)[0]
    return float(d.min() * 1000), float(np.percentile(d, 1) * 1000)       # mm (closest point, 1st percentile)


cad_cam = trimesh.load(f'{SNAP}/cad_mm/d455.stl', process=False)
cad_mnt = trimesh.load(f'{SNAP}/cad_mm/mount.stl', process=False)
cam_cad = np.asarray(cad_cam.sample(200000)) if hasattr(cad_cam, 'sample') else cad_cam.vertices
mnt_cad = np.asarray(cad_mnt.sample(200000))
cam_cad = np.asarray(trimesh.Trimesh(cad_cam.vertices, cad_cam.faces, process=False).sample(200000))
g_cad = cKDTree(mnt_cad).query(cam_cad)[0]

# the same two meshes through the URDF: bracket mesh is already in base_link; camera mesh is in d455_link -> chain to base_link
r = ET.parse(URDF).getroot()
J = {j.find('child').get('link'): (j.find('parent').get('link'), np.array(j.find('origin').get('xyz').split(), float),
                                   np.array(j.find('origin').get('rpy').split(), float)) for j in r.findall('joint')}
ch, link = [], 'd455_link'
while link != 'base_link':
    p, x, y = J[link]
    ch.append((x, y))
    link = p
R, t = np.eye(3), np.zeros(3)
for x, y in reversed(ch):
    t = t + R @ x
    R = R @ Rot.from_euler('xyz', y).as_matrix()
# 1) pure transform consistency, point by point on the full CAD camera mesh:
#    route A  CAD -> base_link directly (the way the bracket is placed)      route B  CAD -> d455_link -> URDF chain -> base_link
rep0 = json.load(open(f'{A}/d455_overlay_report.json'))
ax = rep0['camera_axes_in_cad']
M = np.stack([ax['forward'], ax['left'], ax['up']])
p_link = np.array(rep0['d455_link_in_cad_mm'])
V = np.asarray(cad_cam.vertices)
route_a = (V - BASE_CAD) @ R_SIM.T / 1000.0
route_b = (((V - p_link) @ M.T) / 1000.0) @ R.T + t
transform_diff_mm = float(np.abs(route_a - route_b).max() * 1000)
# 2) closest gap camera <-> bracket measured through the URDF, from the same full CAD mesh (route B points)
mnt_base = np.asarray(trimesh.load(f'{A}/meshes_d455/pelvis_camera_mount.stl', process=False).sample(200000))
cam_base = np.asarray(trimesh.Trimesh(route_b, cad_cam.faces, process=False).sample(200000))
g_urdf = cKDTree(mnt_base).query(cam_base)[0]

rep = json.load(open(f'{A}/d455_overlay_report.json'))['mount']
out = dict(
    camera_mesh_transform_route_difference_mm=round(transform_diff_mm, 6),
    closest_gap_camera_to_bracket_mm=dict(measured_in_CAD=round(float(g_cad.min()), 3), measured_through_URDF=round(float(g_urdf.min() * 1000), 3)),
    p1_gap_mm=dict(CAD=round(float(np.percentile(g_cad, 1)), 3), URDF=round(float(np.percentile(g_urdf, 1) * 1000), 3)),
    bracket_mass_g=round(rep['mass_kg'] * 1000, 2), bracket_cad_mass_g=round(rep['cad_mass_kg'] * 1000, 1),
    bracket_volume_cm3=rep['volume_cm3'], density_g_cm3=rep['density_g_cm3'],
    mass_check_g=round(rep['volume_cm3'] * rep['density_g_cm3'], 2))
json.dump(out, open(f'{REPO}/docs/img/d455/mount_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
