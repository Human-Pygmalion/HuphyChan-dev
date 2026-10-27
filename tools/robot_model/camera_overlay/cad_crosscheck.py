#!/usr/bin/env python3
"""Cross-check the d455_link frame we built from the CAD against two independent sources.

  datasheet  Intel D400 datasheet (via docs/External notes): Left IR is the depth origin; RGB 59 mm, projector 37.5 mm,
             Right IR 95 mm to its right; body 124 x 29 x 26 mm; Z = 0 plane 4.65 mm behind the front plate, back plane 21.35 mm behind
             the origin; tripod hole and USB-C plug positions; IMU at (-16.02, -30.22, +7.4) mm.
  realsense-ros  the official `_d455.urdf.xacro` + `d455.stl` (reference/, downloaded from Intel's repository): mesh placed with
             origin (0.00465, -0.0475, 0) rpy (pi/2, 0, pi/2), scale 0.001 in the depth frame.

What is measured, all in OUR d455_link frame (x forward, y left, z up, origin at the Left IR centre):
  1. extent of our CAD mesh                       vs  datasheet box
  2. lens centres found in OUR CAD front face     vs  datasheet offsets
  3. USB-C receptacle and IMU candidates in OUR CAD vs datasheet / xacro numbers
  4. Intel's official mesh placed by Intel's xacro: distance from its OUTER surface to OUR CAD mesh (should be ~0), with the same distance
     after deliberately moving it by 1 mm / 3 mm / 1 degree as controls (the check must be able to fail).
Writes docs/img/d455/cad_crosscheck.png and .json.   usage (mjlab venv): cad_crosscheck.py
"""
import json
import math
import os

import matplotlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = '/home/syaro/MikuchanRemote/Human-Pygmalion'
HERE = f'{REPO}/tools/robot_model/camera_overlay'
SNAP = f'{HERE}/snapshot_20261028'
rep = json.load(open(f'{REPO}/pygmalion_locomotion/assets/pygmalion_v2/d455_overlay_report.json'))
J = json.load(open(f'{SNAP}/d455_bodies.json'))

fwd, left, up = (np.array(rep['camera_axes_in_cad'][k]) for k in ('forward', 'left', 'up'))
p_link = np.array(rep['d455_link_in_cad_mm'])
M = np.stack([fwd, left, up])                       # CAD mm -> d455_link mm: M @ (v - p_link)


def to_link(v):
    return (np.asarray(v) - p_link) @ M.T


cad = trimesh.load(f'{SNAP}/cad_mm/d455.stl', process=False)
V = to_link(cad.vertices)
out = {}

# 1. box ------------------------------------------------------------------------------------------------------------
lo, hi = V.min(0), V.max(0)
out['extent_mm'] = dict(x=[round(lo[0], 2), round(hi[0], 2)], y=[round(lo[1], 2), round(hi[1], 2)], z=[round(lo[2], 2), round(hi[2], 2)])
out['datasheet_box_mm'] = dict(x=[-21.35, 4.65], y=[-109.5, 14.5], z=[-14.5, 14.5])
out['extent_error_mm'] = round(float(np.abs(np.array([lo, hi]) - np.array([[-21.35, -109.5, -14.5], [4.65, 14.5, 14.5]])).max()), 2)

# 2. front windows ---------------------------------------------------------------------------------------------------
# The front face has four windows. Their glass outlines are found as connected components of the outermost 4.65 - 3.0 mm slab. The RGB window
# and the projector window (a ring split in two halves) are centred on their optics; the two IR windows are wider than the lens and sit
# symmetric about the body centre, so for them only the MIDPOINT of the pair is meaningful (= body centre, y = -47.5).
from scipy import ndimage as ndi  # noqa: E402
slab = V[V[:, 0] > 3.0]
cell = 0.25
g = np.zeros((int(130 / cell), int(40 / cell)), bool)
g[((slab[:, 1] + 115) / cell).astype(int).clip(0, g.shape[0] - 1), ((slab[:, 2] + 20) / cell).astype(int).clip(0, g.shape[1] - 1)] = True
lab, n = ndi.label(ndi.binary_dilation(g, iterations=6))
comps = []
for k in range(1, n + 1):
    ys, zs = np.where(lab == k)
    if len(ys) < 400:
        continue
    comps.append((ys.min() * cell - 115, ys.max() * cell - 115, zs.min() * cell - 20, zs.max() * cell - 20))
comps.sort()
cy = lambda c: (c[0] + c[1]) / 2
ir_right, ir_left = comps[0], comps[-1]
mid = [c for c in comps[1:-1]]
rgb = min(mid, key=lambda c: abs(cy(c) + 59.0))
proj = [c for c in mid if c is not rgb and -52 < cy(c) < -25]
proj_c = (min(c[0] for c in proj) + max(c[1] for c in proj)) / 2
lens = {'RGB window centre': dict(expected_y_mm=-59.0, found_y_mm=round(cy(rgb), 2)),
        'IR projector window centre (2 halves)': dict(expected_y_mm=-37.5, found_y_mm=round(proj_c, 2)),
        'midpoint of the two IR windows': dict(expected_y_mm=-47.5, found_y_mm=round((cy(ir_left) + cy(ir_right)) / 2, 2)),
        'Left IR window centre (wider than the lens)': dict(expected_y_mm=0.0, found_y_mm=round(cy(ir_left), 2)),
        'Right IR window centre (wider than the lens)': dict(expected_y_mm=-95.0, found_y_mm=round(cy(ir_right), 2))}
for v in lens.values():
    v['error_mm'] = round(v['found_y_mm'] - v['expected_y_mm'], 2)
out['front_windows'] = lens
out['front_window_boxes_yz_mm'] = [[round(x, 1) for x in c] for c in comps]

# 3. USB-C and IMU --------------------------------------------------------------------------------------------------
small = {}
for b in J['d455']:
    bb = np.array(b['bb'])
    ctr = to_link((bb[:3] + bb[3:]) / 2)
    small[b['q'].split('/')[-1] + f"#{b['idx']}"] = ctr
usb = small['10137062C_1_1_1:1#0']
out['usb_c_centre_link_mm'] = [round(float(x), 2) for x in usb]
out['usb_c_expected_y_mm'] = round(-47.5 - 37.01, 2)       # tripod centre y = -47.5, plug 37.01 mm to the camera's right (xacro / datasheet)
out['usb_c_is_on_underside'] = bool(usb[2] < -3.0)
imu_exp = np.array([-16.02, -30.22, 7.4])
cand = sorted(((float(np.linalg.norm(v - imu_exp)), k, [round(float(x), 1) for x in v]) for k, v in small.items()))[:3]
out['imu_expected_mm'] = imu_exp.tolist()
out['imu_nearest_cad_parts'] = [dict(dist_mm=round(d, 1), part=k, centre_mm=c) for d, k, c in cand]

# 4. Intel's mesh + xacro vs ours, ICP -------------------------------------------------------------------------------
off = trimesh.load(f'{HERE}/reference/d455.stl', process=False)
Rm = Rot.from_euler('xyz', [math.pi / 2, 0, math.pi / 2]).as_matrix()
OV = (off.vertices * 0.001 @ Rm.T + np.array([0.00465, -0.0475, 0.0])) * 1000.0      # official mesh in the depth frame, mm
olo, ohi = OV.min(0), OV.max(0)
out['official_mesh_extent_mm'] = dict(x=[round(olo[0], 2), round(ohi[0], 2)], y=[round(olo[1], 2), round(ohi[1], 2)], z=[round(olo[2], 2), round(ohi[2], 2)])
ref_pts = trimesh.Trimesh(OV, off.faces, process=False).sample(120000)       # points on Intel's smooth outer surface
tree = cKDTree(trimesh.Trimesh(V, cad.faces, process=False).sample(700000))      # dense points ON the surfaces of our full vendor assembly
# ICP was tried first and rejected: our CAD carries screws, boards and cables inside the housing, which the smooth outer mesh does not
# have, so ICP slides along the flat faces to a wrong optimum (it reported 7 deg of twist). One-sided distance is robust: every point of
# Intel's OUTER surface must lie on a surface of our assembly, whatever is inside it.
def gap(shift=(0, 0, 0), yaw_pitch_roll_deg=(0, 0, 0), mirror=False):
    R = Rot.from_euler('zyx', yaw_pitch_roll_deg, degrees=True).as_matrix()
    q = ref_pts @ R.T + np.array(shift)
    if mirror:
        q = q * np.array([1, -1, 1]) + np.array([0, -95.0, 0])        # left-right mirror about the body centre y = -47.5
    d = tree.query(q)[0]
    return dict(median_mm=round(float(np.median(d)), 3), p90_mm=round(float(np.percentile(d, 90)), 3))
out['official_outer_surface_to_our_cad'] = dict(
    as_placed_by_xacro=gap(),
    control_if_shifted_1mm_forward=gap((1, 0, 0)),
    control_if_shifted_3mm_left=gap((0, 3, 0)),
    control_if_turned_1deg_about_up=gap(yaw_pitch_roll_deg=(1, 0, 0)),
    control_if_mirrored_left_right=gap(mirror=True),
    control_if_turned_upside_down=gap(yaw_pitch_roll_deg=(0, 0, 180)))
json.dump(out, open(f'{REPO}/docs/img/d455/cad_crosscheck.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# figure -------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={'width_ratios': [1.3, 0.8, 1.3]})
a0 = ax[0]
a0.scatter(slab[:, 1], slab[:, 2], s=0.3, c='k')
for n, d in lens.items():
    a0.axvline(d['expected_y_mm'], c='tab:red', lw=1.2, ls='--')
    a0.scatter([d['found_y_mm']], [0], s=60, c='tab:blue', marker='x', zorder=5)
    a0.annotate(n.split(' (')[0].replace(' window centre', '').replace('midpoint of the two IR windows', 'IR-pair midpoint'), (d['expected_y_mm'], 17.5), ha='center', fontsize=6.5)
for c in comps:
    a0.add_patch(plt.Rectangle((c[0], c[2]), c[1] - c[0], c[3] - c[2], fill=False, ec='tab:green', lw=1))
a0.set_aspect('equal'); a0.invert_xaxis(); a0.set_xlabel('y [mm]  (Left IR = 0; the others lie to its right)'); a0.set_ylabel('z [mm]')
a0.set_title('Front face of our CAD mesh\n green box = window found, blue x = its centre, red dashed = datasheet', fontsize=9); a0.grid(alpha=.3)
a1, a2 = ax[1], ax[2]
for axx, (h, v, xl, yl, ttl) in zip((a1, a2), ((0, 2, 'x forward [mm]', 'z up [mm]', 'SIDE'), (1, 2, 'y left [mm]', 'z up [mm]', 'FRONT'))):
    axx.scatter(OV[::6, h], OV[::6, v], s=0.2, c='tab:orange', label="Intel's official mesh + xacro frame")
    axx.scatter(V[::25, h], V[::25, v], s=0.2, c='tab:blue', label='our CAD mesh in d455_link')
    axx.scatter([0], [0], s=90, c='red', marker='+', zorder=5)
    axx.set_aspect('equal'); axx.set_xlabel(xl); axx.set_ylabel(yl); axx.set_title(ttl + ' overlay (red + = d455_link origin)'); axx.grid(alpha=.3)
a2.legend(loc='upper left', fontsize=7, markerscale=20)
g = out['official_outer_surface_to_our_cad']
fig.suptitle(f"D455 frame cross-check: extent error vs datasheet {out['extent_error_mm']} mm; Intel outer surface to our CAD: median gap {g['as_placed_by_xacro']['median_mm']} mm "
             f"(1 mm shift would give {g['control_if_shifted_1mm_forward']['median_mm']} mm)")
fig.tight_layout(); fig.savefig(f'{REPO}/docs/img/d455/cad_crosscheck.png', dpi=100)
print('saved figure')
