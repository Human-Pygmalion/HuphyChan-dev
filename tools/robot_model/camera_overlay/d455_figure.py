"""Draw the D455 on the LegOnly pelvis from the generated URDF (side / front / top, base_link frame, metres).
usage: d455_figure.py [--urdf PATH] [--out PNG]   (mjlab .venv python)"""
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

A = '/home/syaro/MikuchanRemote/Human-Pygmalion/pygmalion_locomotion/assets/pygmalion_v2'
ap = argparse.ArgumentParser()
ap.add_argument('--urdf', default=f'{A}/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_d455_loop.urdf')
ap.add_argument('--out', default='/home/syaro/MikuchanRemote/Human-Pygmalion/docs/img/d455/pelvis_side_front_top.png')
a = ap.parse_args()
r = ET.parse(a.urdf).getroot()
J = {j.find('child').get('link'): (j.find('parent').get('link'), np.array(j.find('origin').get('xyz').split(), float),
                                   np.array(j.find('origin').get('rpy').split(), float)) for j in r.findall('joint')}


def fk(link):
    chain = []
    while link != 'base_link':
        p, x, y = J[link]
        chain.append((x, y))
        link = p
    R, t = np.eye(3), np.zeros(3)
    for x, y in reversed(chain):
        t = t + R @ x
        R = R @ Rot.from_euler('xyz', y).as_matrix()
    return t, R


def cloud(rel):
    return trimesh.load(f'{A}/{rel}', process=False).vertices[::3]


pelvis = cloud('meshes_v30_armfix/pelvis.stl')
mount_pts = cloud('meshes_d455/pelvis_camera_mount.stl')     # PLA bracket, already in the base_link frame
t, R = fk('d455_link')
cam = cloud('meshes_d455/d455.stl') @ R.T + t
views = [('SIDE (robot forward = right)', 0, 2, 'x forward [m]', 'z up [m]'),
         ('FRONT (robot left = right)', 1, 2, 'y left [m]', 'z up [m]'),
         ('TOP', 1, 0, 'y left [m]', 'x forward [m]')]
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
for ax, (title, h, v, xl, yl) in zip(axs, views):
    ax.scatter(pelvis[:, h], pelvis[:, v], s=0.3, c='0.8')
    ax.scatter(mount_pts[:, h], mount_pts[:, v], s=0.4, c='tab:brown', label='PLA mount bracket (fixed to base_link)')
    ax.scatter(cam[:, h], cam[:, v], s=0.4, c='tab:blue')
    ax.scatter([0], [0], s=160, c='red', zorder=5, label='base_link origin')
    ax.scatter([t[h]], [t[v]], s=60, c='green', marker='^', zorder=5, label='d455_link (Left IR)')
    ax.scatter([t[h] + 0.08 * R[h, 0]], [t[v] + 0.08 * R[v, 0]], s=0)
    for col, vec, nm in (('green', R[:, 0], 'view dir (+x)'), ('orange', R[:, 2], 'camera up (+z)')):
        ax.annotate('', xy=(t[h] + 0.08 * vec[h], t[v] + 0.08 * vec[v]), xytext=(t[h], t[v]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2), annotation_clip=False)
    ax.autoscale_view(); ax.margins(0.12); ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
axs[0].legend(loc='lower left', fontsize=8)
fig.suptitle('Realsense D455 + PLA bracket on the LegOnly pelvis (generated URDF): looks forward and 60 deg DOWN. Green arrow = view direction, orange = camera up')
fig.tight_layout(); fig.savefig(a.out, dpi=100); print('saved', a.out)
