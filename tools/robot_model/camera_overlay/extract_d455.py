#!/usr/bin/env python3
"""Read the Realsense D455 under the `Pelvis_Camera` component out of the live Fusion document.

READ ONLY: nothing is saved, edited or recoloured in Fusion. Writes, below --out only:
  d455_bodies.json   document name/version, the occurrence transform, per body: path, index, bbox (mm)
  cad_mm/d455.stl    every live D455 body merged, tessellation in the CAD root frame, millimetres
  cad_mm/mount.stl   Pelvis_Camera_Mount (the bracket), same frame (read for reference, not placed in the URDF)

Skipped on purpose: bodies whose bounding box is longer than 300 mm. The vendor model carries
"..._FOV" / "OP_D6" field-of-view solids that span +-800 mm and are not part of the camera.
Same safety limits as tools/robot_model/shell_overlay/extract_parts.py (small scripts, short replies,
adapter output-capture healing).

Usage (mjlab venv):
  FUSION_MCP=http://192.168.20.161:27182/mcp FUSION_MCP_HOST=127.0.0.1:27182 \
    mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/camera_overlay/extract_d455.py --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E  # noqa: E402  (installs the output-capture healing on M.printed)
M, Fetcher = E.M, E.Fetcher

LIST_SCRIPT = r'''
import adsk.core, adsk.fusion, json
def run(_c):
    app = adsk.core.Application.get()
    d = app.activeDocument
    root = adsk.fusion.Design.cast(d.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    out = []
    T = None
    for i in range(allo.count):
        o = allo.item(i)
        n = o.fullPathName
        if not n.startswith('Pelvis_Camera:1'):
            continue
        if n == 'Pelvis_Camera:1+Realsense D455:1':
            T = [round(v, 6) for v in o.transform2.asArray()]
        for j in range(o.bRepBodies.count):
            b = o.bRepBodies.item(j)
            bb = b.boundingBox
            out.append(['/' + n.replace('+', '/'), j, b.name, int(o.isLightBulbOn and b.isLightBulbOn),
                [round(v * 10, 2) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]])
    ver = None
    try:
        ver = d.dataFile.versionNumber
    except Exception:
        pass
    print(json.dumps({'doc': d.name, 'ver': ver, 'mod': d.isModified, 'T_d455_in_pelvis_camera_cm': T, 'b': out}))
'''


def fetch(fx, q, idx):
    for qual in ('Normal', 'Low'):
        try:
            f = fx if qual == 'Normal' else Fetcher(quality='Low')
            v, tri = f.body(q, idx)
            return trimesh.Trimesh(v.reshape(-1, 3) * 10.0, tri.reshape(-1, 3), process=False), qual
        except M.HostWedged:
            raise
        except Exception as e:  # noqa: BLE001
            print(f'  {q[-50:]} {qual} failed ({str(e)[:60]})', flush=True)
    raise SystemExit(f'no mesh for {q}')


def read_mount_props(out):
    """CAD mass properties of Pelvis_Camera_Mount (one body, same read the shell overlay uses) -> merged into d455_bodies.json."""
    M.connect()
    r = E.read_prefix('/Pelvis_Camera:1/Pelvis_Camera_Mount:1', ('Pelvis_Camera_Mount',))
    assert len(r['b']) == 1, r['b']
    b = r['b'][0]
    j = json.load(open(f'{out}/d455_bodies.json'))
    if (r['doc'], r['ver']) != (j['doc'], j['version']):
        raise SystemExit(f'document changed since the snapshot: {r["doc"]} v{r["ver"]} vs {j["doc"]} v{j["version"]}')
    j['mount_props'] = {k: b[k] for k in ('m', 'v', 'c', 'I', 'mat', 'app')}
    j['mount_props']['units'] = 'm kg (CAD material), v cm3, c cm, I kg*cm2 about the CAD root origin [xx,yy,zz,xy,yz,xz]'
    json.dump(j, open(f'{out}/d455_bodies.json', 'w'), indent=1)
    print('mount props:', j['mount_props'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--props-only', action='store_true', help='only read the mount bracket mass properties into an existing snapshot')
    a = ap.parse_args()
    if a.props_only:
        return read_mount_props(a.out)
    os.makedirs(f'{a.out}/cad_mm', exist_ok=True)
    M.connect()
    r = json.loads(M.printed(LIST_SCRIPT).strip())
    bodies = [dict(q=q, idx=i, name=n, live=l, bb=bb) for q, i, n, l, bb in r['b']]
    for b in bodies:
        b['long_mm'] = round(max(np.array(b['bb'][3:]) - np.array(b['bb'][:3])), 1)
    cam = [b for b in bodies if '/Realsense D455:1' in b['q'] and b['live'] and b['long_mm'] <= 300]
    mount = [b for b in bodies if b['q'].endswith('/Pelvis_Camera_Mount:1') and b['live']]
    skipped = [b for b in bodies if b not in cam and b not in mount]
    print(f'{r["doc"]} v{r["ver"]} (unsaved edits: {r["mod"]}): {len(cam)} D455 bodies, '
          f'{len(mount)} mount bodies, {len(skipped)} skipped', flush=True)
    fx = Fetcher(quality='Normal')
    for tag, group in (('d455', cam), ('mount', mount)):
        meshes = []
        for b in group:
            m, qual = fetch(fx, b['q'], b['idx'])
            b['tris'], b['quality'] = int(len(m.faces)), qual
            meshes.append(m)
            print(f'  {tag}: {b["q"][-46:]:46s} {b["tris"]:6d} tris ({qual})', flush=True)
        trimesh.util.concatenate(meshes).export(f'{a.out}/cad_mm/{tag}.stl')
    json.dump(dict(doc=r['doc'], version=r['ver'], modified_unsaved=r['mod'],
                   T_d455_in_pelvis_camera_cm=r['T_d455_in_pelvis_camera_cm'],
                   units='bb mm in the CAD root frame', d455=cam, mount=mount, skipped=skipped),
              open(f'{a.out}/d455_bodies.json', 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
