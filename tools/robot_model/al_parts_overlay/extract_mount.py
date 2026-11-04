#!/usr/bin/env python3
"""READ ONLY: (1) HipRoll2Yaw_Mount mass properties -> <out>/mount_cad.json; (2) print the occurrences under Joints_UnderBody (one level) and every
occurrence / body whose name contains HipRoll2Yaw, Crank or Mount, with material, volume, mass, bulb, so one can see whether the CAD holds one leg or two."""
import argparse, json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
SRC = r'''
import adsk.core, adsk.fusion, json
KEYS = ('hiproll2yaw', 'piproll2yaw', 'crank', 'mount')
def run(_c):
    root = adsk.fusion.Design.cast(adsk.core.Application.get().activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    top, rows = [], []
    for i in range(root.occurrences.count):
        o = root.occurrences.item(i)
        top.append([o.name, o.childOccurrences.count, int(o.isLightBulbOn)])
    for i in range(allo.count):
        o = allo.item(i); n = o.fullPathName
        if not o.fullPathName.startswith('Joints_UnderBody:1') or o.fullPathName.count('+') > 2:
            continue
        for j in range(o.bRepBodies.count):
            b = o.bRepBodies.item(j)
            if any(k in (n + b.name).lower() for k in KEYS) and 'screw' not in (n + b.name).lower() and 'bolt' not in (n + b.name).lower():
                p = b.physicalProperties
                rows.append([n.split('+', 1)[-1][-48:], b.name, round(p.mass * 1000, 2), round(p.volume, 3), b.material.name if b.material else '?', int(o.isLightBulbOn and b.isLightBulbOn)])
    print(json.dumps({'top': top, 'rows': rows}))
'''
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
M.connect()
r = E.read_prefix('/Joints_UnderBody:1/HipRoll2Yaw:1/HipRoll2Yaw_Mount:1', ('HipRoll2Yaw_Mount',))
assert len(r['b']) == 1
b = r['b'][0]
print(f"HipRoll2Yaw_Mount m={b['m']*1000:.3f} g v={b['v']:.4f} cm3 rho={b['m']*1000/b['v']:.4f} {b['mat']} live={b['live']}")
json.dump(dict(doc=r['doc'], version=r['ver'], modified_unsaved=r['mod'], part=b), open(f'{a.out}/mount_cad.json', 'w'), indent=1)
d = json.loads(M.printed(SRC).strip())
print('root-level occurrences:', d['top'])
for row in d['rows']:
    print(row)
