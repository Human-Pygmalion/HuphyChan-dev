#!/usr/bin/env python3
"""READ ONLY: mass properties (CAD root frame, schema of shell_overlay/extract_parts.py) of the parts converted to aluminium in round 2.
Writes <out>/round2_cad.json."""
import argparse, json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
J = '/Joints_UnderBody:1'
PARTS = [  # (id, occurrence path, body name)
    ('HipRoll2YawA', f'{J}/HipRoll2Yaw:1/HipRoll2YawA:1', 'HipRoll2YawA'),
    ('HipRoll2YawB', f'{J}/HipRoll2Yaw:1/HipRoll2YawB:1', 'HipRoll2YawB'),
    ('Yaw2KneePlateA', f'{J}/HipYaw2Knee:1', 'Yaw2KneePlateA'),
    ('Knee2Ankle_IdleSide_1', f'{J}/Knee2Ankle:1/Knee2Ankle_IdleSide_1:1', 'Knee2Ankle_IdleSide_1'),
    ('Knee2Ankle_RotorSide_1', f'{J}/Knee2Ankle:1/Knee2Ankle_RotorSide_1:1', 'Knee2Ankle_RotorSide_1'),
    ('AnkleFeetPillowA', f'{J}/Ankle2Feet:1', 'AnkleFeetPillowA'),
    ('AnkleFeetPillowB', f'{J}/Ankle2Feet:1/AnkleFeetPillowB:1', 'AnkleFeetPillowB'),
    ('AnkleUniversalJointCore', f'{J}/Ankle2Feet:1/AnkleUniversalJointCore:1/AnkleUniversalJointCore-Revised:1', 'AnkleUniversalJointCore'),
]
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
M.connect()
parts, doc = {}, None
for pid, occ, name in PARTS:
    r = E.read_prefix(occ, (name,))
    doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
    assert (r['doc'], r['ver']) == (doc['doc'], doc['ver']), 'document changed while reading'
    assert len(r['b']) == 1, (pid, [b['name'] for b in r['b']])
    parts[pid] = r['b'][0]
    b = parts[pid]
    print(f"{pid:24s} m={b['m']*1000:8.3f} g  v={b['v']:8.4f} cm3  rho={b['m']*1000/b['v']:.4f}  {b['mat']}  live={b['live']}", flush=True)
json.dump(dict(doc=doc['doc'], version=doc['ver'], modified_unsaved=doc['mod'],
               units='kg, cm3, com cm, inertia kg*cm2 about the CAD root origin [xx,yy,zz,xy,yz,xz], bb mm', parts=parts),
          open(f'{a.out}/round2_cad.json', 'w'), indent=1)
print('document', doc)
