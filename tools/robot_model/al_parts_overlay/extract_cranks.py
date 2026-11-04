#!/usr/bin/env python3
"""READ ONLY: read the four ankle crank bodies (Crank_A, Crank_B and their covers) out of the live Fusion document.
Writes <out>/cranks_cad.json (mass properties about the CAD root origin, same schema as shell_overlay/extract_parts.py)."""
import argparse, json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
M.connect()
A = '/Joints_UnderBody:1/Ankle2Feet:1'
bodies, doc = [], None
for occ, name in ((f'{A}/Crank_A:1', 'Crank_A'), (f'{A}/Crank_B:1', 'Crank_B'), (A, 'Crank_A_Cover'), (A, 'Crank_B_Cover')):
    r = E.read_prefix(occ, (name,))
    doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
    assert (r['doc'], r['ver']) == (doc['doc'], doc['ver']), 'document changed while reading'
    assert len(r['b']) == 1, (occ, name, [b['name'] for b in r['b']])
    bodies += r['b']
r = dict(doc, b=bodies)
for b in r['b']:
    print(f"{b['name']:14s} m={b['m']*1000:8.3f} g  v={b['v']:8.4f} cm3  rho={b['m']*1000/b['v']:.4f} g/cm3  {b['mat']}  live={b['live']}")
json.dump(dict(doc=doc['doc'], version=doc['ver'], modified_unsaved=doc['mod'],
               units='kg, cm3, com cm, inertia kg*cm2 about the CAD root origin [xx,yy,zz,xy,yz,xz]',
               parts={b['name']: b for b in r['b']}), open(f'{a.out}/cranks_cad.json', 'w'), indent=1)
print('document', doc)
