#!/usr/bin/env python3
"""READ ONLY: read named leg parts out of the live Fusion document (any body under Joints_UnderBody, found by body name).
usage: extract_by_name.py --out DIR --names HipRoll2YawA HipRoll2YawB ...
Writes DIR/parts_cad.json: per part the live path, CAD group, volume, CAD mass/material, COM, inertia about the CAD root origin (same schema as extract_parts.py).
Two small Fusion calls per part group (a name search, then the exact read); nothing is edited in Fusion."""
import argparse, json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
FIND = r'''
import adsk.core, adsk.fusion, json
NAMES = __NAMES__
def run(_c):
    root = adsk.fusion.Design.cast(adsk.core.Application.get().activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    out = []
    for i in range(allo.count):
        o = allo.item(i); n = o.fullPathName
        if not n.startswith('Joints_UnderBody:1'):
            continue
        for j in range(o.bRepBodies.count):
            if o.bRepBodies.item(j).name in NAMES:
                out.append(['/' + n.replace('+', '/'), j, o.bRepBodies.item(j).name])
    print(json.dumps(out))
'''
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--names', nargs='+', required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
M.connect()
found = json.loads(M.printed(FIND.replace('__NAMES__', json.dumps(a.names))).strip())
by = {}
for path, idx, name in found:
    by.setdefault(name, []).append(path)
bad = {n: by.get(n, []) for n in a.names if len(by.get(n, [])) != 1}
if bad:
    sys.exit(f'each name must match exactly one body (the CAD holds one leg): {bad}')
parts, doc = {}, None
for name in a.names:
    path = by[name][0]
    r = E.read_prefix(path, (name,))
    doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
    assert (r['doc'], r['ver']) == (doc['doc'], doc['ver']), 'document changed while reading'
    assert len(r['b']) == 1, (path, name)
    b = r['b'][0]
    b['path'] = path
    b['group'] = path.split('/')[2]
    parts[name] = b
    print(f"{name:28s} {b['group']:16s} V={b['v']:8.3f} cm3  CAD {b['m']*1000:8.2f} g {b['mat']}  live={b['live']}")
json.dump(dict(doc=doc['doc'], version=doc['ver'], modified_unsaved=doc['mod'], parts=parts), open(f'{a.out}/parts_cad.json', 'w'), indent=1)
print('document', doc)
