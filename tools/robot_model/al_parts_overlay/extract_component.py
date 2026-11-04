#!/usr/bin/env python3
"""READ ONLY: every body under one Fusion component (all depths, fasteners included, hidden ones flagged) with its mass properties.
usage: extract_component.py <occurrence path, e.g. /Joints_UnderBody:1/Knee2Ankle:1> <out.json>
Same schema as shell_overlay/extract_parts.py (m kg, v cm3, c cm, I kg cm2 about the CAD root origin, bb mm). Small reply per call: bodies are read in chunks by child occurrence."""
import json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
prefix, out = sys.argv[1], sys.argv[2]
M.connect()
# children of the component, one level (names only, tiny reply)
SRC = r'''
import adsk.core, adsk.fusion, json
def run(_c):
    root = adsk.fusion.Design.cast(adsk.core.Application.get().activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    P = PREFIX
    stack = [(root.occurrences.item(i), '') for i in range(root.occurrences.count)]
    node = None
    while stack:
        o, p = stack.pop()
        q = p + '/' + o.name
        if q == P:
            node = o; break
        if P.startswith(q + '/'):
            for i in range(o.childOccurrences.count):
                stack.append((o.childOccurrences.item(i), q))
    kids = [[node.childOccurrences.item(i).name, node.childOccurrences.item(i).childOccurrences.count] for i in range(node.childOccurrences.count)]
    print(json.dumps({'own_bodies': node.bRepBodies.count, 'kids': kids}))
'''.replace('PREFIX', json.dumps(prefix))
info = json.loads(M.printed(SRC).strip())
print('component', prefix, 'own bodies', info['own_bodies'], 'child occurrences', len(info['kids']), flush=True)
rows, doc = [], None
# own bodies (names unknown here -> read the component prefix with the exact-match rule by passing no names would walk all kids; so ask by kid)
for kid, nchild in [(None, 0)] + [(k, n) for k, n in info['kids']]:
    pre = prefix if kid is None else f'{prefix}/{kid}'
    if kid is None:
        # own bodies only: E.read_prefix with NAMES empty matches startswith(PRE) i.e. the whole subtree; so read the subtree of each kid and the own bodies by difference
        continue
    r = E.read_prefix(pre)
    doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
    assert (r['doc'], r['ver']) == (doc['doc'], doc['ver']), 'document changed while reading'
    rows += r['b']
    print(f'  {kid}: {len(r["b"])} bodies', flush=True)
# own bodies of the component itself: whole subtree minus the kids' bodies
r = E.read_prefix(prefix)
doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
seen = {(b['q'], b['idx']) for b in rows}
own = [b for b in r['b'] if (b['q'], b['idx']) not in seen]
rows += own
print(f'  own (not in a child occurrence): {len(own)} bodies; total {len(rows)}', flush=True)
json.dump(dict(doc=doc['doc'], version=doc['ver'], modified_unsaved=doc['mod'], prefix=prefix, bodies=rows), open(out, 'w'))
print('wrote', out)
