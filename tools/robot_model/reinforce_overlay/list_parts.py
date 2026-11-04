#!/usr/bin/env python3
"""READ ONLY: list Fusion occurrences whose path matches Reinforce / HipRoll2Yaw / Knee2Ankle / Pelvis_Camera.
Small script, short reply (connector limits: see shell_overlay/extract_parts.py)."""
import json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E  # installs output-capture healing
M = E.M
SRC = r'''
import adsk.core, adsk.fusion, json
KEYS = ("reinforce", "hiproll2yaw", "knee2ankle", "pelvis_camera")
def run(_c):
    app = adsk.core.Application.get()
    d = app.activeDocument
    root = adsk.fusion.Design.cast(d.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    out = []
    for i in range(allo.count):
        o = allo.item(i)
        n = o.fullPathName
        if any(k in n.lower() for k in KEYS) and 'screw' not in n.lower() and 'nut' not in n.lower() and 'washer' not in n.lower():
            out.append([n.replace('+', '/'), o.bRepBodies.count, int(o.isLightBulbOn)])
    print(json.dumps({'doc': d.name, 'n': allo.count, 'hit': out}))
'''
M.connect()
txt = M.printed(SRC).strip()
r = json.loads(txt)
print(r['doc'], 'occurrences', r['n'])
for p, nb, on in r['hit']:
    print(f'{on} {nb:2d} {p}')
