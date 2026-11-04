#!/usr/bin/env python3
"""READ ONLY: list the bodies of Joints_UnderBody/Ankle2Feet (and any occurrence whose name has 'crank') with mass, volume, material, bulb."""
import json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
SRC = r'''
import adsk.core, adsk.fusion, json
def run(_c):
    root = adsk.fusion.Design.cast(adsk.core.Application.get().activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    out = []
    for i in range(allo.count):
        o = allo.item(i); n = o.fullPathName
        if not (n.startswith('Joints_UnderBody:1+Ankle2Feet:1') or 'rank' in n.lower()):
            continue
        for j in range(o.bRepBodies.count):
            b = o.bRepBodies.item(j); p = b.physicalProperties
            if p.volume < 4.0 and 'rank' not in (n + b.name).lower():
                continue
            out.append([n.split('+')[-1][:40], b.name, round(p.mass * 1000, 2), round(p.volume, 3), b.material.name if b.material else '?', int(o.isLightBulbOn and b.isLightBulbOn)])
    print(json.dumps(out))
'''
M.connect()
for row in json.loads(M.printed(SRC).strip()):
    print(row)
