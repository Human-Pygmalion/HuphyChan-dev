#!/usr/bin/env python3
"""READ ONLY: path, body name, mass, volume, material, bulb of the bodies the aluminium round 2 names (and anything called universal / revised)."""
import json, os, sys
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M = E.M
SRC = r'''
import adsk.core, adsk.fusion, json
KEYS = ('hiproll2yawa', 'hiproll2yawb', 'yaw2kneeplate', 'knee2ankle_idle', 'knee2ankle_rotor', 'universal', 'revised', 'crank_a', 'crank_b')
def run(_c):
    root = adsk.fusion.Design.cast(adsk.core.Application.get().activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    out = []
    for i in range(allo.count):
        o = allo.item(i); n = o.fullPathName
        for j in range(o.bRepBodies.count):
            b = o.bRepBodies.item(j)
            if any(k in (n + '/' + b.name).lower().replace('-', '_') for k in KEYS) and 'screw' not in n.lower():
                p = b.physicalProperties
                out.append(['/' + n.replace('+', '/'), b.name, round(p.mass * 1000, 2), round(p.volume, 3), b.material.name if b.material else '?', int(o.isLightBulbOn and b.isLightBulbOn)])
    print(json.dumps(out))
'''
M.connect()
for r in json.loads(M.printed(SRC).strip()):
    print(r)
