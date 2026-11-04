#!/usr/bin/env python3
"""READ ONLY: compare the live Realsense D455 occurrence transform and its bodies' bounding box with the 2026-10-28 snapshot."""
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
    T = None; lo = [1e9]*3; hi = [-1e9]*3
    for i in range(allo.count):
        o = allo.item(i); n = o.fullPathName
        if n == 'Pelvis_Camera:1+Realsense D455:1':
            T = [round(v, 6) for v in o.transform2.asArray()]
        if n.startswith('Pelvis_Camera:1+Realsense D455:1+MD0645501') and o.bRepBodies.count:
            for j in range(o.bRepBodies.count):
                b = o.bRepBodies.item(j); bb = b.boundingBox
                big = max(bb.maxPoint.x-bb.minPoint.x, bb.maxPoint.y-bb.minPoint.y, bb.maxPoint.z-bb.minPoint.z)*10 > 300
                if o.isLightBulbOn and not big:
                    for k, (a, c) in enumerate(((bb.minPoint.x, bb.maxPoint.x), (bb.minPoint.y, bb.maxPoint.y), (bb.minPoint.z, bb.maxPoint.z))):
                        lo[k] = min(lo[k], a*10); hi[k] = max(hi[k], c*10)
    print(json.dumps({'T': T, 'lo': [round(v, 2) for v in lo], 'hi': [round(v, 2) for v in hi]}))
'''
M.connect()
r = json.loads(M.printed(SRC).strip())
old = json.load(open(f'{REPO}/tools/robot_model/camera_overlay/snapshot_20261028/d455_bodies.json'))
print('new T ', r['T']); print('old T ', old['T_d455_in_pelvis_camera_cm'])
print('new bbox mm', r['lo'], r['hi'])
import numpy as np
b = np.array([x['bb'] for x in old['d455'] if 'MD0645501' in x['q'] and x['live']])
print('old bbox mm', b[:, :3].min(0).tolist(), b[:, 3:].max(0).tolist())
