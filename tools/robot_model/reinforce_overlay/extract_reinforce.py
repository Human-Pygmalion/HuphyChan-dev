#!/usr/bin/env python3
"""READ ONLY: read Hip-Reinforcement, Knee-Reinforcement, Pelvis_Camera_Mount and the Pelvis_Camera total out of Fusion.

Writes below --out: reinforce_cad.json (mass properties, CAD root frame, same schema as shell_overlay/extract_parts.py)
and cad_mm/{hip_reinforcement,knee_reinforcement,camera_mount}.stl (tessellation, CAD root frame, mm).
Small scripts only (connector limits, see extract_parts.py). Run in the mjlab venv with FUSION_MCP / FUSION_MCP_HOST set.
"""
import argparse, json, os, sys
import trimesh
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E  # installs output-capture healing
M, Fetcher = E.M, E.Fetcher

# one number per line: total mass and body count of the live Pelvis_Camera bodies (the FOV solids are hidden or huge, so list them)
SUM_SRC = r'''
import adsk.core, adsk.fusion, json
def run(_c):
    app = adsk.core.Application.get()
    root = adsk.fusion.Design.cast(app.activeDocument.products.itemByProductType('DesignProductType')).rootComponent
    allo = root.allOccurrences
    tot = {}
    for i in range(allo.count):
        o = allo.item(i)
        n = o.fullPathName
        if not n.startswith('Pelvis_Camera:1'):
            continue
        k = 'mount' if n.endswith('Pelvis_Camera_Mount:1') else ('fov' if ('FOV' in n or 'OP_D6' in n) else 'd455')
        for j in range(o.bRepBodies.count):
            b = o.bRepBodies.item(j)
            bb = b.boundingBox
            big = max(bb.maxPoint.x - bb.minPoint.x, bb.maxPoint.y - bb.minPoint.y, bb.maxPoint.z - bb.minPoint.z) * 10 > 300
            live = o.isLightBulbOn and b.isLightBulbOn and not big
            key = k + ('_live' if live else '_off_or_big')
            t = tot.setdefault(key, [0, 0.0])
            t[0] += 1
            t[1] += b.physicalProperties.mass
    print(json.dumps({k: [v[0], round(v[1], 6)] for k, v in tot.items()}))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(f'{a.out}/cad_mm', exist_ok=True)
    M.connect()
    parts = {}
    doc = None
    for pid, occ, body in (('hip_reinforcement', '/Reinforcement:1/Hip-Reinforcement:1', None),
                           ('knee_reinforcement', '/Reinforcement:1/Knee-Reinforcement:1', None),
                           ('camera_mount', '/Pelvis_Camera:1/Pelvis_Camera_Mount:1', 'Pelvis_Camera_Mount')):
        r = E.read_prefix(occ, (body,) if body else ())
        doc = doc or {k: r[k] for k in ('doc', 'ver', 'mod')}
        if (r['doc'], r['ver']) != (doc['doc'], doc['ver']):
            raise SystemExit('document changed while reading')
        assert len(r['b']) == 1, (occ, [b['name'] for b in r['b']])
        parts[pid] = r['b'][0]
        b = parts[pid]
        print(f"{pid}: {b['name']!r} m={b['m']*1000:.2f} g (CAD material {b['mat']}) v={b['v']:.3f} cm3 live={b['live']} bb={b['bb']}", flush=True)
    sums = json.loads(M.printed(SUM_SRC).strip())
    print('Pelvis_Camera totals (count, kg):', sums, flush=True)
    fx = Fetcher(quality='Normal')
    for pid in ('hip_reinforcement', 'knee_reinforcement', 'camera_mount'):
        b = parts[pid]
        v, f = fx.body(b['q'], b['idx'])
        mesh = trimesh.Trimesh(v.reshape(-1, 3) * 10.0, f.reshape(-1, 3), process=False)
        mesh.export(f'{a.out}/cad_mm/{pid}.stl')
        b['tris'] = int(len(mesh.faces))
        print(f'  {pid}: {b["tris"]} tris, mesh bbox mm {mesh.bounds.round(1).tolist()}', flush=True)
    json.dump(dict(doc=doc['doc'], version=doc['ver'], modified_unsaved=doc['mod'],
                   units='m kg (CAD material), v cm3, c cm, I kg*cm2 about the CAD root origin [xx,yy,zz,xy,yz,xz], bb mm',
                   parts=parts, pelvis_camera_totals=sums), open(f'{a.out}/reinforce_cad.json', 'w'), indent=1)
    print('document', doc, '->', a.out)

if __name__ == '__main__':
    main()
