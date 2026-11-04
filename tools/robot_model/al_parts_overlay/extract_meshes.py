#!/usr/bin/env python3
"""READ ONLY: tessellation (CAD root frame, mm) of every aluminium part drawn in the visualization, one leg (the CAD leg), into <out>/cad_mm/<id>.stl.
Small requests through the slice-adaptive Fetcher (connector limits: see shell_overlay/extract_parts.py)."""
import argparse, json, os, sys
import trimesh
REPO = os.environ.get('PYG_REPO', '/home/syaro/MikuchanRemote/Human-Pygmalion')
sys.path.insert(0, f'{REPO}/tools/robot_model/shell_overlay')
import extract_parts as E
M, Fetcher = E.M, E.Fetcher
J = '/Joints_UnderBody:1'
PARTS = [  # (file id, occurrence path, body name, link, group)   group: new = converted to Al in rounds 1-2, old = Al before
    ('HipRoll2YawA', f'{J}/HipRoll2Yaw:1/HipRoll2YawA:1', 'HipRoll2YawA', 'hip_roll_link', 'new'),
    ('HipRoll2YawB', f'{J}/HipRoll2Yaw:1/HipRoll2YawB:1', 'HipRoll2YawB', 'hip_roll_link', 'new'),
    ('HipRoll2Yaw_Mount', f'{J}/HipRoll2Yaw:1/HipRoll2Yaw_Mount:1', 'HipRoll2Yaw_Mount', 'hip_roll_link', 'new'),
    ('Yaw2KneePlateA', f'{J}/HipYaw2Knee:1', 'Yaw2KneePlateA', 'thigh', 'new'),
    ('Knee2Ankle_IdleSide_1', f'{J}/Knee2Ankle:1/Knee2Ankle_IdleSide_1:1', 'Knee2Ankle_IdleSide_1', 'shin', 'new'),
    ('Knee2Ankle_IdleSide_2', f'{J}/Knee2Ankle:1', 'Knee2Ankle_IdleSide_2', 'shin', 'new'),
    ('Knee2Ankle_RotorSide_1', f'{J}/Knee2Ankle:1/Knee2Ankle_RotorSide_1:1', 'Knee2Ankle_RotorSide_1', 'shin', 'new'),
    ('Knee2Ankle_RotorSide_2', f'{J}/Knee2Ankle:1', 'Knee2Ankle_RotorSide_2', 'shin', 'new'),
    ('ankle_core_revised', f'{J}/Ankle2Feet:1/AnkleUniversalJointCore:1/AnkleUniversalJointCore-Revised:1', 'AnkleUniversalJointCore', 'ankle_pitch_link', 'new'),
    ('AnkleFeetPillowA', f'{J}/Ankle2Feet:1', 'AnkleFeetPillowA', 'foot', 'new'),
    ('AnkleFeetPillowB', f'{J}/Ankle2Feet:1/AnkleFeetPillowB:1', 'AnkleFeetPillowB', 'foot', 'new'),
    ('Crank_A', f'{J}/Ankle2Feet:1/Crank_A:1', 'Crank_A', 'crank_A', 'new'),
    ('Crank_B', f'{J}/Ankle2Feet:1/Crank_B:1', 'Crank_B', 'crank_B', 'new'),
    ('Arm_A', f'{J}/Ankle2Feet:1', 'Arm_A', 'rod_A', 'old'),
    ('Arm_B', f'{J}/Ankle2Feet:1', 'Arm_B', 'rod_B', 'old'),
] + [(n.replace(' ', '').replace('(', '_').replace(')', ''), f'{J}/Knee2Ankle:1', n, 'shin', 'old') for n in ('AnkleA2B_A_L', 'AnkleA2B_A_L (1)', 'AnkleA2B_A_R', 'AnkleA2B_A_R (1)', 'AnkleA2B_A_Support', 'AnkleA2B_A_Support (1)')]
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--only', default=''); a = ap.parse_args()
os.makedirs(f'{a.out}/cad_mm', exist_ok=True)
M.connect()
fx = Fetcher(quality='Normal')
meta = {}
for pid, occ, name, link, group in PARTS:
    if a.only and pid not in a.only.split(','):
        continue
    if os.path.exists(f'{a.out}/cad_mm/{pid}.stl') and not a.only:
        print(f'{pid}: already fetched', flush=True); continue
    r = E.read_prefix(occ, (name,))
    if len(r['b']) != 1:
        print(f'{pid}: NOT FOUND at {occ} ({len(r["b"])} bodies), skipped', flush=True); continue
    b = r['b'][0]
    v, f = fx.body(b['q'], b['idx'])
    m = trimesh.Trimesh(v.reshape(-1, 3) * 10.0, f.reshape(-1, 3), process=False)
    m.export(f'{a.out}/cad_mm/{pid}.stl')
    meta[pid] = dict(link=link, group=group, mass_g=round(b['m'] * 1000, 3), volume_cm3=b['v'], tris=int(len(m.faces)), live=b['live'])
    print(f"{pid:24s} {len(m.faces):6d} tris  {b['m']*1000:8.2f} g  {b['v']:8.3f} cm3  link={link}", flush=True)
p = f'{a.out}/mesh_meta.json'
old = json.load(open(p)) if os.path.exists(p) else {}
old.update(meta); json.dump(old, open(p, 'w'), indent=1)
