#!/usr/bin/env python3
"""Add the hip / knee reinforcement plates and the re-measured camera bracket to the combined LegOnly AB model.

Base   : `<name>_footv5_d455_loop`  (URDF + the two MJCF copies; tools/robot_model/foot_overlay/build_footv5_d455.py)
Result : `<name>_footv5_d455_reinf_loop` (same three files + motor sidecar). Nothing shipped is modified.

What changes (user, 2026-11-04):
  * Hip-Reinforcement   160.0 g, bolted to HipRoll2Yaw  -> merged into `hip_roll_link`  (one per leg)
  * Knee-Reinforcement  158.2 g, bolted to Knee2Ankle   -> merged into `shin_link`      (one per leg)
  * Pelvis_Camera       166 g in total = D455 116 g (unchanged) + bracket 50 g           (the bracket CAD changed:
    63.57 cm3 instead of 69.83 cm3, so the old 61.97 g estimate is replaced)
The CAD holds ONE leg. Masses are the user's numbers, not the CAD material (steel placeholder: 1143 g / 1242 g / 499 g); shape,
COM and inertia come from the CAD tensor scaled by mass / CAD mass (uniform density), the same rule as the motor placeholders
(massprops_fusion.py). The CAD leg is the one at -y (R_), the other leg is the y-mirror (build_robot.py `mirror`).
Mass is merged into the link's `<inertial>` (parallel-axis sum), exactly what the rigid-body split in massprops_fusion.py does for
every other part, so no body is added. A visual-only mesh is added per plate. NO collision geometry is added: the plates reach 51 mm (hip) and 57 mm
(knee) outside the existing hull of their link (measured in verify_reinforce.py), so contact with them is not simulated. Ground contact is unaffected (feet only).

usage (mjlab venv): build_reinforce.py --snapshot tools/robot_model/fusion_snapshots/reinforce_20261104
"""
import argparse
import copy
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh
from lxml import etree

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
XMLS = REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_loop"
OUT = BASE.replace("_loop", "_reinf_loop")
MESHDIR = "meshes_reinforce"
LINK_NAME = "assets_v30_reinforce"
R_SIM = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])       # CAD -> sim (build_robot.R)
BASE_CAD = np.array([0.0, 70.0, 60.0])
ORIGIN_CAD = {"hip_roll_link": np.array([-123.7, 70.0, 60.0]), "shin_link": np.array([-123.7, 115.0, -310.0])}
MASS_G = {"hip": 160.0, "knee": 158.2, "camera_total": 166.0, "d455": 116.0}
PLATE_RGBA = "0.95 0.60 0.20 1"
MOUNT_RGBA = "0.86 0.86 0.80 1"
MIRROR = np.diag([1.0, -1.0, 1.0])


def vec(v):
    return " ".join(f"{x:.9g}" for x in np.ravel(v))


def rigid(entry, mass_kg, origin_mm):
    """CAD body -> (mass, COM, inertia about the COM) in a sim-axes frame at `origin_mm` (CAD mm). Mass replaced, shape tensor scaled."""
    k = mass_kg / entry["m"]
    c = np.array(entry["c"], float)                                            # cm, CAD root frame
    xx, yy, zz, xy, yz, xz = entry["I"]                                        # kg cm2 about the CAD root origin
    Io = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * k
    Ic = Io - mass_kg * (c @ c * np.eye(3) - np.outer(c, c))                   # about the COM, kg cm2
    com = R_SIM @ (c * 10.0 - origin_mm) / 1000.0                              # m
    I = R_SIM @ Ic @ R_SIM.T * 1e-4                                            # kg m2
    w = np.linalg.eigvalsh(I)
    assert w.min() > 0 and w[0] + w[1] >= w[2] * (1 - 1e-6), f"inertia is not physical: {w}"
    return mass_kg, com, I


def combine(a, b):
    (m0, c0, I0), (m1, c1, I1) = a, b
    m = m0 + m1
    c = (m0 * c0 + m1 * c1) / m
    I = I0 + I1
    for mm, cc in ((m0, c0), (m1, c1)):
        d = cc - c
        I = I + mm * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, I


def mirrored(p):
    m, c, I = p
    return m, MIRROR @ c, MIRROR @ I @ MIRROR


def plate_mesh(stl, origin_mm, mirror):
    t = trimesh.load(stl, process=False)
    v = (np.asarray(t.vertices) - origin_mm) @ R_SIM.T / 1000.0
    f = np.asarray(t.faces)
    if mirror:
        v = v @ MIRROR
        f = f[:, ::-1]
    return trimesh.Trimesh(v, f, process=False)


# ---------------------------------------------------------------- URDF
def read_urdf_inertial(txt, name):
    blk = re.search(r'<link name="%s">(?:\s*<!--.*?-->)*\s*<inertial>(.*?)</inertial>' % name, txt, re.S).group(1)   # the foot link has a comment before <inertial>
    o = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', blk)
    assert np.allclose([float(x) for x in o.group(2).split()], 0), f"{name}: rotated inertial frame"
    c = np.array([float(x) for x in o.group(1).split()])
    m = float(re.search(r'<mass value="([^"]+)"', blk).group(1))
    g = {k: float(v) for k, v in re.findall(r'(ixx|ixy|ixz|iyy|iyz|izz)="([^"]+)"', blk)}
    I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    return m, c, I


def urdf_inertial_xml(p):
    m, c, I = p
    return (f'\n      <origin xyz="{vec(c)}" rpy="0 0 0"/>\n      <mass value="{m:.9g}"/>\n'
            f'      <inertia ixx="{I[0,0]:.9g}" ixy="{I[0,1]:.9g}" ixz="{I[0,2]:.9g}" iyy="{I[1,1]:.9g}" iyz="{I[1,2]:.9g}" izz="{I[2,2]:.9g}"/>\n    ')


def set_urdf_inertial(txt, name, p):
    pat = re.compile(r'(<link name="%s">(?:\s*<!--.*?-->)*\s*<inertial>).*?(</inertial>)' % name, re.S)
    assert len(pat.findall(txt)) == 1
    return pat.sub(lambda m: m.group(1) + urdf_inertial_xml(p) + m.group(2), txt)


def add_urdf_visual(txt, name, mesh, rgba, mat):
    pat = re.compile(r'(<link name="%s">.*?</visual>)' % name, re.S)
    add = (f'\n    <visual>\n      <geometry><mesh filename="{MESHDIR}/{mesh}.stl"/></geometry>\n'
           f'      <material name="{mat}"><color rgba="{rgba}"/></material>\n    </visual>')
    assert len(pat.findall(txt)) == 1
    return pat.sub(lambda m: m.group(1) + add, txt)


def build_urdf(P, mount):
    txt = (ASSETS / f"{BASE}.urdf").read_text(encoding="utf-8")
    txt = re.sub(r'<robot name="[^"]*">', f'<robot name="{OUT}">', txt, count=1)
    before = {}
    for side in ("R", "L"):
        for link, key in (("hip_roll_link", "hip"), ("shin_link", "knee")):
            n = f"{side}_{link}"
            old = read_urdf_inertial(txt, n)
            before[n] = old
            new = combine(old, P[side][key])
            txt = set_urdf_inertial(txt, n, new)
            txt = add_urdf_visual(txt, n, f"{side}_{key}_reinforcement", PLATE_RGBA, "reinforcement_plate")
    # camera bracket: replace the whole link (mass, COM, inertia, mesh)
    m, c, I = mount
    blk = re.search(r'<link name="pelvis_camera_mount">.*?</link>', txt, re.S).group(0)
    new_blk = (f'<link name="pelvis_camera_mount">\n    <inertial>{urdf_inertial_xml(mount)}</inertial>\n'
               f'    <visual>\n      <geometry><mesh filename="{MESHDIR}/pelvis_camera_mount.stl"/></geometry>\n'
               f'      <material name="pla_mount"><color rgba="{MOUNT_RGBA}"/></material>\n    </visual>\n'
               f'    <collision>\n      <geometry><mesh filename="{MESHDIR}/pelvis_camera_mount_hull.stl"/></geometry>\n    </collision>\n  </link>')
    txt = txt.replace(blk, new_blk)
    txt = txt.replace("<link name=\"base_link\"", "<!-- REINFORCE OVERLAY, generated by tools/robot_model/reinforce_overlay/build_reinforce.py: hip / knee reinforcement plates merged into "
                      "hip_roll_link / shin_link, camera bracket re-measured (Pelvis_Camera 166 g). Do not hand-edit; rebuild. -->\n  <link name=\"base_link\"", 1)
    (ASSETS / f"{OUT}.urdf").write_text(txt, encoding="utf-8")
    shutil.copyfile(ASSETS / f"{BASE}.motor_params.json", ASSETS / f"{OUT}.motor_params.json")
    print("wrote", f"{OUT}.urdf")
    return before


# ---------------------------------------------------------------- MJCF
def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_mjcf_inertial(el):
    assert el.get("fullinertia") is None or el.get("quat") is None
    c = np.array([float(x) for x in el.get("pos").split()])
    m = float(el.get("mass"))
    if el.get("fullinertia"):
        a, b, cc, d, e, f = [float(x) for x in el.get("fullinertia").split()]
        I = np.array([[a, d, e], [d, b, f], [e, f, cc]])
    else:
        Rq = quat_to_R(np.array([float(x) for x in el.get("quat", "1 0 0 0").split()]))
        I = Rq @ np.diag([float(x) for x in el.get("diaginertia").split()]) @ Rq.T
    return m, c, I


def set_mjcf_inertial(el, p):
    m, c, I = p
    for a in ("quat", "diaginertia", "fullinertia", "axisangle", "xyaxes", "zaxis", "euler"):
        if a in el.attrib:
            del el.attrib[a]
    el.set("pos", vec(c))
    el.set("mass", f"{m:.9g}")
    el.set("fullinertia", vec([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))


def build_mjcf(src, dst, prefix, P, mount):
    """`src` base MJCF -> `dst`; `prefix` is the mesh folder as that file sees it (mjlab copy: ../assets_v30_reinforce/, repo copy: ../meshes_reinforce/)."""
    tree = etree.parse(str(src))
    root = tree.getroot()
    root.set("model", OUT)
    asset = root.find("asset")
    for side in ("R", "L"):
        for link, key in (("hip_roll_link", "hip"), ("shin_link", "knee")):
            body = root.find(f".//body[@name='{side}_{link}']")
            inert = body.find("inertial")
            set_mjcf_inertial(inert, combine(read_mjcf_inertial(inert), P[side][key]))
            mname = f"{side}_{key}_reinforcement"
            etree.SubElement(asset, "mesh", name=mname, file=f"{prefix}{mname}.stl")
            body.findall("geom[@class='visual']")[0].addnext(etree.Element("geom", {"class": "visual", "mesh": mname, "rgba": PLATE_RGBA}))
    set_mjcf_inertial(root.find(".//body[@name='pelvis_camera_mount']/inertial"), mount)
    for m in asset.findall("mesh"):
        if m.get("name") == "pelvis_camera_mount":
            m.set("file", f"{prefix}pelvis_camera_mount.stl")
    root.insert(0, etree.Comment(f" {OUT}: {BASE} + hip / knee reinforcement plates merged into hip_roll_link / shin_link, camera bracket re-measured "
                                 "(Pelvis_Camera 166 g). Generated by tools/robot_model/reinforce_overlay/build_reinforce.py. Do not hand-edit; rebuild. "))
    tree.write(str(dst), encoding="utf-8", xml_declaration=False, pretty_print=True)
    print("wrote", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    a = ap.parse_args()
    snap = Path(a.snapshot)
    J = json.load(open(snap / "reinforce_cad.json"))
    parts = J["parts"]
    # --- directly from the CAD (CAD leg = R_, at -y)
    hip = rigid(parts["hip_reinforcement"], MASS_G["hip"] / 1000, ORIGIN_CAD["hip_roll_link"])
    knee = rigid(parts["knee_reinforcement"], MASS_G["knee"] / 1000, ORIGIN_CAD["shin_link"])
    for nm, entry, o in (("hip", parts["hip_reinforcement"], ORIGIN_CAD["hip_roll_link"]), ("knee", parts["knee_reinforcement"], ORIGIN_CAD["shin_link"])):
        assert entry["live"] == 1 and entry["c"][0] < 0, f"{nm}: expected a visible body at CAD x<0 (sim -y = R_ side)"
    P = {"R": {"hip": hip, "knee": knee}, "L": {"hip": mirrored(hip), "knee": mirrored(knee)}}
    mount = rigid(parts["camera_mount"], (MASS_G["camera_total"] - MASS_G["d455"]) / 1000, BASE_CAD)
    # --- meshes
    out = ASSETS / MESHDIR
    out.mkdir(exist_ok=True)
    for key, origin in (("hip", ORIGIN_CAD["hip_roll_link"]), ("knee", ORIGIN_CAD["shin_link"])):
        for side in ("R", "L"):
            m = plate_mesh(snap / f"cad_mm/{key}_reinforcement.stl", origin, mirror=(side == "L"))
            m.export(out / f"{side}_{key}_reinforcement.stl")
    t = trimesh.load(snap / "cad_mm/camera_mount.stl", process=False)
    mm = trimesh.Trimesh((np.asarray(t.vertices) - BASE_CAD) @ R_SIM.T / 1000.0, t.faces, process=False)
    mm.export(out / "pelvis_camera_mount.stl")
    mm.convex_hull.export(out / "pelvis_camera_mount_hull.stl")
    # --- models
    before = build_urdf(P, mount)
    build_mjcf(ASSETS / f"{BASE}.xml", ASSETS / f"{OUT}.xml", f"../{MESHDIR}/", P, mount)
    build_mjcf(XMLS / f"{BASE}.xml", XMLS / f"{OUT}.xml", f"../{LINK_NAME}/", P, mount)
    link = XMLS / LINK_NAME
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(out, link.parent))
    print("link", link.name, "->", os.readlink(link))
    rep = dict(cad_document=J["doc"], cad_version=J["version"], cad_unsaved_edits=J["modified_unsaved"], base=BASE, out=OUT,
               mass_g=MASS_G, mass_source="user 2026-11-04 (CAD material is a steel placeholder: hip 1143.15 g, knee 1241.97 g, bracket 498.99 g)",
               cad_leg="R_ (-y); L_ = y-mirror",
               plates={f"{s}_{k}": dict(mass_kg=P[s][k][0], com_link_m=np.round(P[s][k][1], 7).tolist(),
                                       inertia_com_kg_m2=np.round(P[s][k][2], 10).tolist(), link=("hip_roll_link" if k == "hip" else "shin_link"))
                       for s in P for k in P[s]},
               bracket=dict(mass_kg=mount[0], com_base_link_m=np.round(mount[1], 7).tolist(), inertia_com_kg_m2=np.round(mount[2], 10).tolist(),
                            cad_volume_cm3=parts["camera_mount"]["v"], old_mass_kg=0.0619659),
               link_inertial_before={k: dict(mass=v[0], com=v[1].tolist()) for k, v in before.items()},
               assumptions=["Pelvis_Camera 166 g = D455 116 g (datasheet, unchanged) + bracket 50 g", "plates add no collision geometry (they stick out of the link hulls by up to 57 mm)"])
    json.dump(rep, open(ASSETS / f"{OUT}_report.json", "w"), indent=1)
    print(f"added {2*(MASS_G['hip']+MASS_G['knee'])/1000:.4f} kg of plates; bracket {mount[0]*1000:.2f} g (was 61.97 g)")


if __name__ == "__main__":
    main()
