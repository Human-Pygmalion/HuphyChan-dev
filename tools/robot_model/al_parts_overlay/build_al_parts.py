#!/usr/bin/env python3
"""Make the four ankle cranks and the two HipRoll2Yaw_Mount parts aluminium (Al 6061, 2.70 g/cm3) in the reinforcement model; the crank covers stay 3D-printed.

Base   : `<name>_footv5_d455_reinf_loop`  (URDF + both MJCF copies; tools/robot_model/reinforce_overlay/build_reinforce.py)
Result : `<name>_footv5_d455_reinf_alparts_loop`. Nothing shipped is modified.

Why (user, 2026-11-04): the real crank A and crank B (one pair per leg, 4 parts in total) and the HipRoll2Yaw_Mount (one per leg, 2 parts) are now
machined from aluminium instead of printed.
Rule (user): mass = CAD volume x 2.70 g/cm3, inertia from the CAD tensor of that body scaled by the same factor (uniform density).
A crank link (`crank_A`, `crank_B`) is two CAD bodies: the crank and its cover. Before: both at the printed density 0.8874 g/cm3
(docs/89, 0.329 x 2.70). Now: crank 2.70 g/cm3, cover still 0.8874 g/cm3 (covers are excluded, user). Mass, COM and inertia of each
link are rebuilt from the two CAD bodies (CAD frame -> link frame, CAD leg = R_, L_ = y-mirror); the script first rebuilds the OLD link from
the same CAD numbers and refuses to go on if that does not reproduce the old file (a check of the frame conversion).
HipRoll2Yaw_Mount sits in `hip_roll_link` with many other parts, so it is ADDED as a difference: the model had it at 67.0 g (printed, the weighed value of
docs/89, 1.0989 g/cm3); it becomes CAD volume x 2.70 g/cm3, and the extra mass with the same CAD shape (COM, tensor x mass ratio) is merged into the link.
Geometry, joints, ranges, armature and damping are untouched. The CAD holds ONE leg (one HipRoll2Yaw, one Crank_A, one Crank_B); the other leg is the y-mirror.

usage (mjlab venv): build_al_parts.py --snapshot tools/robot_model/fusion_snapshots/alparts_20261104
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reinforce_overlay"))
import build_reinforce as B  # noqa: E402  (frame conversion, inertia algebra, URDF / MJCF inertial read-write)

BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_loop"
OUT = BASE.replace("_loop", "_alparts_loop")
RHO_AL, RHO_PRINTED = 2.70, 0.8874              # g/cm3 (Al 6061; printed structural part, docs/89)
MOUNT_OLD_KG = 0.066997                          # HipRoll2Yaw_Mount as the shipped models carry it (bodies_printed_prototype_tempmass.json: 60.9674 cm3 x 1.0989 g/cm3)
HIP_ORIGIN = np.array([-123.7, 70.0, 60.0])     # hip_roll_link origin, CAD mm (build_robot.ORIGIN_CAD)
CRANK_ORIGIN = {"A": np.array([-125.418, 144.945, -500.034]), "B": np.array([-121.983, 145.055, -600.034])}   # CAD mm, motor axis point (ankle_loop_points_v3_printed.json)


def link_props(parts, tag, rho_crank):
    c, cov = parts[f"Crank_{tag}"], parts[f"Crank_{tag}_Cover"]
    o = CRANK_ORIGIN[tag]
    a = B.rigid(c, c["v"] * rho_crank / 1000.0, o)
    b = B.rigid(cov, cov["v"] * RHO_PRINTED / 1000.0, o)
    return B.combine(a, b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--snapshot", required=True); a = ap.parse_args()
    J = json.load(open(Path(a.snapshot) / "cranks_cad.json"))
    parts = J["parts"]
    mount = json.load(open(Path(a.snapshot) / "mount_cad.json"))["part"]
    urdf = (B.ASSETS / f"{BASE}.urdf").read_text(encoding="utf-8")
    import re
    urdf = re.sub(r'<robot name="[^"]*">', f'<robot name="{OUT}">', urdf, count=1)
    rows, new_props = {}, {}
    for tag in "AB":
        # frame check: both bodies printed must give the old link
        for s in "RL":
            old = B.read_urdf_inertial(urdf, f"{s}_crank_{tag}")
            chk = link_props(parts, tag, RHO_PRINTED)
            if s == "L":
                chk = B.mirrored(chk)
            dm = abs(chk[0] - old[0]) / old[0]
            dc = np.abs(chk[1] - old[1]).max()
            dI = np.abs(chk[2] - old[2]).max() / np.abs(old[2]).max()
            print(f"old-link check {s}_crank_{tag}: mass {old[0]*1000:.3f} g vs CAD-rebuilt {chk[0]*1000:.3f} g, COM diff {dc*1000:.3f} mm, inertia diff {dI*100:.3f} %")
            assert dm < 2e-3 and dc < 2e-4 and dI < 2e-2, f"{s}_crank_{tag}: frame conversion does not reproduce the shipped link"
            new = link_props(parts, tag, RHO_AL)
            new = B.mirrored(new) if s == "L" else new
            new_props[f"{s}_crank_{tag}"] = new
            rows[f"{s}_crank_{tag}"] = dict(old_mass_g=round(old[0] * 1000, 3), new_mass_g=round(new[0] * 1000, 3),
                                            old_com_m=np.round(old[1], 7).tolist(), new_com_m=np.round(new[1], 7).tolist(),
                                            old_inertia_com_kg_m2=np.round(old[2], 10).tolist(), new_inertia_com_kg_m2=np.round(new[2], 10).tolist())
    d_mount = mount["v"] * RHO_AL / 1000.0 - MOUNT_OLD_KG
    extra = B.rigid(mount, d_mount, HIP_ORIGIN)                 # the added mass, same shape
    for s in "RL":
        n = f"{s}_hip_roll_link"
        old = B.read_urdf_inertial(urdf, n)
        new = B.combine(old, extra if s == "R" else B.mirrored(extra))
        new_props[n] = new
        rows[n] = dict(old_mass_g=round(old[0] * 1000, 3), new_mass_g=round(new[0] * 1000, 3), mount_old_g=MOUNT_OLD_KG * 1000, mount_new_g=round(mount["v"] * RHO_AL, 3),
                       old_com_m=np.round(old[1], 7).tolist(), new_com_m=np.round(new[1], 7).tolist(),
                       old_inertia_com_kg_m2=np.round(old[2], 10).tolist(), new_inertia_com_kg_m2=np.round(new[2], 10).tolist())
    print(f"HipRoll2Yaw_Mount: {MOUNT_OLD_KG*1000:.3f} g -> {mount['v'] * RHO_AL:.3f} g (+{d_mount*1000:.3f} g per leg), CAD volume {mount['v']:.4f} cm3")
    for n, p in new_props.items():
        urdf = B.set_urdf_inertial(urdf, n, p)
    urdf = urdf.replace("<link name=\"base_link\"", "<!-- ALUMINIUM CRANK OVERLAY, generated by tools/robot_model/al_parts_overlay/build_al_parts.py: the four crank bodies and the two HipRoll2Yaw_Mount parts are Al 6061 "
                        "(volume x 2.70 g/cm3), crank covers stay printed. Do not hand-edit; rebuild. -->\n  <link name=\"base_link\"", 1)
    (B.ASSETS / f"{OUT}.urdf").write_text(urdf, encoding="utf-8")
    shutil.copyfile(B.ASSETS / f"{BASE}.motor_params.json", B.ASSETS / f"{OUT}.motor_params.json")
    print("wrote", f"{OUT}.urdf")
    for src, dst in ((B.ASSETS / f"{BASE}.xml", B.ASSETS / f"{OUT}.xml"), (B.XMLS / f"{BASE}.xml", B.XMLS / f"{OUT}.xml")):
        tree = etree.parse(str(src)); root = tree.getroot(); root.set("model", OUT)
        for n, p in new_props.items():
            B.set_mjcf_inertial(root.find(f".//body[@name='{n}']/inertial"), p)
        root.insert(0, etree.Comment(f" {OUT}: {BASE} with the four ankle cranks and the two hip-roll mounts in aluminium (Al 6061, volume x 2.70 g/cm3; crank covers stay printed). "
                                     "Generated by tools/robot_model/al_parts_overlay/build_al_parts.py. Do not hand-edit; rebuild. "))
        tree.write(str(dst), encoding="utf-8", xml_declaration=False, pretty_print=True)
        print("wrote", dst)
    # the mesh folder links of the mjlab copy are unchanged (no new mesh); the new file reuses ../assets_v30_reinforce/ etc.
    dm = sum(r["new_mass_g"] - r["old_mass_g"] for r in rows.values())
    rep = dict(cad_document=J["doc"], cad_version=J["version"], cad_unsaved_edits=J["modified_unsaved"], base=BASE, out=OUT,
               density_g_cm3=dict(crank=RHO_AL, hip_roll_mount=RHO_AL, cover=RHO_PRINTED), rule="mass = CAD volume x density; inertia = CAD tensor x mass / CAD mass",
               cad_volumes_cm3={**{k: v["v"] for k, v in parts.items()}, "HipRoll2Yaw_Mount": mount["v"]}, links=rows, total_mass_change_g=round(dm, 3))
    json.dump(rep, open(B.ASSETS / f"{OUT}_report.json", "w"), indent=1)
    print(f"total mass change {dm:.3f} g")


if __name__ == "__main__":
    main()
