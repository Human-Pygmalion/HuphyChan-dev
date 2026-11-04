#!/usr/bin/env python3
"""Round 2 of the aluminium conversion: HipRoll2YawA/B, Yaw2KneePlateA, Knee2Ankle_{Idle,Rotor}Side_{1,2}, and the revised AnkleUniversalJointCore
(together with the round-1 parts: cranks A/B and HipRoll2Yaw_Mount) are Al 6061, both legs (user, 2026-11-04).

Base   : `<name>_footv5_d455_reinf_alparts_loop` (round 1)      Result : `<name>_footv5_d455_reinf_alparts2_loop`. Nothing shipped is modified.

Method (the project's own mass pipeline, not a hand calculation):
  1. take the body dump the shipped model was built from (tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json);
  2. state A = that dump with the round-1 parts (cranks, mount) at their current Fusion values   (must reproduce the round-1 model: checked);
  3. state B = A with the round-2 parts replaced by their current Fusion values (CAD material Al 6061 = volume x 2.70 g/cm3, CAD inertia);
  4. run tools/robot_model/massprops_fusion.py on A and on B (the loop-ankle LegOnly aggregation the model uses);
  5. per rigid link: delta = B - A, taken about the link origin (mass, first moment, inertia), added to the round-1 model's link,
     R_ direct (the CAD leg) and L_ y-mirrored. Plates, camera and everything else in the model stay as they were.
Also adds a visual mesh of the revised AnkleUniversalJointCore to ankle_pitch_link (the model drew no cross before).
The part list is PARTS below: edit it and rebuild to change what is aluminium.
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reinforce_overlay"))
import build_reinforce as B  # noqa: E402

REPO = B.REPO
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts_loop"
OUT = BASE.replace("_alparts_loop", "_alparts2_loop")
DUMP = REPO / "tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json"
SNAP = REPO / "tools/robot_model/fusion_snapshots/alparts_20261104"
PY = REPO / "mujoco-sim/mjlab/.venv/bin/python3"
SCRATCH = Path(os.environ.get("PYG_SCRATCH", "/tmp/pyg_alparts2")); SCRATCH.mkdir(parents=True, exist_ok=True)
ROUND1 = {"Ankle2Feet:1::Crank_A": "Crank_A", "Ankle2Feet:1::Crank_B": "Crank_B", "HipRoll2Yaw:1::HipRoll2Yaw_Mount": "HipRoll2Yaw_Mount"}
ROUND2 = ["HipRoll2YawA", "HipRoll2YawB", "Yaw2KneePlateA", "Knee2Ankle_IdleSide_1", "Knee2Ankle_IdleSide_2", "Knee2Ankle_RotorSide_1", "Knee2Ankle_RotorSide_2", "AnkleUniversalJointCore",
          "AnkleFeetPillowA", "AnkleFeetPillowB"]   # exactly the parts the user named (Knee2Ankle_IdleSide / _RotorSide: both numbered bodies); nothing else in Knee2Ankle is touched
LINKS = {"hip_roll_link": ("hip_roll_link", np.array([-123.7, 70.0, 60.0])), "thigh": ("thigh_link", np.array([-123.7, 70.0, 60.0])),
         "shin": ("shin_link", np.array([-123.7, 115.0, -310.0])), "ankle_pitch_link": ("ankle_pitch_link", np.array([-123.7, 145.0, -800.0])),
         "foot": ("foot_link", np.array([-123.7, 145.0, -800.0]))}   # foot: the v5 foot link = the old foot (pillows included) + new parts, so a delta can be added
ANKLE_ORIGIN = LINKS["ankle_pitch_link"][1]
CORE_RGBA = "0.75 0.75 0.78 1"                # plain aluminium grey; the red/translucent colouring is only in the visualization (render_al_parts.py)


def load_round2():
    """round-2 part properties from the Fusion snapshot, with the parts the user said to take from the ORIGINAL geometry replaced (use_original_plate.py)"""
    r = json.load(open(SNAP / "round2_cad.json"))
    ov = SNAP / "original_geometry.json"
    if ov.exists():
        r["parts"].update(json.load(open(ov)))
    return r


def find(dump, tail):
    ks = [k for k in dump if k.endswith("::" + tail) or k.endswith(tail)]
    ks = [k for k in ks if k.split("::")[-1] == tail.split("::")[-1]]
    assert len(ks) == 1, (tail, ks)
    return ks[0]


def swap(dump, body, entry):
    k = find(dump, body)
    dump[k] = dict(dump[k], m=entry["m"], v=entry["v"], c=entry["c"], I=entry["I"], mat="Aluminum 6061 (CAD)", live=True)


def run_massprops(dump, tag):
    f = SCRATCH / f"bodies_{tag}.json"; out = SCRATCH / f"mp_{tag}.json"
    json.dump(dump, open(f, "w"))
    env = dict(os.environ, PYG_BODIES=str(f), PYG_MASSPROPS=str(out), PYG_ANKLE_LOOP="1", PYG_MODEL_VARIANT="LegOnly")
    subprocess.run([str(PY), str(REPO / "tools/robot_model/massprops_fusion.py")], env=env, check=True, capture_output=True)
    return json.load(open(out))["bodies"]


def link_frame(d, origin):
    com = B.R_SIM @ (np.array(d["com"]) - origin) / 1000.0
    I = B.R_SIM @ np.array(d["I_com"]) @ B.R_SIM.T * 1e-6
    return d["mass"], com, I


def about_origin(p):
    m, c, I = p
    return m, m * c, I + m * (c @ c * np.eye(3) - np.outer(c, c))


def from_origin(m, f, Io):
    c = f / m
    return m, c, Io - m * (c @ c * np.eye(3) - np.outer(c, c))


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    d0 = json.load(open(DUMP))
    cr = json.load(open(SNAP / "cranks_cad.json"))["parts"]; mt = json.load(open(SNAP / "mount_cad.json"))["part"]; r2 = load_round2()
    parts2 = r2["parts"]
    dA = json.loads(json.dumps(d0))
    for tail, name in ROUND1.items():
        swap(dA, tail, mt if name == "HipRoll2Yaw_Mount" else cr[name])
    dB = json.loads(json.dumps(dA))
    for name in ROUND2:
        swap(dB, name, parts2[name])
    mpA, mpB = run_massprops(dA, "A"), run_massprops(dB, "B")
    base_urdf = (B.ASSETS / f"{BASE}.urdf").read_text(encoding="utf-8")
    # ---- check: state A reproduces the round-1 model (link masses, with the overlay parts taken out)
    plate = {"hip_roll_link": B.MASS_G["hip"] / 1000, "shin": B.MASS_G["knee"] / 1000}
    for lk, (mname, origin) in LINKS.items():
        if lk == "foot":
            continue                       # the shipped foot link is the v5 foot (old foot + carrier etc.), not comparable with the old dump
        got = mpA[lk]["mass"]
        want = B.read_urdf_inertial(base_urdf, f"R_{mname}")[0] - plate.get(lk, 0.0)
        assert abs(got - want) < 2e-4, f"state A does not reproduce the round-1 model: {lk} {got} vs {want}"
        print(f"check A vs round-1 model: {lk:17s} {got*1000:9.3f} g vs {want*1000:9.3f} g")
    # ---- deltas
    delta, rows = {}, {}
    for lk, (mname, origin) in LINKS.items():
        a_, b_ = about_origin(link_frame(mpA[lk], origin)), about_origin(link_frame(mpB[lk], origin))
        delta[lk] = (b_[0] - a_[0], b_[1] - a_[1], b_[2] - a_[2])
        rows[lk] = dict(mass_A_g=round(mpA[lk]["mass"] * 1000, 3), mass_B_g=round(mpB[lk]["mass"] * 1000, 3), delta_g=round(delta[lk][0] * 1000, 3))
        print(f"{lk:17s} A {mpA[lk]['mass']*1000:9.3f} g -> B {mpB[lk]['mass']*1000:9.3f} g  (+{delta[lk][0]*1000:.3f} g)")
    assert mpB["ankle_pitch_link"]["mass"] > mpA["ankle_pitch_link"]["mass"], "the revised core must stay on the ankle axis (it is classified by its COM z)"
    for k in ("crank_A", "crank_B", "foot", "pelvis", "hip_pitch_link", "thigh"):
        pass
    others = {k: abs(mpB[k]["mass"] - mpA[k]["mass"]) for k in mpA if k not in LINKS}
    assert max(others.values()) < 1e-9, others
    # ---- apply to the round-1 model
    new_props = {}
    for s in "RL":
        for lk, (mname, origin) in LINKS.items():
            n = f"{s}_{mname}"
            m0, c0, I0 = B.read_urdf_inertial(base_urdf, n)
            o = about_origin((m0, c0, I0))
            dm, df, dI = delta[lk]
            if s == "L":
                df, dI = B.MIRROR @ df, B.MIRROR @ dI @ B.MIRROR
            new_props[n] = from_origin(o[0] + dm, o[1] + df, o[2] + dI)
    # ---- visual mesh of the revised core, in the ankle_pitch_link frame (R direct, L y-mirrored)
    mesh_src = SNAP / "cad_mm/ankle_core_revised.stl"
    have_mesh = mesh_src.exists()
    if have_mesh:
        for s in "RL":
            m = B.plate_mesh(mesh_src, ANKLE_ORIGIN, mirror=(s == "L"))
            m.export(B.ASSETS / B.MESHDIR / f"{s}_ankle_core_revised.stl")
    urdf = re.sub(r'<robot name="[^"]*">', f'<robot name="{OUT}">', base_urdf, count=1)
    for n, p in new_props.items():
        urdf = B.set_urdf_inertial(urdf, n, p)
    if have_mesh:
        for s in "RL":
            pat = re.compile(r'(<link name="%s_ankle_pitch_link">.*?</inertial>)' % s, re.S)
            assert len(pat.findall(urdf)) == 1
            urdf = pat.sub(lambda m: m.group(1) + f'\n    <visual>\n      <geometry><mesh filename="{B.MESHDIR}/{s}_ankle_core_revised.stl"/></geometry>\n'
                           f'      <material name="al_core"><color rgba="{CORE_RGBA}"/></material>\n    </visual>', urdf)
    urdf = urdf.replace("<link name=\"base_link\"", "<!-- ALUMINIUM ROUND 2, generated by tools/robot_model/al_parts_overlay/build_al_round2.py: HipRoll2YawA/B, Yaw2KneePlateA, "
                        "Knee2Ankle_{Idle,Rotor}Side_{1,2} and the revised AnkleUniversalJointCore are Al 6061 (volume x 2.70 g/cm3). Do not hand-edit; rebuild. -->\n  <link name=\"base_link\"", 1)
    (B.ASSETS / f"{OUT}.urdf").write_text(urdf, encoding="utf-8")
    shutil.copyfile(B.ASSETS / f"{BASE}.motor_params.json", B.ASSETS / f"{OUT}.motor_params.json")
    print("wrote", f"{OUT}.urdf")
    for src, dst, prefix in ((B.ASSETS / f"{BASE}.xml", B.ASSETS / f"{OUT}.xml", f"../{B.MESHDIR}/"), (B.XMLS / f"{BASE}.xml", B.XMLS / f"{OUT}.xml", f"../{B.LINK_NAME}/")):
        tree = etree.parse(str(src)); root = tree.getroot(); root.set("model", OUT)
        for n, p in new_props.items():
            B.set_mjcf_inertial(root.find(f".//body[@name='{n}']/inertial"), p)
        if have_mesh:
            asset = root.find("asset")
            for s in "RL":
                etree.SubElement(asset, "mesh", name=f"{s}_ankle_core_revised", file=f"{prefix}{s}_ankle_core_revised.stl")
                body = root.find(f".//body[@name='{s}_ankle_pitch_link']")
                body.find("inertial").addnext(etree.Element("geom", {"class": "visual", "mesh": f"{s}_ankle_core_revised", "rgba": CORE_RGBA}))
        root.insert(0, etree.Comment(f" {OUT}: {BASE} + aluminium round 2 (HipRoll2YawA/B, Yaw2KneePlateA, Knee2Ankle Idle/Rotor side 1+2, revised AnkleUniversalJointCore; Al 6061, volume x 2.70). "
                                     "Generated by tools/robot_model/al_parts_overlay/build_al_round2.py. Do not hand-edit; rebuild. "))
        tree.write(str(dst), encoding="utf-8", xml_declaration=False, pretty_print=True)
        print("wrote", dst)
    # ---- report: every converted part, old (as the original model had it) and new
    part_rows = {}
    for name in list(ROUND1.values()) + ROUND2:
        tail = next((t for t, n in ROUND1.items() if n == name), name)
        k = find(d0, tail)
        e = (mt if name == "HipRoll2Yaw_Mount" else cr.get(name) if name in cr else parts2[name])
        part_rows[name] = dict(round=1 if name in ROUND1.values() else 2, old_mass_g=round(d0[k]["m"] * 1000, 3), old_density_g_cm3=round(d0[k]["m"] * 1000 / d0[k]["v"], 4),
                               new_mass_g=round(e["m"] * 1000, 3), cad_volume_cm3=round(e["v"], 4), increase_g_per_leg=round((e["m"] - d0[k]["m"]) * 1000, 3))
    rep = dict(cad_document=r2["doc"], cad_version=r2["version"], cad_unsaved_edits=r2["modified_unsaved"], base=BASE, out=OUT, parts=part_rows, links=rows,
               round2_increase_g=round(sum(r["delta_g"] for r in rows.values()) * 2, 3),
               note="old_mass_g = the part as the original (pre-aluminium) model carried it; round 1 = cranks A/B and HipRoll2Yaw_Mount")
    json.dump(rep, open(B.ASSETS / f"{OUT}_report.json", "w"), indent=1)
    print(f"round-2 mass increase (both legs): {rep['round2_increase_g']:.3f} g; revised-core mesh: {have_mesh}")


if __name__ == "__main__":
    main()
