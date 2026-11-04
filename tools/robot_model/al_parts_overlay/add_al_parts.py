#!/usr/bin/env python3
"""Turn named leg parts into aluminium (Al 6061, 2.70 g/cm3) in a model, as a new file. General form of build_al_parts.py (which did M18).

For each part: new mass = live CAD volume x 2.70 g/cm3; the model carries it at OLD mass (the printed value of the body dump the models were built from,
bodies_printed_prototype_tempmass.json). The EXTRA mass (new - old, same CAD shape: COM and tensor x mass ratio) is merged into the link the part belongs to
(parallel-axis sum), for the CAD leg (R_) directly and for the other leg mirrored in y. Nothing else changes.

usage (mjlab venv):
  add_al_parts.py --snapshot DIR --base <model name ending _loop> --suffix alparts2 [--dry-run]
DIR is written by extract_by_name.py. Output model name: <base without _loop>_<suffix>_loop, with URDF, both MJCF copies, motor sidecar, report json.
Links: HipPitch2Roll -> hip_pitch_link, HipRoll2Yaw -> hip_roll_link, HipYaw2Knee -> thigh_link, Knee2Ankle -> shin_link. Ankle2Feet parts need --link NAME=link
(shin_link | ankle_pitch_link; the foot link is refused: the v5 foot models replace its inertial, and crank links are rebuilt by build_al_parts.py).
--old NAME=grams overrides the model's old mass of a part (the dump value is used otherwise).
Run verify_al_parts.py-style checks after building (mass delta, other bodies unchanged, URDF vs MJCF); this script prints the numbers it can check itself."""
import argparse, json, re, shutil, sys
from pathlib import Path
import numpy as np
from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reinforce_overlay"))
import build_reinforce as B  # noqa: E402

RHO_AL = 2.70
DUMP = B.REPO / "tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json"
LINK_OF = {"HipPitch2Roll:1": "hip_pitch_link", "HipRoll2Yaw:1": "hip_roll_link", "HipYaw2Knee:1": "thigh_link", "Knee2Ankle:1": "shin_link"}
ORIGIN = {"hip_pitch_link": B.ORIGIN_CAD["hip_roll_link"], "hip_roll_link": B.ORIGIN_CAD["hip_roll_link"], "thigh_link": B.ORIGIN_CAD["hip_roll_link"],
          "shin_link": B.ORIGIN_CAD["shin_link"], "ankle_pitch_link": np.array([-123.7, 145.0, -800.0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True); ap.add_argument("--base", required=True); ap.add_argument("--suffix", required=True)
    ap.add_argument("--link", action="append", default=[]); ap.add_argument("--old", action="append", default=[]); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    J = json.load(open(Path(a.snapshot) / "parts_cad.json"))
    dump = json.load(open(DUMP))
    links = dict(x.split("=") for x in a.link); olds = {k: float(v) / 1000 for k, v in (x.split("=") for x in a.old)}
    assert a.base.endswith("_loop")
    out = a.base[:-5] + f"_{a.suffix}_loop"
    urdf = (B.ASSETS / f"{a.base}.urdf").read_text(encoding="utf-8")
    urdf = re.sub(r'<robot name="[^"]*">', f'<robot name="{out}">', urdf, count=1)
    adds = {}                                                   # link base name -> list of (mass, com, I) for the CAD leg
    rows = {}
    for name, p in J["parts"].items():
        link = links.get(name) or LINK_OF.get(p["group"])
        if link is None or link == "foot_link" or link.startswith(("crank", "rod")):
            sys.exit(f"{name}: no usable link ({p['group']}); give --link {name}=shin_link|ankle_pitch_link. Foot and crank/rod links are not handled here.")
        key = next((k for k in dump if k.endswith("::" + name)), None)
        old = olds.get(name, dump[key]["m"] if key else None)
        if old is None:
            sys.exit(f"{name}: not in the body dump, give --old {name}=grams")
        new = p["v"] * RHO_AL / 1000
        extra = B.rigid(p, new - old, ORIGIN[link])
        adds.setdefault(link, []).append(extra)
        rows[name] = dict(group=p["group"], link=link, volume_cm3=p["v"], old_g=round(old * 1000, 3), new_g=round(new * 1000, 3), delta_g=round((new - old) * 1000, 3))
        print(f"{name:26s} {link:16s} V={p['v']:8.3f} cm3  {old*1000:7.2f} g -> {new*1000:7.2f} g  (+{(new-old)*1000:.2f} g per leg)")
    total = 2 * sum(r["delta_g"] for r in rows.values())
    print(f"robot mass change {total/1000:+.4f} kg (both legs)")
    if a.dry_run:
        return
    props = {}
    for link, lst in adds.items():
        for s in "RL":
            n = f"{s}_{link}"
            cur = B.read_urdf_inertial(urdf, n)
            for e in lst:
                cur = B.combine(cur, e if s == "R" else B.mirrored(e))
            props[n] = cur
    for n, pr in props.items():
        urdf = B.set_urdf_inertial(urdf, n, pr)
    (B.ASSETS / f"{out}.urdf").write_text(urdf, encoding="utf-8")
    shutil.copyfile(B.ASSETS / f"{a.base}.motor_params.json", B.ASSETS / f"{out}.motor_params.json")
    for src, dst in ((B.ASSETS / f"{a.base}.xml", B.ASSETS / f"{out}.xml"), (B.XMLS / f"{a.base}.xml", B.XMLS / f"{out}.xml")):
        tree = etree.parse(str(src)); root = tree.getroot(); root.set("model", out)
        for n in props:
            cur = B.read_mjcf_inertial(root.find(f".//body[@name='{n}']/inertial"))
            for e in adds[n[2:]]:
                cur = B.combine(cur, e if n[0] == "R" else B.mirrored(e))
            B.set_mjcf_inertial(root.find(f".//body[@name='{n}']/inertial"), cur)
        root.insert(0, etree.Comment(f" {out}: {a.base} with more parts at Al 6061 (volume x 2.70 g/cm3): {', '.join(rows)}. Generated by tools/robot_model/al_parts_overlay/add_al_parts.py. Do not hand-edit; rebuild. "))
        tree.write(str(dst), encoding="utf-8", xml_declaration=False, pretty_print=True)
        print("wrote", dst)
    json.dump(dict(base=a.base, out=out, cad_document=J["doc"], cad_version=J["version"], cad_unsaved_edits=J["modified_unsaved"], parts=rows, total_mass_change_g=round(total, 3)),
              open(B.ASSETS / f"{out}_report.json", "w"), indent=1)
    print("wrote", f"{out}.urdf", "and the report")


if __name__ == "__main__":
    main()
