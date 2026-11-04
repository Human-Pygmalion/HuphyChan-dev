#!/usr/bin/env python3
"""Build the VISUALIZATION of which parts are aluminium (not part of the delivered model).
  * red       = converted to Al 6061 in rounds 1-2 (cranks A/B, HipRoll2Yaw A/B/Mount, Yaw2KneePlateA, Knee2Ankle Idle/Rotor side 1+2, revised AnkleUniversalJointCore)
  * dark red  = was already aluminium before (AnkleA2B_A_* braces, Arm_A / Arm_B rods; the 'prototype-tempmass' rule)
  * grey translucent = everything else (printed parts, motors, bearings, fasteners, the v5 foot, camera, plates ...)
Writes <out>/vis/<model>_alvis.xml (absolute mesh paths) and <out>/vis/meshes/*.stl, then `render_al_parts.py` renders it on the 3090.
Part meshes come from the CAD (one leg, tools/robot_model/fusion_snapshots/alparts_20261104/cad_mm); the other leg is the y-mirror."""
import json, os, sys
from pathlib import Path
import numpy as np
import trimesh
from lxml import etree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_al_round2 as R2  # noqa: E402
B = R2.B
OUTDIR = Path(sys.argv[1] if len(sys.argv) > 1 else R2.SNAP / "vis")
CAD = R2.SNAP / "cad_mm"
LINK = {  # link key -> (model body suffix, origin CAD mm)
    "hip_roll_link": ("hip_roll_link", np.array([-123.7, 70.0, 60.0])), "thigh": ("thigh_link", np.array([-123.7, 70.0, 60.0])),
    "shin": ("shin_link", np.array([-123.7, 115.0, -310.0])), "ankle_pitch_link": ("ankle_pitch_link", np.array([-123.7, 145.0, -800.0])),
    "foot": ("foot_link", np.array([-123.7, 145.0, -800.0])),
    "crank_A": ("crank_A", np.array([-125.418, 144.945, -500.034])), "crank_B": ("crank_B", np.array([-121.983, 145.055, -600.034])),
    "rod_A": ("rod_A", np.array([-82.7, 206.439, -521.219])), "rod_B": ("rod_B", np.array([-164.7, 205.094, -615.254]))}
RED, DARK = "0.95 0.08 0.08 1", "0.55 0.03 0.03 1"
GREY = "0.72 0.76 0.82 0.12"


def main():
    meta = json.load(open(R2.SNAP / "mesh_meta.json"))
    (OUTDIR / "meshes").mkdir(parents=True, exist_ok=True)
    tree = etree.parse(str(B.ASSETS / f"{R2.OUT}.xml")); root = tree.getroot(); asset = root.find("asset")
    root.find("compiler").set("meshdir", str(B.ASSETS / "meshes_v30_armfix") + "/")
    for m in asset.findall("mesh"):                                  # absolute paths so the folder is self-contained once rsync -aL'd
        f = m.get("file")
        if f:
            m.set("file", os.path.normpath(str(B.ASSETS / "meshes_v30_armfix" / f)))   # lexical: the meshes_v30_armfix link must not be followed before ../
    for g in root.iter("geom"):                                      # everything drawn so far becomes translucent
        if g.get("class") == "visual":
            g.set("rgba", GREY)
    n_red = 0
    for pid, info in meta.items():
        suffix, O = LINK[info["link"]]
        for s in "RL":
            t = trimesh.load(CAD / f"{pid}.stl", process=False)
            v = (np.asarray(t.vertices) - O) @ B.R_SIM.T / 1000.0
            f = np.asarray(t.faces)
            if s == "L":
                v, f = v @ B.MIRROR, f[:, ::-1]
            name = f"{s}_{pid}"
            trimesh.Trimesh(v, f, process=False).export(OUTDIR / "meshes" / f"{name}.stl")
            etree.SubElement(asset, "mesh", name=f"vis_{name}", file=str((OUTDIR / "meshes" / f"{name}.stl").resolve()))
            body = root.find(f".//body[@name='{s}_{suffix}']")
            etree.SubElement(body, "geom", {"class": "visual", "group": "1", "mesh": f"vis_{name}", "rgba": RED if info["group"] == "new" else DARK})   # group 1: drawn in its own pass
            n_red += 1
    # the revised core is already in the model (grey): hide that copy so only the red one shows
    for g in root.iter("geom"):
        if (g.get("mesh") or "").endswith("_ankle_core_revised") and not (g.get("mesh") or "").startswith("vis_"):
            g.set("rgba", "0 0 0 0")
    tree.write(str(OUTDIR / f"{R2.OUT}_alvis.xml"), encoding="utf-8", xml_declaration=False, pretty_print=True)
    print(f"{n_red} red meshes -> {OUTDIR}")


if __name__ == "__main__":
    main()
