#!/usr/bin/env python3
"""List the leg parts that the current models carry at PRINTED density, with the mass they would have as Al 6061 (volume x 2.70 g/cm3).
Source: the body dump the models were built from (tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json).
Parts already aluminium (density >= 2.5), fasteners, bearings and motor placeholders are left out. Writes candidates_printed_leg_parts.md next to this file.
The dump is one CAD revision behind the live Fusion document: read the volumes again from Fusion (extract_by_name.py) before building."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
B = json.load(open(REPO / "tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json"))
LINK = {"HipPitch2Roll:1": "hip_pitch_link", "HipRoll2Yaw:1": "hip_roll_link", "HipYaw2Knee:1": "thigh_link", "Knee2Ankle:1": "shin_link", "Ankle2Feet:1": "(ankle group, see below)"}
SKIP = ("Hexagon", "Screw", "Washer", "Nut", "RS0", "RobStride", "Bearing", "6814", "6900", "6810", "Inner Ball", "Outer Shell", "EBIMU", "JMC")
DONE = {"Crank_A", "Crank_B", "HipRoll2Yaw_Mount"}      # already aluminium in model M18 (MODEL_VERSION_LEDGER.md)
rows = []
for k, v in B.items():
    if not k.startswith("/Joints_UnderBody:1/") or not v.get("live", True):
        continue
    occ, name = k.split("::")[0], k.split("::")[-1]
    grp = occ.split("/")[2] if occ.count("/") >= 2 else ""
    if grp not in LINK or any(s in k for s in SKIP):
        continue
    rho = v["m"] * 1000 / v["v"]
    if rho > 2.5 or v["v"] < 1.0:
        continue
    al = v["v"] * 2.70
    rows.append((grp, name + (" (done in M18)" if name in DONE else ""), v["v"], v["m"] * 1000, rho, al, 0.0 if name in DONE else al - v["m"] * 1000, LINK[grp], v["mat"][:34]))
rows.sort(key=lambda r: (r[0], -r[6]))
out = ["# Leg parts still at printed density in the current models (candidates for Al 6061)", "",
       "Mass as Al = CAD volume x 2.70 g/cm3. Delta is per leg (the other leg is the mirror, so the robot gains twice the sum).",
       "Source dump is one CAD revision behind the live Fusion document; volumes are re-read from Fusion before any build.", "",
       "| CAD group | part | volume cm3 | now g | now g/cm3 | as Al g | +g per leg | link |", "|---|---|---|---|---|---|---|---|"]
for g, n, v, m, r, a, d, l, mat in rows:
    out.append(f"| {g[:-2]} | {n} | {v:.2f} | {m:.1f} | {r:.4f} | {a:.1f} | {'-' if 'done' in n else '+%.1f' % d} | {l} |")
tot = sum(r[6] for r in rows)
out += ["", f"The {len([r for r in rows if 'done' not in r[1]])} parts not yet changed, as Al: +{tot:.0f} g per leg, +{2*tot/1000:.3f} kg for the robot.",
        "Rows marked 'done in M18' are already aluminium in the newest model. AnkleA2B_A_L / _A_R show a density above 0.89 because the `tempmass` step (ledger M06) already gave them aluminium mass from an older volume.",
        "Ankle2Feet parts are split over shin / ankle_pitch / foot / crank / rod by position (massprops_fusion.py); the foot link of the v5 foot models is replaced by the v5 foot, so foot-link parts need a separate decision.", ""]
(HERE / "candidates_printed_leg_parts.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
