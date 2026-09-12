#!/usr/bin/env python3
"""Apply the established printed-part density survey to a read-only Fusion body dump.

Unlike ``tools/fusion/set_printed_density.py`` this program never edits or saves the CAD.
It reproduces that script's scope and ratio rules on JSON mass properties, scales the full
origin inertia tensor with mass, and writes a provenance manifest.  Prototype aluminium
overrides are deliberately a later stage: run ``prototype_tempmass.py`` on this output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
AL = 2.70
KEEP_AL = ("Arm_A", "Arm_B")
SKIP_OCC = ("CenterPin", "NoSim", "DR2020", "DF2020")
SHOULDER_SUBTREES = ("Shoulder-Pitch2Roll", "Shoulder-Roll2Yaw-dummy", "DummyHand")
ALT_BRANCH = ("NotUse", "fullDoF", "REF", "NoSim")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    bodies = json.loads(args.source.read_text())
    measured = json.loads((REPO / "tools/robot_model/alu_parts_measured.json").read_text())
    stats = json.loads((REPO / "tools/robot_model/alu_parts_ratio_stats.json").read_text())
    v5 = json.loads(Path("/home/syaro/pyg_fea/fusion/alu_parts_v5/index.json").read_text())
    mean_ratio = float(stats["confident"]["mean"])
    ratio_by_body = {}
    for entry in measured["entries"]:
        if entry["g"] is None:
            continue
        rows = [row for row in v5 if row["body"] == entry["body"]]
        if len(rows) == 1:
            ratio_by_body[entry["body"]] = {
                "ratio": entry["g"] / rows[0]["mass_g"], "confidence": entry["conf"],
            }

    changes = []
    for path, rec in bodies.items():
        body = path.split("::")[-1]
        occ = path.split("::")[0].split("/")[-1]
        material = rec.get("mat", "")
        in_leg = path.startswith("/Joints_UnderBody")
        in_shoulder = any(token in path for token in SHOULDER_SUBTREES)
        if not (in_leg or in_shoulder) or any(token in path for token in ALT_BRANCH):
            continue
        accepted = "Alumin" in material or material.startswith("PLA ")
        if in_shoulder:
            accepted = accepted or material.strip() == "Steel"
        if not accepted or body in KEEP_AL or any(token in occ for token in SKIP_OCC):
            continue

        own = ratio_by_body.get(body)
        if own and own["confidence"] not in ("high", "med"):
            own = None
        ratio = own["ratio"] if own else mean_ratio
        density = ratio * AL
        old_mass = float(rec["m"])
        new_mass = float(rec["v"]) * density / 1000.0
        scale = new_mass / old_mass
        rec["m"] = new_mass
        rec["I"] = [float(value) * scale for value in rec["I"]]
        rec["mat"] = f"PLA effective {density:.4f} g/cm3 (docs/89 survey)"
        changes.append({
            "path": path, "body": body, "old_mass_kg": old_mass,
            "new_mass_kg": new_mass, "density_g_cm3": density,
            "ratio_source": f"measured-{own['confidence']}" if own else "confident-mean",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bodies, indent=1), encoding="utf-8")
    manifest = {
        "tag": "printed-density-snapshot", "status": "offline; Fusion CAD unmodified",
        "source_dump": str(args.source), "output_dump": str(args.out),
        "survey": "docs/89; same scope/rules as tools/fusion/set_printed_density.py",
        "mean_ratio": mean_ratio, "changed_body_count": len(changes), "changes": changes,
        "next_stage": "prototype_tempmass.py overrides the known aluminium hardware",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    before = sum(change["old_mass_kg"] for change in changes)
    after = sum(change["new_mass_kg"] for change in changes)
    print(f"changed {len(changes)} printed bodies: {before:.3f} -> {after:.3f} kg")
    print(f"body dump -> {args.out}")
    print(f"manifest  -> {args.manifest}")


if __name__ == "__main__":
    main()
