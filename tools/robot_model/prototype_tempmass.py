#!/usr/bin/env python3
"""Build the temporary aluminium-body input for ``prototype-tempmass``.

This is intentionally a provenance-preserving override of the v4 printed Fusion dump.
The target values are the existing aluminium-6061 column (D) in
``docs/aluminium_parts_3dprint_masses.xlsx``.  They are CAD-density estimates, except
Arm_A/Arm_B which the sheet identifies as existing aluminium parts; they are *not* being
promoted to final physical weighings by this script.

Only the nine explicitly named CAD bodies are changed.  The simulator mirrors the
one-leg CAD into left/right links when aggregating the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/syaro/pyg_fea/fusion/bodies_v4_printed.json")
DEFAULT_OUT = Path("/tmp/pygmalion_prototype-tempmass_bodies.json")
DEFAULT_MANIFEST = REPO / "tools/robot_model/prototype_tempmass_manifest.json"

# Snapshot of workbook sheet "알루미늄 부품", column D, rows shown below (grams).
TARGET_G = {
    "AnkleA2B_A_Support": (40.82, 40),
    "AnkleA2B_A_Support (1)": (40.82, 41),
    "AnkleA2B_A_R (1)": (32.27, 42),
    "AnkleA2B_A_L (1)": (32.27, 43),
    "AnkleA2B_A_L": (32.03, 44),
    "AnkleA2B_A_R": (31.84, 46),
    "FEET": (617.72, 14),
    "Arm_A": (54.76, 26),
    "Arm_B": (41.55, 27),
}


def in_scope(path: str, body: str) -> bool:
    if body.startswith("AnkleA2B_"):
        return "/Knee2Ankle:1::" in path
    return body in {"FEET", "Arm_A", "Arm_B"} and "/Ankle2Feet:1::" in path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    bodies = json.loads(args.source.read_text())
    changed = []
    found = set()
    for path, rec in bodies.items():
        body = path.split("::")[-1]
        if body not in TARGET_G or not in_scope(path, body):
            continue
        if body in found:
            raise RuntimeError(f"duplicate prototype target: {body}: {path}")
        found.add(body)
        target_g, row = TARGET_G[body]
        old_m = float(rec["m"])
        new_m = target_g / 1000.0
        scale = new_m / old_m
        # Fusion I is about the assembly origin.  Fixed geometry+density means both mass
        # and the complete origin tensor scale by the same factor; COM is unchanged.
        rec["m"] = new_m
        rec["I"] = [float(v) * scale for v in rec["I"]]
        rec["mat"] = "Prototype Aluminum 6061 (worksheet column D)"
        changed.append({
            "body": body,
            "path": path,
            "workbook_row": row,
            "old_mass_kg": old_m,
            "prototype_mass_kg": new_m,
            "tensor_scale": scale,
        })

    missing = set(TARGET_G) - found
    if missing or len(changed) != 9:
        raise RuntimeError(f"prototype scope mismatch: changed={len(changed)}, missing={sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bodies, indent=1))
    manifest = {
        "tag": "prototype-tempmass",
        "status": "CONDITIONAL prototype only; not final Sim2Real mass evidence",
        "source_dump": str(args.source),
        "source_workbook": str(REPO / "docs/aluminium_parts_3dprint_masses.xlsx"),
        "source_column": "D: 질량 (알루 6061, g)",
        "important_evidence_note": (
            "Column D is CAD geometry at 6061 density, not a new physical aluminium "
            "weighing. Arm_A/Arm_B are marked existing aluminium/CAD-mass in the sheet."
        ),
        "scope_assumption": (
            "Latest user wildcard AnkleA2B_*_* is interpreted as all six matching CAD "
            "bodies, plus FEET/Arm_A/Arm_B. The build mirrors the one-leg CAD."
        ),
        "output_bodies": str(args.out),
        "changes": sorted(changed, key=lambda x: x["body"]),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"changed {len(changed)} bodies -> {args.out}")
    for item in sorted(changed, key=lambda x: x["body"]):
        print(f"  {item['body']:28s} {item['old_mass_kg']:.6f} -> {item['prototype_mass_kg']:.5f} kg")
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
