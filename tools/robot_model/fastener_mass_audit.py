#!/usr/bin/env python3
"""Audit screw bodies included in the mirrored Pygmalion mass aggregation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import massprops_fusion as mp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BODIES = ROOT / "tools/robot_model/fusion_snapshots/v30_inspection/bodies_printed_prototype_tempmass.json"
DEFAULT_OUT = ROOT / "tools/robot_model/fusion_snapshots/v30_inspection/fastener_mass_audit.json"


def is_fastener(path: str) -> bool:
    return any(token in path.lower() for token in ("screw", "bolt", "nut", "washer"))


def main() -> None:
    bodies_path = Path(os.environ.get("PYG_BODIES", DEFAULT_BODIES))
    out_path = Path(os.environ.get("PYG_FASTENER_AUDIT_OUT", DEFAULT_OUT))
    source = json.loads(bodies_path.read_text())
    grouped, _, _ = mp.collect(source)
    links = {}
    for link, items in grouped.items():
        multiplicity = 1 if link in ("pelvis", "torso") else 2
        selected = [item for item in items if is_fastener(item["path"])]
        if selected:
            links[link] = {
                "physical_count": len(selected) * multiplicity,
                "physical_mass_kg": sum(item["m"] for item in selected) * multiplicity,
                "multiplicity": multiplicity,
            }
    unassigned = []
    for path, record in source.items():
        if not is_fastener(path) or mp.is_alternative(path) or mp.classify(path) is not None:
            continue
        unassigned.append({"path": path, "mass_kg": record["m"],
                           "visible_in_cad": bool(record.get("live", True))})
    report = {
        "schema": "pygmalion-fastener-mass-audit-v1",
        "source": str(bodies_path),
        "model_mode": "AB loop aggregation; one-side leg/arm CAD mirrored L/R",
        "included": {
            "physical_count": sum(row["physical_count"] for row in links.values()),
            "physical_mass_kg": sum(row["physical_mass_kg"] for row in links.values()),
            "by_link": links,
        },
        "unassigned_raw_cad": {
            "count": len(unassigned),
            "mass_kg": sum(row["mass_kg"] for row in unassigned),
            "visible_count": sum(row["visible_in_cad"] for row in unassigned),
            "visible_mass_kg": sum(row["mass_kg"] for row in unassigned if row["visible_in_cad"]),
            "items": unassigned,
        },
        "comparison_to_user_2kg": {
            "user_value_kg": 2.0,
            "included_minus_user_kg": sum(row["physical_mass_kg"] for row in links.values()) - 2.0,
            "status": "NOT A CONTRADICTION: user confirmed 2 kg was rough memory, not a scale measurement",
            "remaining_uncertainty": (
                "SCM/CAD screw population may be only about 50% complete; evaluate a correlated "
                "fastener-completeness event instead of forcing the nominal total to 2.000 kg"
            ),
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["included"], ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
