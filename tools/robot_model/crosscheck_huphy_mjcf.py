#!/usr/bin/env python3
"""Cross-check Pygmalion hip/knee axis *lines* against Human-Pygmalion/huphy_mjcf.

This deliberately compares infinite hinge lines, not the XML anchor coordinates.  Any point
along a hinge axis is an equivalent joint anchor, so comparing anchor XYZ directly creates a
false geometry alarm.

The independent repository uses a serial pitch/roll ankle.  Its ankle joints therefore cannot
validate this project's 2-RSU crank A/B topology or motor ordering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OURS = (ROOT / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/"
                "FullDoF_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml")


def ref_point_to_ours(p: np.ndarray) -> np.ndarray:
    """Reference world -> this project's base frame, metres."""
    x, y, z = p
    return np.array([0.070 - y, x, z - 0.060], dtype=float)


def ref_dir_to_ours(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([-y, x, z], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def vec(v: np.ndarray, scale: float = 1.0) -> list[float]:
    return [round(float(x * scale), 4) for x in v]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--reference-commit", default="unknown")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tools/robot_model/huphy_mjcf_crosscheck.json")
    args = ap.parse_args()

    ours = mujoco.MjModel.from_xml_path(str(args.ours))
    ref = mujoco.MjModel.from_xml_path(str(args.reference))
    od, rd = mujoco.MjData(ours), mujoco.MjData(ref)
    if ours.nq >= 7 and ours.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        od.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    if ref.nq >= 7 and ref.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        rd.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    mujoco.mj_forward(ours, od)
    mujoco.mj_forward(ref, rd)

    result = {
        "schema": "pygmalion-independent-mjcf-crosscheck-v1",
        "ours": str(args.ours.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_repository": "https://github.com/Human-Pygmalion/huphy_mjcf",
        "reference_commit": args.reference_commit,
        "frame_transform": {
            "point_m": "[0.070-y_ref, x_ref, z_ref-0.060]",
            "direction": "[-y_ref, x_ref, z_ref]",
            "side_map": {"reference_R": "ours_L", "reference_L": "ours_R"},
        },
        "scope": {
            "validated": "hip pitch/roll/yaw and knee hinge axis lines at neutral",
            "not_validated": "2-RSU crank A/B topology, motor IDs, offsets, or hardware signs",
        },
        "joints": {},
    }

    for ref_side, our_side in (("R", "L"), ("L", "R")):
        for stem in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
            rname, oname = f"{ref_side}_{stem}", f"{our_side}_{stem}_joint"
            rjid, ojid = ref.joint(rname).id, ours.joint(oname).id
            rp = ref_point_to_ours(rd.xanchor[rjid].copy())
            ra = unit(ref_dir_to_ours(rd.xaxis[rjid].copy()))
            op, oa = od.xanchor[ojid].copy(), unit(od.xaxis[ojid].copy())

            # Align the two line directions before measuring their perpendicular separation.
            axis_dot = float(oa @ ra)
            ra_line = ra if axis_dot >= 0 else -ra
            delta = rp - op
            perpendicular = delta - oa * float(delta @ oa)

            aid = ref.actuator(rname).id
            gear = float(ref.actuator_gear[aid, 0])
            motor_plus_axis = unit(ra * gear)
            sign_dot = float(oa @ motor_plus_axis)
            sign = 1 if sign_dot > 0.9 else -1 if sign_dot < -0.9 else None

            result["joints"][oname] = {
                "reference_joint": rname,
                "ours_anchor_mm": vec(op, 1000),
                "reference_anchor_transformed_mm": vec(rp, 1000),
                "ours_axis": vec(oa),
                "reference_axis_transformed": vec(ra),
                "axis_abs_dot": round(abs(axis_dot), 6),
                "axis_line_perpendicular_distance_mm": round(float(np.linalg.norm(perpendicular) * 1000), 4),
                "reference_motor_gear": gear,
                "reference_motor_plus_axis_transformed": vec(motor_plus_axis),
                "raw_to_model_sign_candidate": sign,
                "sign_evidence": "independent MJCF gear; raw hardware log provenance not committed",
            }

    distances = [j["axis_line_perpendicular_distance_mm"] for j in result["joints"].values()]
    result["verdict"] = {
        "axis_lines": "PASS" if max(distances) <= 1.0 else "REVIEW",
        "max_perpendicular_distance_mm": max(distances),
        "motor_signs": "PROVISIONAL_CROSSCHECK_ONLY",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
