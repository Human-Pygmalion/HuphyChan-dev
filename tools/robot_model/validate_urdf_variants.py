#!/usr/bin/env python3
"""Hard gate for the FullDoF, SemiFullDoF and LegOnly URDF contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


EXPECTED_ROM_DEG = {
    "hip_pitch": (-120, 25), "hip_roll": (-85, 25), "hip_yaw": (-45, 45),
    "knee": (-120, 0), "ankle_pitch": (-50, 30), "ankle_roll": (-20, 20),
    "waist_yaw": (-45, 45), "shoulder_pitch": (-90, 60),
    "shoulder_roll": (-32, 30),
}
UPPER_TOKENS = ("torso", "shoulder", "arm", "waist")


def suffix(name: str) -> str | None:
    stem = name.removesuffix("_joint")
    for key in EXPECTED_ROM_DEG:
        if stem == key or stem.endswith("_" + key):
            return key
    return None


def audit(path: Path, variant: str) -> dict:
    root = ET.parse(path).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    movable = [j for j in joints if j.get("type") in ("revolute", "continuous", "prismatic")]
    fixed = [j for j in joints if j.get("type") == "fixed"]
    expected = {"full": 17, "semi": 13, "leg": 12}[variant]
    assert len(movable) == expected, (path, len(movable), expected)

    link_names = {link.get("name") for link in links}
    children = {j.find("child").get("link") for j in joints}
    roots = link_names - children
    assert roots == {"base_link"}, (path, roots)
    assert len(link_names) == len(links), f"{path}: duplicate link name"
    assert len({j.get('name') for j in joints}) == len(joints), f"{path}: duplicate joint name"

    for link in links:
        inertial = link.find("inertial")
        assert inertial is not None, f"{path}: {link.get('name')} has no inertial"
        mass = float(inertial.find("mass").get("value"))
        it = inertial.find("inertia")
        I = np.array([
            [float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
            [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
            [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))],
        ])
        eig = np.linalg.eigvalsh(I)
        assert mass > 0 and eig.min() > 0, (path, link.get("name"), mass, eig)
        principal = eig
        assert principal.max() <= principal.sum() - principal.max() + 1e-8, (
            path, link.get("name"), "inertia triangle inequality", principal)

    for joint in movable:
        key = suffix(joint.get("name"))
        assert key is not None, (path, "unknown movable joint", joint.get("name"))
        target = np.radians(EXPECTED_ROM_DEG[key])
        limit = joint.find("limit")
        actual = np.array([float(limit.get("lower")), float(limit.get("upper"))])
        assert np.allclose(actual, target, atol=2e-6), (path, joint.get("name"), actual, target)

    if variant == "semi":
        shoulder = {j.get("name"): j for j in fixed if "shoulder" in j.get("name")}
        assert len(shoulder) == 4, shoulder
        assert np.isclose(float(shoulder["L_shoulder_roll_joint"].find("origin").get("rpy").split()[0]),
                          np.radians(-15), atol=1e-8)
        assert np.isclose(float(shoulder["R_shoulder_roll_joint"].find("origin").get("rpy").split()[0]),
                          np.radians(15), atol=1e-8)
    if variant == "leg":
        forbidden_links = [name for name in link_names if any(t in name.lower() for t in UPPER_TOKENS)]
        assert not forbidden_links, (path, forbidden_links)
        filenames = [mesh.get("filename", "") for mesh in root.findall(".//mesh")]
        forbidden_meshes = [name for name in filenames if any(t in name.lower() for t in UPPER_TOKENS)]
        assert not forbidden_meshes, (path, forbidden_meshes)

    return {
        "path": str(path), "links": len(links), "joints": len(joints),
        "movable": len(movable), "fixed": len(fixed), "root": "base_link",
        "verdict": "PASS",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--semi", type=Path, required=True)
    ap.add_argument("--leg", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {key: audit(getattr(args, key), key) for key in ("full", "semi", "leg")}
    report["verdict"] = "PASS"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
