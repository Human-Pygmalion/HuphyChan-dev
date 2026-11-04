#!/usr/bin/env python3
"""Every robot model file (URDF / MJCF) must be named in the version ledger docs/186_urdf_mjcf_version_ledger.md.
Exit code 1 and a list when a file name is missing. Run after building a model, before committing."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
DIRS = [REPO / "pygmalion_locomotion/assets/pygmalion_v2", REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"]
ledger = (REPO / "docs/186_urdf_mjcf_version_ledger.md").read_text(encoding="utf-8")
names = sorted({p.stem for d in DIRS for p in list(d.glob("*.urdf")) + list(d.glob("*.xml"))})
mentioned = lambda n: f"`{n}`" in ledger or f"`{n}." in ledger   # the full name, in backticks (no abbreviations with ...)
missing = [n for n in names if not mentioned(n)]
print(f"{len(names)} model names, {len(missing)} missing from docs/186")
for n in missing:
    print("  MISSING", n)
sys.exit(1 if missing else 0)
