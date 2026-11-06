#!/bin/bash
# Put the M19 robot model into a Union-DR training checkout (release 49999 or later) and check that it loads.
#   bash install_m19.sh <path to mjlab_policy/pygmalion_mjlab/assets> [python with mujoco]
# Copies: pygmalion_legonly_ab_m19.xml (= the ledger MJCF LegOnly_..._footv5_d455_reinf_alparts2_loop.xml) and the four mesh folders it reads
# (meshes_v30_armfix, meshes_footv5, meshes_reinforce, meshes_d455), as real files (the source folder holds symlinks for meshes_v30_*).
set -euo pipefail
DST="${1:?target assets dir (mjlab_policy/pygmalion_mjlab/assets)}"; PY="${2:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../../assets/pygmalion_v2" && pwd)"
NAME=LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_loop
[ -d "$DST" ] || { echo "no such dir: $DST" >&2; exit 2; }
cp -f "$SRC/$NAME.xml" "$DST/pygmalion_legonly_ab_m19.xml"
for d in meshes_v30_armfix meshes_footv5 meshes_reinforce meshes_d455; do rm -rf "$DST/$d"; cp -rL "$SRC/$d" "$DST/$d"; done
"$PY" - "$DST/pygmalion_legonly_ab_m19.xml" <<'P'
import sys, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1])
mass = float(m.body_mass.sum())
print(f"loaded: bodies={m.nbody} joints={m.njnt} geoms={m.ngeom} mass={mass:.3f} kg (expected 26.911)")
assert abs(mass - 26.911) < 0.01, f"mass {mass:.3f} != 26.911"
P
echo "next: apply robot_py_on_49999_release.patch to pygmalion_mjlab/robot.py if your robot.py lacks PYG_FOOT_XML / PYG_INIT_DZ / foot\\d+ (see README)"
