#!/usr/bin/env python3
"""Copy a foot-variant MJCF and replace both <side>_foot_link inertials with the ones in a URDF (same foot_link frame).
Use: the v5 foot URDF (tools/robot_model/foot_overlay/build_footv5.py) carries the full measured foot mass, including the
Feet_Up toe-box cover (80 g) that v3_collision.py does not add. Geometry is untouched; only the two <inertial> lines change.
  python3 tools/foot_sole/foot_inertial_from_urdf.py <src.xml> <foot.urdf> <out.xml>
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC, URDF, OUT = (Path(a) for a in sys.argv[1:4])
links = {l.get("name"): l.find("inertial") for l in ET.parse(URDF).getroot().findall("link")}
s = SRC.read_text()
for side in ("R", "L"):
  i = links[f"{side}_foot_link"]
  pos = i.find("origin").get("xyz")
  m = i.find("mass").get("value")
  I = {k: float(v) for k, v in i.find("inertia").attrib.items()}
  full = " ".join(f"{I[k]:.6e}" for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
  new = f'<inertial pos="{pos}" mass="{m}" fullinertia="{full}"/>'
  pat = re.compile(rf'(<body name="{side}_foot_link"[^>]*>\s*)<inertial[^>]*/>')
  s, n = pat.subn(lambda mo: mo.group(1) + new, s, count=1)
  assert n == 1, f"{side}_foot_link inertial not found in {SRC}"
  print(side, new)
OUT.write_text(s)
print("wrote", OUT)
