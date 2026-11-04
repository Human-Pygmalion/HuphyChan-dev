#!/usr/bin/env python3
"""Combine the two overlays of the LegOnly AB (_loop) model: v5 foot (CAD sole + Feet_Up) and the pelvis D455 camera + PLA bracket.

Both overlays were built separately from the same base (`..._proxyfix_loop`): `build_footv5.py` changes only the two foot links,
`build_d455.py` / `build_d455_mjcf.py` only append bodies to base_link. They do not overlap, so the combination is exact:
  URDF  `<name>_footv5_d455_loop.urdf` = the footv5 URDF + the D455 block (copied character for character from the d455 URDF; the
        script asserts that the d455 URDF is base + block, so nothing else is taken from it).
  MJCF  `<name>_footv5_d455_loop.xml`  = the shipped base MJCF with, per foot: the `<inertial>` and the collision geoms (3 upper boxes + 29 pieces) of the
        `v5fullm` MJCF (Feet_design study; inertia = the URDF values), plus the four CAD visual meshes (carrier, tape, rubber, Feet_Up);
        then the D455 body, bracket and camera exactly as build_d455_mjcf.py puts them.
Nothing shipped is modified. Two MJCF copies, like the D455 build: one in the mjlab `xmls/` folder (meshdir assets_v30_armfix, the new
mesh folder is reached through the relative link `xmls/assets_v30_footv5`), one next to the URDF in this repo (meshdir
meshes_v30_armfix, plain MuJoCo). Also turns the two absolute mesh symbolic links of the asset folder into relative ones.
The camera pose comes from `d455_overlay_report.json`; the foot numbers from `..._footv5_loop.urdf` (written by build_footv5.py).

usage (mjlab venv): build_footv5_d455.py
"""
import copy
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from lxml import etree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools/robot_model/camera_overlay"))
sys.path.insert(0, str(HERE))
import build_d455 as B  # noqa: E402  (constants, MASS, SIZE, ASSETS)
import build_footv5 as F  # noqa: E402  (part colours)

ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
XMLS = REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
V5FULLM = REPO / "Feet_design/Feet_Design_Workspace/mjcf_sim/union_branch/mjlab_policy/pygmalion_mjlab/assets/pygmalion_legonly_ab_v5fullm.xml"
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop"
FOOT = BASE.replace("_loop", "_footv5_loop")
D455 = BASE.replace("_loop", "_d455_loop")
OUT = BASE.replace("_loop", "_footv5_d455_loop")
LINK_D455, LINK_FOOT = "assets_v30_d455", "assets_v30_footv5"


def relink(link, target):
  """absolute symbolic link -> relative one (same target)"""
  link = Path(link)
  rel = os.path.relpath(Path(target), link.parent)
  if link.is_symlink():
    link.unlink()
  link.symlink_to(rel)
  return rel


def build_urdf():
  base = (ASSETS / f"{BASE}.urdf").read_text(encoding="utf-8")
  d455 = (ASSETS / f"{D455}.urdf").read_text(encoding="utf-8")
  foot = (ASSETS / f"{FOOT}.urdf").read_text(encoding="utf-8")
  head = re.sub(r'<robot name="[^"]*">', f'<robot name="{D455}">', base, count=1).rstrip()[: -len("</robot>")]
  assert d455.startswith(head) and d455.endswith("</robot>\n"), "the d455 URDF is not base + appended block"
  block = d455[len(head): -len("</robot>\n")]
  assert foot.rstrip().endswith("</robot>") and "<link name=\"d455_link\">" not in foot
  out = re.sub(r'<robot name="[^"]*">', f'<robot name="{OUT}">', foot, count=1).rstrip()[: -len("</robot>")] + block + "</robot>\n"
  (ASSETS / f"{OUT}.urdf").write_text(out, encoding="utf-8")
  shutil.copyfile(ASSETS / f"{BASE}.motor_params.json", ASSETS / f"{OUT}.motor_params.json")
  print("wrote", f"{OUT}.urdf", f"(+{len(block)} chars of D455 block)")


def fmt(v):
  return " ".join(f"{x:.9g}" for x in v)


def build_mjcf():
  tree = etree.parse(str(XMLS / f"{BASE}.xml"))
  root = tree.getroot()
  root.set("model", OUT)
  asset = root.find("asset")
  src = etree.parse(str(V5FULLM)).getroot()
  # ---- feet: collision pieces + inertial from v5fullm, CAD visuals
  n_pieces = 0
  for S in ("R", "L"):
    for m in src.find("asset").findall("mesh"):
      if m.get("name").startswith(f"{S}_v5full_"):
        asset.append(copy.deepcopy(m))
        n_pieces += 1
    for part in F.PARTS:
      etree.SubElement(asset, "mesh", name=f"{S}_v5_{part}", file=f"../{LINK_FOOT}/{S}_v5_{part}.stl")
    bb = root.find(f".//body[@name='{S}_foot_link']")
    bs = src.find(f".//body[@name='{S}_foot_link']")
    old = bb.find("inertial")
    bb.replace(old, copy.deepcopy(bs.find("inertial")))
    old_cols = [g for g in bb.findall("geom") if re.fullmatch(rf"{S}_foot\d+_collision", g.get("name") or "")]
    new = [g for g in bs.findall("geom") if re.fullmatch(rf"{S}_foot\d+_collision", g.get("name") or "")]
    assert len(old_cols) == 3 and len(new) == 32, (len(old_cols), len(new))   # v5fullm also lifts the 3 upper boxes by 4 mm: take all 32
    idx = list(bb).index(old_cols[0])
    for g in old_cols:
      bb.remove(g)
    for k, g in enumerate(new):
      bb.insert(idx + k, copy.deepcopy(g))
    vis = bb.findall("geom[@class='visual']")[-1]
    vidx = list(bb).index(vis)
    for k, (part, (_, _, rgba)) in enumerate(F.PARTS.items()):
      g = etree.Element("geom", {"class": "visual", "mesh": f"{S}_v5_{part}", "rgba": rgba})
      bb.insert(vidx + 1 + k, g)
  # ---- D455 body, bracket, camera: same as build_d455_mjcf.py
  rep = json.load(open(ASSETS / "d455_overlay_report.json"))
  pos, pitch = rep["d455_link_in_base_link_m"], rep["d455_mount_rpy_rad"][1]
  quat = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
  W, H, HFOV, PIXEL = 848, 480, 87.0, 3.0e-6
  fx = (W / 2) / math.tan(math.radians(HFOV / 2))
  w, d, h = B.SIZE[1], B.SIZE[0], B.SIZE[2]
  diag = [B.MASS * (w * w + h * h) / 12, B.MASS * (d * d + h * h) / 12, B.MASS * (d * d + w * w) / 12]
  etree.SubElement(asset, "mesh", name="d455", file=f"../{LINK_D455}/d455.stl")
  etree.SubElement(asset, "mesh", name="pelvis_camera_mount", file=f"../{LINK_D455}/pelvis_camera_mount.stl")
  base = root.find(".//body[@name='base_link']")
  body = etree.SubElement(base, "body", name="d455_link", pos=fmt(pos), quat=fmt(quat))
  etree.SubElement(body, "inertial", pos=fmt(B.COM_IN_LINK), mass=str(B.MASS), diaginertia=fmt(diag))
  etree.SubElement(body, "geom", {"class": "visual", "mesh": "d455", "rgba": "0.10 0.35 0.85 1"})
  etree.SubElement(body, "camera", name="d455_depth", pos="0 0 0", xyaxes="0 -1 0 0 0 1", resolution=f"{W} {H}",
                   sensorsize=f"{W * PIXEL:.6g} {H * PIXEL:.6g}", focal=f"{fx * PIXEL:.7g} {fx * PIXEL:.7g}")
  mnt = rep["mount"]
  mI = np.array(mnt["inertia_base_link_kg_m2"])
  mb = etree.SubElement(base, "body", name="pelvis_camera_mount", pos="0 0 0")
  etree.SubElement(mb, "inertial", pos=fmt(mnt["com_base_link_m"]), mass=f"{mnt['mass_kg']:.9g}",
                   fullinertia=fmt([mI[0, 0], mI[1, 1], mI[2, 2], mI[0, 1], mI[0, 2], mI[1, 2]]))
  etree.SubElement(mb, "geom", {"class": "visual", "mesh": "pelvis_camera_mount", "rgba": "0.86 0.86 0.80 1"})
  root.insert(0, etree.Comment(f" {OUT}: v5 foot (collision + inertial from v5fullm, CAD visuals) + D455 overlay. Generated by "
                               "tools/robot_model/foot_overlay/build_footv5_d455.py. Do not hand-edit; rebuild. "))
  # ---- two copies
  XMLS_LINK = XMLS / LINK_FOOT
  print("link", XMLS_LINK.name, "->", relink(XMLS_LINK, ASSETS / "meshes_footv5"))
  # the D455 link of the mjlab folder is an absolute link written by build_d455_mjcf.py: it keeps working, left as it is
  tree.write(str(XMLS / f"{OUT}.xml"), encoding="utf-8", xml_declaration=False, pretty_print=True)
  print("wrote", XMLS / f"{OUT}.xml", f"({n_pieces} collision mesh assets)")
  root.find("compiler").set("meshdir", "meshes_v30_armfix/")
  for m in asset.findall("mesh"):
    f = m.get("file") or ""
    for a, b in ((f"../{LINK_D455}/", "../meshes_d455/"), (f"../{LINK_FOOT}/", "../meshes_footv5/")):
      if f.startswith(a):
        m.set("file", f.replace(a, b))
  tree.write(str(ASSETS / f"{OUT}.xml"), encoding="utf-8", xml_declaration=False, pretty_print=True)
  print("wrote", ASSETS / f"{OUT}.xml")


if __name__ == "__main__":
  for n in ("meshes_v30_armfix", "meshes_v30_legonly"):
    tgt = REPO / "tools/robot_model/fusion_snapshots/v30_inspection" / n.replace("meshes_v30_", "meshes_")
    assert tgt.is_dir(), tgt
    print(n, "->", relink(ASSETS / n, tgt))
  build_urdf()
  build_mjcf()
