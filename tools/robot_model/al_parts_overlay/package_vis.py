#!/usr/bin/env python3
"""Make the visualization folder self-contained (meshes copied with links followed, relative paths) -> <pkg dir>. usage: package_vis.py <pkg dir>"""
import shutil, sys
from pathlib import Path
from lxml import etree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_al_round2 as R2
pkg = Path(sys.argv[1]); (pkg / "meshes").mkdir(parents=True, exist_ok=True)
src = R2.SNAP / "vis" / f"{R2.OUT}_alvis.xml"
tree = etree.parse(str(src)); root = tree.getroot()
root.find("compiler").set("meshdir", "meshes/")
for m in root.find("asset").findall("mesh"):
    f = m.get("file")
    if f:
        shutil.copyfile(f, pkg / "meshes" / f"{m.get('name')}.stl")
        m.set("file", f"{m.get('name')}.stl")
tree.write(str(pkg / "alvis.xml"), encoding="utf-8", xml_declaration=False, pretty_print=True)
print("packaged", pkg)
