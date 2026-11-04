#!/usr/bin/env python3
"""Put the v5 sole module (CAD component `Pygmalion_Foot_v5_RightFoot`) on the feet of the LegOnly AB (_loop) URDF.

The shipped models are NOT touched. This writes a sibling `<name>_footv5_loop.urdf` (+ the unchanged motor sidecar under the
new name) and new meshes in `meshes_footv5/`. Only the two `<link name="?_foot_link">` blocks differ from the base URDF; the
script asserts that every other character is identical. Generated files are never hand-edited; change this script and rebuild.

What is added to each foot link (frames: foot_link, metres; left foot = mirror y of the right foot):
  visual    carrier, tape, rubber sheet, Feet_Up: STEP files exported from the live CAD component on 2026-11-01
            (Feet_design/Feet_Design_Workspace/inputs/fusion_upload_20261101, via tools/foot_sole/upload_receiver.py) and
            tessellated by tools/foot_sole/step_to_foot_stl.py. The carrier is the user-edited one (160.1 cm3, 264 x 119 mm, a rim
            down to 2.1 mm above the ground). The existing plate/bracket visual and its convex hull stay as they are.
  collision the 29 convex pieces of the v5full MJCF (plate + 14 rubber strips + 14 exposed-carrier strips), checked against the
            CAD solid by tools/foot_sole/v5_collision_review.py (flat contact 155.6 vs 156.4 cm2, rolling height error 0.07 mm;
            the carrier rim is not modelled: it touches the ground only beyond ~40 deg of tilt).
            Feet_Up (the toe-box cover) gets no collision: no published humanoid foot model collides its shoe upper.
  inertial  existing foot link (0.84518 kg) + parts below, combined about the new centre of mass:
              Rubber_Sole_4mm  rubber (user)                 1.15 g/cm3
              DS_Tape_1mm      acrylic foam tape             0.72 g/cm3
              Sole_Carrier_v5  printed ABS: printed structural density 0.8874 g/cm3 (PLA, docs/89) x ABS/PLA 1.04/1.24
              Feet_Up          PLA, 80 g (user): the 80 g spread uniformly over the CAD solid (488.3 cm3).
usage (mjlab venv): build_footv5.py
"""
import json
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop"
OUTNAME = BASE.replace("_loop", "_footv5_loop")
MESHDIR = "meshes_footv5"
REVIEW = REPO / "Feet_design/Feet_Design_Workspace/foot_cad/v5/review_cad"  # step_to_foot_stl.py output of the CAD export
VAR = "v5cad_MedialPlusX"
V5FULL = REPO / "Feet_design/Feet_Design_Workspace/mjcf_sim/union_branch/mjlab_policy/pygmalion_mjlab/assets/pygmalion_legonly_ab_v5full.xml"
PRINTED_PLA = 887.4  # kg/m3, structural printed density (shell_overlay manifest, docs/89)
FEET_UP_KG = 0.080
PARTS = {  # part -> (density kg/m3 or None = fixed mass, fixed mass kg, URDF rgba)
  "carrier": (PRINTED_PLA * 1.04 / 1.24, None, "0.85 0.80 0.25 1"),
  "tape": (720.0, None, "0.95 0.95 0.95 1"),
  "sheet": (1150.0, None, "0.15 0.55 0.75 1"),
  "feet_up": (None, FEET_UP_KG, "0.45 0.80 0.35 1"),
}


def parse_inertial(block):
  o = re.search(r'<origin xyz="([^"]+)"', block.split("</inertial>")[0]).group(1)
  m = float(re.search(r'<mass value="([^"]+)"', block).group(1))
  I = {k: float(v) for k, v in re.findall(r'(i[xyz]{2})="([^"]+)"', block)}
  T = np.array([[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])
  return m, np.array([float(v) for v in o.split()]), T


def combine(items):
  M = sum(m for m, _, _ in items)
  c = sum(m * p for m, p, _ in items) / M
  T = np.zeros((3, 3))
  for m, p, I in items:
    r = p - c
    T += I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
  return M, c, T


def g(v):
  return f"{v:.6g}"


def main():
  src = (ASSETS / f"{BASE}.urdf").read_text()
  out = src
  (ASSETS / MESHDIR).mkdir(exist_ok=True)
  mv = mujoco.MjModel.from_xml_path(str(V5FULL))
  report = dict(base=f"{BASE}.urdf", output=f"{OUTNAME}.urdf", feet={})
  for S, sign in (("R", 1.0), ("L", -1.0)):
    blk = re.search(rf'  <link name="{S}_foot_link">.*?</link>\n', src, re.S).group(0)
    m0, c0, I0 = parse_inertial(blk)
    items, parts_rep, visuals = [(m0, c0, I0)], {}, []
    for part, (rho, kg, rgba) in PARTS.items():
      mesh = trimesh.load(REVIEW / f"{S}_{part}_{VAR}.stl")
      if not mesh.is_watertight:  # OCC face-by-face tessellation can leave hairline seams; the volume is what matters here
        mesh.merge_vertices()
        trimesh.repair.fill_holes(mesh)
      assert mesh.volume > 0, (S, part)
      if rho is None:
        rho = kg / mesh.volume
      mesh.density = rho
      items.append((mesh.mass, mesh.center_mass.copy(), mesh.moment_inertia.copy()))
      fn = f"{MESHDIR}/{S}_v5_{part}.stl"
      mesh.export(ASSETS / fn)
      parts_rep[part] = dict(watertight=bool(mesh.is_watertight), volume_cm3=round(mesh.volume * 1e6, 3), density_g_cm3=round(rho / 1000, 4), mass_kg=round(float(mesh.mass), 5),
                             com_mm=(mesh.center_mass * 1000).round(2).tolist())
      visuals.append(f'    <visual>\n      <geometry><mesh filename="{fn}"/></geometry>\n'
                     f'      <material name="footv5_{part}"><color rgba="{rgba}"/></material>\n    </visual>\n')
    M, c, T = combine(items)
    cols = []
    gids = [i for i in range(mv.ngeom) if mv.geom_bodyid[i] == mv.body(f"{S}_foot_link").id and mv.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            and mv.geom_group[i] == 3]
    for k, gi in enumerate(gids):
      mat = np.zeros(9)
      mujoco.mju_quat2Mat(mat, mv.geom_quat[gi])
      d = mv.geom_dataid[gi]
      v = mv.mesh_vert[mv.mesh_vertadr[d]: mv.mesh_vertadr[d] + mv.mesh_vertnum[d]] @ mat.reshape(3, 3).T + mv.geom_pos[gi]
      fn = f"{MESHDIR}/{S}_v5_col{k:02d}.stl"
      trimesh.Trimesh(v).convex_hull.export(ASSETS / fn)
      cols.append(f'    <collision>\n      <geometry><mesh filename="{fn}"/></geometry>\n    </collision>\n')
    assert len(cols) == 29, len(cols)
    keep_vis = re.findall(r"    <visual>.*?</visual>\n", blk, re.S)
    keep_col = re.findall(r"    <collision>.*?</collision>\n", blk, re.S)
    new = (f'  <link name="{S}_foot_link">\n'
           f'    <!-- 2026-10-31 v5 sole module (CAD Pygmalion_Foot_v5_RightFoot; left = mirror). Built by '
           f'tools/robot_model/foot_overlay/build_footv5.py, do not hand-edit. Base link: {m0} kg; + carrier/tape/rubber/Feet_Up '
           f'= {M:.5f} kg. Sole bottom z = -0.057 (was -0.043). Feet_Up (toe-box cover, PLA 80 g): visual and mass, no collision. -->\n'
           f'    <inertial>\n      <origin xyz="{g(c[0])} {g(c[1])} {g(c[2])}" rpy="0 0 0"/>\n      <mass value="{M:.6g}"/>\n'
           f'      <inertia ixx="{g(T[0,0])}" ixy="{g(T[0,1])}" ixz="{g(T[0,2])}" iyy="{g(T[1,1])}" iyz="{g(T[1,2])}" izz="{g(T[2,2])}"/>\n'
           f'    </inertial>\n' + "".join(keep_vis) + "".join(visuals) + "".join(keep_col) + "".join(cols) + "  </link>\n")
    assert np.all(np.linalg.eigvalsh(T) > 0)
    out = out.replace(blk, new)
    report["feet"][S] = dict(base_mass_kg=m0, new_mass_kg=round(M, 6), added_kg=round(M - m0, 6), com_mm=(c * 1000).round(3).tolist(),
                             base_com_mm=(c0 * 1000).round(3).tolist(), parts=parts_rep, collision_pieces=len(cols) + len(keep_col))
  # every character outside the two foot links is unchanged
  back = out
  for S in ("R", "L"):
    back = back.replace(re.search(rf'  <link name="{S}_foot_link">.*?</link>\n', out, re.S).group(0),
                        re.search(rf'  <link name="{S}_foot_link">.*?</link>\n', src, re.S).group(0))
  assert back == src
  (ASSETS / f"{OUTNAME}.urdf").write_text(out)
  shutil.copyfile(ASSETS / f"{BASE}.motor_params.json", ASSETS / f"{OUTNAME}.motor_params.json")

  # independent read-back with MuJoCo's URDF loader: same kinematics, mass changes on the feet only
  def load(name):
    t = (ASSETS / f"{name}.urdf").read_text()
    i = t.index(">", t.index("<robot")) + 1
    t = t[:i] + f'\n  <mujoco><compiler discardvisual="false" meshdir="{ASSETS}"/></mujoco>' + t[i:]
    m = mujoco.MjModel.from_xml_string(t)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d
  ma, da = load(BASE)
  mb, db = load(OUTNAME)
  names = [ma.body(i).name for i in range(ma.nbody)]
  assert names == [mb.body(i).name for i in range(mb.nbody)]
  dpos = max(float(np.abs(da.xpos[i] - db.xpos[mb.body(n).id]).max()) for i, n in enumerate(names))
  dmass = {n: round(float(mb.body_mass[mb.body(n).id] - ma.body_mass[i]), 6) for i, n in enumerate(names)
           if abs(mb.body_mass[mb.body(n).id] - ma.body_mass[i]) > 1e-9}
  fz = {}
  for S in ("R", "L"):
    b = mb.body(f"{S}_foot_link").id
    zs = [db.geom_xpos[i][2] - db.xpos[b][2] for i in range(mb.ngeom) if mb.geom_bodyid[i] == b]
    lo = min(float((mb.mesh_vert[mb.mesh_vertadr[mb.geom_dataid[i]]: mb.mesh_vertadr[mb.geom_dataid[i]] + mb.mesh_vertnum[mb.geom_dataid[i]]]
                    @ db.geom_xmat[i].reshape(3, 3).T + db.geom_xpos[i] - db.xpos[b])[:, 2].min())
             for i in range(mb.ngeom) if mb.geom_bodyid[i] == b and mb.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and mb.geom_contype[i])
    fz[S] = dict(n_geoms=len(zs), lowest_collision_z_mm=round(lo * 1000, 3))
  report["readback"] = dict(body_position_max_diff_m=dpos, mass_changed_bodies=dmass,
                            total_mass_kg=dict(base=round(float(ma.body_mass.sum()), 5), footv5=round(float(mb.body_mass.sum()), 5)), feet=fz,
                            outside_foot_links_identical=True)
  (ASSETS / f"{OUTNAME}_report.json").write_text(json.dumps(report, indent=1))
  print(json.dumps(report, indent=1))


if __name__ == "__main__":
  main()
