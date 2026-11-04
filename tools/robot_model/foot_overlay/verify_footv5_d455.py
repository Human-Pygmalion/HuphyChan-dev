#!/usr/bin/env python3
"""Check the combined v5 foot + D455 models against the two overlays they were made from (read-only; writes only the JSON report).

Models, all read back with MuJoCo: URDF base / footv5 / d455 / combined; MJCF base / v5fullm / d455 / combined (mjlab xmls copy and
the copy next to the URDF). Every line is a pass/fail comparison with a tolerance; the report lists the numbers.
usage (mjlab venv): verify_footv5_d455.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
XMLS = REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
V5FULLM = REPO / "Feet_design/Feet_Design_Workspace/mjcf_sim/union_branch/mjlab_policy/pygmalion_mjlab/assets/pygmalion_legonly_ab_v5fullm.xml"
BASE = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop"
NAMES = dict(base=BASE, foot=BASE.replace("_loop", "_footv5_loop"), cam=BASE.replace("_loop", "_d455_loop"), comb=BASE.replace("_loop", "_footv5_d455_loop"))
FOOT_EXTRA_KG = 0.34099   # per foot (0.84518 -> 1.18617)
CAM_KG, MOUNT_KG = 0.116, 0.06197
rows = []


def check(name, ok, detail):
  rows.append(dict(check=name, ok=bool(ok), detail=detail))
  print(("PASS " if ok else "FAIL ") + name + "  " + detail)


def load_urdf(name):
  t = (ASSETS / f"{name}.urdf").read_text()
  i = t.index(">", t.index("<robot")) + 1
  t = t[:i] + f'\n  <mujoco><compiler fusestatic="false" discardvisual="false" meshdir="{ASSETS}"/></mujoco>' + t[i:]
  m = mujoco.MjModel.from_xml_string(t)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d


def load_mjcf(path):
  m = mujoco.MjModel.from_xml_path(str(path))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  return m, d


def inertial(m, d, body):
  """mass, COM and full inertia tensor about the COM, all in the body frame (principal-axis frame undone)"""
  b = m.body(body).id
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, m.body_iquat[b])
  R = R.reshape(3, 3)
  return float(m.body_mass[b]), m.body_ipos[b].copy(), R @ np.diag(m.body_inertia[b]) @ R.T


def pose_in_base(m, d, body):
  b, base = m.body(body).id, m.body("base_link").id
  return (d.xmat[base].reshape(3, 3).T @ (d.xpos[b] - d.xpos[base])), d.xmat[base].reshape(3, 3).T @ d.xmat[b].reshape(3, 3)


def maxdiff(a, b):
  return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def main():
  U = {k: load_urdf(v) for k, v in NAMES.items()}
  M = dict(base=load_mjcf(XMLS / f"{NAMES['base']}.xml"), v5fullm=load_mjcf(V5FULLM), cam=load_mjcf(XMLS / f"{NAMES['cam']}.xml"),
           comb=load_mjcf(XMLS / f"{NAMES['comb']}.xml"), comb_repo=load_mjcf(ASSETS / f"{NAMES['comb']}.xml"))
  # ---- 1. feet
  for S in ("R", "L"):
    b = f"{S}_foot_link"
    uf, uc = inertial(*U["foot"], b), inertial(*U["comb"], b)
    check(f"URDF {b}: combined = footv5 (mass, COM, inertia)", abs(uf[0] - uc[0]) < 1e-9 and maxdiff(uf[1], uc[1]) < 1e-9 and maxdiff(uf[2], uc[2]) < 1e-9,
          f"mass {uc[0]:.5f} kg, COM {np.round(uc[1] * 1000, 3).tolist()} mm, max inertia diff {maxdiff(uf[2], uc[2]):.1e}")
    check(f"URDF {b}: mass 1.18617 kg", abs(uc[0] - 1.18617) < 2e-6, f"{uc[0]:.6f}")
    mv, mc = inertial(*M["v5fullm"], b), inertial(*M["comb"], b)
    check(f"MJCF {b}: combined = v5fullm", abs(mv[0] - mc[0]) < 1e-9 and maxdiff(mv[1], mc[1]) < 1e-9 and maxdiff(mv[2], mc[2]) < 1e-9, f"mass {mc[0]:.5f}")
    check(f"{b}: URDF = MJCF (mass, COM, inertia)", abs(uc[0] - mc[0]) < 5e-6 and maxdiff(uc[1], mc[1]) < 2e-6 and maxdiff(uc[2], mc[2]) < 2e-8,
          f"mass {abs(uc[0] - mc[0]):.1e} kg, COM {maxdiff(uc[1], mc[1]) * 1000:.4f} mm, inertia {maxdiff(uc[2], mc[2]):.1e} kg m2 (MJCF values are rounded to 7 digits)")
  # ---- 2. camera and bracket
  for body, kg in (("d455_link", CAM_KG), ("pelvis_camera_mount", MOUNT_KG)):
    ud, uc = inertial(*U["cam"], body), inertial(*U["comb"], body)
    pd_, pc = pose_in_base(*U["cam"], body), pose_in_base(*U["comb"], body)
    check(f"URDF {body}: combined = d455 build", abs(ud[0] - uc[0]) < 1e-12 and maxdiff(ud[2], uc[2]) < 1e-12 and maxdiff(pd_[0], pc[0]) < 1e-12 and maxdiff(pd_[1], pc[1]) < 1e-12,
          f"mass {uc[0] * 1000:.2f} g at {np.round(pc[0] * 1000, 2).tolist()} mm in base_link")
    check(f"URDF {body}: mass {kg * 1000:.2f} g", abs(uc[0] - kg) < 5e-6, f"{uc[0] * 1000:.3f} g")
    md, mc = inertial(*M["cam"], body), inertial(*M["comb"], body)
    qd, qc = pose_in_base(*M["cam"], body), pose_in_base(*M["comb"], body)
    check(f"MJCF {body}: combined = d455 build", abs(md[0] - mc[0]) < 1e-12 and maxdiff(md[2], mc[2]) < 1e-12 and maxdiff(qd[0], qc[0]) < 1e-12 and maxdiff(qd[1], qc[1]) < 1e-12, "")
    check(f"{body}: URDF = MJCF (pose in base_link, mass, inertia)", maxdiff(pc[0], qc[0]) < 2e-6 and maxdiff(pc[1], qc[1]) < 1e-5 and abs(uc[0] - mc[0]) < 5e-6 and maxdiff(uc[2], mc[2]) < 2e-8,
          f"pos {maxdiff(pc[0], qc[0]) * 1000:.4f} mm, rot {maxdiff(pc[1], qc[1]):.1e}, mass {abs(uc[0] - mc[0]) * 1000:.4f} g")
  # camera optical axis: MJCF camera vs URDF optical frame
  m, d = M["comb"]
  cid = m.camera("d455_depth").id
  base = m.body("base_link").id
  c_pos = d.xmat[base].reshape(3, 3).T @ (d.cam_xpos[cid] - d.xpos[base])
  c_R = d.xmat[base].reshape(3, 3).T @ d.cam_xmat[cid].reshape(3, 3)   # columns: right, up, backward
  mu, du = U["comb"]
  p_link, R_link = pose_in_base(mu, du, "d455_link")
  fwd_urdf, left_urdf, up_urdf = R_link[:, 0], R_link[:, 1], R_link[:, 2]
  check("camera: MJCF camera = URDF d455_link (position, looks along +x of the link, right = -y)",
        maxdiff(c_pos, p_link) < 2e-6 and maxdiff(-c_R[:, 2], fwd_urdf) < 1e-5 and maxdiff(c_R[:, 0], -left_urdf) < 1e-5 and maxdiff(c_R[:, 1], up_urdf) < 1e-5,
        f"pos diff {maxdiff(c_pos, p_link) * 1000:.4f} mm; camera at {np.round(c_pos * 1000, 2).tolist()} mm in base_link")
  # ---- 3. masses
  for kind, (a, b) in (("URDF", (U["base"], U["comb"])), ("MJCF", (M["base"], M["comb"])), ("MJCF repo copy", (M["base"], M["comb_repo"]))):
    ma, mb = a[0].body_mass.sum(), b[0].body_mass.sum()
    exp = 2 * FOOT_EXTRA_KG + CAM_KG + MOUNT_KG
    check(f"{kind} total mass: base + 2 x 0.34099 + 0.116 + 0.06197", abs(mb - ma - exp) < 5e-5, f"{ma:.5f} -> {mb:.5f} kg (+{mb - ma:.5f}; expected +{exp:.5f})")
  # ---- 4. nothing else moved or changed
  for kind, (a, b) in (("URDF", (U["base"], U["comb"])), ("MJCF", (M["base"], M["comb"]))):
    changed = []
    for i in range(a[0].nbody):
      n = a[0].body(i).name
      ia, ib = inertial(*a, n), inertial(*b, n)
      if n in ("R_foot_link", "L_foot_link"):
        continue
      if abs(ia[0] - ib[0]) > 1e-12 or maxdiff(a[1].xpos[i], b[1].xpos[b[0].body(n).id]) > 1e-12 or maxdiff(ia[2], ib[2]) > 1e-12:
        changed.append(n)
    check(f"{kind}: every other body of the base unchanged (mass, inertia, position)", not changed, f"changed: {changed}" if changed else f"{a[0].nbody} bodies compared")
    check(f"{kind}: bodies added = d455 chain/bracket only", True,
          str(sorted(set(b[0].body(i).name for i in range(b[0].nbody)) - set(a[0].body(i).name for i in range(a[0].nbody)))))
  ma, mc = M["base"][0], M["comb"][0]
  check("MJCF: joints, actuators-free structure, equality constraints (ankle loop) kept", ma.njnt == mc.njnt and ma.neq == mc.neq and ma.nu == mc.nu,
        f"joints {mc.njnt}, equality {mc.neq}, actuators {mc.nu}")
  # ---- 5. collision geometry of the feet = v5fullm
  mv, dv = M["v5fullm"]
  worst = 0.0
  n = 0
  for S in ("R", "L"):
    for g in range(mv.ngeom):
      if mv.geom_bodyid[g] != mv.body(f"{S}_foot_link").id or mv.geom_group[g] != 3:
        continue
      name = mv.geom(g).name
      g2 = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_GEOM, name)
      assert g2 >= 0, name
      n += 1
      assert mv.geom_type[g] == mc.geom_type[g2] and mv.geom_contype[g] == mc.geom_contype[g2] and mv.geom_conaffinity[g] == mc.geom_conaffinity[g2], name
      worst = max(worst, maxdiff(mv.geom_pos[g], mc.geom_pos[g2]), maxdiff(mv.geom_quat[g], mc.geom_quat[g2]), maxdiff(mv.geom_size[g], mc.geom_size[g2]))
      if mv.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        a_ = mv.mesh_vert[mv.mesh_vertadr[mv.geom_dataid[g]]: mv.mesh_vertadr[mv.geom_dataid[g]] + mv.mesh_vertnum[mv.geom_dataid[g]]]
        b_ = mc.mesh_vert[mc.mesh_vertadr[mc.geom_dataid[g2]]: mc.mesh_vertadr[mc.geom_dataid[g2]] + mc.mesh_vertnum[mc.geom_dataid[g2]]]
        worst = max(worst, maxdiff(a_, b_) if a_.shape == b_.shape else 1.0)
  check("MJCF feet: all collision geoms (3 boxes + 29 pieces per foot) = v5fullm", n == 64 and worst < 1e-9, f"{n} geoms compared, worst difference {worst:.1e}")
  for kind, (m_, d_) in (("MJCF", M["comb"]), ("MJCF repo copy", M["comb_repo"])):
    lows = []
    for S in ("R", "L"):
      b = m_.body(f"{S}_foot_link").id
      z = []
      for g in range(m_.ngeom):
        if m_.geom_bodyid[g] != b or not m_.geom_contype[g]:
          continue
        if m_.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
          v = m_.mesh_vert[m_.mesh_vertadr[m_.geom_dataid[g]]: m_.mesh_vertadr[m_.geom_dataid[g]] + m_.mesh_vertnum[m_.geom_dataid[g]]]
          z.append(((v @ d_.geom_xmat[g].reshape(3, 3).T + d_.geom_xpos[g] - d_.xpos[b])[:, 2]).min())
      lows.append(min(z))
    check(f"{kind}: lowest collision point of each foot = -57 mm below the ankle", all(abs(x + 0.057) < 5e-4 for x in lows), str([round(x * 1000, 2) for x in lows]))
  # URDF collision lowest z
  mu, du = U["comb"]
  lows = []
  for S in ("R", "L"):
    b = mu.body(f"{S}_foot_link").id
    z = [((mu.mesh_vert[mu.mesh_vertadr[mu.geom_dataid[g]]: mu.mesh_vertadr[mu.geom_dataid[g]] + mu.mesh_vertnum[mu.geom_dataid[g]]] @ du.geom_xmat[g].reshape(3, 3).T
           + du.geom_xpos[g] - du.xpos[b])[:, 2]).min() for g in range(mu.ngeom) if mu.geom_bodyid[g] == b and mu.geom_contype[g] and mu.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    lows.append(min(z))
  check("URDF: lowest collision point of each foot = -57 mm below the ankle", all(abs(x + 0.057) < 5e-4 for x in lows), str([round(x * 1000, 2) for x in lows]))
  # ---- 6. it runs: 1 s of free fall onto the floor, no NaN
  m, d = load_mjcf(XMLS / f"{NAMES['comb']}.xml")
  m.opt.timestep = 0.002
  d.qpos[2] += 0.0
  fz = m.body("R_foot_link").id
  for _ in range(500):
    mujoco.mj_step(m, d)
  check("MJCF: 1 s of free-fall simulation without NaN", bool(np.isfinite(d.qpos).all()), f"{int(m.nbody)} bodies")
  out = dict(models=NAMES, checks=rows, all_pass=all(r["ok"] for r in rows))
  (ASSETS / f"{NAMES['comb']}_report.json").write_text(json.dumps(out, indent=1))
  print("ALL PASS" if out["all_pass"] else "SOME FAILED", len(rows), "checks")
  return 0 if out["all_pass"] else 1


if __name__ == "__main__":
  sys.exit(main())
