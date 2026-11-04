#!/usr/bin/env python3
"""Figures for the combined v5 foot + D455 model (docs/185): ray-cast renders (no OpenGL needed), a whole-body URDF vs MJCF comparison,
and the list of verification checks.  Output: docs/img/footv5_d455/{model_views,collision_pieces,whole_body_check,checks}.png
usage (mjlab venv): report_footv5_d455_figs.py
"""
import json
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "pygmalion_locomotion/assets/pygmalion_v2"
NAME = "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_loop"
OUT = REPO / "docs/img/footv5_d455"
OUT.mkdir(parents=True, exist_ok=True)
INIT = {"L_hip_pitch_joint": 0.175, "R_hip_pitch_joint": -0.175, "L_knee_joint": -0.35, "R_knee_joint": 0.35}  # the colleague's stand pose


def load_mjcf():
  m = mujoco.MjModel.from_xml_path(str(ASSETS / f"{NAME}.xml"))
  d = mujoco.MjData(m)
  d.qpos[2] = 0.9
  for k, v in INIT.items():
    d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
  mujoco.mj_forward(m, d)
  return m, d


def render(m, d, center, azim, elev, dist, fov, groups, size=(520, 520), by="rgba", bodies=None):
  """perspective ray cast; returns RGB. azim 0 = looking along -x from +x (front view), 90 = from +y (left side)."""
  a, e = np.radians(azim), np.radians(elev)
  fwd = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
  right = np.cross(fwd, [0, 0, 1.0])
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  eye = np.asarray(center) - fwd * dist
  h, w = size[1], size[0]
  f = (w / 2) / np.tan(np.radians(fov) / 2)
  u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
  vec = (fwd[None, None] * f + right[None, None] * u[..., None] - up[None, None] * v[..., None]).reshape(-1, 3)
  vec /= np.linalg.norm(vec, axis=1, keepdims=True)
  gid = np.zeros(len(vec), np.int32)
  dist_ = np.zeros(len(vec))
  nrm = np.zeros(len(vec) * 3)
  mask = np.zeros(6, np.uint8)
  for g in groups:
    mask[g] = 1
  mujoco.mj_multiRay(m, d, eye, vec.ravel(), mask, True, -1, gid, dist_, nrm, len(vec), 1e9)
  nrm = nrm.reshape(-1, 3)
  hit = gid >= 0
  col = np.ones((len(vec), 3)) * 0.97
  light = np.array([0.4, -0.5, 0.75])
  light /= np.linalg.norm(light)
  shade = np.clip(np.abs(nrm @ -vec.T).diagonal() if False else np.einsum("ij,ij->i", nrm, -vec), 0, 1)
  lam = 0.45 + 0.55 * np.clip(np.einsum("ij,j->i", nrm, light), 0, 1) * 0.6 + 0.25 * shade
  for g in np.unique(gid[hit]):
    sel = gid == g
    if by == "rgba":
      mid = m.geom_matid[g]
      rgba = m.mat_rgba[mid] if mid >= 0 else m.geom_rgba[g]
      c = rgba[:3] if not np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]) or mid < 0 else rgba[:3]
      if mid >= 0 and not np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
        c = m.geom_rgba[g][:3]
    else:
      rng = np.random.default_rng(int(g) * 7 + 3)
      c = 0.25 + 0.7 * rng.random(3)
    col[sel] = np.clip(np.outer(lam[sel], c), 0, 1)
  return col.reshape(h, w, 3)


def fig_views(m, d):
  base = d.xpos[m.body("base_link").id]
  foot = d.xpos[m.body("R_foot_link").id]
  cam = d.xpos[m.body("d455_link").id]
  views = [("front", dict(center=[0, 0, 0.45], azim=0, elev=8, dist=2.4, fov=28, groups=[2])),
           ("left side", dict(center=[0, 0, 0.45], azim=90, elev=8, dist=2.4, fov=28, groups=[2])),
           ("3/4 view", dict(center=[0, 0, 0.45], azim=35, elev=18, dist=2.4, fov=28, groups=[2])),
           ("right foot: CAD sole (carrier yellow, tape white, rubber blue, Feet_Up green)", dict(center=foot + [0.05, 0, -0.03], azim=60, elev=-12, dist=0.9, fov=30, groups=[2])),
           ("pelvis: D455 (blue) + PLA bracket", dict(center=cam + [0.0, 0, -0.03], azim=40, elev=15, dist=0.9, fov=30, groups=[2])),
           ("pelvis from the side (camera tilted 60 deg down)", dict(center=cam + [0.0, 0, -0.03], azim=90, elev=3, dist=0.9, fov=30, groups=[2]))]
  fig, ax = plt.subplots(2, 3, figsize=(15, 10.4))
  for a, (t, kw) in zip(ax.ravel(), views):
    a.imshow(render(m, d, **kw, size=(480, 520)))
    a.set_title(t, fontsize=9)
    a.axis("off")
  fig.suptitle(f"{NAME}  (MJCF visual meshes, ray-cast render; stand pose of the 99998 policy; the D455 mesh looks speckled only in this ray-cast render)", fontsize=11)
  fig.tight_layout()
  fig.savefig(OUT / "model_views.png", dpi=90)
  plt.close(fig)


def fig_collision(m, d):
  foot = d.xpos[m.body("R_foot_link").id]
  views = [("right foot: collision geoms, from below (each piece one colour)", dict(center=foot + [0.05, 0, -0.04], azim=0, elev=-80, dist=0.9, fov=30, groups=[3], by="id")),
           ("same, 3/4 view from below", dict(center=foot + [0.05, 0, -0.04], azim=50, elev=-35, dist=0.9, fov=30, groups=[3], by="id")),
           ("both feet + visual (side)", dict(center=foot + [0.05, 0, -0.03], azim=90, elev=0, dist=0.8, fov=30, groups=[2, 3], by="rgba"))]
  fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
  for a, (t, kw) in zip(ax, views):
    a.imshow(render(m, d, **kw, size=(520, 520)))
    a.set_title(t, fontsize=9)
    a.axis("off")
  fig.suptitle("Foot collision model in the combined MJCF: 3 upper boxes + 29 convex pieces per foot (= v5fullm)", fontsize=11)
  fig.tight_layout()
  fig.savefig(OUT / "collision_pieces.png", dpi=90)
  plt.close(fig)


def load_urdf():
  t = (ASSETS / f"{NAME}.urdf").read_text()
  i = t.index(">", t.index("<robot")) + 1
  t = t[:i] + f'\n  <mujoco><compiler fusestatic="false" discardvisual="false" meshdir="{ASSETS}"/></mujoco>' + t[i:]
  m = mujoco.MjModel.from_xml_string(t)
  d = mujoco.MjData(m)
  return m, d


def fig_whole_body():
  mm_ = mujoco.MjModel.from_xml_path(str(ASSETS / f"{NAME}.xml"))
  dm_ = mujoco.MjData(mm_)
  mu, du = load_urdf()
  jn = [mu.joint(i).name for i in range(mu.njnt) if mu.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  names = [mu.body(i).name for i in range(1, mu.nbody) if mujoco.mj_name2id(mm_, mujoco.mjtObj.mjOBJ_BODY, mu.body(i).name) >= 0]
  rng = np.random.default_rng(0)
  poses = [dict(INIT)] + [{n: float(rng.uniform(*mu.jnt_range[mu.joint(n).id]) * 0.5) for n in jn if mu.jnt_limited[mu.joint(n).id]} for _ in range(3)]
  com_d = {n: 0.0 for n in names}
  ax_d = {n: 0.0 for n in jn if mujoco.mj_name2id(mm_, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}
  for pz in poses:   # pose 0 = the stand pose, 1..3 = random angles (half range): forward kinematics must agree in every pose
    for m, d in ((mm_, dm_), (mu, du)):
      mujoco.mj_resetData(m, d)
      for k, v in pz.items():
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, k) >= 0:
          d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
    bm, bu = mm_.body("base_link").id, mu.body("base_link").id
    for n in names:
      im, iu = mm_.body(n).id, mu.body(n).id
      pm = dm_.xmat[bm].reshape(3, 3).T @ (dm_.xipos[im] - dm_.xpos[bm])
      pu = du.xmat[bu].reshape(3, 3).T @ (du.xipos[iu] - du.xpos[bu])
      com_d[n] = max(com_d[n], float(np.abs(pm - pu).max() * 1000))
    for n in ax_d:
      ax_d[n] = max(ax_d[n], float(np.degrees(np.arccos(np.clip(abs(dm_.xaxis[mm_.joint(n).id] @ du.xaxis[mu.joint(n).id]), -1, 1)))))
  mass_m = {n: mm_.body_mass[mm_.body(n).id] * 1000 for n in names}
  mass_u = {n: mu.body_mass[mu.body(n).id] * 1000 for n in names}
  rng_d = max(float(np.abs(mm_.jnt_range[mm_.joint(n).id] - mu.jnt_range[mu.joint(n).id]).max()) for n in ax_d if mm_.jnt_limited[mm_.joint(n).id] and mu.jnt_limited[mu.joint(n).id])
  fig, ax = plt.subplots(1, 2, figsize=(16, 8.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
  y = np.arange(len(names))
  ax[0].barh(y, [mass_m[n] for n in names], color="#4c78a8", label="MJCF mass")
  ax[0].plot([mass_u[n] for n in names], y, "k|", ms=11, mew=2, label="URDF mass")
  for yi, n in zip(y, names):
    ax[0].text(max(mass_m[n], mass_u[n]) * 1.02 + 20, yi, f"{mass_m[n]:.2f} g | diff {abs(mass_m[n] - mass_u[n]):.4f} g | COM diff {com_d[n]:.5f} mm", va="center", fontsize=7)
  ax[0].set_yticks(y, names, fontsize=8)
  ax[0].set_xlim(0, max(mass_m.values()) * 1.9)
  ax[0].invert_yaxis()
  ax[0].legend(loc="lower right", fontsize=8)
  ax[0].set_xlabel("body mass [g]")
  ax[0].set_title(f"{len(names)} bodies: mass, and COM position in base_link (max over 4 poses)\n"
                  f"total {mm_.body_mass.sum():.4f} kg (MJCF) vs {mu.body_mass.sum():.4f} kg (URDF); the 4.0 g gap exists in the base models too (docs/181 5)", fontsize=9)
  jy = np.arange(len(ax_d))
  ax[1].barh(jy, [max(ax_d[n], 1e-7) for n in ax_d], color="#54a24b")
  for yi, n in zip(jy, ax_d):
    ax[1].text(max(ax_d[n], 1e-7) * 1.5, yi, f"{ax_d[n]:.1e} deg", va="center", fontsize=7)
  ax[1].set_yticks(jy, list(ax_d), fontsize=8)
  ax[1].set_xscale("log")
  ax[1].set_xlim(1e-7, 1e-1)
  ax[1].invert_yaxis()
  ax[1].set_xlabel("joint axis angle difference MJCF vs URDF [deg] (log, floor 1e-7)")
  ax[1].set_title(f"{len(ax_d)} joints: axis direction, max over 4 poses; range diff max {rng_d:.1e} rad", fontsize=9)
  fig.suptitle("Whole body, combined URDF vs combined MJCF: stand pose + 3 random poses, forward kinematics, same joint angles", fontsize=11)
  fig.tight_layout()
  fig.savefig(OUT / "whole_body_check.png", dpi=85)
  plt.close(fig)
  res = dict(n_bodies=len(names), n_joints=len(ax_d), poses=len(poses), max_mass_diff_g=max(abs(mass_m[n] - mass_u[n]) for n in names), max_com_diff_mm=max(com_d.values()),
             max_axis_diff_deg=max(ax_d.values()), max_range_diff_rad=rng_d, total_kg=dict(mjcf=float(mm_.body_mass.sum()), urdf=float(mu.body_mass.sum())))
  (OUT / "whole_body_check.json").write_text(json.dumps(res, indent=1))
  print(json.dumps(res, indent=1))


def fig_checks():
  rep = json.load(open(ASSETS / f"{NAME}_report.json"))
  ck = rep["checks"]
  fig, ax = plt.subplots(figsize=(15, 0.36 * len(ck) + 1.4))
  ax.axis("off")
  for i, c in enumerate(ck):
    y = len(ck) - i
    ax.text(0.0, y, "PASS" if c["ok"] else "FAIL", color="white", fontsize=8, fontweight="bold", va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#2e8b57" if c["ok"] else "#c0392b", ec="none"))
    ax.text(0.04, y, c["check"], fontsize=8, va="center")
    ax.text(0.62, y, c["detail"][:95], fontsize=7, va="center", color="#444")
  ax.set_xlim(-0.03, 1.2)
  ax.set_ylim(0, len(ck) + 1)
  ax.set_title(f"Verification of {NAME}: {sum(c['ok'] for c in ck)}/{len(ck)} checks pass", fontsize=11)
  fig.tight_layout()
  fig.savefig(OUT / "checks.png", dpi=85)
  plt.close(fig)


if __name__ == "__main__":
  m, d = load_mjcf()
  fig_views(m, d)
  fig_collision(m, d)
  fig_whole_body()
  fig_checks()
  print("saved", [p.name for p in OUT.glob("*.png")])
