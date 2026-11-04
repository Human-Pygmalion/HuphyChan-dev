#!/usr/bin/env python3
"""Is the latest robot model inside the domain-randomization (DR) range the policy was trained with?

Two levels (a point can be inside every single range and still be outside the cloud, so both are needed):
 1. Per link: mass ratio exp(2*alpha) and centre-of-mass shift (t1..t3, link frame) of the new model against the training model,
    compared with the DR bounds read from the policy's own env.yaml (events union_inertia_*).
 2. Whole robot: draw N DR samples on the TRAINING model (same law as mjlab dr.pseudo_inertia: mass and inertia x exp(2 alpha),
    CoM shift t in the link frame, independent per body; payload on base_link) and compute robot-level features at the default pose
    (total mass, leg mass, whole-body CoM, leg inertia about the hip-pitch axis). Report where the new model falls in each feature's
    sample distribution (percentile; 0 or 100 = outside the sampled cloud) and the Mahalanobis distance of the feature vector.

  mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/m19_transfer/dr_coverage.py --env-yaml <env.yaml> --xml-train <M10.xml> \
      --xml-new <M19.xml> --out docs/sim2real/data/2026-11-04_dr_coverage.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import yaml


class _L(yaml.SafeLoader):
  pass


def _tag(loader, suffix, node):
  if isinstance(node, yaml.ScalarNode):
    return loader.construct_scalar(node)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return loader.construct_mapping(node, deep=True)


_L.add_multi_constructor("", _tag)


def dr_groups(env_yaml):
  """-> list of (regex list, alpha_range, t_ranges[3] or None, event name) for the pseudo-inertia events + payload range."""
  ev = yaml.load(open(env_yaml), _L)["events"]
  groups = []
  for k, v in ev.items():
    if not k.startswith("union_inertia_"):
      continue
    p = v["params"]
    pats = v["params"]["asset_cfg"]["body_names"]
    groups.append(dict(name=k, patterns=list(pats), alpha=tuple(p["alpha_range"]),
                       t=[tuple(p.get(f"t{i}_range", (0.0, 0.0))) for i in (1, 2, 3)], d=tuple(p.get("d_range", (0.0, 0.0)))))
  payload = tuple(ev["payload"]["params"]["mass_range"])
  return groups, payload


def body_table(xml):
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
  return m, d, names


def fold_extra(m, names, base_names):
  """Bodies that exist only in the new model (camera, camera mount) are folded into their parent for the per-link comparison."""
  fold = {}
  for i, n in enumerate(names):
    if n and n not in base_names:
      j = i
      while names[j] not in base_names:
        j = int(m.body_parentid[j])
      fold[i] = j
  return fold


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--env-yaml", required=True)
  ap.add_argument("--xml-train", required=True)
  ap.add_argument("--xml-new", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=50000)
  a = ap.parse_args()
  groups, payload = dr_groups(a.env_yaml)
  mt, dt, nt = body_table(a.xml_train)
  mn, dn, nn = body_table(a.xml_new)
  base_names = set(n for n in nt if n)
  fold = fold_extra(mn, nn, base_names)
  idx_t = {n: i for i, n in enumerate(nt) if n}
  idx_n = {n: i for i, n in enumerate(nn) if n}

  # ---- level 1: per link --------------------------------------------------------------
  def group_of(name):
    for g in groups:
      if any(re.fullmatch(p, name) for p in g["patterns"]):
        return g
    return None

  per_link = []
  new_mass = {n: float(mn.body_mass[idx_n[n]]) for n in base_names if n in idx_n}
  new_com_local = {n: np.array(mn.body_ipos[idx_n[n]], float) * 1.0 for n in base_names if n in idx_n}
  # fold camera bodies: add mass, move the parent's CoM to the combined CoM (parent frame)
  for i, j in fold.items():
    pn = nn[j]
    mi = float(mn.body_mass[i])
    # CoM of the extra body in the parent frame
    R = dn.xmat[j].reshape(3, 3)
    pos_local = R.T @ (dn.xipos[i] - dn.xpos[j])
    tot = new_mass[pn] + mi
    new_com_local[pn] = (new_com_local[pn] * new_mass[pn] + pos_local * mi) / tot
    new_mass[pn] = tot
  for n in sorted(base_names):
    if n in ("world",) or mt.body_mass[idx_t[n]] <= 1e-9:
      continue
    g = group_of(n)
    m0, m1 = float(mt.body_mass[idx_t[n]]), new_mass[n]
    r = m1 / m0
    dc = (new_com_local[n] - np.array(mt.body_ipos[idx_t[n]], float)) * 1000.0  # mm, link frame
    row = dict(body=n, mass_train_kg=round(m0, 4), mass_new_kg=round(m1, 4), ratio=round(r, 3), dcom_mm=[round(float(x), 2) for x in dc],
               group=g["name"] if g else None)
    if g:
      lo, hi = math.exp(2 * g["alpha"][0]), math.exp(2 * g["alpha"][1])
      row.update(dr_ratio_range=[round(lo, 3), round(hi, 3)], mass_inside=bool(lo <= r <= hi),
                 dr_t_range_mm=[[round(x * 1000, 1) for x in t] for t in g["t"]],
                 com_inside=bool(all(g["t"][k][0] * 1000 - 1e-6 <= dc[k] <= g["t"][k][1] * 1000 + 1e-6 for k in range(3))) if any(g["t"][k] != (0.0, 0.0) for k in range(3)) else None)
    per_link.append(row)

  # ---- level 2: whole-robot features over DR samples ---------------------------------
  rng = np.random.default_rng(0)
  bodies = [n for n in sorted(base_names) if n != "world" and mt.body_mass[idx_t[n]] > 1e-9]
  ii = np.array([idx_t[n] for n in bodies])
  m0 = mt.body_mass[ii]
  xip = dt.xipos[ii].copy()  # world CoM of each body at the default pose
  Rb = dt.xmat[ii].reshape(-1, 3, 3)
  alpha = np.zeros((a.n, len(bodies)))
  tt = np.zeros((a.n, len(bodies), 3))
  for k, n in enumerate(bodies):
    g = group_of(n)
    if g:
      alpha[:, k] = rng.uniform(*g["alpha"], a.n)
      for ax in range(3):
        tt[:, k, ax] = rng.uniform(*g["t"][ax], a.n)
  pay = rng.uniform(*payload, a.n)
  ms = m0[None, :] * np.exp(2 * alpha)
  base_k = bodies.index("base_link")
  ms[:, base_k] += pay
  pos = xip[None, :, :] + np.einsum("kij,nkj->nki", Rb, tt)
  # payload sits at the pelvis CoM (shifted with it)

  leg = np.array([n.startswith(("R_", "L_")) for n in bodies])
  hip_axis_pt = {}
  for s in "RL":
    jid = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_hip_pitch_joint")
    hip_axis_pt[s] = (dt.xanchor[jid].copy(), dt.xaxis[jid].copy())

  def features(ms_, pos_, bodies_, legmask):
    tot = ms_.sum(-1)
    com = (ms_[..., None] * pos_).sum(-2) / tot[..., None]
    legm = (ms_ * legmask).sum(-1)
    # leg (right) mass moment about the hip-pitch axis: sum m r_perp^2 (point-mass part; own inertia is a few percent)
    p0, ax = hip_axis_pt["R"]
    isR = np.array([n.startswith("R_") and n != "R_hip_pitch_link" for n in bodies_])
    rel = pos_ - p0
    perp = rel - (rel @ ax)[..., None] * ax
    Ihip = (ms_ * isR * (perp ** 2).sum(-1)).sum(-1)
    return np.stack([tot, legm, com[..., 0] * 1000, com[..., 2] * 1000, Ihip], -1)

  names_f = ["total_mass_kg", "legs_mass_kg", "com_x_mm", "com_z_mm", "swing_leg_inertia_about_hip_kgm2"]
  F = features(ms, pos, bodies, leg)
  # new model features at the same pose: use its own bodies (folded camera) at the training bodies' world positions (same pose)
  ms_new = np.array([new_mass[n] for n in bodies])[None, :]
  pos_new = []
  for n in bodies:
    # CoM in the world: parent frame of the TRAINING model at default pose (frames identical across models)
    pos_new.append(dt.xpos[idx_t[n]] + dt.xmat[idx_t[n]].reshape(3, 3) @ new_com_local[n])
  pos_new = np.array(pos_new)[None, :, :]
  Fn = features(ms_new, pos_new, bodies, leg)[0]
  F0 = features(m0[None, :], xip[None, :, :], bodies, leg)[0]
  mu, sd = F.mean(0), F.std(0)
  pct = [float((F[:, k] < Fn[k]).mean() * 100) for k in range(F.shape[1])]
  cov = np.cov(F.T)
  maha = float(np.sqrt((Fn - mu) @ np.linalg.solve(cov, Fn - mu)))
  maha_train = float(np.sqrt((F0 - mu) @ np.linalg.solve(cov, F0 - mu)))
  m95 = float(np.percentile(np.sqrt(np.einsum("ni,ij,nj->n", F - mu, np.linalg.inv(cov), F - mu)), 99))
  feat = []
  for k, nme in enumerate(names_f):
    feat.append(dict(feature=nme, training_model=round(float(F0[k]), 4), new_model=round(float(Fn[k]), 4), dr_min=round(float(F[:, k].min()), 4),
                     dr_p1=round(float(np.percentile(F[:, k], 1)), 4), dr_p99=round(float(np.percentile(F[:, k], 99)), 4), dr_max=round(float(F[:, k].max()), 4),
                     new_percentile=round(pct[k], 2), outside_sampled_cloud=bool(Fn[k] < F[:, k].min() or Fn[k] > F[:, k].max())))
  # counterfactual: the new model's EXTRA mass (+delta) put on the torso instead of the legs is covered by the payload range
  extra = float(sum(new_mass.values()) - m0.sum())
  out = dict(env_yaml=a.env_yaml, xml_train=a.xml_train, xml_new=a.xml_new, n_samples=a.n, payload_range_kg=payload, extra_mass_kg=round(extra, 3),
             per_link=per_link, whole_robot=feat, mahalanobis=dict(new=round(maha, 2), training_model=round(maha_train, 2), dr_p99=round(m95, 2)),
             notes=["mass multiplier = exp(2 alpha) (mjlab dr.pseudo_inertia); sampling independent per body; d_range (shape) and shear not used in features",
                    "camera bodies folded into base_link", "features at the model default pose (qpos0)"])
  Path(a.out).write_text(json.dumps(out, indent=1))
  np.savez_compressed(Path(a.out).with_suffix(".samples.npz"), F=F[:20000].astype(np.float32), Fn=Fn, F0=F0, names=np.array(names_f))
  print(f"extra mass {extra:.3f} kg; payload range {payload}")
  print("%-22s %8s %8s %6s  %-16s %-8s %-8s" % ("body", "m_train", "m_new", "ratio", "DR ratio range", "mass_in", "com_in"))
  for r in per_link:
    print("%-22s %8.4f %8.4f %6.3f  %-16s %-8s %-8s  dCoM %s" % (r["body"], r["mass_train_kg"], r["mass_new_kg"], r["ratio"], r.get("dr_ratio_range"), r.get("mass_inside"), r.get("com_inside"), r["dcom_mm"]))
  for f in feat:
    print(f)
  print("mahalanobis", out["mahalanobis"])


if __name__ == "__main__":
  main()
