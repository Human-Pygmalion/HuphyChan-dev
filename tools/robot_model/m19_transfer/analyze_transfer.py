#!/usr/bin/env python3
"""Compare the Union-DR 49999 policy on the training model and on newer robot models (outputs of foot_variant_eval.py).

Per run (tag) and per command segment (15 s, first 3 s dropped) it reports: falls, velocity / yaw tracking, body tilt,
base height, motor torque RMS and P99 per joint group, torque-limit saturation (|unclipped| > |applied|), foot slip P95,
stance width, and the vertical ground force of each foot in body weights.

  mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/m19_transfer/analyze_transfer.py <dir with npz+json> <out.json> [tag ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DT = 0.02
PROGRAM = [("stand", 0.0, 0.0), ("vx0.3", 0.3, 0.0), ("vx0.6", 0.6, 0.0), ("vx1.0", 1.0, 0.0), ("vx1.5", 1.5, 0.0),
           ("vx-0.5", -0.5, 0.0), ("wz+0.6", 0.0, 0.6), ("wz-0.6", 0.0, -0.6), ("vx0.6wz0.5", 0.6, 0.5)]
MOTORS = [f"{s}_{j}" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
GROUPS = {"hip_pitch": [0, 6], "hip_roll": [1, 7], "hip_yaw": [2, 8], "knee": [3, 9], "crank": [4, 5, 10, 11]}
LIMIT = {"hip_pitch": 120.0, "hip_roll": 120.0, "hip_yaw": 60.0, "knee": 120.0, "crank": 60.0}  # effort limits of the task (N m)
G = 9.81


def yaw_of(q):  # q = (w, x, y, z)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(q):  # angle between body z and world z [rad]
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  cz = 1 - 2 * (x * x + y * y)
  return np.arccos(np.clip(cz, -1, 1))


def analyse(npz, meta):
  D = np.load(npz)
  T, E = D["tau"].shape[:2]
  n_per = int(round(15.0 / DT))
  mass = meta.get("robot_mass_kg", float("nan"))
  out = dict(tag=meta["tag"], xml=Path(meta["xml"]).name, robot_mass_kg=round(mass, 3), falls=int(D["terminated"].any(0).sum()), n_envs=int(E), segments={})
  term = D["terminated"]
  fall_t = [float(D["t"][np.argmax(term[:, e])]) if term[:, e].any() else None for e in range(E)]
  out["fall_times_s"] = fall_t
  out["fall_events"] = int(term.sum())
  out["fall_events_per_env_minute"] = round(float(term.sum()) / (E * T * DT / 60.0), 3)
  for si, (name, vx, wz) in enumerate(PROGRAM):
    a, b = si * n_per + int(3.0 / DT), min((si + 1) * n_per, T)
    if a >= b:
      continue
    sl = slice(a, b)
    alive = ~np.maximum.accumulate(term[:b], axis=0)[a:b]  # env still not fallen
    seg = {}
    v_b = D["base_lin_vel_b"][sl]
    w_b = D["base_ang_vel_b"][sl]
    m = alive
    def mean(x):
      xx = np.where(m, x, np.nan)
      return float(np.nanmean(xx)) if np.isfinite(xx).any() else None
    seg["alive_frac"] = round(float(m.mean()), 3)
    seg["vx_mean"] = mean(v_b[..., 0])
    seg["vx_err_rms"] = float(np.sqrt(np.nanmean(np.where(m, (v_b[..., 0] - vx) ** 2, np.nan)))) if m.any() else None
    seg["wz_mean"] = mean(w_b[..., 2])
    seg["wz_err_rms"] = float(np.sqrt(np.nanmean(np.where(m, (w_b[..., 2] - wz) ** 2, np.nan)))) if m.any() else None
    tl = tilt_of(D["base_quat"][sl])
    seg["tilt_rms_deg"] = float(np.degrees(np.sqrt(np.nanmean(np.where(m, tl ** 2, np.nan))))) if m.any() else None
    seg["base_height_m"] = mean(D["base_pos"][sl][..., 2])
    tau = D["tau"][sl]
    unc = D["tau_unc"][sl]
    for g, cols in GROUPS.items():
      t_ = tau[..., cols]
      mm = m[..., None]
      seg[f"tau_rms_{g}"] = float(np.sqrt(np.nanmean(np.where(mm, t_ ** 2, np.nan)))) if m.any() else None
      seg[f"tau_p99abs_{g}"] = float(np.nanpercentile(np.where(mm, np.abs(t_), np.nan), 99)) if m.any() else None
      sat = (np.abs(unc[..., cols]) > np.abs(t_) + 1e-3) & mm
      seg[f"sat_pct_{g}"] = float(100.0 * sat.sum() / max(mm.sum() * len(cols), 1))
    # signed mean torque (static load) for the standing segment
    for k, nme in enumerate(MOTORS):
      seg[f"tau_mean_{nme}"] = float(np.nanmean(np.where(m, tau[..., k], np.nan))) if m.any() else None
    # stance width: foot centres in the body yaw frame
    fp = D["foot_pos_w"][sl]  # T,E,2,3 (R, L)
    yaw = yaw_of(D["base_quat"][sl])
    dxy = fp[..., 1, :2] - fp[..., 0, :2]  # L - R
    lat = -np.sin(yaw) * dxy[..., 0] + np.cos(yaw) * dxy[..., 1]
    seg["stance_width_cm"] = float(np.nanmean(np.where(m, lat, np.nan)) * 100) if m.any() else None
    # foot slip (only when loaded > 0.2 BW) and vertical force
    sole = D["sole"][sl]  # T,E,2,10: Fn, Ft, ...
    fn = sole[..., 0]
    bw = mass * G
    loaded = (fn > 0.2 * bw) & m[..., None]
    fv = D["foot_lin_vel_w"][sl][..., :2]
    sp = np.linalg.norm(fv, axis=-1)
    seg["slip_p95_mm_s"] = float(np.nanpercentile(np.where(loaded, sp, np.nan), 95) * 1000) if loaded.any() else None
    seg["fn_total_bw_mean"] = float(np.nanmean(np.where(m, np.nansum(fn, -1), np.nan)) / bw) if m.any() else None
    seg["fn_peak_bw_p99"] = float(np.nanpercentile(np.where(m[..., None], fn, np.nan), 99) / bw) if m.any() else None
    # fall and stuck flags
    out["segments"][name] = seg
  return out


def main():
  d = Path(sys.argv[1])
  tags = sys.argv[3:] or sorted(p.stem for p in d.glob("*.json"))
  res = {}
  for t in tags:
    meta = json.load(open(d / f"{t}.json"))
    res[t] = analyse(d / f"{t}.npz", meta)
    print(t, "mass", res[t]["robot_mass_kg"], "falls", res[t]["falls"], "/", res[t]["n_envs"], flush=True)
  Path(sys.argv[2]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
