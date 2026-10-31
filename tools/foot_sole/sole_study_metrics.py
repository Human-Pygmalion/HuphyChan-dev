#!/usr/bin/env python3
"""Per-environment metrics of one foot_variant_eval.py run (the unit of the statistics in sole_study_report.py).

Three groups, all computed on the moving segments (commands 2-9, first 3 s of every 15 s segment dropped) and only while the
environment is alive (an environment is dropped from its first termination onward):

Impact (net foot force at every 5 ms physics substep, grf_sub)
  td_peak_bw        median over touchdowns of the peak vertical force in the first 100 ms after touchdown [body weights]
  td_peak_p95_bw    95th percentile of the same
  td_rate_bw_s      median over touchdowns of the largest 5 ms force rise in the first 50 ms (loading rate) [BW/s]
  grf_p99_bw        99th percentile of the vertical force over all loaded substeps [BW]
  touchdown = force below 0.02 BW for >= 60 ms, then above 0.05 BW
Energy (12 motor joints, 50 Hz)
  cot_pos           positive mechanical work / (m g distance)   (distance = base path length in the horizontal plane)
  cot_abs           absolute mechanical work / (m g distance)
  tau2_per_m        integral of the summed squared motor torque per metre walked [N^2 m^2 s / m] (copper-loss proxy)
Stability
  fell              1 if the environment terminated
  survival_s        time until the first fall (135 s = never fell); with many falls this separates feet better than the count
  dist_to_fall_m    horizontal path length walked until the first fall
  alive_frac        fraction of the evaluated time the environment was alive
  tilt_rms_deg      RMS of the base tilt (angle between base z and world z)
  angvel_rms        RMS of the base roll/pitch rate [rad/s]
  vx_err, wz_err    mean absolute command tracking error [m/s], [rad/s]
  slip_p95_mm_s     95th percentile horizontal speed of the foot material point under the CoP while the foot carries > 0.5 BW
  foot_tilt_p95_deg 95th percentile foot tilt (sole normal vs world z) while the foot carries > 0.3 BW
  cop_margin_p5_mm  5th percentile distance from the CoP to the sole outline edge (heel -80, toe +180 mm, side 50 box / 57 sole)
                    while the foot carries > 0.3 BW: how close the load comes to tipping over the foot edge
  onsets_per_s      stance onsets per second per foot (contact chatter; normal walking ~1.5-2)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DT, H = 0.02, 4  # policy step, substeps per policy step
SKIP = 3.0
MOTORS = 12


def quat_to_mat(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def sole_z(variant):
  return -0.043 if variant in ("base", "v3same") else -0.057  # sole bottom in foot_link [m]


def load(path_stem: Path):
  D = dict(np.load(f"{path_stem}.npz"))
  meta = json.loads(Path(f"{path_stem}.json").read_text())
  T, E = D["terminated"].shape
  alive = np.ones((T, E), bool)
  for e in range(E):
    if D["terminated"][:, e].any():
      alive[np.argmax(D["terminated"][:, e]):, e] = False
  D["alive"] = alive
  return D, meta


def moving_mask(meta, T):
  n = int(round(meta["seconds_per_cmd"] / DT))
  m = np.zeros(T, bool)
  for k, (name, _, _) in enumerate(meta["program"]):
    if name == "stand":
      continue
    m[k * n + int(SKIP / DT):(k + 1) * n] = True
  return m


def stand_mask(meta, T):
  n = int(round(meta["seconds_per_cmd"] / DT))
  m = np.zeros(T, bool)
  for k, (name, _, _) in enumerate(meta["program"]):
    if name == "stand":
      m[k * n + int(SKIP / DT):(k + 1) * n] = True
  return m


def touchdowns(fz, ok, bw):
  """fz: substep force of one foot [N] (time,), ok: usable substeps. Returns (peak_bw, rate_bw_s) per touchdown."""
  quiet = int(round(0.060 / (DT / H)))
  win, rwin = int(round(0.100 / (DT / H))), int(round(0.050 / (DT / H)))
  low = (fz < 0.02 * bw).astype(int)
  c = np.concatenate([[0], np.cumsum(low)])
  i_ = np.arange(len(fz))
  quiet_before = (i_ >= quiet) & (c[i_] - c[np.clip(i_ - quiet, 0, None)] == quiet)  # all of [i-quiet, i) unloaded
  idx = np.flatnonzero((fz >= 0.05 * bw) & quiet_before)
  peaks, rates = [], []
  for i in idx:
    if i + win >= len(fz) or not ok[i:i + win].all():
      continue
    seg = fz[i - 1:i + win]
    peaks.append(seg.max() / bw)
    rates.append(np.max(np.diff(seg[:rwin + 1])) / (DT / H) / bw)
  return np.array(peaks), np.array(rates)


def env_metrics(D, meta):
  T, E = D["terminated"].shape
  bw = meta["robot_mass_kg"] * 9.81
  mv = moving_mask(meta, T)[:, None] & D["alive"]                  # T,E
  # sign of the world-frame net force: standing must carry + body weight
  g = D["grf_sub"][..., 2]                                         # T,E,2,H
  st = stand_mask(meta, T)[:, None] & D["alive"]
  sgn = np.sign(np.nanmedian(g.sum(2).mean(-1)[st])) or 1.0
  fz_sub = sgn * g                                                  # T,E,2,H
  S = D["sole"]
  fn = S[..., 0]
  out = {k: np.full(E, np.nan) for k in ("td_peak_bw", "td_peak_p95_bw", "td_rate_bw_s", "grf_p99_bw", "cot_pos", "cot_abs", "tau2_per_m",
                                           "fell", "alive_frac", "survival_s", "dist_to_fall_m", "tilt_rms_deg", "angvel_rms", "vx_err", "wz_err", "slip_p95_mm_s",
                                           "foot_tilt_p95_deg", "cop_margin_p5_mm", "onsets_per_s", "n_touchdowns", "distance_m")}
  Rf = quat_to_mat(D["foot_quat_w"])                               # T,E,2,3,3
  Rb = quat_to_mat(D["base_quat"])                                 # T,E,3,3
  zb = sole_z(meta["variant"])
  for e in range(E):
    m = mv[:, e]
    out["fell"][e] = float(D["terminated"][:, e].any())
    out["alive_frac"][e] = float(D["alive"][:, e][moving_mask(meta, T)].mean())
    # time and horizontal distance walked until the first fall (censored at the end of the program)
    k_end = int(np.argmax(D["terminated"][:, e])) if D["terminated"][:, e].any() else T
    out["survival_s"][e] = k_end * DT
    out["dist_to_fall_m"][e] = float(np.linalg.norm(np.diff(D["base_pos"][:k_end, e, :2], axis=0), axis=-1).sum()) if k_end > 1 else 0.0
    if m.sum() < 50:
      continue
    # impact
    ok_sub = np.repeat(m, H)
    pk, rt = [], []
    for f in range(2):
      p, r = touchdowns(fz_sub[:, e, f, :].reshape(-1), ok_sub, bw)
      pk.append(p)
      rt.append(r)
    pk, rt = np.concatenate(pk), np.concatenate(rt)
    out["n_touchdowns"][e] = len(pk)
    if len(pk):
      out["td_peak_bw"][e], out["td_peak_p95_bw"][e], out["td_rate_bw_s"][e] = np.median(pk), np.percentile(pk, 95), np.median(rt)
    fzl = fz_sub[m, e].reshape(-1)
    fzl = fzl[fzl > 0.1 * bw]
    out["grf_p99_bw"][e] = np.percentile(fzl, 99) / bw if fzl.size else np.nan
    # energy
    pw = D["tau"][m, e] * D["qd16"][m, e, :MOTORS]
    dxy = np.linalg.norm(np.diff(D["base_pos"][:, e, :2], axis=0), axis=-1)
    dist = float(dxy[m[1:] & m[:-1]].sum())
    out["distance_m"][e] = dist
    if dist > 1.0:
      out["cot_pos"][e] = np.clip(pw, 0, None).sum() * DT / (bw * dist)
      out["cot_abs"][e] = np.abs(pw).sum() * DT / (bw * dist)
      out["tau2_per_m"][e] = (D["tau"][m, e] ** 2).sum() * DT / dist
    # stability
    tilt = np.degrees(np.arccos(np.clip(Rb[m, e, 2, 2], -1, 1)))
    out["tilt_rms_deg"][e] = np.sqrt(np.mean(tilt ** 2))
    out["angvel_rms"][e] = np.sqrt(np.mean(np.sum(D["base_ang_vel_b"][m, e, :2] ** 2, -1)))
    out["vx_err"][e] = np.mean(np.abs(D["base_lin_vel_b"][m, e, 0] - D["command"][m, e, 0]))
    out["wz_err"][e] = np.mean(np.abs(D["base_ang_vel_b"][m, e, 2] - D["command"][m, e, 2]))
    ld = (fn[:, e] > 0.3 * bw) & m[:, None]                         # T,2
    # slip of the material point under the CoP (rolling feet move their origin without slipping)
    r = np.concatenate([np.nan_to_num(S[:, e, :, 2:4]), np.full((T, 2, 1), zb)], -1)
    pw_ = D["foot_pos_w"][:, e] + np.einsum("tfij,tfj->tfi", Rf[:, e], r)
    pn_ = D["foot_pos_w"][1:, e] + np.einsum("tfij,tfj->tfi", Rf[1:, e], r[:-1])
    vxy = np.linalg.norm((pn_ - pw_[:-1])[..., :2], axis=-1) / DT * 1000
    heavy = (fn[:, e] > 0.5 * bw) & m[:, None]
    sl = vxy[heavy[:-1] & heavy[1:]]
    out["slip_p95_mm_s"][e] = np.percentile(sl, 95) if sl.size else np.nan
    ft = np.degrees(np.arccos(np.clip(Rf[:, e, :, 2, 2], -1, 1)))
    out["foot_tilt_p95_deg"][e] = np.percentile(ft[ld], 95) if ld.any() else np.nan
    # distance from the CoP to the edge of the sole outline (tipping over the foot edge): x -80..+180 mm, |y| <= half width
    cx, cy = S[:, e, :, 2] * 1000, S[:, e, :, 3] * 1000
    hw = 50.0 if meta["variant"] in ("base", "v3same") else 57.0
    marg = np.minimum.reduce([cx + 80.0, 180.0 - cx, hw - np.abs(cy)])
    mg = marg[ld & np.isfinite(marg)]
    out["cop_margin_p5_mm"][e] = np.percentile(mg, 5) if mg.size else np.nan
    stn = (fn[:, e] > 0.1 * bw)
    on = (~stn[:-1]) & stn[1:] & m[1:, None]
    out["onsets_per_s"][e] = on.sum() / 2 / (m.sum() * DT)
  return out


if __name__ == "__main__":
  import sys
  D, meta = load(Path(sys.argv[1]))
  o = env_metrics(D, meta)
  print(json.dumps({k: [round(float(np.nanmean(v)), 4), round(float(np.nanstd(v)), 4)] for k, v in o.items()}, indent=1))
