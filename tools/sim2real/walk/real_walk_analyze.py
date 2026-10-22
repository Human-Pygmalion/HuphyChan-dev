#!/usr/bin/env python3
"""Numerical analysis of real-robot (or simulated) walking logs in the HUPHY `policy_*.csv` layout.

Layout-driven: every column is found by name (`obs/gyro/x`, `obs/q/<leg>/<joint>`, `act/...`, `tgt/...`, `pos/...`), so a
simulator export written in the same layout runs through the same code. Units assumed from the layout (checked by the
consistency block): pos and tgt in degrees, obs/q in rad minus default, act raw policy output, gyro rad/s, grav unit vector.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/real_walk_analyze.py \
      --manifest tools/sim2real/walk/real_walk_runs_expr_data.json \
      --out docs/sim2real/data/2026-10-22_real_walk_metrics.json \
      --fig-prefix docs/sim2real/img/2026-10-22_real_walk_
  # ad hoc:  ... --run simR1=path/policy.csv[,path/latency.csv]

Manifest: {"root": "...", "runs": [{"label","policy","latency"(opt),"body_gain","ankle_gain","kd_scale","note"}],
           "contrasts": [{"name","a":label,"b":label}], "fig_runs": [label,...]}.

Method notes (also written into the JSON under "method"):
  * Policy rows are sampled at the policy calls (one row per call), at a non-uniform interval on the real robot. Spectra use
    linear interpolation onto a uniform grid at the run's median interval (Welch, Hann window, 50 % overlap, linear detrend).
    Dominant frequencies of single channels use Lomb-Scargle on the raw (non-uniform) times, no resampling.
  * IMU frame is assumed: x forward, y left, z up (gz = -1 upright). tilt_fwd = atan2(gx, -gz) (+ = leaning forward),
    tilt_left = atan2(gy, -gz) (+ = leaning to the body left). Yaw rate = gyro z, + = counter-clockwise seen from above.
  * Heading = trapezoid integral of gyro z over the logged rows (policy-call sampling). It is a gyro integral, not an absolute
    heading: a constant gyro bias adds bias*T to it.
  * Mirror sign per joint type: from the `# limits` header (R limits == -L limits reversed => -1), else from the default pose
    sign, else -1 (audit rule: left = -right for all six types).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

DEG = 180.0 / math.pi
ACTION_SCALE_DEG = 0.25 * DEG
BANDS = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, None)]
AXES = "xyz"
JT = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]


def pct(a, q):
  return float(np.percentile(a, q)) if len(a) else float("nan")


def stats(a):
  a = np.asarray(a, float)
  a = a[np.isfinite(a)]
  if not len(a):
    return {}
  return {"n": int(len(a)), "mean": float(a.mean()), "median": pct(a, 50), "p5": pct(a, 5), "p95": pct(a, 95),
          "p99": pct(a, 99), "max": float(a.max()), "min": float(a.min()), "std": float(a.std())}


# ----------------------------------------------------------------------------------------------- loading
def load_policy(path):
  limits = {}
  with open(path, encoding="utf-8") as fh:
    first = fh.readline()
  if first.startswith("# limits"):
    for tok in first.split()[2:]:
      n, v = tok.split("=")
      lo, hi = v.split(":")
      limits[n] = (float(lo), float(hi))
  d = pd.read_csv(path, comment="#")
  joints = [c[4:] for c in d.columns if c.startswith("pos/")]

  def blk(prefix):
    return d[[f"{prefix}/{j}" for j in joints]].to_numpy(float)

  S = dict(t=d["t"].to_numpy(float), gyro=d[[f"obs/gyro/{a}" for a in AXES]].to_numpy(float),
           grav=d[[f"obs/grav/{a}" for a in AXES]].to_numpy(float), qprev=blk("obs/q_prev"), q=blk("obs/q"),
           lastact=blk("obs/last_act"), act=blk("act"), tgt=blk("tgt"), pos=blk("pos"),
           cmd_vx=d["obs/cmd/vx"].to_numpy(float) if "obs/cmd/vx" in d else np.zeros(len(d)),
           cmd_wz=d["obs/cmd/wz"].to_numpy(float) if "obs/cmd/wz" in d else np.zeros(len(d)))
  known = set(["t"] + [c for c in d.columns if re.match(r"(obs/(gyro|grav|q_prev|q|last_act|cmd)|act|tgt|pos)/", c)])
  S["extra_columns"] = [c for c in d.columns if c not in known]
  S["joints"], S["limits"], S["ncols"] = joints, limits, d.shape[1]
  return S


def load_latency(path):
  hz = None
  with open(path, encoding="utf-8") as fh:
    m = re.search(r"hz=(\d+)", fh.readline())
    hz = int(m.group(1)) if m else None
  d = pd.read_csv(path, comment="#")
  joints = list(dict.fromkeys(d["joint"]))
  per = {}
  for j in joints:
    x = d[d["joint"] == j]
    per[j] = (x["t"].to_numpy(float), x["latency_ms"].to_numpy(float), x["status"].to_numpy(str))
  return dict(hz=hz, joints=joints, per=per)


# ----------------------------------------------------------------------------------------------- 1 timing + pairing
def timing(S, L):
  dt = np.diff(S["t"]) * 1e3
  out = {"n_rows": int(len(S["t"])), "duration_s": float(S["t"][-1] - S["t"][0]), "policy_interval_ms": stats(dt),
         "policy_interval_frac_gt_30ms": float((dt > 30).mean()), "policy_interval_frac_gt_40ms": float((dt > 40).mean()),
         "policy_rate_hz_mean": float(1e3 / dt.mean())}
  if L is None:
    return out, None
  n = min(len(v[0]) for v in L["per"].values())
  T = np.stack([L["per"][j][0][:n] for j in L["joints"]])  # joints x cycles
  lat = np.stack([L["per"][j][1][:n] for j in L["joints"]])
  st = np.stack([L["per"][j][2][:n] for j in L["joints"]])
  c0 = np.nanmin(T, axis=0)  # cycle start = first reply of the cycle
  per_ms = np.diff(c0) * 1e3
  hz = L["hz"]
  lo = {"hz_setting": hz, "n_cycles": int(n), "duration_s": float(c0[-1] - c0[0]), "loop_period_ms": stats(per_ms),
        "loop_rate_hz_mean": float(1e3 / per_ms.mean()),
        "loop_frac_gt_1p5x_setting": float((per_ms > 1.5e3 / hz).mean()) if hz else None,
        "reply_spread_ms": stats((np.nanmax(T, 0) - np.nanmin(T, 0)) * 1e3), "per_joint": {}, "leg": {}}
  for i, j in enumerate(L["joints"]):
    bad = st[i] != "ontime"
    lo["per_joint"][j] = {"latency_ms": stats(lat[i]), "non_ontime_frac": float(bad.mean()), "n_late": int((st[i] == "late").sum()),
                          "n_lost": int((st[i] == "lost").sum())}
  for leg in ("right", "left"):
    ix = [i for i, j in enumerate(L["joints"]) if j.startswith(leg)]
    if ix:
      lo["leg"][leg] = {"latency_ms": stats(lat[ix].ravel()), "non_ontime_frac": float((st[ix] != "ontime").mean())}
  out["latency"] = lo
  # pairing check: policy is called every N cycles; compare policy dt with the sum of N cycle periods
  N = int(round(hz / 50)) if hz else None
  pair = {"N_cycles_per_call": N, "cycles_per_row": float(n / len(S["t"]))}
  if N:
    best = None
    for o in range(N):
      idx = N * np.arange(len(S["t"])) + o
      idx = idx[idx < n]
      if len(idx) < 3:
        continue
      d_lat = np.diff(c0[idx])
      d_pol = np.diff(S["t"][:len(idx)])
      r = d_pol - d_lat
      rn = d_pol - np.roll(d_lat, 7)  # null: same series shifted by 7 policy calls
      cand = {"offset": o, "mae_ms": float(np.abs(r - np.median(r)).mean() * 1e3), "mae_null_shift7_ms": float(np.abs(rn - np.median(rn)).mean() * 1e3), "median_diff_ms": float(np.median(r) * 1e3),
              "corr": float(np.corrcoef(d_pol, d_lat)[0, 1]), "t0_latency_s": float(c0[idx[0]])}
      if best is None or cand["mae_ms"] < best["mae_ms"]:
        best = cand
    pair.update(best or {})
    pair["rows_match"] = bool(abs(n / N - len(S["t"])) <= 2)
    pair["duration_diff_s"] = float(lo["duration_s"] - out["duration_s"])
    pair["verdict"] = "paired" if pair["rows_match"] and best and best["mae_ms"] < 0.5 * best["mae_null_shift7_ms"] and abs(pair["duration_diff_s"]) < 0.2 else "NOT paired"
  out["pairing"] = pair
  return out, dict(cycle_start=c0, period_ms=per_ms)


# ----------------------------------------------------------------------------------------------- 2 consistency
def default_pose(S):
  return np.median(S["pos"] - np.degrees(S["q"]), axis=0)


def consistency(S, dflt):
  J = S["joints"]
  lim = np.array([S["limits"].get(j, (-np.inf, np.inf)) for j in J])
  raw = dflt + ACTION_SCALE_DEG * S["act"]
  exp = np.clip(raw, lim[:, 0], lim[:, 1])
  clipped = np.abs(raw - exp) > 1e-6
  near = (S["tgt"] < lim[:, 0] + 3.0) | (S["tgt"] > lim[:, 1] - 3.0)  # within the 3 deg safety margin (audit F6)
  at = (S["tgt"] <= lim[:, 0] + 1e-3) | (S["tgt"] >= lim[:, 1] - 1e-3)
  d_q = np.abs(S["pos"] - dflt - np.degrees(S["q"])).max(0)
  d_qn = np.abs(S["pos"][:-1] - dflt - np.degrees(S["q"][1:])).max(0)
  pr = {
    "default_pose_deg": dict(zip(J, dflt.round(4).tolist())),
    "q_prev_k_vs_q_k-1_max_abs_rad": float(np.abs(S["qprev"][1:] - S["q"][:-1]).max()),
    "last_act_k_vs_act_k-1_max_abs": float(np.abs(S["lastact"][1:] - S["act"][:-1]).max()),
    "pos_k_vs_default+deg(q_k)_max_abs_deg": float(d_q.max()),
    "pos_k_vs_default+deg(q_k+1)_max_abs_deg": float(d_qn.max()),
    "newer_reading": "pos(k) == obs/q(k) (same sample)" if d_q.max() < 1e-2 else ("pos(k) == obs/q(k+1)" if d_qn.max() < 1e-2 else "neither"),
    "tgt_vs_clip(default+0.25*act)_max_abs_deg": float(np.abs(S["tgt"] - exp).max()),
    "tgt_vs_unclipped_max_abs_deg_where_not_clipped": float(np.abs(S["tgt"] - raw)[~clipped].max()) if (~clipped).any() else 0.0,
    "per_joint": {j: {"n_clipped": int(clipped[:, i].sum()), "frac_clipped": float(clipped[:, i].mean()),
                      "frac_tgt_within_3deg_of_limit": float(near[:, i].mean()), "frac_tgt_at_limit": float(at[:, i].mean()),
                      "limit_lo_hi_deg": list(map(float, lim[i]))} for i, j in enumerate(J)},
    "n_rows": int(len(S["t"])), "n_columns": int(S["ncols"]), "extra_columns": S["extra_columns"],
    "t_monotonic": bool((np.diff(S["t"]) > 0).all()),
    "gravity_norm_minmax": [float(np.linalg.norm(S["grav"], axis=1).min()), float(np.linalg.norm(S["grav"], axis=1).max())],
  }
  return pr


# ----------------------------------------------------------------------------------------------- legs / mirror
def leg_index(S, dflt):
  J = S["joints"]
  types = list(dict.fromkeys(j.split("/")[-1] for j in J))
  idx, mir = {}, {}
  for ty in types:
    r = next(i for i, j in enumerate(J) if j.startswith("right") and j.endswith("/" + ty))
    l = next(i for i, j in enumerate(J) if j.startswith("left") and j.endswith("/" + ty))
    idx[ty] = (r, l)
    lr, ll = S["limits"].get(J[r]), S["limits"].get(J[l])
    if lr and ll and abs(lr[0] + lr[1]) > 1 and abs(lr[0] - ll[0]) + abs(lr[1] - ll[1]) > 1:
      mir[ty] = (-1, "limits") if abs(lr[0] + ll[1]) + abs(lr[1] + ll[0]) < 1e-3 else (1, "limits")
    elif lr and ll and abs(lr[0] + lr[1]) > 1:
      mir[ty] = (1, "limits")
    elif abs(dflt[r]) > 1.0:
      mir[ty] = (int(np.sign(dflt[r] * dflt[l])) or -1, "default pose")
    else:
      mir[ty] = (-1, "audit rule (symmetric limits, zero default)")
  return types, idx, mir


# ----------------------------------------------------------------------------------------------- IMU / encoder freeze
def repeat_masks(S):
  """rep[k] is True when row k is an exact copy of row k-1 (row 0 False). Exact repeats of float readings = no new sample."""
  def rep(X):
    r = np.zeros(len(X), bool)
    r[1:] = (np.diff(X, axis=0) == 0).all(1)
    return r
  return {"gyro": rep(S["gyro"]), "grav": rep(S["grav"]), "q": rep(S["q"])}


def runs_of(mask):
  out, i, n = [], 0, len(mask)
  while i < n:
    if mask[i]:
      j = i
      while j < n and mask[j]:
        j += 1
      out.append((i, j - i))
      i = j
    else:
      i += 1
  return out


def freeze(S, rep):
  """Exact-repeat statistics per channel group and whether gyro freezes coincide with motion."""
  t, n = S["t"], len(S["t"])
  out = {"definition": "row k equals row k-1 exactly in all axes (gyro, gravity) or all 12 joints (q)"}
  for k, m in rep.items():
    rr = runs_of(m)
    long = [(i, L) for i, L in rr if L >= 3]
    lg = max(rr, key=lambda x: x[1]) if rr else None
    out[k] = {"frac_rows": float(m.mean()), "n_rows": int(m.sum()), "n_runs": len(rr), "n_runs_ge3_rows": len(long),
              "frac_rows_in_runs_ge3": float(sum(L for _, L in long) / n), "longest_rows": int(lg[1]) if lg else 0,
              "longest_s": float(t[lg[0] + lg[1] - 1] - t[lg[0] - 1]) if lg else 0.0,
              "longest_starts_at_s": float(t[lg[0] - 1]) if lg else None}
  g = rep["gyro"]
  out["gyro"]["frac_of_repeats_with_all_zero_value"] = float((S["gyro"][g] == 0).all(1).mean()) if g.any() else None
  out["overlap"] = {"gyro_repeat_also_grav_repeat": float(rep["grav"][g].mean()) if g.any() else None,
                    "gyro_repeat_also_q_repeat": float(rep["q"][g].mean()) if g.any() else None,
                    "grav_repeat_also_gyro_repeat": float(g[rep["grav"]].mean()) if rep["grav"].any() else None}
  # coincidence with motion: joint motion per row and gravity-direction change per row, for frozen vs live gyro rows
  dpos = np.zeros(n)
  dpos[1:] = np.abs(np.diff(S["pos"], axis=0)).mean(1)
  gv = S["grav"] / np.linalg.norm(S["grav"], axis=1, keepdims=True)
  dtilt = np.zeros(n)
  dtilt[1:] = np.degrees(np.arccos(np.clip((gv[1:] * gv[:-1]).sum(1), -1, 1)))
  rows = np.arange(n) > 0
  fz, lv = g & rows, ~g & rows
  gl = lv & ~rep["grav"]
  gf = fz & ~rep["grav"]
  if fz.any() and lv.any():
    out["motion_vs_gyro_freeze"] = {
      "mean_abs_dpos_deg_per_row_gyro_frozen": float(dpos[fz].mean()), "mean_abs_dpos_deg_per_row_gyro_live": float(dpos[lv].mean()),
      "frac_frozen_rows_dpos_above_live_median": float((dpos[fz] > np.median(dpos[lv])).mean()),
      "mean_dtilt_deg_per_row_gyro_frozen_grav_live": float(dtilt[gf].mean()) if gf.any() else None,
      "mean_dtilt_deg_per_row_gyro_live_grav_live": float(dtilt[gl].mean()) if gl.any() else None,
      "note": "frac ~0.5 means the freeze is unrelated to joint motion; the dtilt pair tells whether the body keeps moving while the gyro is frozen"}
  return out


# ----------------------------------------------------------------------------------------------- 3 heading
def trapz_cum(t, y):
  return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


def quiet_window(t, g, width=2.0, min_n=8):
  cs = np.vstack([np.zeros(3), np.cumsum(g, 0)])
  cs2 = np.vstack([np.zeros(3), np.cumsum(g * g, 0)])
  best = None
  for i in range(len(t)):
    j = int(np.searchsorted(t, t[i] + width))
    if j - i < min_n or t[i] + width > t[-1] + 1e-9:
      continue
    m = (cs[j] - cs[i]) / (j - i)
    v = (cs2[j] - cs2[i]) / (j - i) - m * m
    score = float(np.sqrt(np.maximum(v, 0)).sum())
    if best is None or score < best[0]:
      best = (score, i, j)
  return best


def tilt_deg(grav):
  return np.degrees(np.arctan2(grav[:, 0], -grav[:, 2])), np.degrees(np.arctan2(grav[:, 1], -grav[:, 2]))


def heading(S, rep, win=10.0):
  """Heading from gyro z, raw and with frozen intervals handled. Interval k (rows k-1 -> k) is frozen when gyro row k repeats row k-1."""
  t, g = S["t"], S["gyro"]
  rate = g * DEG
  hd = trapz_cum(t, rate[:, 2])
  T = t[-1] - t[0]
  fwd, left = tilt_deg(S["grav"])
  dt = np.diff(t)
  inc = 0.5 * (rate[1:, 2] + rate[:-1, 2]) * dt
  fzi = rep["gyro"][1:]
  live_t, froz_t = dt[~fzi].sum(), dt[fzi].sum()
  rate_live = inc[~fzi].sum() / live_t if live_t > 0 else float("nan")
  glive = ~rep["gyro"]
  gravlive = ~rep["grav"]
  out = {"heading_change_deg": float(hd[-1]), "mean_yaw_rate_deg_s": float(hd[-1] / T), "duration_s": float(T),
         "heading_frozen_handling": {
           "raw_deg": float(hd[-1]), "frozen_time_frac": float(froz_t / T),
           "contribution_of_frozen_intervals_deg": float(inc[fzi].sum()),
           "live_intervals_only_deg (frozen counted as zero)": float(inc[~fzi].sum()),
           "live_mean_rate_deg_s": float(rate_live), "live_rate_times_duration_deg (frozen filled with live mean)": float(rate_live * T),
           "note": "yaw is not observable from gravity or joint angles; no independent heading estimate is possible from these columns"},
         "mean_roll_rate_deg_s": float(trapz_cum(t, rate[:, 0])[-1] / T), "mean_pitch_rate_deg_s": float(trapz_cum(t, rate[:, 1])[-1] / T),
         "gyro_rms_deg_s": {a: float(np.sqrt((rate[:, i] ** 2).mean())) for i, a in enumerate(AXES)},
         "gyro_rms_deg_s_live_rows": {a: float(np.sqrt((rate[glive, i] ** 2).mean())) for i, a in enumerate(AXES)},
         "gyro_frac_exact_zero": float((g == 0).all(1).mean()),
         "gyro_frac_repeat_of_previous": float(rep["gyro"].mean()),
         "tilt_fwd_deg_run": {"mean": float(fwd[gravlive].mean()), "std": float(fwd[gravlive].std()), "note": "rows whose gravity repeats the previous row excluded",
                              "mean_all_rows": float(fwd.mean())},
         "tilt_left_deg_run": {"mean": float(left[gravlive].mean()), "std": float(left[gravlive].std()), "mean_all_rows": float(left.mean())}}
  n0 = max(int(np.searchsorted(t, t[0] + 0.3)), 1)
  out["start_0p3s"] = {"n": int(n0), "n_gyro_repeat_rows": int(rep["gyro"][:n0].sum()), "gyro_mean_deg_s": (g[:n0].mean(0) * DEG).round(4).tolist(),
                       "tilt_fwd_deg": float(fwd[:n0].mean()), "tilt_left_deg": float(left[:n0].mean())}
  # quiet 2 s: search only live gyro rows (exact repeats removed), need >= half the expected row density
  il = np.where(glive)[0]
  dtm = float(np.median(dt))
  qw = quiet_window(t[il], g[il], min_n=max(8, int(0.5 * 2.0 / dtm))) if len(il) > 16 else None
  if qw:
    _, i0, j0 = qw
    idx = il[i0:j0]
    gi = idx[gravlive[idx]]
    t0q, t1q = t[idx[0]], t[idx[-1]]
    n_all = int(((t >= t0q) & (t <= t1q)).sum())
    out["quiet_2s"] = {"t_start": float(t0q), "n_live_rows": int(len(idx)), "n_rows_in_span_incl_frozen": n_all,
                       "gyro_mean_deg_s": (g[idx].mean(0) * DEG).round(4).tolist(), "gyro_std_deg_s": (g[idx].std(0) * DEG).round(4).tolist(),
                       "tilt_fwd_deg": float(fwd[gi].mean()), "tilt_left_deg": float(left[gi].mean()),
                       "note": "frozen (exactly repeated) gyro rows excluded"}
  else:
    out["quiet_2s"] = None
  # command vs achieved
  vx, wz = S["cmd_vx"], S["cmd_wz"]
  bias_z = (out["quiet_2s"] or {}).get("gyro_mean_deg_s", [0, 0, 0])[2]
  out["command"] = {"vx_median": float(np.median(vx)), "wz_median": float(np.median(wz)), "vx_unique": sorted(set(np.round(vx, 3).tolist()))[:6],
                    "wz_unique": sorted(set(np.round(wz, 3).tolist()))[:6], "wz_cmd_deg_s": float(np.median(wz) * DEG),
                    "yaw_rate_achieved_minus_bias_deg_s": out["mean_yaw_rate_deg_s"] - bias_z}
  # segments by command change and fixed windows
  key = np.round(vx, 3) * 1000 + np.round(wz, 3) * 7
  cuts = [0] + [i for i in range(1, len(t)) if key[i] != key[i - 1]] + [len(t) - 1]

  def window(a, b):
    f = fzi[a:b]
    dd = dt[a:b]
    live = dd[~f].sum()
    return {"heading_change_deg": float(hd[b] - hd[a]), "mean_yaw_rate_deg_s": float((hd[b] - hd[a]) / (t[b] - t[a])),
            "frozen_time_frac": float(dd[f].sum() / dd.sum()),
            "mean_yaw_rate_live_deg_s": float(inc[a:b][~f].sum() / live) if live > 0 else None}
  segs = []
  for a, b in zip(cuts[:-1], cuts[1:]):
    if t[b] - t[a] > 0.5:
      segs.append({"t0": float(t[a]), "t1": float(t[b]), "vx": float(vx[a]), "wz": float(wz[a]), **window(a, b)})
  out["segments_by_command"] = segs
  wins = []
  for t0 in np.arange(t[0], t[-1] - win / 2, win):
    a, b = int(np.searchsorted(t, t0)), min(int(np.searchsorted(t, t0 + win)), len(t) - 1)
    if b - a >= 3:
      wins.append({"t0": float(t0), "t1": float(t[b]), **window(a, b)})
  out["windows_10s"] = wins
  return out, hd


# ----------------------------------------------------------------------------------------------- spectra helpers
def uniform(t, Y, dt):
  tu = np.arange(t[0], t[-1], dt)
  return tu, np.stack([np.interp(tu, t, Y[:, i]) for i in range(Y.shape[1])], 1)


def ls_peaks(t, y, fmin, fmax, k=3):
  df = max(0.01, 0.25 / (t[-1] - t[0]))
  f = np.arange(fmin, fmax + df / 2, df)
  p = signal.lombscargle(t, y, 2 * np.pi * f, precenter=True, normalize=True)
  pk, _ = signal.find_peaks(p, distance=max(int(0.15 / df), 1))
  pk = pk[np.argsort(p[pk])[::-1]][:k]
  return [{"f_hz": float(f[i]), "norm_power": float(p[i])} for i in pk]


def band_power(f, P):
  fn = f[-1]
  out = []
  df = f[1] - f[0]
  for lo, hi in BANDS:
    m = (f >= lo) & (f < (hi if hi else fn + 1))
    out.append(P[m].sum(0) * df)
  return np.array(out)  # bands x channels


def psd_peaks(f, p, k=3, fmin=0.2):
  m = f >= fmin
  pk, _ = signal.find_peaks(p[m])
  pk = pk[np.argsort(p[m][pk])[::-1]][:k]
  return [{"f_hz": float(f[m][i]), "psd": float(p[m][i])} for i in pk]


def near(f, x, f0):
  return x[int(np.argmin(np.abs(f - f0)))]


# ----------------------------------------------------------------------------------------------- 4 gait
def stat_lr(x, y, m):
  """x right, y left (raw). Returns raw + mirrored stats and asymmetry indices."""
  ym = m * y
  r = dict(mean_R=float(x.mean()), mean_L=float(y.mean()), mean_L_mirrored=float(ym.mean()), rms_R=float(x.std()), rms_L=float(y.std()),
           min_R=float(x.min()), max_R=float(x.max()), min_L=float(y.min()), max_L=float(y.max()), pp_R=float(np.ptp(x)), pp_L=float(np.ptp(y)))
  r["offset_mirrored_L_minus_R"] = r["mean_L_mirrored"] - r["mean_R"]
  r["asym_rms"] = (r["rms_L"] - r["rms_R"]) / (0.5 * (r["rms_L"] + r["rms_R"]) + 1e-12)
  r["asym_abs_mean"] = (abs(r["mean_L"]) - abs(r["mean_R"])) / (0.5 * (abs(r["mean_L"]) + abs(r["mean_R"])) + 1e-12)
  return r


def phase_average(t, X, ref, nb=50):
  """X: n x ch. Cycles from rising zero crossings (hysteresis 0.1 std) of the reference signal."""
  r = ref - ref.mean()
  h = 0.1 * r.std()
  cross, armed = [], False
  for i in range(1, len(r)):
    if r[i] < -h:
      armed = True
    if armed and r[i - 1] <= 0 < r[i]:
      cross.append(t[i - 1] + (t[i] - t[i - 1]) * (0 - r[i - 1]) / (r[i] - r[i - 1]))
      armed = False
  cross = np.array(cross)
  if len(cross) < 4:
    return None
  per = np.diff(cross)
  ok = (per > 0.25) & (per < 6.0)
  sums = np.zeros((nb, X.shape[1]))
  sq = np.zeros_like(sums)
  cnt = np.zeros(nb)
  for a, b, good in zip(cross[:-1], cross[1:], ok):
    if not good:
      continue
    i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
    if i1 - i0 < 4:
      continue
    ph = (t[i0:i1] - a) / (b - a)
    bi = np.minimum((ph * nb).astype(int), nb - 1)
    for k in range(nb):
      sel = bi == k
      if sel.any():
        v = X[i0:i1][sel].mean(0)
        sums[k] += v
        sq[k] += v * v
        cnt[k] += 1
  c = np.maximum(cnt, 1)[:, None]
  mean = sums / c
  return dict(mean=mean, std=np.sqrt(np.maximum(sq / c - mean ** 2, 0)), n_cycles=int(ok.sum()), period_mean_s=float(per[ok].mean()),
              period_std_s=float(per[ok].std()), period_cv=float(per[ok].std() / per[ok].mean()))


def gait(S, types, idx, mir, tm, t_lo, t_hi):
  t, pos, tgt, act = S["t"], S["pos"], S["tgt"], S["act"]
  fn = 0.5 / np.median(np.diff(t))
  out = {"nyquist_hz_at_median_interval": float(fn), "mirror_sign": {ty: {"sign": mir[ty][0], "basis": mir[ty][1]} for ty in types},
         "per_type": {}}
  for ty in types:
    r, l = idx[ty]
    m = mir[ty][0]
    out["per_type"][ty] = {"pos": stat_lr(pos[:, r], pos[:, l], m), "tgt": stat_lr(tgt[:, r], tgt[:, l], m),
                           "act_mean_R": float(act[:, r].mean()), "act_mean_L": float(act[:, l].mean()),
                           "act_std_R": float(act[:, r].std()), "act_std_L": float(act[:, l].std())}
  # ankle common/differential modes (assumption: pitch ~ a - b, roll ~ a + b, from default pose a=-b)
  if "ankle_a" in idx and "ankle_b" in idx:
    for nm, sg in (("ankle_diff(a-b)/2", -1), ("ankle_sum(a+b)/2", 1)):
      xr = 0.5 * (pos[:, idx["ankle_a"][0]] + sg * pos[:, idx["ankle_b"][0]])
      xl = 0.5 * (pos[:, idx["ankle_a"][1]] + sg * pos[:, idx["ankle_b"][1]])
      out["per_type"][nm] = {"pos": stat_lr(xr, xl, -1)}
  # step frequency
  rh, lh = idx["hip_pitch"]
  rk, lk = idx["knee"]
  fmax = fn * 0.999
  fr = {}
  for name, y in (("R_hip_pitch", pos[:, rh]), ("L_hip_pitch", pos[:, lh]), ("R_knee", pos[:, rk]), ("L_knee", pos[:, lk]),
                  ("gyro_x", S["gyro"][:, 0]), ("gyro_y", S["gyro"][:, 1]), ("gyro_z", S["gyro"][:, 2])):
    fr[name] = ls_peaks(t, y, 0.3, fmax)
  out["dominant_freq_lomb_scargle"] = fr
  # alias check: Lomb-Scargle on the non-uniform stamps up to 2x the mean-rate Nyquist; a peak above Nyquist means the <=Nyquist peak may be an alias
  out["dominant_freq_lomb_scargle_to_2x_nyquist"] = {n: ls_peaks(t, y, 0.3, 2 * fn, k=2) for n, y in
                                                     (("R_hip_pitch", pos[:, rh]), ("R_knee", pos[:, rk]), ("gyro_x", S["gyro"][:, 0]), ("gyro_y", S["gyro"][:, 1]))}
  f_gait = fr["R_hip_pitch"][0]["f_hz"] if fr["R_hip_pitch"] else float("nan")
  out["f_gait_hz"] = f_gait
  # L-R phase at f_gait (cross-spectrum) and cross-correlation, on the uniform grid
  dtm = float(np.median(np.diff(t)))
  tu, U = uniform(t, np.stack([pos[:, rh], mir["hip_pitch"][0] * pos[:, lh]], 1), dtm)
  nper = min(256, len(tu) // 2)
  if nper >= 32 and np.isfinite(f_gait):
    f, Pxy = signal.csd(U[:, 0], U[:, 1], fs=1 / dtm, nperseg=nper, noverlap=nper // 2, detrend="linear")
    ph = float(np.degrees(np.angle(near(f, Pxy, f_gait))))
    out["LR_hip_pitch_phase_deg_at_f_gait"] = ph  # + = mirrored-left leads right
    a, b = U[:, 0] - U[:, 0].mean(), U[:, 1] - U[:, 1].mean()
    cc = signal.correlate(a, b, mode="full") / (len(a) * a.std() * b.std() + 1e-12)
    lags = signal.correlation_lags(len(a), len(b), mode="full") * dtm
    sel = np.abs(lags) <= 0.75 / f_gait
    k = int(np.argmax(cc[sel]))
    out["LR_hip_pitch_xcorr"] = {"max_coef": float(cc[sel][k]), "lag_s_right_minus_left": float(lags[sel][k]),
                                 "lag_deg_of_gait_cycle": float(lags[sel][k] * f_gait * 360)}
    # also zero-lag correlation (mirrored left vs right)
    out["LR_hip_pitch_corr_zero_lag"] = float(np.corrcoef(U[:, 0], U[:, 1])[0, 1])
  # phase average from R hip_pitch crossings, both legs (left mirrored)
  Xm = pos.copy()
  for ty in types:
    Xm[:, idx[ty][1]] *= mir[ty][0]
  pa = phase_average(t, Xm, pos[:, rh])
  arr = None
  if pa:
    out["phase_average"] = {"n_cycles": pa["n_cycles"], "period_mean_s": pa["period_mean_s"], "period_std_s": pa["period_std_s"],
                            "period_cv": pa["period_cv"], "n_bins": int(pa["mean"].shape[0]),
                            "pp_deg": {S["joints"][i]: float(np.ptp(pa["mean"][:, i])) for i in range(len(S["joints"]))}}
    # phase-average difference between mirrored L and R per type
    out["phase_average"]["LR_rms_diff_deg"] = {ty: float(np.sqrt(((pa["mean"][:, idx[ty][0]] - pa["mean"][:, idx[ty][1]]) ** 2).mean())) for ty in types}
    arr = pa
  return out, arr


# ----------------------------------------------------------------------------------------------- 5/6 jitter, frequency response
def live_interp(t, Y, rep):
  """Replace frozen (exactly repeated) rows by linear interpolation between the neighbouring live rows."""
  live = ~rep
  if live.sum() < 10:
    return Y.copy()
  return np.stack([np.interp(t, t[live], Y[live, i]) for i in range(Y.shape[1])], 1)


def spectra(S, types, idx, gait_out, rep):
  t = S["t"]
  dtm = float(np.median(np.diff(t)))
  fs = 1 / dtm
  J = S["joints"]
  nj = len(J)
  pos, tgt, act, gy = S["pos"], S["tgt"], S["act"], S["gyro"]
  err0 = tgt - pos
  gy_live = live_interp(t, gy, rep["gyro"])
  pos_live = live_interp(t, pos, rep["q"])
  tu, U = uniform(t, np.hstack([pos, tgt, act, gy, err0, gy_live, pos_live]), dtm)
  nper = min(256, len(tu) // 2)
  if len(t) < 128 or nper < 64:
    return {"skipped": f"too short ({len(t)} rows, {len(tu)} uniform samples; need >=128 rows)"}, None
  kseg = (len(tu) - nper) // (nper // 2) + 1
  sl = {"pos": slice(0, nj), "tgt": slice(nj, 2 * nj), "act": slice(2 * nj, 3 * nj), "gyro": slice(3 * nj, 3 * nj + 3),
        "err": slice(3 * nj + 3, 4 * nj + 3), "gyro_live": slice(4 * nj + 3, 4 * nj + 6), "pos_live": slice(4 * nj + 6, 5 * nj + 6)}
  kw = dict(fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, detrend="linear")
  f, P = signal.welch(U, axis=0, **kw)
  df = f[1] - f[0]
  out = {"method": {"resample": f"linear interpolation to uniform grid dt={dtm * 1e3:.2f} ms (median policy interval), fs={fs:.2f} Hz",
                    "welch": f"Hann, nperseg={nper} ({nper * dtm:.1f} s), noverlap={nper // 2}, linear detrend, {kseg} segments, df={df:.4f} Hz",
                    "nyquist_hz": float(f[-1]), "bands_hz": ["0-1", "1-3", "3-6", "6-Nyq"], "band_unit": "channel unit^2 (deg^2, act^2, (rad/s)^2)"},
         "band_power": {}, "peaks": {}}
  bp = band_power(f, P)
  names = {"pos": J, "tgt": J, "act": J, "gyro": ["x", "y", "z"], "err": J, "gyro_live": ["x", "y", "z"], "pos_live": J}
  for grp, s in sl.items():
    out["band_power"][grp] = {n: bp[:, s.start + i].tolist() for i, n in enumerate(names[grp])}
    out["peaks"][grp] = {n: psd_peaks(f, P[:, s.start + i]) for i, n in enumerate(names[grp])}
  # summed pos spectrum -> dominant and high-band peak
  Ppos = P[:, sl["pos"]].sum(1)
  pk_all = psd_peaks(f, Ppos, 3)
  hi = f >= 3.0
  pk_hi = psd_peaks(f[hi], Ppos[hi], 2, fmin=3.0)
  f_dom = pk_all[0]["f_hz"] if pk_all else float("nan")
  f_hi = pk_hi[0]["f_hz"] if pk_hi else float("nan")
  out["summed_pos_psd_peaks"] = pk_all
  out["summed_pos_psd_peaks_above_3hz"] = pk_hi
  out["f_dominant_hz"], out["f_hi_hz"] = f_dom, f_hi
  tot = bp[:, sl["pos"]].sum(1)
  out["pos_band_fraction_sum_over_joints"] = (tot / tot.sum()).tolist()
  out["gyro_band_fraction_sum_over_axes"] = (bp[:, sl["gyro"]].sum(1) / bp[:, sl["gyro"]].sum()).tolist()
  out["gyro_live_interp_band_fraction_sum_over_axes"] = (bp[:, sl["gyro_live"]].sum(1) / bp[:, sl["gyro_live"]].sum()).tolist()
  out["pos_live_interp_band_fraction_sum_over_joints"] = (bp[:, sl["pos_live"]].sum(1) / bp[:, sl["pos_live"]].sum()).tolist()
  out["freeze_note"] = (f"gyro_live / pos_live = frozen (exactly repeated) rows replaced by interpolation between live rows; "
                        f"frozen rows: gyro {100 * rep['gyro'].mean():.1f} %, q {100 * rep['q'].mean():.1f} %. Raw channels keep the frozen rows.")
  # tracking error
  out["tracking_error_deg"] = {J[i]: {"rms_tgt_k_minus_pos_k": float(np.sqrt((err0[:, i] ** 2).mean())), "mean_tgt_k_minus_pos_k": float(err0[:, i].mean()),
                                       "rms_tgt_k-1_minus_pos_k": float(np.sqrt(((tgt[:-1, i] - pos[1:, i]) ** 2).mean()))} for i in range(nj)}
  out["act_step_rms"] = {J[i]: float(np.sqrt((np.diff(act[:, i]) ** 2).mean())) for i in range(nj)}
  # coherence and frequency response at the frequency points
  pts = {"f_gait": gait_out.get("f_gait_hz"), "f_dominant": f_dom, "f_hi": f_hi}
  out["freq_points_hz"] = pts
  coh = {"act_pos": {}, "act_gyro": {}, "tgt_pos": {}}
  H = {}
  for i, j in enumerate(J):
    _, c_ap = signal.coherence(U[:, 2 * nj + i], U[:, i], **{k: v for k, v in kw.items() if k != "window"}, window="hann")
    _, c_tp = signal.coherence(U[:, nj + i], U[:, i], **{k: v for k, v in kw.items() if k != "window"}, window="hann")
    _, Pxy = signal.csd(U[:, nj + i], U[:, i], **kw)
    Pxx = P[:, nj + i]
    h = Pxy / (Pxx + 1e-30)
    m_mean = (f >= 0.3)
    coh["act_pos"][j] = {"mean_0p3_nyq": float(c_ap[m_mean].mean()), **{k: float(near(f, c_ap, v)) for k, v in pts.items() if v and np.isfinite(v)}}
    coh["tgt_pos"][j] = {"mean_0p3_nyq": float(c_tp[m_mean].mean())}
    H[j] = {"coh_mean_0p3_nyq": float(c_tp[m_mean].mean())}
    for k, v in pts.items():
      if v and np.isfinite(v):
        hv = near(f, h, v)
        H[j][k] = {"mag": float(abs(hv)), "phase_deg": float(np.degrees(np.angle(hv))), "coh": float(near(f, c_tp, v))}
    cg = {}
    for ai, a in enumerate(AXES):
      _, c_ag = signal.coherence(U[:, 2 * nj + i], U[:, 3 * nj + ai], **{k: v for k, v in kw.items() if k != "window"}, window="hann")
      cg[a] = {"mean_0p3_nyq": float(c_ag[m_mean].mean()), **{k: float(near(f, c_ag, v)) for k, v in pts.items() if v and np.isfinite(v)}}
    coh["act_gyro"][j] = cg
  out["coherence"] = coh
  out["freq_response_tgt_to_pos"] = H
  out["freq_response_note"] = "H = Pxy/Pxx, x = tgt, y = pos, both interpolated on their own time stamps. Closed loop (tgt depends on pos through the policy), so H mixes plant and policy feedback; it is not the plant."
  return out, dict(f=f, P=P, sl=sl, names=names)


# ----------------------------------------------------------------------------------------------- per run
def analyze(entry, root):
  S = load_policy(root / entry["policy"])
  L = load_latency(root / entry["latency"]) if entry.get("latency") else None
  t_lo, t_hi = entry.get("trim_s", [0.0, 0.0])
  dflt = default_pose(S)
  res = {"label": entry["label"], "policy_file": entry["policy"], "latency_file": entry.get("latency"),
         "conditions": {k: entry.get(k) for k in ("body_gain", "ankle_gain", "kd_scale", "note", "date")}}
  tm, lat_arr = timing(S, L)
  res["timing"] = tm
  res["consistency"] = consistency(S, dflt)
  types, idx, mir = leg_index(S, dflt)
  rep = repeat_masks(S)
  res["imu_freeze"] = freeze(S, rep)
  hd_out, hd = heading(S, rep)
  res["heading"] = hd_out
  g_out, pa = gait(S, types, idx, mir, tm, t_lo, t_hi)
  res["gait"] = g_out
  sp_out, sp = spectra(S, types, idx, g_out, rep)
  res["jitter"] = sp_out
  res["_arr"] = dict(rep=rep, S=S, dflt=dflt, types=types, idx=idx, mir=mir, hd=hd, pa=pa, sp=sp, lat=lat_arr)
  return res


def flat(res):
  """Key metrics for the cross-run table."""
  f = {}
  tm, h, g, j = res["timing"], res["heading"], res["gait"], res["jitter"]
  f["duration_s"] = tm["duration_s"]
  f["policy_interval_median_ms"] = tm["policy_interval_ms"]["median"]
  f["policy_interval_p95_ms"] = tm["policy_interval_ms"]["p95"]
  f["policy_interval_max_ms"] = tm["policy_interval_ms"]["max"]
  f["frac_interval_gt_30ms"] = tm["policy_interval_frac_gt_30ms"]
  if "latency" in tm:
    la = tm["latency"]
    f["loop_hz_setting"] = la["hz_setting"]
    f["loop_period_median_ms"] = la["loop_period_ms"]["median"]
    f["loop_rate_mean_hz"] = la["loop_rate_hz_mean"]
    f["reply_latency_median_ms_R"] = la["leg"]["right"]["latency_ms"]["median"]
    f["reply_latency_median_ms_L"] = la["leg"]["left"]["latency_ms"]["median"]
    f["reply_latency_p99_ms_R"] = la["leg"]["right"]["latency_ms"]["p99"]
    f["reply_latency_p99_ms_L"] = la["leg"]["left"]["latency_ms"]["p99"]
    f["non_ontime_frac"] = max(la["leg"]["right"]["non_ontime_frac"], la["leg"]["left"]["non_ontime_frac"])
    f["pairing"] = tm["pairing"].get("verdict")
  fz = res["imu_freeze"]
  f["gyro_repeat_frac"], f["gyro_repeat_longest_rows"], f["gyro_repeat_longest_s"] = fz["gyro"]["frac_rows"], fz["gyro"]["longest_rows"], fz["gyro"]["longest_s"]
  f["grav_repeat_frac"], f["q_repeat_frac"] = fz["grav"]["frac_rows"], fz["q"]["frac_rows"]
  hf = h["heading_frozen_handling"]
  f["heading_live_only_deg"] = hf["live_intervals_only_deg (frozen counted as zero)"]
  f["heading_live_rate_x_duration_deg"] = hf["live_rate_times_duration_deg (frozen filled with live mean)"]
  f["frozen_time_frac"] = hf["frozen_time_frac"]
  f["cmd_vx"], f["cmd_wz"] = h["command"]["vx_median"], h["command"]["wz_median"]
  f["heading_change_deg"], f["yaw_rate_mean_deg_s"] = h["heading_change_deg"], h["mean_yaw_rate_deg_s"]
  if h.get("quiet_2s"):
    f["gyro_z_bias_quiet2s_deg_s"] = h["quiet_2s"]["gyro_mean_deg_s"][2]
    f["tilt_fwd_quiet_deg"], f["tilt_left_quiet_deg"] = h["quiet_2s"]["tilt_fwd_deg"], h["quiet_2s"]["tilt_left_deg"]
  f["tilt_fwd_run_deg"], f["tilt_left_run_deg"] = h["tilt_fwd_deg_run"]["mean"], h["tilt_left_deg_run"]["mean"]
  f["gyro_z_rms_deg_s"] = h["gyro_rms_deg_s"]["z"]
  f["f_gait_hz"] = g["f_gait_hz"]
  f["LR_phase_deg"] = g.get("LR_hip_pitch_phase_deg_at_f_gait")
  f["LR_xcorr_coef"] = g.get("LR_hip_pitch_xcorr", {}).get("max_coef")
  for ty, v in g["per_type"].items():
    if "/" in ty or "(" in ty or ty in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"):
      p = v["pos"]
      f[f"asym_rms.{ty}"] = p["asym_rms"]
      f[f"offset_mirror_deg.{ty}"] = p["offset_mirrored_L_minus_R"]
      f[f"rms_deg.{ty}"] = 0.5 * (p["rms_R"] + p["rms_L"])
  f["frac_tgt_clipped_max_joint"] = max(v["frac_clipped"] for v in res["consistency"]["per_joint"].values())
  if "f_dominant_hz" in j:
    f["f_dominant_hz"], f["f_hi_hz"] = j["f_dominant_hz"], j["f_hi_hz"]
    for i, (lo, hi) in enumerate(BANDS):
      f[f"pos_band_frac.{lo:g}-{'Nyq' if hi is None else format(hi, 'g')}"] = j["pos_band_fraction_sum_over_joints"][i]
    f["pos_var_total_deg2"] = float(sum(sum(v) for v in j["band_power"]["pos"].values()))
    f["gyro_live_band_frac.3-6"] = j["gyro_live_interp_band_fraction_sum_over_axes"][2]
    f["gyro_band_frac.3-6"] = j["gyro_band_fraction_sum_over_axes"][2]
    f["track_err_rms_deg_mean"] = float(np.mean([v["rms_tgt_k_minus_pos_k"] for v in j["tracking_error_deg"].values()]))
    f["act_step_rms_mean"] = float(np.mean(list(j["act_step_rms"].values())))
  return f


# ----------------------------------------------------------------------------------------------- figures
def figures(results, prefix, fig_runs):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
  cols = plt.get_cmap("tab20")
  made = []

  # 1 timing histogram
  n = len(results)
  nc = 4
  nr = math.ceil(n / nc)
  fig, ax = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.4 * nr), squeeze=False)
  for a in ax.ravel():
    a.axis("off")
  for k, r in enumerate(results):
    a = ax.ravel()[k]
    a.axis("on")
    dt = np.diff(r["_arr"]["S"]["t"]) * 1e3
    a.hist(dt, bins=np.arange(0, 160, 2), color=cols(k % 20))
    a.axvline(20, color="k", ls="--", lw=0.8)
    a.axvline(30, color="r", ls=":", lw=0.8)
    a.set_yscale("log")
    a.set_title(r["label"], fontsize=7)
    a.text(0.97, 0.92, f"median {np.median(dt):.0f} ms\np95 {np.percentile(dt, 95):.0f} ms\n>30 ms: {100 * (dt > 30).mean():.0f}%", transform=a.transAxes, ha="right", va="top", fontsize=7)
    a.set_xlabel("policy-call interval [ms]")
  fig.suptitle("Policy-call interval per run (dashed: 20 ms training period, dotted: 30 ms)", fontsize=9)
  fig.tight_layout()
  p = f"{prefix}timing_hist.png"
  fig.savefig(p, dpi=130)
  plt.close(fig)
  made.append(p)

  # 2 heading vs time
  fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
  for k, r in enumerate(results):
    t = r["_arr"]["S"]["t"]
    ax[0].plot(t - t[0], r["_arr"]["hd"], color=cols(k % 20), lw=1.2, label=r["label"])
    fzr = r["_arr"]["rep"]["gyro"]
    m2 = fzr.copy()
    m2[:-1] |= fzr[1:]  # an interval k (rows k-1,k) is frozen: mark both end rows
    ax[0].plot(t - t[0], np.where(m2, r["_arr"]["hd"], np.nan), color="k", lw=2.4, alpha=0.45)
  ax[0].set_xlabel("time since first policy call [s]")
  ax[0].set_ylabel("integrated gyro-z heading [deg]  (+ = counter-clockwise)")
  ax[0].set_title("thick dark overlay = intervals where the gyro row repeats the previous row (frozen)", fontsize=8)
  ax[0].legend(fontsize=6, ncol=2)
  lab = [r["label"] for r in results]
  rate = [r["heading"]["mean_yaw_rate_deg_s"] for r in results]
  ax[1].barh(range(n), rate, color=[cols(k % 20) for k in range(n)])
  ax[1].set_yticks(range(n))
  ax[1].set_yticklabels(lab, fontsize=6)
  ax[1].invert_yaxis()
  ax[1].set_xlabel("mean yaw rate over the run [deg/s]")
  fig.tight_layout()
  p = f"{prefix}heading.png"
  fig.savefig(p, dpi=130)
  plt.close(fig)
  made.append(p)

  # per-run figures
  for r in results:
    if r["label"] not in fig_runs:
      continue
    A = r["_arr"]
    S, types, idx, mir = A["S"], A["types"], A["idx"], A["mir"]
    tag = re.sub(r"[^A-Za-z0-9.]+", "_", r["label"]).strip("_")
    # 3 PSD
    if A["sp"]:
      sp = A["sp"]
      fig, ax = plt.subplots(4, len(types), figsize=(2.6 * len(types), 9.2), squeeze=False)
      for c, ty in enumerate(types):
        for rr, grp in enumerate(("pos", "act", "err")):
          for leg, i, ls in (("R", idx[ty][0], "-"), ("L", idx[ty][1], "--")):
            ax[rr, c].semilogy(sp["f"], sp["P"][:, sp["sl"][grp].start + i], ls, label=leg)
          ax[rr, c].set_title(f"{ty} {('pos [deg^2/Hz]', 'act [raw^2/Hz]', 'tgt-pos [deg^2/Hz]')[rr]}", fontsize=7)
      for c in range(len(types)):
        ax[2, c].set_xlabel("frequency [Hz]")
        if c < 3:
          for key, ls in (("gyro", "-"), ("gyro_live", "--")):
            ax[3, c].semilogy(sp["f"], sp["P"][:, sp["sl"][key].start + c], ls, label="raw" if key == "gyro" else "frozen rows interpolated")
          ax[3, c].set_title(f"gyro {AXES[c]} [(rad/s)^2/Hz]", fontsize=7)
          ax[3, c].set_xlabel("frequency [Hz]")
        else:
          ax[3, c].axis("off")
      ax[3, 0].legend(fontsize=6)
      for rr in range(3):
        for c in range(len(types)):
          ax[rr, c].set_xlim(0, sp["f"][-1])
      ax[0, 0].legend()
      fig.suptitle(f"Welch PSD, right (solid) vs left (dashed): {r['label']}  (gyro frozen rows: {100 * A['rep']['gyro'].mean():.1f} %)", fontsize=9)
      fig.tight_layout()
      p = f"{prefix}psd_{tag}.png"
      fig.savefig(p, dpi=130)
      plt.close(fig)
      made.append(p)
    # 4 gait phase
    if A["pa"]:
      pa = A["pa"]
      x = (np.arange(pa["mean"].shape[0]) + 0.5) / pa["mean"].shape[0]
      fig, ax = plt.subplots(1, len(types), figsize=(2.6 * len(types), 3.2), squeeze=False)
      for c, ty in enumerate(types):
        for leg, i, ls, col in (("R", idx[ty][0], "-", "C0"), ("L mirrored", idx[ty][1], "--", "C3")):
          ax[0, c].plot(x, pa["mean"][:, i], ls, color=col, label=leg)
          ax[0, c].fill_between(x, pa["mean"][:, i] - pa["std"][:, i], pa["mean"][:, i] + pa["std"][:, i], color=col, alpha=0.15)
        ax[0, c].set_title(f"{ty} [deg]", fontsize=8)
        ax[0, c].set_xlabel("gait phase (right hip_pitch rising zero)")
      ax[0, 0].legend(fontsize=7)
      fig.suptitle(f"Gait-phase average of measured position, {pa['n_cycles']} cycles, period {pa['period_mean_s']:.2f} s: {r['label']}", fontsize=9)
      fig.tight_layout()
      p = f"{prefix}gait_phase_{tag}.png"
      fig.savefig(p, dpi=130)
      plt.close(fig)
      made.append(p)
    # 5 tgt vs pos slice
    t = S["t"]
    t0 = t[0] + 0.45 * (t[-1] - t[0])
    m = (t >= t0) & (t < t0 + 8)
    fig, ax = plt.subplots(2, len(types), figsize=(2.6 * len(types), 5), squeeze=False, sharex=True)
    for c, ty in enumerate(types):
      for rr, (leg, i) in enumerate((("right", idx[ty][0]), ("left", idx[ty][1]))):
        ax[rr, c].plot(t[m], S["tgt"][m, i], ".-", ms=3, lw=0.8, label="tgt")
        ax[rr, c].plot(t[m], S["pos"][m, i], ".-", ms=3, lw=0.8, label="pos")
        ax[rr, c].set_title(f"{leg} {ty} [deg]", fontsize=8)
      ax[1, c].set_xlabel("time [s]")
    ax[0, 0].legend(fontsize=7)
    for i0, L in runs_of(A["rep"]["gyro"]):
      if L >= 3 and t[i0 + L - 1] > t0 and t[i0 - 1] < t0 + 8:
        for a_ in ax.ravel():
          a_.axvspan(t[i0 - 1], t[i0 + L - 1], color="0.5", alpha=0.25, lw=0)
    fig.suptitle(f"Target vs measured position, 8 s slice: {r['label']} (grey = gyro frozen >= 3 rows)", fontsize=9)
    fig.tight_layout()
    p = f"{prefix}tgt_vs_pos_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)
  return made


# ----------------------------------------------------------------------------------------------- tables / cli
TABS = {
  "timing": ["duration_s", "loop_hz_setting", "policy_interval_median_ms", "policy_interval_p95_ms", "policy_interval_max_ms", "frac_interval_gt_30ms",
             "loop_period_median_ms", "loop_rate_mean_hz", "reply_latency_median_ms_R", "reply_latency_median_ms_L", "reply_latency_p99_ms_R",
             "reply_latency_p99_ms_L", "non_ontime_frac", "pairing"],
  "freeze": ["gyro_repeat_frac", "gyro_repeat_longest_rows", "gyro_repeat_longest_s", "grav_repeat_frac", "q_repeat_frac", "frozen_time_frac",
             "heading_change_deg", "heading_live_only_deg", "heading_live_rate_x_duration_deg"],
  "heading": ["cmd_vx", "cmd_wz", "heading_change_deg", "yaw_rate_mean_deg_s", "gyro_z_bias_quiet2s_deg_s", "tilt_fwd_quiet_deg", "tilt_left_quiet_deg",
              "tilt_fwd_run_deg", "tilt_left_run_deg", "gyro_z_rms_deg_s"],
  "gait": ["f_gait_hz", "LR_phase_deg", "LR_xcorr_coef"] + [f"{k}.{t}" for k in ("rms_deg", "asym_rms", "offset_mirror_deg") for t in ("hip_pitch", "knee")],
  "jitter": ["f_dominant_hz", "f_hi_hz", "pos_var_total_deg2", "pos_band_frac.0-1", "pos_band_frac.1-3", "pos_band_frac.3-6", "pos_band_frac.6-Nyq",
             "track_err_rms_deg_mean", "act_step_rms_mean", "gyro_band_frac.3-6", "gyro_live_band_frac.3-6"],
}


def jdefault(o):
  if isinstance(o, (np.floating, float)):
    return None if not np.isfinite(o) else round(float(o), 6)
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, np.bool_):
    return bool(o)
  raise TypeError(type(o))


def clean(o):
  if isinstance(o, dict):
    return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
  if isinstance(o, (list, tuple)):
    return [clean(v) for v in o]
  if isinstance(o, (float, np.floating)):
    return None if not np.isfinite(o) else round(float(o), 6)
  if isinstance(o, np.integer):
    return int(o)
  if isinstance(o, np.bool_):
    return bool(o)
  if isinstance(o, np.ndarray):
    return clean(o.tolist())
  return o


def md_table(results, keys):
  rows = ["| run | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
  for r in results:
    f = r["flat"]
    cells = []
    for k in keys:
      v = f.get(k)
      cells.append("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)))
    rows.append(f"| {r['label']} | " + " | ".join(cells) + " |")
  return "\n".join(rows)


def selftest():
  """Synthetic 40 s log at 50 Hz: right hip_pitch 1.5 Hz sine, left = mirrored half-cycle later, gyro z 10 deg/s. Asserts the headline numbers."""
  import tempfile
  n, dt = 2000, 0.02
  t = np.arange(n) * dt
  J = [f"{l}_leg/{j}" for l in ("right", "left") for j in JT]
  dflt = dict(zip(J, [-10.03, 0, 0, 20.05, 8.25, -8.43, 10.03, 0, 0, -20.05, -8.24, 8.42]))
  pos = np.tile([dflt[j] for j in J], (n, 1)).astype(float)
  pos[:, 0] += 8 * np.sin(2 * np.pi * 1.5 * t)
  pos[:, 6] -= 8 * np.sin(2 * np.pi * 1.5 * t + np.pi)  # mirror of right shifted half a cycle: L = -R(t+T/2) -> mirrored-left = R shifted
  q = np.radians(pos - np.array([dflt[j] for j in J]))
  act = np.vstack([np.zeros(12), np.random.default_rng(0).normal(0, 0.3, (n - 1, 12))])
  tgt = np.array([dflt[j] for j in J]) + ACTION_SCALE_DEG * act
  cols = {"t": t}
  for i, a in enumerate(AXES):
    cols[f"obs/gyro/{a}"] = np.radians(10.0) * np.ones(n) if a == "z" else np.zeros(n)
    cols[f"obs/grav/{a}"] = -np.ones(n) if a == "z" else np.zeros(n)
  for nm, A in (("obs/q_prev", np.vstack([q[:1], q[:-1]])), ("obs/q", q), ("obs/last_act", np.vstack([np.zeros(12), act[:-1]]))):
    for i, j in enumerate(J):
      cols[f"{nm}/{j}"] = A[:, i]
  cols["obs/cmd/vx"], cols["obs/cmd/wz"] = np.zeros(n), np.zeros(n)
  for nm, A in (("act", act), ("tgt", tgt), ("pos", pos)):
    for i, j in enumerate(J):
      cols[f"{nm}/{j}"] = A[:, i]
  d = Path(tempfile.mkdtemp())
  pd.DataFrame(cols).to_csv(d / "syn.csv", index=False)
  r = analyze({"label": "syn", "policy": "syn.csv"}, d)
  assert abs(r["gait"]["f_gait_hz"] - 1.5) < 0.05, r["gait"]["f_gait_hz"]
  assert abs(r["heading"]["mean_yaw_rate_deg_s"] - 10) < 1e-6
  assert r["consistency"]["q_prev_k_vs_q_k-1_max_abs_rad"] == 0 and r["consistency"]["tgt_vs_clip(default+0.25*act)_max_abs_deg"] < 1e-9
  assert r["consistency"]["newer_reading"].startswith("pos(k) == obs/q(k)")
  assert abs(r["gait"]["per_type"]["hip_pitch"]["pos"]["asym_rms"]) < 0.05
  assert r["gait"]["phase_average"]["n_cycles"] > 50
  assert r["imu_freeze"]["gyro"]["frac_rows"] > 0.99 and r["imu_freeze"]["q"]["frac_rows"] == 0  # constant gyro = all repeats; moving joints = none
  print("selftest ok")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--manifest")
  ap.add_argument("--run", action="append", default=[], help="LABEL=POLICY[,LATENCY]")
  ap.add_argument("--root", default=None, help="base dir for relative manifest paths (default: manifest 'root' or cwd)")
  ap.add_argument("--selftest", action="store_true", help="run the synthetic-log check and exit")
  ap.add_argument("--out", required=False, help="metrics JSON path; summary tables go next to it as .md")
  ap.add_argument("--fig-prefix", default="")
  ap.add_argument("--fig-runs", default="", help="comma list of labels for PSD/gait/slice figures (default: manifest fig_runs or longest run)")
  args = ap.parse_args()
  if args.selftest:
    return selftest()
  assert args.out, "--out is required"
  entries, contrasts, root, fig_runs = [], [], Path(args.root or ".").resolve(), []
  if args.manifest:
    man = json.load(open(args.manifest, encoding="utf-8"))
    root = Path(args.root or man.get("root", ".")).resolve()
    entries += man["runs"]
    contrasts = man.get("contrasts", [])
    fig_runs = man.get("fig_runs", [])
  for s in args.run:
    lab, rest = s.split("=", 1)
    pol, _, lat = rest.partition(",")
    entries.append({"label": lab, "policy": pol, "latency": lat or None})
  results = []
  for e in entries:
    print("analysing", e["label"], file=sys.stderr)
    r = analyze(e, root)
    r["flat"] = flat(r)
    results.append(r)
  if args.fig_runs:
    fig_runs = args.fig_runs.split(",")
  if not fig_runs:
    fig_runs = [max(results, key=lambda r: r["timing"]["duration_s"])["label"]]
  # contrasts
  by = {r["label"]: r["flat"] for r in results}
  con = []
  for c in contrasts:
    a, b = by[c["a"]], by[c["b"]]
    d = {k: {"a": a[k], "b": b[k], "b_minus_a": b[k] - a[k], "ratio": (b[k] / a[k] if a[k] not in (0, None) else None)}
         for k in (c.get("keys") or [k for ks in TABS.values() for k in ks]) if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) and k in a and k in b}
    con.append({"name": c["name"], "a": c["a"], "b": c["b"], "metrics": d})
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  meta = {"method": __doc__.split("Method notes", 1)[1].strip() if "Method notes" in __doc__ else "", "tool": "tools/sim2real/walk/real_walk_analyze.py"}
  json.dump(clean({"meta": meta, "runs": [{k: v for k, v in r.items()} for r in results], "contrasts": con}), open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
  tabs = TABS
  with open(out.with_suffix(".md"), "w", encoding="utf-8") as fh:
    for k, keys in tabs.items():
      fh.write(f"## {k}\n\n{md_table(results, keys)}\n\n")
    for c in con:
      fh.write(f"## contrast {c['name']}: {c['a']} -> {c['b']}\n\n| metric | a | b | b-a | ratio |\n|---|---|---|---|---|\n")
      for k, v in c["metrics"].items():
        fh.write(f"| {k} | {v['a']:.4g} | {v['b']:.4g} | {v['b_minus_a']:.4g} | {'' if v['ratio'] is None else format(v['ratio'], '.3g')} |\n")
      fh.write("\n")
  if args.fig_prefix:
    Path(args.fig_prefix).parent.mkdir(parents=True, exist_ok=True)
    for p in figures(results, args.fig_prefix, fig_runs):
      print("figure", p, file=sys.stderr)
  print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
  main()
