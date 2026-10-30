#!/usr/bin/env python3
"""Audit of the Union-DR checkpoint 49999 against its first real-robot logs (HUPHY `main` 90bbd1f, 2026-10-28).

One script, five blocks. Every number in docs/sim2real/2026-10-30_union49999_dr_coverage_and_risks.md comes from here.

  B  contract : policy.onnx and model_49999.pt (actor + embedded normalizer) re-run on every logged 44-input row and compared
                with the logged `act/`; log self-consistency (last_act, q_prev, tgt = clip(default + 0.25 act));
                deploy.yaml vs HUPHY deploy_contract.json vs robot.yaml gains vs calibration limits.
  C  health   : policy interval, loop period, reply latency per joint, IMU repeat / exact-zero, 3-8 Hz and 5 Hz band, heading drift
                and wz tracking, command statistics (ramp, standing fraction).
  F  clip     : how often the target is cut by the training clip and by the HUPHY guard window (clip narrowed by 3 deg), per joint,
                with hip_roll inner side separated.
  G  stance   : lateral distance of the two ankle-joint origins from the logged hip/knee angles (MuJoCo forward kinematics of the
                training model), standing vs walking.

Run (CPU only):
  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/union49999_audit.py \
      --pol <dir with deploy.yaml, policy.onnx, model_49999.pt> --huphy <dir with config/> --xml <pygmalion_legonly_ab.xml> \
      --logs Expr_Data/huphy_main_90bbd1f --old Expr_Data --out docs/sim2real/data/2026-10-30_union49999_audit.json

The directories are extracted with `git archive` (policy branch origin/generalized-union-dr, HUPHY origin/main); nothing is checked out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_walk_analyze as rwa  # noqa: E402

DEG = 180.0 / math.pi
JOINTS = [f"{s}_leg/{j}" for s in ("right", "left") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")]
CONTRACT_JOINTS = [f"{s}_{j}_joint" for s in ("R", "L") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
OBS_NAMES = (["obs/gyro/x", "obs/gyro/y", "obs/gyro/z", "obs/grav/x", "obs/grav/y", "obs/grav/z"]
             + [f"obs/q_prev/{j}" for j in JOINTS] + [f"obs/q/{j}" for j in JOINTS]
             + [f"obs/last_act/{j}" for j in JOINTS] + ["obs/cmd/vx", "obs/cmd/wz"])


def r(x, n=4):
  return None if x is None else (round(float(x), n) if np.isfinite(x) else None)


# ------------------------------------------------------------------------------------------------ loading
def load_yaml_min(path):
  import yaml
  return yaml.safe_load(open(path, encoding="utf-8"))


def load_log(path):
  d = pd.read_csv(path, comment="#")
  return d


def runs_where(mask):
  """[(start, stop)] index pairs of True runs (stop exclusive)."""
  m = np.asarray(mask, bool)
  if not m.any():
    return []
  edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
  return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


# ------------------------------------------------------------------------------------------------ B contract
def block_contract(a, logs):
  out = {}
  pol = Path(a.pol)
  dep = load_yaml_min(pol / "deploy.yaml")
  hu = Path(a.huphy)
  con = json.load(open(hu / "config/policies/generalized_walk/deploy_contract.json", encoding="utf-8"))
  # --- hashes
  sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
  out["sha256"] = {"model_49999.pt": sha(pol / "model_49999.pt"), "policy.onnx": sha(pol / "policy.onnx"),
                   "deploy.yaml_says_pt": dep["artifacts"]["model_49999.pt"]["sha256"], "deploy.yaml_says_onnx": dep["artifacts"]["policy.onnx"]["sha256"],
                   "HUPHY_contract_says_pt": con["checkpoints"]["49999"]["pt_sha256"], "HUPHY_contract_says_onnx": con["checkpoints"]["49999"].get("onnx_sha256")}
  hp = hu / "config/policies/generalized_walk/generalized_walk_49999.pt"
  if hp.exists():
    out["sha256"]["HUPHY_file_generalized_walk_49999.pt"] = sha(hp)
  # --- deploy.yaml vs HUPHY contract
  d_def = np.array(dep["actions"]["default_position_rad"])
  c_def = np.array([con["default_pose_rad"][j] for j in CONTRACT_JOINTS])
  d_clip = np.array(dep["actions"]["clip_rad"])
  acts = {x["joint"]: x for x in con["actions"]}
  c_clip = np.array([acts[j]["clip_rad"] for j in CONTRACT_JOINTS])
  c_scale = np.array([acts[j]["scale_rad_per_unit"] for j in CONTRACT_JOINTS])
  out["deploy_vs_huphy_contract"] = {
      "joint_order_equal": dep["actions"]["joint_order"] == CONTRACT_JOINTS and [x["joint"] for x in sorted(con["actions"], key=lambda z: z["index"])] == CONTRACT_JOINTS,
      "default_pose_max_abs_diff_deg": r(np.abs(d_def - c_def).max() * DEG, 5),
      "clip_max_abs_diff_deg": r(np.abs(d_clip - c_clip).max() * DEG, 5),
      "scale_deploy": dep["actions"]["scale_rad_per_unit"], "scale_huphy_all_equal": bool(np.allclose(c_scale, dep["actions"]["scale_rad_per_unit"])),
      "obs_dim": [dep["policy"]["actor_input_dim"], con["obs_dim"]], "action_dim": [dep["policy"]["actor_output_dim"], con["action_dim"]],
      "control_hz": [dep["policy"]["control_rate_hz"], con["control_rate_hz"]],
      "huphy_velocity_command_ranges_key": [t.get("ranges") for t in con["observations"] if t["func"] == "velocity_command"],
  }
  # --- gains: robot.yaml vs deploy.yaml
  ry = load_yaml_min(hu / "config/robot.yaml")
  gains = {}
  names = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]
  kp_d = dep["pd_gains"]["stiffness_nm_per_rad"]
  kd_d = dep["pd_gains"]["damping_nm_s_per_rad"]
  mism = []
  for si, leg in enumerate(("right_leg", "left_leg")):
    for ji, jn in enumerate(names):
      g = ry["limbs"][leg]["motors"][jn]
      i = si * 6 + ji
      gains[f"{leg}/{jn}"] = {"robot_kp": g["kp"], "robot_kd": g["kd"], "deploy_kp": kp_d[i], "deploy_kd": kd_d[i]}
      if g["kp"] != kp_d[i] or g["kd"] != kd_d[i]:
        mism.append(f"{leg}/{jn}")
  out["gains"] = {"mismatch": mism, "table": gains, "run_gain_scale": logs["run_info"].get("gain"), "command_margin_deg": ry["safety"]["command_margin_deg"],
                  "max_delta_deg": ry["safety"]["max_delta_deg"], "link_loss_cycles": ry["safety"]["link_loss_cycles"]}
  # --- calibration placeholders and limits vs clip
  cal = {}
  for leg in ("right", "left"):
    c = json.load(open(hu / f"config/calibration/{leg}_leg_v1.0.json", encoding="utf-8"))
    cal[leg] = {k: {"sign": v["sign"], "offset_deg": v["offset_deg"], "zero_reference": v["zero_reference"], "limits_deg": v["limits_deg"]} for k, v in c["motors"].items()}
  lim_vs_clip = {}
  for i, j in enumerate(JOINTS):
    leg, jn = j.split("/")
    lim = cal[leg.split("_")[0]][jn]["limits_deg"]
    clip = (np.array(dep["actions"]["clip_rad"][i]) * DEG)
    lim_vs_clip[j] = {"calibration_limits_deg": lim, "training_clip_deg": [r(clip[0], 3), r(clip[1], 3)],
                      "guard_window_deg": [r(lim[0] + 3.0, 3), r(lim[1] - 3.0, 3)]}
  out["calibration"] = {"all_offsets_zero": all(v["offset_deg"] == 0.0 for l in cal.values() for v in l.values()),
                        "all_zero_reference_is_string_0": all(v["zero_reference"] == "0" for l in cal.values() for v in l.values()),
                        "all_sign_one": all(v["sign"] == 1.0 for l in cal.values() for v in l.values()), "limits_vs_clip": lim_vs_clip}
  # --- replay
  import onnxruntime as ort
  import torch
  sys.path.insert(0, str(HERE))
  import policy_replay_check as prc
  actor, ck = prc.load_actor(str(pol / "model_49999.pt"))
  sess = ort.InferenceSession(str(pol / "policy.onnx"), providers=["CPUExecutionProvider"])
  in_name = sess.get_inputs()[0].name
  out["onnx_io"] = {"input": [in_name, sess.get_inputs()[0].shape], "output": [sess.get_outputs()[0].name, sess.get_outputs()[0].shape]}
  rep = {}
  for lab, d in logs["frames"].items():
    X = d[OBS_NAMES].to_numpy(np.float32)
    A = d[[f"act/{j}" for j in JOINTS]].to_numpy(np.float64)
    ao = np.concatenate([sess.run(None, {in_name: X[i:i + 1]})[0] for i in range(len(X))], 0).astype(np.float64)
    with torch.no_grad():
      at = actor(torch.from_numpy(X)).numpy().astype(np.float64)
    # log self-consistency
    la = d[[f"obs/last_act/{j}" for j in JOINTS]].to_numpy()
    qp = d[[f"obs/q_prev/{j}" for j in JOINTS]].to_numpy()
    q = d[[f"obs/q/{j}" for j in JOINTS]].to_numpy()
    tg = d[[f"tgt/{j}" for j in JOINTS]].to_numpy()
    ps = d[[f"pos/{j}" for j in JOINTS]].to_numpy()
    dflt = np.array(dep["actions"]["default_position_rad"]) * DEG
    lo = np.array(dep["actions"]["clip_rad"])[:, 0] * DEG
    hi = np.array(dep["actions"]["clip_rad"])[:, 1] * DEG
    raw = dflt + 0.25 * DEG * A
    rep[lab] = {"n_rows": int(len(X)),
                "onnx_vs_log_act_max": r(np.abs(ao - A).max(), 6), "onnx_vs_log_act_mean": r(np.abs(ao - A).mean(), 7),
                "torch_vs_log_act_max": r(np.abs(at - A).max(), 6), "torch_vs_log_act_mean": r(np.abs(at - A).mean(), 7),
                "onnx_vs_torch_max": r(np.abs(ao - at).max(), 6),
                "last_act_next_eq_act_max": r(np.abs(la[1:] - A[:-1]).max(), 6),
                "q_prev_next_eq_q_max_rad": r(np.abs(qp[1:] - q[:-1]).max(), 6),
                "tgt_eq_clip_default_plus_act_max_deg": r(np.abs(tg - np.clip(raw, lo, hi)).max(), 4),
                "obs_q_eq_pos_minus_default_max_deg": r(np.abs(q * DEG - (ps - dflt)).max(), 4),
                "first_row_q_deg_from_default_abs_max": r(np.abs(q[0] * DEG).max(), 3),
                "first_row_pos_deg": [r(v, 2) for v in ps[0]], "default_deg": [r(v, 2) for v in dflt]}
  out["replay"] = rep
  return out, dict(dflt=dflt, lo=lo, hi=hi)


# ------------------------------------------------------------------------------------------------ C health
def block_health(lab, d, lat, dep_info):
  t = d["t"].to_numpy()
  dt = np.diff(t) * 1e3
  vx = d["obs/cmd/vx"].to_numpy()
  wz = d["obs/cmd/wz"].to_numpy()
  g = d[[f"obs/gyro/{a}" for a in "xyz"]].to_numpy()
  gr = d[[f"obs/grav/{a}" for a in "xyz"]].to_numpy()
  q = d[[f"obs/q/{j}" for j in JOINTS]].to_numpy()
  out = {"duration_s": r(t[-1] - t[0], 2), "n_rows": int(len(t))}
  out["policy_interval_ms"] = {k: r(v, 3) for k, v in rwa.stats(dt).items() if k in ("median", "mean", "std", "p5", "p95", "p99", "max", "min")}
  out["policy_interval_frac_outside_19_21ms"] = r(((dt < 19) | (dt > 21)).mean(), 4)
  out["policy_interval_frac_gt_25ms"] = r((dt > 25).mean(), 5)
  # latency
  if lat is not None:
    L = lat
    per = {}
    for j in L["joints"]:
      tt, ll, ss = L["per"][j]
      per[j] = {"median_ms": r(np.median(ll), 3), "p95_ms": r(np.percentile(ll, 95), 3), "p99_ms": r(np.percentile(ll, 99), 3), "max_ms": r(ll.max(), 3),
                "n": int(len(ll)), "late": int((ss == "late").sum()), "lost": int((ss == "lost").sum())}
    n = min(len(v[0]) for v in L["per"].values())
    T = np.stack([L["per"][j][0][:n] for j in L["joints"]])
    c0 = np.nanmin(T, 0)
    per_ms = np.diff(c0) * 1e3
    out["latency"] = {"loop_period_ms": {k: r(v, 3) for k, v in rwa.stats(per_ms).items() if k in ("median", "mean", "std", "p95", "p99", "max")},
                      "n_cycles": int(n), "duration_s": r(c0[-1] - c0[0], 2), "per_joint": per,
                      "reply_spread_ms": {k: r(v, 3) for k, v in rwa.stats((np.nanmax(T, 0) - np.nanmin(T, 0)) * 1e3).items() if k in ("median", "p95", "max")},
                      "worst_joint_by_p99": max(per, key=lambda k: per[k]["p99_ms"]), "worst_p99_ms": max(v["p99_ms"] for v in per.values()),
                      "all_ontime": all(v["late"] == 0 and v["lost"] == 0 for v in per.values())}
  # IMU
  zero = (g == 0).all(1)
  rep_g = np.zeros(len(t), bool)
  rep_g[1:] = (g[1:] == g[:-1]).all(1)
  rep_gr = np.zeros(len(t), bool)
  rep_gr[1:] = (gr[1:] == gr[:-1]).all(1)
  moving = (np.abs(vx) + np.abs(wz)) >= 0.1
  rep_runs = [(b - a) for a, b in runs_where(rep_g)]
  out["imu"] = {"gyro_all_three_exactly_zero_frac": r(zero.mean(), 5), "gyro_repeat_prev_frac": r(rep_g.mean(), 4), "grav_repeat_prev_frac": r(rep_gr.mean(), 4),
                "gyro_repeat_frac_when_cmd_moving": r(rep_g[moving].mean(), 4) if moving.any() else None,
                "gyro_repeat_frac_when_cmd_zero": r(rep_g[~moving].mean(), 4) if (~moving).any() else None,
                "longest_gyro_repeat_rows": int(max(rep_runs) if rep_runs else 0), "longest_gyro_repeat_s": r((max(rep_runs) if rep_runs else 0) * np.median(dt) / 1e3, 3),
                "n_repeat_runs_ge_5_rows": int(sum(1 for x in rep_runs if x >= 5)),
                "grav_norm_mean": r(np.linalg.norm(gr, axis=1).mean(), 5)}
  # quiet window: first stand segment (before the command starts moving)
  stand = ~moving
  sr = [x for x in runs_where(stand) if (x[1] - x[0]) * np.median(dt) / 1e3 >= 3.0]
  if sr:
    a, b = sr[0]
    live = ~rep_g[a:b]
    gg = g[a:b][live]
    grq = gr[a:b]
    out["imu"]["quiet_stand"] = {"t0": r(t[a], 2), "t1": r(t[b - 1], 2), "gyro_mean_deg_s": [r(v * DEG, 4) for v in gg.mean(0)], "gyro_std_deg_s": [r(v * DEG, 3) for v in gg.std(0)],
                                  "gyro_std_rad_s_max_axis": r(gg.std(0).max(), 4), "train_gyro_noise_std_rad_s": r(0.2 / math.sqrt(3), 4),
                                  "tilt_fwd_deg": r(math.degrees(math.atan2(grq[:, 0].mean(), -grq[:, 2].mean())), 2), "tilt_left_deg": r(math.degrees(math.atan2(grq[:, 1].mean(), -grq[:, 2].mean())), 2),
                                  "grav_std": [r(v, 5) for v in grq.std(0)]}
  # standing / command statistics
  tot = np.abs(vx) + np.abs(wz)
  out["command"] = {"vx_min": r(vx.min(), 3), "vx_max": r(vx.max(), 3), "wz_min": r(wz.min(), 3), "wz_max": r(wz.max(), 3),
                    "frac_time_standing_total_lt_0p1": r((tot < 0.1).mean(), 4), "frac_time_vx_gt_0p1": r((vx > 0.1).mean(), 4), "frac_time_vx_lt_minus0p1": r((vx < -0.1).mean(), 4),
                    "frac_time_abs_wz_gt_0p1": r((np.abs(wz) > 0.1).mean(), 4), "frac_time_vx_gt_0p1_and_abs_wz_lt_0p05": r(((vx > 0.1) & (np.abs(wz) < 0.05)).mean(), 4)}
  dv = np.diff(vx) / (dt / 1e3)
  dw = np.diff(wz) / (dt / 1e3)
  out["command"].update({"max_abs_dvx_dt": r(np.abs(dv).max(), 3), "p99_abs_dvx_dt": r(np.percentile(np.abs(dv), 99), 3), "max_abs_dwz_dt": r(np.abs(dw).max(), 3),
                         "p99_abs_dwz_dt": r(np.percentile(np.abs(dw), 99), 3), "max_single_step_dvx": r(np.abs(np.diff(vx)).max(), 3), "max_single_step_dwz": r(np.abs(np.diff(wz)).max(), 3)})
  # hold lengths: time between command changes larger than 0.02
  moving_cmd = np.concatenate([[False], (np.abs(np.diff(vx)) > 1e-4) | (np.abs(np.diff(wz)) > 1e-4)])
  held_runs = [(b - a) * np.median(dt) / 1e3 for a, b in runs_where(~moving_cmd)]
  out["command"]["frac_time_command_changing"] = r(moving_cmd.mean(), 4)
  out["command"]["longest_constant_command_s"] = r(max(held_runs), 2) if held_runs else None
  out["command"]["median_constant_command_hold_s"] = r(float(np.median(held_runs)), 3) if held_runs else None
  return out


def band_metrics(d, sel, lab):
  """Band power of the 12 logged joint angles (deg^2, summed over joints) in the policy-row time base (uniform 50 Hz after interpolation)."""
  t = d["t"].to_numpy()[sel]
  P = d[[f"pos/{j}" for j in JOINTS]].to_numpy()[sel]
  if len(t) < 100:
    return None
  dtm = float(np.median(np.diff(t)))
  tu, Pu = rwa.uniform(t, P, dtm)
  nper = int(min(len(tu), round(4.0 / dtm)))
  f, S = signal.welch(Pu, fs=1 / dtm, nperseg=nper, noverlap=nper // 2, window="hann", detrend="linear", axis=0)
  df = f[1] - f[0]
  bp = lambda lo, hi: float(S[(f >= lo) & (f < hi)].sum() * df)
  tot = float(S[f >= 0.3].sum() * df)
  res = {"n_s": r(t[-1] - t[0], 1), "df_hz": r(df, 3), "total_ge_0p3Hz_deg2": r(tot, 4), "band_3_8_deg2": r(bp(3, 8), 4), "band_4p5_5p7_deg2": r(bp(4.5, 5.7), 4),
         "band_3_8_frac_of_total": r(bp(3, 8) / tot, 4), "band_8_25_deg2": r(bp(8, 25), 5)}
  tot_psd = S.sum(1)
  m = (f >= 3.0) & (f <= 8.0)
  k = np.argmax(tot_psd[m])
  res["peak_3_8_hz"] = r(f[m][k], 2)
  nb = ((f >= 3.5) & (f < 4.5)) | ((f >= 5.7) & (f < 7.0))
  res["peak_4p5_5p7_over_neighbour_mean"] = r(tot_psd[(f >= 4.5) & (f < 5.7)].max() / tot_psd[nb].mean(), 2)
  # dominant gait frequency (0.8-3 Hz) and 3x
  mg = (f >= 0.8) & (f <= 3.0)
  res["gait_peak_hz"] = r(f[mg][np.argmax(tot_psd[mg])], 2)
  res["three_times_gait_hz"] = r(3 * res["gait_peak_hz"], 2)
  # action-change rms between policy calls
  A = d[[f"act/{j}" for j in JOINTS]].to_numpy()[sel]
  res["act_step_rms"] = r(np.sqrt(((np.diff(A, axis=0)) ** 2).mean()), 4)
  return res


def block_spectral(lab, d, old=False):
  t = d["t"].to_numpy()
  vx = d["obs/cmd/vx"].to_numpy() if "obs/cmd/vx" in d else np.zeros(len(t))
  wz = d["obs/cmd/wz"].to_numpy() if "obs/cmd/wz" in d else np.zeros(len(t))
  tot = np.abs(vx) + np.abs(wz)
  out = {"all": band_metrics(d, np.ones(len(t), bool), lab)}
  out["standing_cmd_lt_0p1"] = band_metrics(d, tot < 0.1, lab)
  out["walking_vx_gt_0p3"] = band_metrics(d, vx > 0.3, lab)
  # 10 s windows of the 3-8 Hz band and the neighbour ratio, to see whether any window stands out
  wins = []
  for t0 in np.arange(t[0], t[-1] - 10, 10.0):
    sel = (t >= t0) & (t < t0 + 10.0)
    m = band_metrics(d, sel, lab)
    if m:
      wins.append((r(t0, 0), m["band_3_8_deg2"], m["peak_4p5_5p7_over_neighbour_mean"], m["peak_3_8_hz"], r(float(np.mean(vx[sel])), 2)))
  out["windows_10s_t0_band38_ratio_peakhz_meanvx"] = wins
  return out


def block_heading(lab, d, dt_med):
  t = d["t"].to_numpy()
  vx = d["obs/cmd/vx"].to_numpy()
  wz = d["obs/cmd/wz"].to_numpy()
  g = d[[f"obs/gyro/{a}" for a in "xyz"]].to_numpy()
  gz = g[:, 2] * DEG
  rep = np.zeros(len(t), bool)
  rep[1:] = (g[1:] == g[:-1]).all(1)
  tot = np.abs(vx) + np.abs(wz)
  stand = tot < 0.1
  bias = float(gz[stand & ~rep].mean()) if (stand & ~rep).sum() > 50 else 0.0
  gzc = gz - bias
  # smooth command by one second to avoid ramp edges
  k = int(round(1.0 / dt_med))
  ker = np.ones(k) / k
  vxs = np.convolve(vx, ker, mode="same")
  wzs = np.convolve(wz, ker, mode="same")
  out = {"gyro_z_bias_from_standing_deg_s": r(bias, 4)}
  sel = (vx > 0.1) & (np.abs(wz) < 0.05) & (vxs > 0.1) & (np.abs(wzs) < 0.05)
  # contiguous segments of at least 2 s
  segs = []
  for a, b in runs_where(sel):
    if (t[b - 1] - t[a]) >= 2.0:
      live = ~rep[a:b]
      segs.append({"t0": r(t[a], 1), "t1": r(t[b - 1], 1), "dur_s": r(t[b - 1] - t[a], 1), "mean_vx": r(vx[a:b].mean(), 3),
                   "yaw_rate_deg_s": r(gzc[a:b][live].mean(), 3), "heading_change_deg": r(np.trapezoid(gzc[a:b], t[a:b]), 2)})
  out["straight_segments_ge_2s"] = segs
  if segs:
    w = np.array([s["dur_s"] for s in segs])
    y = np.array([s["yaw_rate_deg_s"] for s in segs])
    out["straight_time_s"] = r(w.sum(), 1)
    out["straight_yaw_rate_time_weighted_deg_s"] = r((w * y).sum() / w.sum(), 3)
    out["straight_n_segments"] = len(segs)
    out["straight_n_negative_right"] = int((y < 0).sum())
    out["straight_total_heading_change_deg"] = r(sum(s["heading_change_deg"] for s in segs), 1)
  # by vx bin, all rows with |wz|<0.05
  bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
  out["by_vx_bin"] = {}
  for lo, hi in bins:
    s2 = (vxs >= lo) & (vxs < hi) & (np.abs(wzs) < 0.05) & ~rep
    out["by_vx_bin"][f"{lo}-{hi}"] = {"n_rows": int(s2.sum()), "yaw_rate_deg_s": r(gzc[s2].mean(), 3) if s2.sum() > 25 else None}
  # wz tracking: regression of 1 s-smoothed gz on smoothed wz for |wz|>0.1 (and sign split)
  gzs = np.convolve(gzc, ker, mode="same")
  tr = {}
  for name, m in (("all", np.abs(wzs) > 0.1), ("wz_positive_left", wzs > 0.1), ("wz_negative_right", wzs < -0.1)):
    m = m & (np.abs(vxs) < 1e9)
    if m.sum() > 50:
      gain = float((gzs[m] * wzs[m]).sum() / (wzs[m] ** 2).sum()) / DEG
      tr[name] = {"n_rows": int(m.sum()), "gain_yawrate_over_wz": r(gain, 3), "mean_wz_cmd_deg_s": r(wzs[m].mean() * DEG, 2), "mean_gz_deg_s": r(gzs[m].mean(), 2)}
  # plateau only: smoothed |wz| >= 0.4 and the command did not change by more than 0.02 over +-1 s
  hold = np.array([np.ptp(wz[max(0, i - k):i + k + 1]) < 0.02 for i in range(len(wz))])
  for name, m in (("plateau_all", (np.abs(wzs) >= 0.4) & hold), ("plateau_left", (wzs >= 0.4) & hold), ("plateau_right", (wzs <= -0.4) & hold)):
    if m.sum() > 25:
      gain = float((gzs[m] * wzs[m]).sum() / (wzs[m] ** 2).sum()) / DEG
      tr[name] = {"n_rows": int(m.sum()), "gain_yawrate_over_wz": r(gain, 3), "mean_wz_cmd_deg_s": r(wzs[m].mean() * DEG, 2), "mean_gz_deg_s": r(gzs[m].mean(), 2)}
  # same but only with vx<0.1 (turn in place) and vx>0.1 (turn while walking)
  for name, m in (("turn_in_place", (np.abs(wzs) > 0.1) & (np.abs(vxs) < 0.1)), ("turn_walking", (np.abs(wzs) > 0.1) & (vxs > 0.1))):
    if m.sum() > 50:
      gain = float((gzs[m] * wzs[m]).sum() / (wzs[m] ** 2).sum()) / DEG
      tr[name] = {"n_rows": int(m.sum()), "gain_yawrate_over_wz": r(gain, 3)}
  out["wz_tracking"] = tr
  # lagged correlation lag of gz vs wz
  x = wz - wz.mean()
  y = gzc - gzc.mean()
  best = None
  for lag in range(0, int(1.5 / dt_med)):
    c = np.corrcoef(x[: len(x) - lag], y[lag:])[0, 1]
    if best is None or c > best[1]:
      best = (lag, c)
  out["gz_vs_wz_best_lag_s"] = r(best[0] * dt_med, 2)
  out["gz_vs_wz_corr_at_best_lag"] = r(best[1], 3)
  return out


# ------------------------------------------------------------------------------------------------ F clip
def block_clip(lab, d, info, guard_margin=3.0):
  dflt, lo, hi = info["dflt"], info["lo"], info["hi"]
  A = d[[f"act/{j}" for j in JOINTS]].to_numpy()
  ps = d[[f"pos/{j}" for j in JOINTS]].to_numpy()
  tg = d[[f"tgt/{j}" for j in JOINTS]].to_numpy()
  raw = dflt + 0.25 * DEG * A
  glo, ghi = lo + guard_margin, hi - guard_margin
  out = {}
  for i, j in enumerate(JOINTS):
    sent = np.clip(tg[:, i], glo[i], ghi[i])
    cut_train = (raw[:, i] < lo[i]) | (raw[:, i] > hi[i])
    cut_guard = np.abs(sent - tg[:, i]) > 1e-9
    e = {"train_clip_cut_frac": r(cut_train.mean(), 5), "guard_cut_frac": r(cut_guard.mean(), 5),
         "guard_cut_mean_excess_deg": r(np.abs(sent - tg[cut_guard, i]).mean(), 3) if cut_guard.any() else 0.0,
         "guard_cut_max_excess_deg": r(np.abs(sent - tg[:, i]).max(), 3), "raw_target_min_deg": r(raw[:, i].min(), 2), "raw_target_max_deg": r(raw[:, i].max(), 2),
         "pos_min_deg": r(ps[:, i].min(), 2), "pos_max_deg": r(ps[:, i].max(), 2), "pos_mean_deg": r(ps[:, i].mean(), 2)}
    out[j] = e
  # hip roll inner side
  ri, li = JOINTS.index("right_leg/hip_roll"), JOINTS.index("left_leg/hip_roll")
  inner = {"right_inner_is": "negative", "left_inner_is": "positive"}
  rr, ll = raw[:, ri], raw[:, li]
  inner["R_target_below_guard_inner_limit_frac"] = r((rr < glo[ri]).mean(), 5)
  inner["L_target_above_guard_inner_limit_frac"] = r((ll > ghi[li]).mean(), 5)
  inner["R_guard_inner_limit_deg"] = r(glo[ri], 2)
  inner["L_guard_inner_limit_deg"] = r(ghi[li], 2)
  inner["R_target_min_deg"] = r(rr.min(), 2)
  inner["L_target_max_deg"] = r(ll.max(), 2)
  inner["R_pos_min_deg"] = r(ps[:, ri].min(), 2)
  inner["L_pos_max_deg"] = r(ps[:, li].max(), 2)
  # inward excursion distribution (mirror: inward = -R, +L)
  inw = np.stack([-ps[:, ri], ps[:, li]], 1)
  inner["inward_pos_p50_p95_p99_max_deg_R"] = [r(np.percentile(inw[:, 0], q), 2) for q in (50, 95, 99, 100)]
  inner["inward_pos_p50_p95_p99_max_deg_L"] = [r(np.percentile(inw[:, 1], q), 2) for q in (50, 95, 99, 100)]
  inwt = np.stack([-raw[:, ri], raw[:, li]], 1)
  inner["inward_target_p50_p95_p99_max_deg_R"] = [r(np.percentile(inwt[:, 0], q), 2) for q in (50, 95, 99, 100)]
  inner["inward_target_p50_p95_p99_max_deg_L"] = [r(np.percentile(inwt[:, 1], q), 2) for q in (50, 95, 99, 100)]
  out["hip_roll_inner"] = inner
  return out


def block_track(lab, d, dep):
  """Static tracking error (target minus measured, deg) and the implied PD torque Kp*err per joint, for standing and walking rows."""
  kp = np.array(dep["pd_gains"]["stiffness_nm_per_rad"])
  vx = d["obs/cmd/vx"].to_numpy()
  wz = d["obs/cmd/wz"].to_numpy()
  tot = np.abs(vx) + np.abs(wz)
  ps = d[[f"pos/{j}" for j in JOINTS]].to_numpy()
  tg = d[[f"tgt/{j}" for j in JOINTS]].to_numpy()
  out = {}
  for name, sel in (("standing", tot < 0.1), ("walking_vx_gt_0p3", vx > 0.3)):
    if sel.sum() < 20:
      continue
    e = tg[sel] - ps[sel]
    out[name] = {j: {"mean_err_deg": r(e[:, i].mean(), 2), "rms_err_deg": r(np.sqrt((e[:, i] ** 2).mean()), 2), "implied_mean_torque_Nm": r(kp[i] * math.radians(e[:, i].mean()), 2),
                     "tgt_mean_deg": r(tg[sel][:, i].mean(), 2), "pos_mean_deg": r(ps[sel][:, i].mean(), 2)} for i, j in enumerate(JOINTS)}
  return out


def block_frf(lab, d):
  """tgt -> pos transfer function per joint at the gait frequency (same idea as frf_compare_real_sim.py, uniform 50 Hz time base).
  H = Sxy/Sxx (Welch, 256 samples, Hann), read at the bin nearest the gait peak; gain and phase lag of pos relative to tgt."""
  t = d["t"].to_numpy()
  dtm = float(np.median(np.diff(t)))
  tg = d[[f"tgt/{j}" for j in JOINTS]].to_numpy()
  ps = d[[f"pos/{j}" for j in JOINTS]].to_numpy()
  tu, TG = rwa.uniform(t, tg, dtm)
  _, PS = rwa.uniform(t, ps, dtm)
  nper = 256
  f, Pxx = signal.welch(TG, fs=1 / dtm, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=0)
  _, Pyy = signal.welch(PS, fs=1 / dtm, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=0)
  out = {}
  # cross spectrum by hand
  from scipy.signal import csd
  _, Pxy = csd(TG, PS, fs=1 / dtm, nperseg=nper, noverlap=nper // 2, detrend="linear", axis=0)
  tot = Pxx[(f > 0.8) & (f < 3.0)].sum(1)
  fg = f[(f > 0.8) & (f < 3.0)][np.argmax(tot)]
  k = int(np.argmin(np.abs(f - fg)))
  for i, j in enumerate(JOINTS):
    H = Pxy[k, i] / Pxx[k, i]
    coh = np.abs(Pxy[k, i]) ** 2 / (Pxx[k, i] * Pyy[k, i])
    out[j] = {"gain": r(abs(H), 3), "phase_deg": r(math.degrees(np.angle(H)), 1), "coherence": r(coh, 3)}
  out["gait_bin_hz"] = r(f[k], 3)
  return out


def block_disturb(lab, d):
  """Proxy for pushes / hand holding: tilt and gyro magnitude distribution (no force is logged)."""
  gr = d[[f"obs/grav/{a}" for a in "xyz"]].to_numpy()
  g = d[[f"obs/gyro/{a}" for a in "xyz"]].to_numpy()
  tilt = np.degrees(np.arccos(np.clip(-gr[:, 2], -1, 1)))
  gn = np.linalg.norm(g, axis=1) * DEG
  return {"tilt_deg": {"p50": r(np.percentile(tilt, 50), 2), "p99": r(np.percentile(tilt, 99), 2), "max": r(tilt.max(), 2)},
          "gyro_norm_deg_s": {"p50": r(np.percentile(gn, 50), 1), "p99": r(np.percentile(gn, 99), 1), "max": r(gn.max(), 1)},
          "frac_tilt_gt_5deg": r((tilt > 5).mean(), 5), "frac_gyro_gt_60deg_s": r((gn > 60).mean(), 5)}


# ------------------------------------------------------------------------------------------------ G stance
def block_stance(lab, d, xml, dflt):
  import mujoco
  m = mujoco.MjModel.from_xml_path(xml)
  dat = mujoco.MjData(m)
  jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
  bR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "R_ankle_pitch_link")
  bL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "L_ankle_pitch_link")
  names = {"hip_pitch": "hip_pitch", "hip_roll": "hip_roll", "hip_yaw": "hip_yaw", "knee": "knee"}

  def width(angles_deg):
    """angles_deg: dict joint(robot name) -> deg. Returns lateral ankle-origin distance (m), also x offset difference."""
    dat.qpos[:] = m.qpos0
    for leg, s in (("right_leg", "R"), ("left_leg", "L")):
      for jn, mn in names.items():
        dat.qpos[m.jnt_qposadr[jid[f"{s}_{mn}_joint"]]] = math.radians(angles_deg[f"{leg}/{jn}"])
    mujoco.mj_kinematics(m, dat)
    return float(dat.xpos[bL][1] - dat.xpos[bR][1])

  base = {j: 0.0 for j in JOINTS}
  base.update({JOINTS[i]: float(dflt[i]) for i in range(12)})
  w_default = width(base)
  ps = d[[f"pos/{j}" for j in JOINTS]].to_numpy()
  vx = d["obs/cmd/vx"].to_numpy()
  wz = d["obs/cmd/wz"].to_numpy()
  W = np.array([width({j: ps[k, i] for i, j in enumerate(JOINTS)}) for k in range(len(ps))])
  tot = np.abs(vx) + np.abs(wz)
  out = {"default_width_m": r(w_default, 4), "hip_spacing_m": r(2 * 0.1237, 4),
         "all": {k: r(v, 4) for k, v in {"mean": W.mean(), "p5": np.percentile(W, 5), "p50": np.percentile(W, 50), "p95": np.percentile(W, 95), "min": W.min(), "max": W.max()}.items()}}
  for name, sel in (("standing", tot < 0.1), ("walking_vx_gt_0p3", vx > 0.3)):
    if sel.sum() > 20:
      w = W[sel]
      out[name] = {k: r(v, 4) for k, v in {"mean": w.mean(), "p5": np.percentile(w, 5), "p50": np.percentile(w, 50), "p95": np.percentile(w, 95), "min": w.min(), "max": w.max()}.items()}
      out[name]["mean_minus_default_cm"] = r((w.mean() - w_default) * 100, 2)
      out[name]["hip_roll_mean_R_inward_deg"] = r(-ps[sel, JOINTS.index("right_leg/hip_roll")].mean(), 2)
      out[name]["hip_roll_mean_L_inward_deg"] = r(ps[sel, JOINTS.index("left_leg/hip_roll")].mean(), 2)
      out[name]["hip_yaw_mean_R_L_deg"] = [r(ps[sel, JOINTS.index("right_leg/hip_yaw")].mean(), 2), r(ps[sel, JOINTS.index("left_leg/hip_yaw")].mean(), 2)]
  # pair-inward statistic and the model's own foot-foot contact onset
  iR = -ps[:, JOINTS.index("right_leg/hip_roll")]
  iL = ps[:, JOINTS.index("left_leg/hip_roll")]
  pair = 0.5 * (iR + iL)
  onset = contact_onset(m, dat, jid)
  out["model_leg_contact_onset"] = onset
  out["pair_mean_inward_deg"] = {"p50": r(np.percentile(pair, 50), 2), "p95": r(np.percentile(pair, 95), 2), "p99": r(np.percentile(pair, 99), 2), "max": r(pair.max(), 2),
                                "frac_gt_3deg": r((pair > 3).mean(), 4), "frac_gt_onset": r((pair > onset["symmetric_deg"]).mean(), 5)}
  out["R_inward_minus_L_inward_mean_deg_walking"] = r(iR[vx > 0.3].mean() - iL[vx > 0.3].mean(), 2) if (vx > 0.3).any() else None
  # sensitivity: width change per degree of inward roll on both legs from default
  s = dict(base)
  s["right_leg/hip_roll"] = -1.0
  s["left_leg/hip_roll"] = 1.0
  out["width_change_cm_per_deg_both_inward"] = r((width(s) - w_default) * 100, 3)
  # training reset spread corners (both legs, inward 0.10 rad / outward 0.06 rad)
  for nm, (a, b) in {"reset_inward_0p10": (-0.10, 0.10), "reset_outward_0p06": (0.06, -0.06)}.items():
    s = dict(base)
    s["right_leg/hip_roll"] = math.degrees(a)
    s["left_leg/hip_roll"] = math.degrees(b)
    out[f"width_{nm}_m"] = r(width(s), 4)
  # clip-edge widths
  s = dict(base)
  s["right_leg/hip_roll"] = -16.5
  s["left_leg/hip_roll"] = 16.5
  out["width_at_guard_inner_limit_16p5deg_m"] = r(width(s), 4)
  s["right_leg/hip_roll"] = -19.5
  s["left_leg/hip_roll"] = 19.5
  out["width_at_training_clip_inner_limit_19p5deg_m"] = r(width(s), 4)
  return out


def contact_onset(m, dat, jid):
  """Inward hip-roll angle (deg) at which the training model's two legs first touch (default hip pitch/knee, other joints qpos0)."""
  import mujoco
  d0 = {"R_hip_pitch_joint": -0.175, "L_hip_pitch_joint": 0.175, "R_knee_joint": 0.35, "L_knee_joint": -0.35}
  bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])

  def pen(thR, thL):
    dat.qpos[:] = m.qpos0
    for k, v in d0.items():
      dat.qpos[m.jnt_qposadr[jid[k]]] = v
    dat.qpos[m.jnt_qposadr[jid["R_hip_roll_joint"]]] = -math.radians(thR)
    dat.qpos[m.jnt_qposadr[jid["L_hip_roll_joint"]]] = math.radians(thL)
    mujoco.mj_forward(m, dat)
    worst, who = 0.0, None
    for i in range(dat.ncon):
      c = dat.contact[i]
      n1, n2 = bn(c.geom1), bn(c.geom2)
      if n1[0] != n2[0] and n1[0] in "RL" and n2[0] in "RL" and c.dist < worst and m.geom_contype[c.geom1] and m.geom_contype[c.geom2]:
        worst, who = c.dist, (n1, n2)
    return worst, who

  res = {}
  for name, f in (("symmetric", lambda x: pen(x, x)), ("one_leg_other_at_0", lambda x: pen(x, 0.0))):
    lo, hi = 0.0, 25.0
    for _ in range(25):
      mid = 0.5 * (lo + hi)
      if f(mid)[0] < -1e-6:
        hi = mid
      else:
        lo = mid
    res[name + "_deg"] = r(hi, 2)
    res[name + "_first_pair"] = f(hi)[1]
  return res


# ------------------------------------------------------------------------------------------------ main
def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--pol", required=True)
  ap.add_argument("--huphy", required=True)
  ap.add_argument("--xml", required=True)
  ap.add_argument("--logs", required=True)
  ap.add_argument("--old", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  logs = Path(a.logs)
  runinfo = json.load(open(logs / "1028_policy_generalized_walk_teleop.json", encoding="utf-8"))["run"]
  new = {"1028_teleop": ("1028_teleop_ob.csv", "1028_teleop_laten.csv"), "1028_teleop_extra2": ("1028_teleop_extra2_ob.csv", "1028_teleop_extra2_laten.csv")}
  frames = {k: load_log(logs / v[0]) for k, v in new.items()}
  lats = {k: rwa.load_latency(str(logs / v[1])) for k, v in new.items()}
  res = {"run_info": runinfo}
  con, info = block_contract(a, {"frames": frames, "run_info": {"gain": runinfo.get("gain")}})
  res["B_contract"] = con
  old = Path(a.old)
  oldruns = {"1021_vib_stand_g1_hz200": ("vx0gain1walkob(진동확인용)1021.csv", "vx0gain1walklatency(진동확인용)1021.csv"),
             "1021_kdx1.5_stand_hz200": ("vx0gain1kdx1.5walkob1021.csv", "vx0gain1kdx1.5walklaten1021.csv")}
  res["C_health"] = {}
  res["C_spectral"] = {}
  res["C_heading"] = {}
  for k in new:
    d = frames[k]
    res["C_health"][k] = block_health(k, d, lats[k], info)
    res["C_spectral"][k] = block_spectral(k, d)
    res["C_heading"][k] = block_heading(k, d, float(np.median(np.diff(d["t"].to_numpy()))))
  for k, (pf, lf) in oldruns.items():
    S = pd.read_csv(old / pf, comment="#")
    # old logs have the same column layout (82 cols) without cmd columns in some; fill zeros
    for c in ("obs/cmd/vx", "obs/cmd/wz"):
      if c not in S:
        S[c] = 0.0
    res["C_spectral"][k] = block_spectral(k, S, old=True)
    res.setdefault("F_track_old", {})[k] = block_track(k, S, load_yaml_min(Path(a.pol) / "deploy.yaml"))
    res.setdefault("G_stance_old", {})[k] = block_stance(k, S, a.xml, info["dflt"])
    t = S["t"].to_numpy()
    dtm = np.diff(t) * 1e3
    res["C_health"][k] = {"duration_s": r(t[-1] - t[0], 1), "policy_interval_ms": {kk: r(v, 3) for kk, v in rwa.stats(dtm).items() if kk in ("median", "p95", "p99", "max")},
                          "imu_gyro_repeat_frac": r((np.diff(S[[f"obs/gyro/{x}" for x in "xyz"]].to_numpy(), axis=0) == 0).all(1).mean(), 4)}
  res["F_frf"] = {k: block_frf(k, frames[k]) for k in new}
  res["F_disturb"] = {k: block_disturb(k, frames[k]) for k in new}
  dep_yaml = load_yaml_min(Path(a.pol) / "deploy.yaml")
  res["F_track"] = {k: block_track(k, frames[k], dep_yaml) for k in new}
  res["F_clip"] = {k: block_clip(k, frames[k], info) for k in new}
  res["G_stance"] = {k: block_stance(k, frames[k], a.xml, info["dflt"]) for k in new}
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
  print("wrote", a.out)


if __name__ == "__main__":
  main()
