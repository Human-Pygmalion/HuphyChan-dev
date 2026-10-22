#!/usr/bin/env python3
"""Baseline numbers + gait figure from the npz files written by sim_walk_record.py.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/sim_walk_metrics.py \
      --data mujoco-sim/mjlab/analysis/out/s2r_walk --fig docs/mujoco/assets/2026-10-22_sim_walk_reference.png

Writes <data>/baseline_metrics.json (every number, per env where relevant) and <data>/baseline_tables.md.
Windows skip the first 2 s after the start and after the 5 s command switch (settling).
Definitions (also printed in the doc):
  forward speed   displacement along the INITIAL heading / window length (drops below the body speed when the heading turns)
  body-frame vx   mean of the base velocity along the body x axis
  yaw drift       least-squares slope of the unwrapped base yaw [deg/s]; heading change = yaw(end) - yaw(start) [deg]
  lateral drift   displacement perpendicular to the initial heading over the window [m], + = left
  touchdown       rising edge of the foot-contact flag after bridging air gaps <= 3 samples (60 ms) and dropping contacts < 2 samples (40 ms)
  step rate       touchdowns of both feet per second; stride frequency = step rate / 2
  step length     distance along the heading between the two feet at the instant one foot touches down
  asymmetry       (R - L) / mean(R, L) * 100 [%]
  tracking error  RMS over the window of target[k] - q[k] per joint (50 Hz rows)
  action rate     RMS of (a[k] - a[k-1]) over all 12 outputs, raw action units
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import welch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--fig", default="")
ap.add_argument("--runs", default="R1,R2,R3,R4,R9,R10,R6,R7,R8,V0_20ms,V0_52ms,V0_real,V3_20ms,V3_52ms,V3_real,W5_20ms,W5_52ms,W5_real")
args = ap.parse_args()
D = Path(args.data)
WINDOWS = {  # run -> [(label, t0, t1)]
  "R1": [("R1 vx0.5", 2, 5), ("R1 vx1.0", 7, 10)], "R2": [("R2 vx0.5", 2, 5), ("R2 vx1.0", 7, 10)],
  "R3": [("R3 vx0.5", 2, 20)], "R4": [("R4 vx1.0", 2, 20)], "R9": [("R9 vx0.5 DR", 2, 20)], "R10": [("R10 vx1.0 DR", 2, 20)],
  "R6": [("R6 vx0.5 52ms", 2, 20)], "R7": [("R7 vx1.0 52ms", 2, 20)], "R8": [("R8 vx0.5 real", 2, 20)],
}
for _n, _c in (("V0", "vx0"), ("V3", "vx0.3"), ("W5", "wz0.5")):
  for _t in ("20ms", "52ms", "real"):
    WINDOWS[f"{_n}_{_t}"] = [(f"{_n} {_c} {_t}", 2, 20)]
DEG = 180 / np.pi


def runs_filter(x, max_gap=3, min_stance=2):
  """Clean a 1-D bool contact flag: air gaps of <= max_gap samples (60 ms) between contacts are bridged (the foot
  re-touches during one landing), then contact runs shorter than min_stance samples (40 ms) are dropped."""
  x = np.asarray(x, bool).copy()
  n = len(x)
  i = 0
  while i < n:
    j = i
    while j < n and x[j] == x[i]:
      j += 1
    if not x[i] and 0 < i and j < n and (j - i) <= max_gap:
      x[i:j] = True
    i = j
  i = 0
  while i < n:
    j = i
    while j < n and x[j] == x[i]:
      j += 1
    if x[i] and (j - i) < min_stance:
      x[i:j] = False
    i = j
  return x


def stats(v):
  v = np.asarray(v, float)
  v = v[np.isfinite(v)]
  return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))} if len(v) else {"mean": np.nan, "std": np.nan, "n": 0}


def peaks(f, p, n=3, fmin=0.2, fmax=None, sep=0.4):
  m = (f >= fmin) & (f <= (fmax or f[-1]))
  ff, pp = f[m], p[m]
  loc = [i for i in range(1, len(pp) - 1) if pp[i] >= pp[i - 1] and pp[i] > pp[i + 1]]
  loc.sort(key=lambda i: -pp[i])
  out = []
  for i in loc:
    if all(abs(ff[i] - o) > sep for o in out):
      out.append(float(ff[i]))
    if len(out) == n:
      break
  return out


def psd_mean(x, fs, t0n, nperseg_max):
  """x: (n, E, J) -> mean over envs of welch PSD, shape (F, J)."""
  n = x.shape[0]
  nper = min(nperseg_max, n)
  f, P = welch(x, fs=fs, nperseg=nper, nfft=max(2048, nper), detrend="constant", axis=0)
  return f, P.mean(1)


def uniform_view(d):
  """TIMING runs: 50 Hz view (every 4th 5 ms row) of the per-step arrays + the per-call arrays under call_*."""
  v = {"t": d["t200"][::4], "call_action": d["action"], "call_t": d["t_call"], "n_calls": d["n_calls"], "default_q12": d["default_q12"]}
  for k in ("base_pos", "base_quat", "base_yaw", "base_lin_vel_w", "base_lin_vel_b", "base_ang_vel_w", "base_ang_vel_b", "proj_gravity",
            "foot_contact", "foot_n_contacts", "foot_force_w", "foot_pos_w", "command", "action", "target", "terminated"):
    v[k] = d[k + "200"][::4]
  v["q"], v["qd"] = d["q200"][::4], d["qd200"][::4]
  for k in ("q200", "qd200", "target_eff200", "tau200", "tau_req200"):
    v[k] = d[k]
  return v


def analyse(d, meta, label, t0, t1):
  t = d["t"]
  idx = np.where((t >= t0 - 1e-6) & (t < t1 - 1e-6))[0]
  term = d["terminated"]
  fell = np.cumsum(term, 0)[idx[-1]] > 0  # env terminated at or before window end
  ok = ~fell
  E = d["q"].shape[1]
  r = {"label": label, "window_s": [t0, t1], "n_envs": int(E), "n_valid": int(ok.sum())}
  if ok.sum() == 0:
    return r
  sl = slice(idx[0], idx[-1] + 1)
  yaw0 = np.unwrap(d["base_yaw"], axis=0)[0]
  p = d["base_pos"]
  dp = p[idx[-1]] - p[idx[0]]
  dur = t[idx[-1]] - t[idx[0]]
  h = np.stack([np.cos(yaw0), np.sin(yaw0)], -1)
  along = (dp[:, :2] * h).sum(-1)
  lat = dp[:, 0] * (-h[:, 1]) + dp[:, 1] * h[:, 0]
  yaw = np.unwrap(d["base_yaw"], axis=0)
  slope = np.array([np.polyfit(t[sl], yaw[sl, e], 1)[0] for e in range(E)]) * DEG
  r["forward_speed_mps"] = stats(along[ok] / dur)
  r["mean_body_vx_mps"] = stats(d["base_lin_vel_b"][sl, :, 0].mean(0)[ok])
  r["command_vx_mps"] = float(d["command"][sl, 0, 0].mean())
  r["yaw_drift_degps"] = stats(slope[ok])
  r["heading_change_deg"] = stats(((yaw[idx[-1]] - yaw[idx[0]]) * DEG)[ok])
  r["lateral_drift_m"] = stats(lat[ok])
  r["lateral_drift_mps"] = stats(lat[ok] / dur)
  r["base_height_m"] = stats(p[sl, :, 2].mean(0)[ok] - p[0, :, 2].mean() + 0.0)  # relative, see doc
  r["base_height_abs_note"] = "base z is world z (terrain surface at z=0)"
  r["base_height_m"] = stats(p[sl, :, 2].mean(0)[ok])
  r["roll_pitch_proj_gravity_rms"] = stats(np.sqrt((d["proj_gravity"][sl, :, :2] ** 2).sum(-1).mean(0))[ok])
  # gait from contacts
  td = {"R": [], "L": []}
  st = {"R": [], "L": []}
  sw = {"R": [], "L": []}
  frac = {"R": [], "L": []}
  slen = {"R": [], "L": []}
  rate = []
  dsup = []
  for e in np.where(ok)[0]:
    c = [runs_filter(d["foot_contact"][:, e, k]) for k in (0, 1)]
    ntd = 0
    for k, nm in enumerate("RL"):
      cc = c[k][sl]
      edges = np.where(cc[1:] & ~cc[:-1])[0] + 1
      ntd += len(edges)
      td[nm].append(len(edges) / dur)
      frac[nm].append(cc.mean())
      # complete contact / air runs inside the window
      ch = np.where(np.diff(cc.astype(int)) != 0)[0] + 1
      b = np.concatenate([[0], ch, [len(cc)]])
      for a, z in zip(b[:-1], b[1:]):
        if a == 0 or z == len(cc):
          continue
        (st if cc[a] else sw)[nm].append((z - a) * 0.02)
      # step length at touchdown of this foot
      for i in edges + idx[0]:
        pa, pb = d["foot_pos_w"][i, e, k, :2], d["foot_pos_w"][i, e, 1 - k, :2]
        yh = d["base_yaw"][i, e]
        slen[nm].append(float(((pa - pb) * np.array([np.cos(yh), np.sin(yh)])).sum()))
    rate.append(ntd / dur)
    dsup.append((c[0][sl] & c[1][sl]).mean())
  asym = lambda a, b: float((np.mean(a) - np.mean(b)) / (0.5 * (np.mean(a) + np.mean(b))) * 100) if len(a) and len(b) else np.nan  # noqa: E731
  r["step_rate_hz"] = stats(rate)
  r["stride_freq_hz"] = stats(np.array(rate) / 2)
  r["stance_fraction"] = {k: stats(v) for k, v in frac.items()}
  r["swing_fraction"] = {k: stats(1 - np.array(v)) for k, v in frac.items()}
  r["double_support_fraction"] = stats(dsup)
  r["stance_time_s"] = {k: stats(v) for k, v in st.items()}
  r["swing_time_s"] = {k: stats(v) for k, v in sw.items()}
  r["step_length_m"] = {k: stats(v) for k, v in slen.items()}
  r["step_length_asym_pct"] = asym(slen["R"], slen["L"])
  r["stance_time_asym_pct"] = asym(st["R"], st["L"])
  r["step_length_asym_per_env_pct"] = None
  # tracking, actions
  q12 = d["q"][sl, :, :12]
  err = d["target"][sl] - q12
  r["tracking_rms_rad"] = np.sqrt((err[:, ok] ** 2).mean(0)).mean(0).tolist()  # mean over valid envs of per-env rms
  err200 = d["target_eff200"] - d["q200"][:, :, :12]
  s200 = slice(idx[0] * 4, (idx[-1] + 1) * 4)
  r["tracking_rms_200hz_rad"] = np.sqrt((err200[s200][:, ok] ** 2).mean(0)).mean(0).tolist()
  a = d["action"]
  if "call_t" in d:  # TIMING run: statistics over the policy calls inside the window (rows of the real logs)
    A, dA = [], []
    for e in np.where(ok)[0]:
      n_ = int(d["n_calls"][e])
      m_ = (d["call_t"][:n_, e] >= t0) & (d["call_t"][:n_, e] < t1)
      ae = d["call_action"][:n_, e][m_]
      A.append(ae)
      dA.append(np.diff(ae, axis=0))
    A, dA = np.concatenate(A), np.concatenate(dA)
    r["action_rms"] = float(np.sqrt((A ** 2).mean()))
    r["action_rms_per_joint"] = np.sqrt((A ** 2).mean(0)).tolist()
    r["action_rate_rms"] = float(np.sqrt((dA ** 2).mean()))
    r["action_rate_rms_per_joint"] = np.sqrt((dA ** 2).mean(0)).tolist()
    r["call_interval_ms"] = meta["timing"]["interval_stats_ms"]
  else:
    r["action_rms"] = float(np.sqrt((a[sl][:, ok] ** 2).mean()))
    r["action_rms_per_joint"] = np.sqrt((a[sl][:, ok] ** 2).mean((0, 1))).tolist()
    da = np.diff(a, axis=0)[idx[0] - 1 : idx[-1]]
    r["action_rate_rms"] = float(np.sqrt((da[:, ok] ** 2).mean()))
    r["action_rate_rms_per_joint"] = np.sqrt((da[:, ok] ** 2).mean((0, 1))).tolist()
    r["call_interval_ms"] = {"mean": 20.0, "median": 20.0}
  r["tau_rms_Nm_per_joint"] = np.sqrt((d["tau200"][s200][:, ok] ** 2).mean((0, 1))).tolist()
  r["tau_peak_Nm_per_joint"] = np.abs(d["tau200"][s200][:, ok]).max((0, 1)).tolist()
  # spectra
  f50, P50 = psd_mean(q12[:, ok], 50.0, 0, 512)
  f200, P200 = psd_mean(d["q200"][s200][:, ok, :12], 200.0, 0, 2048)
  r["psd50_peaks_hz"] = [peaks(f50, P50[:, j]) for j in range(12)]
  r["psd200_peaks_hz"] = [peaks(f200, P200[:, j]) for j in range(12)]
  r["psd200_peaks_above25_hz"] = [peaks(f200, P200[:, j], n=2, fmin=25.0, fmax=100.0, sep=1.0) for j in range(12)]
  tot = P200.sum(0)
  hi = P200[f200 > 25].sum(0)
  r["var_fraction_above_25hz"] = (hi / tot).tolist()
  r["psd_freq_resolution_hz"] = {"50Hz": float(50.0 / min(512, len(idx))), "200Hz": float(200.0 / min(2048, len(idx) * 4))}
  r["psd50"] = {"f": f50[f50 <= 25].tolist(), "P_rad2_per_hz": P50[f50 <= 25].T.tolist()}
  return r


def fmt(s, nd=3):
  return f"{s['mean']:.{nd}f} ± {s['std']:.{nd}f}" if s and s["n"] else "n/a"


def tables(R, motors, joints=True):
  L = []
  labels = [r["label"] for r in R]
  L.append("| metric | " + " | ".join(labels) + " |")
  L.append("|---|" + "---|" * len(labels))
  rows = [
    ("policy call interval mean / median [ms]", lambda r: f"{r['call_interval_ms']['mean']:.1f} / {r['call_interval_ms']['median']:.1f}"),
    ("valid envs (of 8, no fall)", lambda r: f"{r['n_valid']}/{r['n_envs']}"),
    ("command vx [m/s]", lambda r: f"{r['command_vx_mps']:.2f}"),
    ("forward speed [m/s]", lambda r: fmt(r["forward_speed_mps"])),
    ("body-frame vx, mean [m/s]", lambda r: fmt(r["mean_body_vx_mps"])),
    ("yaw drift [deg/s]", lambda r: fmt(r["yaw_drift_degps"])),
    ("heading change over window [deg]", lambda r: fmt(r["heading_change_deg"], 1)),
    ("lateral drift over window [m] (+left)", lambda r: fmt(r["lateral_drift_m"])),
    ("step rate, both feet [Hz]", lambda r: fmt(r["step_rate_hz"], 2)),
    ("stance fraction R / L", lambda r: f"{r['stance_fraction']['R']['mean']:.2f} / {r['stance_fraction']['L']['mean']:.2f}"),
    ("double-support fraction", lambda r: fmt(r["double_support_fraction"], 2)),
    ("stance time R / L [s]", lambda r: f"{r['stance_time_s']['R']['mean']:.3f} / {r['stance_time_s']['L']['mean']:.3f}"),
    ("swing time R / L [s]", lambda r: f"{r['swing_time_s']['R']['mean']:.3f} / {r['swing_time_s']['L']['mean']:.3f}"),
    ("step length R / L [m]", lambda r: f"{r['step_length_m']['R']['mean']:.3f} / {r['step_length_m']['L']['mean']:.3f}"),
    ("step-length asymmetry [%]", lambda r: f"{r['step_length_asym_pct']:+.1f}"),
    ("stance-time asymmetry [%]", lambda r: f"{r['stance_time_asym_pct']:+.1f}"),
    ("action RMS", lambda r: f"{r['action_rms']:.3f}"),
    ("action-rate RMS (per policy call)", lambda r: f"{r['action_rate_rms']:.3f}"),
    ("base height [m]", lambda r: fmt(r["base_height_m"], 3)),
  ]
  for name, fn in rows:
    L.append(f"| {name} | " + " | ".join(fn(r) for r in R) + " |")
  if not joints:
    return "\n".join(L)
  L.append("")
  L.append("RMS of (target - q) per joint [deg], 200 Hz rows (target = the position target the PD law used), mean over envs")
  L.append("")
  L.append("| joint | " + " | ".join(labels) + " |")
  L.append("|---|" + "---|" * len(labels))
  for j, m in enumerate(motors):
    L.append(f"| {m} | " + " | ".join(f"{r['tracking_rms_200hz_rad'][j] * DEG:.2f}" for r in R) + " |")
  L.append("")
  L.append("Joint-angle PSD peaks [Hz], 50 Hz rows (top 3 by power, separated by 0.4 Hz; resolution in JSON)")
  L.append("")
  L.append("| joint | " + " | ".join(labels) + " |")
  L.append("|---|" + "---|" * len(labels))
  for j, m in enumerate(motors):
    L.append(f"| {m} | " + " | ".join(", ".join(f"{x:.2f}" for x in r["psd50_peaks_hz"][j]) for r in R) + " |")
  L.append("")
  L.append("Joint-angle PSD at 200 Hz: top 3 peaks [Hz] over 0.2-100 Hz / fraction of the angle variance above 25 Hz [%] / strongest peak in 25-100 Hz")
  L.append("")
  L.append("| joint | " + " | ".join(labels) + " |")
  L.append("|---|" + "---|" * len(labels))
  for j, m in enumerate(motors):
    cells = []
    for r in R:
      hi = r["psd200_peaks_above25_hz"][j]
      cells.append(f"{', '.join(f'{x:.2f}' for x in r['psd200_peaks_hz'][j])} / {100 * r['var_fraction_above_25hz'][j]:.2f} / {hi[0]:.1f}" if hi else "n/a")
    L.append(f"| {m} | " + " | ".join(cells) + " |")
  L.append("")
  L.append("Applied torque per joint [N*m]: RMS / peak (200 Hz)")
  L.append("")
  L.append("| joint | " + " | ".join(labels) + " |")
  L.append("|---|" + "---|" * len(labels))
  for j, m in enumerate(motors):
    L.append(f"| {m} | " + " | ".join(f"{r['tau_rms_Nm_per_joint'][j]:.1f} / {r['tau_peak_Nm_per_joint'][j]:.1f}" for r in R) + " |")
  return "\n".join(L)


def figure(path, data):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  names = ["hip pitch", "hip roll", "hip yaw", "knee", "ankle crank A", "ankle crank B"]
  cols = plt.cm.tab10.colors
  fig, axs = plt.subplots(4, 2, figsize=(14, 11), sharex="col", gridspec_kw={"height_ratios": [3, 3, 2.2, 1.2]})
  for c, (run, title, t0) in enumerate((("R3", "Nominal, vx = 0.5 m/s, wz = 0 (R3, env 0)", 10.0), ("R4", "Nominal, vx = 1.0 m/s, wz = 0 (R4, env 0)", 10.0))):
    d = data[run]
    t = d["t"]
    m = (t >= t0) & (t < t0 + 3.0)
    tt = t[m] - t0
    e = 0
    q = d["q"][m, e] * DEG
    dflt = np.concatenate([d["default_q12"], d["q"][0, e, 12:] * 0]) * DEG
    for r_, (side, off) in enumerate((("Right leg", 0), ("Left leg", 6))):
      ax = axs[r_, c]
      for j in range(6):
        ax.plot(tt, q[:, off + j] - dflt[off + j], color=cols[j], label=names[j])
      ax.set_ylabel(f"{side}: joint angle - default [deg]")
      ax.grid(alpha=0.3)
      if c == 0 and r_ == 0:
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax = axs[2, c]
    for j, (nm, ls) in enumerate((("R ankle pitch", "-"), ("R ankle roll", "-"), ("L ankle pitch", "--"), ("L ankle roll", "--"))):
      ax.plot(tt, q[:, 12 + j] - 0 * dflt[12 + j], ls=ls, color=cols[6 + j % 2], label=nm)
    ax.set_ylabel("Passive ankle angle [deg]")
    ax.grid(alpha=0.3)
    if c == 0:
      ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax = axs[3, c]
    for k, (nm, y) in enumerate((("R foot", 1.0), ("L foot", 0.0))):
      cc = runs_filter(d["foot_contact"][:, e, k])[m]
      ax.fill_between(tt, y, y + 0.8, where=cc, step="post", color=cols[3 if k == 0 else 0], alpha=0.8, linewidth=0)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["L foot", "R foot"])
    ax.set_xlabel("Time [s]")
    ax.set_xlim(0, 3)
    ax.set_ylabel("Ground contact")
    axs[0, c].set_title(title)
  fig.suptitle("Simulated gait of terrain10_2048_model_37400 (3 s windows starting at t = 10 s, flat ground, no DR / noise / delay). Shaded = foot in contact.", fontsize=10)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def main():
  data, R, motors = {}, [], None
  for run in args.runs.split(","):
    p = D / f"{run}.npz"
    if not p.exists():
      continue
    d = dict(np.load(p))
    meta = json.loads((D / f"{run}.json").read_text())
    motors = meta["joint_order_motors12"]
    data[run] = d
    if "n_calls" in d:
      d = uniform_view(d)
    base = re.sub(r"_s\d+$", "", run)  # seed repeats: R3_s1 uses the windows of R3
    for lab, a, b in WINDOWS[base]:
      R.append(analyse(d, meta, lab + run[len(base):], a, b))
  out = {"windows": R, "motors": motors}
  (D / "baseline_metrics.json").write_text(json.dumps(out, indent=1, default=lambda o: None if o != o else float(o)))
  SECTIONS = [  # (title, label prefixes, with per-joint tables)
    ("Deployment profile (vx 0.5 for 5 s, then 1.0 for 5 s): nominal (R1) and training noise/DR/lag/delay (R2)", ("R1 ", "R2 "), False),
    ("Steady walk, nominal, 20 ms policy step (R3 vx 0.5, R4 vx 1.0) and with training DR/noise/lag/delay (R9, R10)", ("R3 ", "R4 ", "R9 ", "R10 "), False),
    ("Policy called every 52 ms (R6 vx 0.5, R7 vx 1.0) and with the real call intervals (R8 vx 0.5); nominal otherwise", ("R6 ", "R7 ", "R8 "), False),
    ("Command set for the real-log comparison, 20 ms / 52 ms / real intervals", ("V0 ", "V3 ", "W5 "), False),
    ("Seed repeats (seeds 1 and 2) of R3, R6, R8", ("R3 vx0.5_s", "R6 vx0.5 52ms_s", "R8 vx0.5 real_s"), False),
  ]
  chunks = []
  for title, pref, jt in SECTIONS:
    seed_sec = pref[0].endswith("_s")
    sub = [r for r in R if r["label"].startswith(pref) and (seed_sec or not re.search(r"_s\d$", r["label"]))]
    if sub:
      chunks.append(f"### {title}\n\n" + tables(sub, motors, joints=jt))
  full = "\n\n".join(chunks)
  jt_runs = [r for r in R if r["label"] in ("R3 vx0.5", "R4 vx1.0")]
  if jt_runs:
    t = tables(jt_runs, motors, joints=True)
    full += "\n\n### Per-joint tables, steady nominal 20 ms (R3, R4)\n\n" + "RMS of (target - q)" + t.split("RMS of (target - q)", 1)[1]
  (D / "baseline_tables.md").write_text(full)
  print(full)
  if args.fig and "R3" in data and "R4" in data:
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    figure(args.fig, data)
    print("figure ->", args.fig)


if __name__ == "__main__":
  main()
