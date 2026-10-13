#!/usr/bin/env python3
"""Real-log signature with the same definitions as tools/sim2real/real_condition_sim.py, plus the
comparison figure (real runs vs simulated conditions).

  python tools/sim2real/huphy_log_signature.py --logs <archivedlogs> --sim a.json b.json ... --fig out.png
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True)
ap.add_argument("--sim", nargs="*", default=[])
ap.add_argument("--fig", default="")
a = ap.parse_args()
LO = np.array([-112.75, -19.5, -40.5, 6.0, -61.88, -61.88, -17.75, -79.5, -40.5, -114.0, -61.88, -61.88])
HI = np.array([17.75, 79.5, 40.5, 114.0, 61.88, 61.88, 112.75, 19.5, 40.5, -6.0, 61.88, 61.88])
real = {}
print(f"{'run':24s} {'s':>6} {'act99':>6} {'actRMS':>6} {'t|a|>3':>6} {'kneeSD':>6} {'kneeHz':>6} {'gyro x/y/z dps':>15} {'lim%':>5} {'jump%':>5} {'grav0':>16}")
for f in sorted(glob.glob(os.path.join(a.logs, "*ob*.csv"))):
  d = np.genfromtxt(f, delimiter=",", skip_header=1)
  t = d[:, 0] - d[0, 0]; act = d[:, 46:58]; tgt = d[:, 58:70]; pos = d[:, 70:82]; gy = d[:, 1:4]
  aa = np.abs(act)
  t3 = t[np.argmax(aa.max(1) > 3)] if (aa.max(1) > 3).any() else np.nan
  lim = np.clip(tgt, LO + 3, HI - 3); sent = np.clip(lim, pos - 50, pos + 50)
  tu = np.arange(0, t[-1], 0.02) if t[-1] > 1 else t
  kn = np.interp(tu, t, pos[:, 3])
  x = kn - kn.mean(); fr = np.fft.rfftfreq(len(x), 0.02); band = (fr > 0.3) & (fr < 12)
  pk = fr[band][np.argmax(np.abs(np.fft.rfft(x * np.hanning(len(x))))[band])] if band.any() else np.nan
  r = dict(seconds=float(t[-1]), act_p99=float(np.percentile(aa, 99)), act_rms=float(np.sqrt((act ** 2).mean())),
           t_act3=float(t3), knee_std_deg=float(pos[:, 3].std()), knee_hz=float(pk),
           gyro_rms_dps=np.degrees(gy.std(0)).tolist(),
           clip_limit=float((np.abs(lim - tgt) > 1e-6).mean()), clip_jump=float((np.abs(sent - lim) > 1e-6).mean()),
           grav0=d[0, 4:7].tolist())
  name = os.path.basename(f)[:-4]
  real[name] = r
  g = r["gyro_rms_dps"]
  print(f"{name:24s} {r['seconds']:6.1f} {r['act_p99']:6.2f} {r['act_rms']:6.2f} {r['t_act3']:6.2f} {r['knee_std_deg']:6.1f} {r['knee_hz']:6.2f} "
        f"{g[0]:4.0f}/{g[1]:4.0f}/{g[2]:4.0f}  {100*r['clip_limit']:5.1f} {100*r['clip_jump']:5.1f} {np.round(r['grav0'], 3)}")

if a.fig:
  import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
  sim = {}
  for p in a.sim:
    J = json.load(open(p)); tag = " (40 Hz)" if J["args"].get("decimation") == 5 else ""
    for c in dict.fromkeys(x["cond"] for x in J["rows"]):
      rs = [x for x in J["rows"] if x["cond"] == c]
      sim[c + tag] = {k: float(np.nanmean([x[k] for x in rs])) for k in ("act_rms", "act_p99", "knee_std_deg", "survived_s", "knee_hz")}
      sim[c + tag]["gyro"] = float(np.mean([np.linalg.norm(x["gyro_rms_dps"]) for x in rs]))
      sim[c + tag]["fell"] = sum(x["fell"] for x in rs) / len(rs)
  pick = ["nominal", "real timing (delay+guards+loss)", "real timing (delay+guards+loss) (40 Hz)",
          "imu gyro_yz flip", "imu grav_x flip", "imu yaw180", "imu grav_x bias +0.1",
          "0.05/0.3 +timing", "0.05/0.3 +timing crane 0.5", "0.05/0.3 +timing crane 1->0", "0.05/0.5 +timing crane 1->0",
          "0.1/0.1 +timing crane 1->0", "0.5/0.5 +timing", "0.7 +timing", "0.8 +timing", "1.0 +timing crane 1->0",
          "1.0 +timing crane 1->0 +gyro_yz", "1.0 +timing crane 1->0 +grav_x"]
  pick = [p for p in pick if p in sim]
  rl = [k for k in real if real[k]["seconds"] > 5]
  labels = [f"REAL {k.replace('바닥서기', '_floor_stand')}" for k in rl] + [f"sim {p}" for p in pick]
  def col(key, skey=None, gyro=False):
    rv = [np.linalg.norm(real[k]["gyro_rms_dps"]) if gyro else real[k][key] for k in rl]
    sv = [sim[p]["gyro" if gyro else (skey or key)] for p in pick]
    return rv + sv
  metrics = [("action RMS", col("act_rms")), ("action |p99|", col("act_p99")),
             ("knee angle std (deg)", col("knee_std_deg")), ("gyro RMS norm (deg/s)", col(None, gyro=True)),
             ("knee peak freq (Hz)", col("knee_hz"))]
  fig, axs = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 0.32 * len(labels) + 1.6), sharey=True)
  y = np.arange(len(labels))
  colors = ["#d62728"] * len(rl) + ["#1f77b4" if sim[p]["fell"] == 0 else "#7f7f7f" for p in pick]
  for ax, (title, v) in zip(axs, metrics):
    ax.barh(y, v, color=colors)
    ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.3)
    for i in range(len(rl)):
      ax.axvspan(0, 0, color="none")
    lo, hi = min(v[:len(rl)]), max(v[:len(rl)])
    ax.axvspan(lo, hi, color="#d62728", alpha=.12)
  axs[0].set_yticks(y); axs[0].set_yticklabels(labels, fontsize=8); axs[0].invert_yaxis()
  fig.suptitle("Real runs (red, band = real range) vs simulated conditions (blue = all 8 stood 20 s, grey = fell); sim metrics until fall", fontsize=10)
  fig.tight_layout(); fig.savefig(a.fig, dpi=110)
  print("wrote", a.fig)
