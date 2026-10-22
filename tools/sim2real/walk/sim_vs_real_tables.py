#!/usr/bin/env python3
"""Real-vs-simulation comparison tables from ONE analysis code (real_walk_analyze.py) applied to
  * the simulator runs exported to the real-log CSV layout (sim_walk_export_csv.py), every env separately, and
  * the matching real logs in Expr_Data/.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/sim_vs_real_tables.py \
      --csv mujoco-sim/mjlab/analysis/out/s2r_walk/csv --npz mujoco-sim/mjlab/analysis/out/s2r_walk --real Expr_Data \
      --out mujoco-sim/mjlab/analysis/out/s2r_walk/sim_vs_real --fig-prefix docs/mujoco/assets/2026-10-22_sim_walk_

Writes <out>.json (all numbers, sim = mean and std over envs) and <out>.md (tables). The 3-8 Hz band is computed here, the
same way for every source: pos [deg] of the 12 motors is linearly interpolated onto a 50 ms grid (Nyquist 10 Hz, the
Nyquist of the real 52 ms policy interval), Welch (Hann, nperseg 200, 50 % overlap, linear detrend), summed over the 12
joints, integrated over 3-8 Hz. Sim-only extra: the same band on the true 200 Hz joint angle (not limited by the call rate).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_walk_analyze as RW  # noqa: E402

DEG = RW.DEG
GROUPS = {
  "vx0": dict(title="command vx 0, wz 0 (stepping in place)", sim=["V0_20ms", "V0_52ms", "V0_real"],
              real=[("real vx0 gain1", "vx0gain1walkob(진동확인용)1021.csv"), ("real vx0 gain0.6", "1021vx0obgain0.61.csv")]),
  "vx0.3": dict(title="command vx 0.3", sim=["V3_20ms", "V3_52ms", "V3_real"], real=[("real vx0.3", "1021vx0.3ob.csv")]),
  "vx0.5": dict(title="command vx 0.5", sim=["R3", "R6", "R8"], real=[("real vx0.5", "1021vx0.5ob.csv")]),
  "wz0.5": dict(title="command wz 0.5 (turn)", sim=["W5_20ms", "W5_52ms", "W5_real"], real=[("real wz0.5", "1021vz0.5ob.csv")]),
}
SIMLAB = {"20ms": "sim 20 ms", "52ms": "sim 52 ms", "real": "sim real-interval"}


def simlabel(run):
  if run in ("R3", "V0_20ms", "V3_20ms", "W5_20ms"):
    return "sim 20 ms"
  if run in ("R6", "V0_52ms", "V3_52ms", "W5_52ms"):
    return "sim 52 ms"
  return "sim real-interval"


def band38(S):
  t, Y = S["t"], S["pos"]
  tu, Yu = RW.uniform(t, Y, 0.05)
  n = min(200, len(tu))
  f, P = signal.welch(Yu, fs=20.0, window="hann", nperseg=n, noverlap=n // 2, detrend="linear", axis=0, nfft=2048)
  df = f[1] - f[0]
  Ps = P.sum(1)
  m = (f >= 3) & (f <= 8)
  tot = Ps[(f >= 0.2) & (f <= 10)].sum() * df
  pk = f[m][np.argmax(Ps[m])]
  return float(Ps[m].sum() * df), float(Ps[m].sum() * df / tot), float(pk), f, Ps


def extract(res):
  S = res["_arr"]["S"]
  J = S["joints"]
  out = dict(
    joints=J,
    interval_median_ms=res["timing"]["policy_interval_ms"]["median"], interval_p95_ms=res["timing"]["policy_interval_ms"]["p95"],
    f_gait_hz=res["gait"].get("f_gait_hz", np.nan), stride_period_s=res["gait"].get("phase_average", {}).get("period_mean_s", np.nan), f_dom_hz=res["jitter"].get("f_dominant_hz", np.nan),
    f_hi_hz=res["jitter"].get("f_hi_hz", np.nan), band_3_6_frac=res["jitter"].get("pos_band_fraction_sum_over_joints", [np.nan] * 4)[2],
    act_step_rms=float(np.sqrt((np.diff(S["act"], axis=0) ** 2).mean(0)).mean()), heading_deg=res["heading"]["heading_change_deg"],
    yaw_rate_deg_s=res["heading"]["mean_yaw_rate_deg_s"], yaw_err_deg_s=res["heading"]["mean_yaw_rate_deg_s"] - res["heading"]["command"]["wz_cmd_deg_s"], duration_s=res["timing"]["duration_s"],
    amp_deg=S["pos"].std(0).tolist(), trk_rms_deg=np.sqrt(((S["tgt"] - S["pos"]) ** 2).mean(0)).tolist(),
    trk_mean_deg=(S["tgt"] - S["pos"]).mean(0).tolist(), pos_minus_default_deg=(np.degrees(S["q"]).mean(0)).tolist(),
  )
  p38, f38, pk38, f, Ps = band38(S)
  out.update(band38_deg2=p38, band38_frac=f38, peak38_hz=pk38)
  return out, (f, Ps)


def band200(npz_path):
  d = np.load(npz_path)
  q = np.degrees(d["q200"][:, :, :12])[int(2 / 0.005):]  # after 2 s
  f, P = signal.welch(q, fs=200.0, nperseg=2048, detrend="linear", axis=0)
  Ps = P.sum(2)  # (F, E)
  df = f[1] - f[0]
  m = (f >= 3) & (f <= 8)
  tot = Ps[(f >= 0.2) & (f <= 25)].sum(0) * df
  return (Ps[m].sum(0) * df / tot).tolist()


def agg(lst, key):
  v = np.array([x[key] for x in lst], float)
  return np.nanmean(v, 0), np.nanstd(v, 0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv", required=True)
  ap.add_argument("--npz", required=True)
  ap.add_argument("--real", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--fig-prefix", default="")
  ap.add_argument("--envs", type=int, default=8)
  a = ap.parse_args()
  R = {}  # group -> source label -> dict
  PSD = {}
  for g, spec in GROUPS.items():
    R[g], PSD[g] = {}, {}
    for run in spec["sim"]:
      lst, ps = [], []
      for e in range(a.envs):
        f = Path(a.csv) / f"{run}_env{e}.csv"
        if not f.exists():
          continue
        res = RW.analyze({"label": f"{run}/{e}", "policy": f.name}, Path(a.csv))
        res["flat"] = RW.flat(res)
        x, p = extract(res)
        lst.append(x)
        ps.append(p)
      lab = simlabel(run)
      R[g][lab] = dict(kind="sim", run=run, n=len(lst), per_env=lst)
      R[g][lab]["band38_true200_frac"] = band200(Path(a.npz) / f"{run}.npz")
      f0 = ps[0][0]
      PSD[g][lab] = (f0, np.mean([p[1] for p in ps], 0))
    for lab, fn in spec["real"]:
      res = RW.analyze({"label": lab, "policy": fn}, Path(a.real))
      res["flat"] = RW.flat(res)
      x, p = extract(res)
      R[g][lab] = dict(kind="real", file=fn, n=1, per_env=[x])
      PSD[g][lab] = p
  json.dump(R, open(a.out + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=lambda o: float(o))

  def cell(src, key, nd=2):
    m, s = agg(src["per_env"], key)
    return f"{m:.{nd}f}" if src["n"] == 1 else f"{m:.{nd}f} ± {s:.{nd}f}"

  L = []
  for g, spec in GROUPS.items():
    srcs = R[g]
    names = list(srcs)
    L.append(f"### {spec['title']}\n")
    L.append("| metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    rows = [
      ("policy interval median / p95 [ms]", lambda s: f"{agg(s['per_env'], 'interval_median_ms')[0]:.1f} / {agg(s['per_env'], 'interval_p95_ms')[0]:.1f}"),
      ("gait frequency f_gait [Hz]", lambda s: cell(s, "f_gait_hz")),
      ("stride period, phase average [s]", lambda s: cell(s, "stride_period_s", 3)),
      ("dominant frequency of summed joint PSD [Hz]", lambda s: cell(s, "f_dom_hz", 1)),
      ("strongest peak above 3 Hz [Hz]", lambda s: cell(s, "f_hi_hz", 1)),
      ("3-8 Hz peak of summed PSD [Hz]", lambda s: cell(s, "peak38_hz", 1)),
      ("3-8 Hz band power, sum of 12 joints [deg^2]", lambda s: cell(s, "band38_deg2", 3)),
      ("3-8 Hz share of joint variance (0.2-10 Hz)", lambda s: cell(s, "band38_frac", 3)),
      ("sim only: 3-8 Hz share, true 200 Hz angle (0.2-25 Hz)", lambda s: (f"{np.mean(s['band38_true200_frac']):.3f} ± {np.std(s['band38_true200_frac']):.3f}" if s["kind"] == "sim" else "n/a")),
      ("action step RMS (mean of 12 joints, per call)", lambda s: cell(s, "act_step_rms", 3)),
      ("heading change over the log [deg] (gyro integral)", lambda s: cell(s, "heading_deg", 1)),
      ("mean yaw rate [deg/s]", lambda s: cell(s, "yaw_rate_deg_s", 2)),
      ("mean yaw rate minus commanded [deg/s]", lambda s: cell(s, "yaw_err_deg_s", 2)),
      ("log length [s]", lambda s: cell(s, "duration_s", 1)),
    ]
    for nm, fn in rows:
      L.append(f"| {nm} | " + " | ".join(fn(srcs[n]) for n in names) + " |")
    L.append("")
    L.append("Per joint: amplitude (std of pos, deg) / RMS of (tgt - pos) (deg) / mean of (tgt - pos) (deg)\n")
    L.append("| joint | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    J = srcs[names[0]]["per_env"][0]["joints"]
    for j, jn in enumerate(J):
      cells = []
      for n in names:
        pe = srcs[n]["per_env"]
        v = [np.mean([x[k][j] for x in pe]) for k in ("amp_deg", "trk_rms_deg", "trk_mean_deg")]
        cells.append(f"{v[0]:.1f} / {v[1]:.1f} / {v[2]:+.1f}")
      L.append(f"| {jn} | " + " | ".join(cells) + " |")
    L.append("")
    if g == "vx0":
      L.append("Mean of (pos - default pose) per joint [deg], standing command\n")
      L.append("| joint | " + " | ".join(names) + " |")
      L.append("|---|" + "---|" * len(names))
      for j, jn in enumerate(J):
        L.append(f"| {jn} | " + " | ".join(f"{np.mean([x['pos_minus_default_deg'][j] for x in srcs[n]['per_env']]):+.1f}" for n in names) + " |")
      L.append("")
  open(a.out + ".md", "w", encoding="utf-8").write("\n".join(L))
  print("\n".join(L))

  if a.fig_prefix:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"sim 20 ms": "tab:blue", "sim 52 ms": "tab:orange", "sim real-interval": "tab:green", "real vx0 gain1": "k", "real vx0 gain0.6": "0.5",
           "real vx0.3": "k", "real vx0.5": "k", "real wz0.5": "k"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (g, spec) in zip(axs.ravel(), GROUPS.items()):
      for n, (f, P) in PSD[g].items():
        m = (f > 0.2) & (f <= 12)
        ax.semilogy(f[m], P[m], color=col[n], lw=2.0 if n.startswith("real") else 1.3, ls="-" if not n.endswith("0.6") else "--", label=n)
      ax.axvspan(3, 8, color="y", alpha=0.12)
      ax.set_title(spec["title"])
      ax.grid(alpha=0.3, which="both")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel("Joint angle PSD, sum of 12 joints [deg^2/Hz]")
    axs[1, 0].set_ylabel("Joint angle PSD, sum of 12 joints [deg^2/Hz]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Joint-angle spectra on a 50 ms grid: simulation (mean of 8 envs) vs real log. Yellow = 3-8 Hz band.", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.fig_prefix + "psd.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    labs, vals, errs, cols = [], [], [], []
    for g in GROUPS:
      for n, s in R[g].items():
        m, sd = agg(s["per_env"], "yaw_err_deg_s")
        labs.append(f"{g}\n{n.replace('sim real-interval', 'sim real-int.')}")
        vals.append(m), errs.append(sd), cols.append(col[n])
    ax.bar(range(len(vals)), vals, yerr=errs, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs, rotation=70, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Mean yaw rate (gyro z) minus commanded [deg/s]")
    ax.set_title("Yaw drift per condition (sim: mean +- std of 8 envs; black/grey = real)", fontsize=9)
    ax = axs[1]
    jn = ["R ankle_a", "R ankle_b", "L ankle_a", "L ankle_b"]
    ix = [4, 5, 10, 11]
    w = 0.16
    srcs = list(R["vx0"])
    for k, n in enumerate(srcs):
      v = [np.mean([x["trk_mean_deg"][i] for x in R["vx0"][n]["per_env"]]) for i in ix]
      ax.bar(np.arange(4) + (k - 2) * w, v, w, color=col[n], label=n, alpha=0.9 if n.startswith("sim") else 1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(jn)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Mean of (target - measured) [deg]")
    ax.set_title("Ankle crank steady offset, command vx 0")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.fig_prefix + "drift_ankle.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
  main()
