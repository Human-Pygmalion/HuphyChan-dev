#!/usr/bin/env python3
"""Aggregate the deploy-fix simulation (docs/sim2real/2026-10-30_union49999_deploy_fix_sim.md): per variant, mean over the simulated envs of
stance width (all / stand / walk), step width, hip_roll abduction (measured), target-measured, crank A/B error, falls, hip_roll torque.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/deploy_fix_report.py --xml <xml> --real-csv <1028_teleop_ob.csv> --sim-dir <dir with csv/ and *.npz> \
      --prefix F --seg 9:29 --out-json <json> --out-fig <png> [--tmp dir] [--cache dir]
Rows after the first fall of an env are already dropped from the CSV (sim_walk_export_csv); torque uses the same rows.
For rolloutN variants the recorded target INCLUDES the +-N deg offset, so target-measured includes it too ("policy" columns remove it).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from stance_fk import FK
from stance_report import cut_csv, run_group, flat
from stance_analyze import summarise

LIMIT = 120.0  # hip_roll torque limit [N m]


def torque_stats(npz):
  d = np.load(npz)
  term = d["terminated"]
  vals = {0: [], 1: []}
  for e in range(term.shape[1]):
    n = int(np.argmax(term[:, e])) + 1 if term[:, e].any() else term.shape[0]
    for si in (0, 1):
      vals[si].append(np.abs(d["tau"][:n, e, 1 + 6 * si]))
  out = {}
  for si, s in enumerate("RL"):
    a = np.concatenate(vals[si])
    out[s] = {"p95": float(np.percentile(a, 95)), "max": float(a.max())}
  out["p95"] = max(out["R"]["p95"], out["L"]["p95"])
  out["max"] = max(out["R"]["max"], out["L"]["max"])
  return out


def row(fl, falls, fall_t, tq, offset_deg):
  a, s, w = fl["all"], fl["stand"], fl["walk"]
  cm = lambda x: float(x) * 100.0  # noqa: E731
  r = dict(width_all=cm(a["width_pos"]), width_stand=cm(s["width_pos"]), width_walk=cm(w["width_pos"]),
           step_width=cm(fl["step_width_pos"]),
           abd_R=a["right_abd_pos"], abd_L=a["left_abd_pos"],
           err_R=a["right_roll_err"], err_L=a["left_roll_err"],
           err_policy_R=a["right_roll_err"] - offset_deg, err_policy_L=a["left_roll_err"] + offset_deg,
           crankA_R=a["j4_err"], crankB_R=a["j5_err"], crankA_L=a["j10_err"], crankB_L=a["j11_err"],
           falls=falls, fall_times_s=fall_t, torque=tq)
  return r


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--real-csv", required=True)
  ap.add_argument("--sim-dir", required=True)
  ap.add_argument("--prefix", required=True)
  ap.add_argument("--seg", default="9:29")
  ap.add_argument("--out-json", required=True)
  ap.add_argument("--out-fig", default="")
  ap.add_argument("--tmp", default="/tmp/deploy_fix_report")
  ap.add_argument("--cache", default="")
  a = ap.parse_args()
  fk = FK(a.xml)
  tmp = Path(a.tmp)
  tmp.mkdir(parents=True, exist_ok=True)
  t0, t1 = map(float, a.seg.split(":"))
  seg = tmp / "real_seg.csv"
  cut_csv(a.real_csv, t0, t1, seg)
  real_o = summarise("real_seg", seg, fk, None)[0]
  real_fl = {rg: real_o["regimes"][rg] for rg in real_o["regimes"]}
  real_fl["step_width_pos"] = real_o["step_width_pos"]["mean"]
  res = {"seg": a.seg, "real": row(real_fl, None, None, None, 0.0), "variants": {}}
  names = sorted({p.name.rsplit("_env", 1)[0] for p in (Path(a.sim_dir) / "csv").glob(f"{a.prefix}_*_env*.csv")} )
  names = [n for n in names if not n.endswith("_true")]
  for name in names:
    v = name[len(a.prefix) + 1:]
    per, falls, _ = run_group(fk, a.sim_dir, name, a.cache or None)
    meta = json.loads((Path(a.sim_dir) / f"{name}.json").read_text())
    tq = torque_stats(Path(a.sim_dir) / f"{name}.npz")
    off = sum(float(x) for x in re.findall(r"rollout(-?\d+(?:\.\d+)?)", v))
    res["variants"][v] = row(flat(per), falls, meta["fall_times_s"], tq, off)
    res["variants"][v]["n_env"] = len(per)
    r = res["variants"][v]
    print(f"{v:24s} width all/stand/walk {r['width_all']:.1f}/{r['width_stand']:.1f}/{r['width_walk']:.1f}  falls {falls}  tau p95 {tq['p95']:.0f} max {tq['max']:.0f}", flush=True)
  Path(a.out_json).write_text(json.dumps(res, indent=1))

  if a.out_fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = res["variants"]
    groups = [
      ("Condition: hip_roll Kp x0.7 (\"Kp low\")", ["kpR_0.7", "kpR_0.7+kpR_1.43", "kpR_0.7+kpR_1.7", "kpR_0.7+rollout1", "kpR_0.7+rollout2", "kpR_0.7+rollout3"]),
      ("Condition: pelvis +6 kg (\"heavy\")", ["payload6", "payload6+kpR_1.2", "payload6+kpR_1.4", "payload6+rollout1", "payload6+rollout2", "payload6+rollout3"]),
      ("Nominal robot + fix alone", ["base", "kpR_1.2", "kpR_1.4", "rollout1", "rollout2", "rollout3"]),
    ]
    def label(v):
      p = v.split("+")
      if len(p) == 1:
        return {"base": "no change", "kpR_0.7": "Kp x0.7\n(no fix)", "payload6": "+6 kg\n(no fix)"}.get(v, v.replace("kpR_", "Kp x").replace("rollout", "roll-out\n+") + (" deg" if v.startswith("rollout") else ""))
      f = p[1]
      return f.replace("kpR_", "fix Kp x").replace("rollout", "fix roll-out\n+") + (" deg" if f.startswith("rollout") else "")
    col = {"all": "#2a6fb0", "walk": "#d98a1c"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    for ax, (title, vs) in zip(axs, groups):
      x = np.arange(len(vs))
      w_all = [V[v]["width_all"] if v in V else np.nan for v in vs]
      w_stand = [V[v]["width_stand"] if v in V else np.nan for v in vs]
      w_walk = [V[v]["width_walk"] if v in V else np.nan for v in vs]
      ax.bar(x - 0.27, w_all, 0.27, color="#2a6fb0", label="all rows")
      ax.bar(x, w_stand, 0.27, color="#7fb069", label="standing")
      ax.bar(x + 0.27, w_walk, 0.27, color="#d98a1c", label="walking")
      for xi, v in zip(x, vs):
        if v in V and V[v]["falls"]:
          ax.text(xi, 3.0, f"falls {V[v]['falls']}/{V[v]['n_env']}", ha="center", va="bottom", fontsize=8, color="#b00020", rotation=90)
      ax.axhline(22.0, color="#b00020", ls="--", lw=1.4, label="real robot, same 20 s: 22.0 cm")
      ax.axhline(24.7, color="#444444", ls=":", lw=1.4, label="default pose geometry: 24.7 cm")
      ax.axhline(25.0, color="#2e7d32", ls="-", lw=0.8, alpha=0.7, label="25 cm goal")
      ax.set_xticks(x)
      ax.set_xticklabels([label(v) for v in vs], fontsize=8)
      ax.set_title(title, fontsize=10)
      ax.set_ylim(0, 40)
      ax.grid(axis="y", alpha=0.25)
    axs[0].set_ylabel("Stance width (foot centre to foot centre, pelvis frame) [cm]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Deploy fixes for the narrow stance, simulation, log 1 t = 9-29 s, 4 envs, nominal mode (policy terrain10_union_dr_49999)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(a.out_fig, dpi=140)
