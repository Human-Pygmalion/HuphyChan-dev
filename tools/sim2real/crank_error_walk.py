"""Ankle crank target-minus-actual error and torque while walking (docs/168 check 5).

Reads the rollouts `mujoco-sim/mjlab/analysis/gait_kinematics_probe.py` writes (the walk npz files of
`analysis/matched_set.sh`) and reports, for the four ankle crank servos pooled
(R/L x A/B), the absolute position error |qtarget - q| in degrees (P50, P99) and the
absolute torque |tau| P99 in N*m. The action window is +-62 deg, so an error P99 close to 62
means the target sits at the window edge and the servo cannot ask for more.

  (cwd = mujoco-sim/mjlab; the script only needs numpy)
  CUDA_VISIBLE_DEVICES="" .venv/bin/python3 <repo>/tools/sim2real/crank_error_walk.py \
      --npz 'analysis/out/m_trialv11a_final_walk_s*.npz' --label trial --warm 2.0 [--out x.json]
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

CRANKS = [f"{s}_crank_{ab}_joint" for s in "RL" for ab in "AB"]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True, help="glob of walk rollouts")
  ap.add_argument("--label", required=True)
  ap.add_argument("--warm", type=float, default=2.0, help="seconds dropped at the start")
  ap.add_argument("--out", default="")
  a = ap.parse_args()
  files = sorted(glob.glob(a.npz))
  assert files, f"no files match {a.npz}"
  err_all, tau_all, per_file = [], [], []
  for f in files:
    d = np.load(f)
    keep = d["time"] >= d["time"][0] + a.warm
    err = np.concatenate([np.abs(d[f"qtarget_{j}"] - d[f"q_{j}"])[keep] for j in CRANKS]) * 180 / np.pi
    tau = np.concatenate([np.abs(d[f"tau_{j}"])[keep] for j in CRANKS])
    err_all.append(err)
    tau_all.append(tau)
    per_file.append(dict(file=f, err_p50=float(np.percentile(err, 50)), err_p99=float(np.percentile(err, 99)),
                         tau_p99=float(np.percentile(tau, 99)), n=int(len(err))))
  err = np.concatenate(err_all)
  tau = np.concatenate(tau_all)
  res = dict(label=a.label, warm_s=a.warm, files=per_file, pooled=dict(
      err_p50_deg=float(np.percentile(err, 50)), err_p99_deg=float(np.percentile(err, 99)),
      err_max_deg=float(err.max()), tau_p99_nm=float(np.percentile(tau, 99)),
      tau_max_nm=float(tau.max()), n=int(len(err))))
  for p in per_file:
    print(f"{a.label} {p['file'].split('/')[-1]}: err P50 {p['err_p50']:.1f} P99 {p['err_p99']:.1f} deg, tau P99 {p['tau_p99']:.1f} N*m")
  r = res["pooled"]
  print(f"{a.label} POOLED: err P50 {r['err_p50_deg']:.1f} P99 {r['err_p99_deg']:.1f} max {r['err_max_deg']:.1f} deg | "
        f"tau P99 {r['tau_p99_nm']:.1f} max {r['tau_max_nm']:.1f} N*m (n={r['n']})")
  if a.out:
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
  main()
