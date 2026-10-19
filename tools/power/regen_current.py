#!/usr/bin/env python3
"""regen_current.py -- battery charge (regeneration) current from simulated joint power.

Question (2026-10-19): what regeneration (charge) current must the battery accept?

Input: measure_full .npz files (analysis/out/<tag>_fc.npz / _fcp.npz).  They hold, per joint,
Pmech_<joint> = tau * omega [W] sampled at the policy rate (50 Hz).
Actuated joints (the 12 motors): hip_pitch, hip_roll, hip_yaw, knee, crank_A, crank_B per leg.
The ankle_pitch/roll and rod_* joints are passive linkage joints: counting them would count
the crank power twice, so they are excluded.

Model (stated, not hidden):
  bus power        P_bus(t) = sum_j P_mech_j(t) + losses
  net regeneration P_regen(t) = max(0, -P_bus(t))      (only the NET reaches the battery: one
                                                        joint braking while another drives is
                                                        absorbed on the bus)
  LOSSLESS upper bound: losses = 0.  Real copper/inverter losses only REDUCE regeneration, so the
  numbers here are an upper bound for the same motion.
  charge current   I(t) = P_regen(t) / V_batt

Windows: the battery spec is usually "continuous" and "pulse (e.g. 10 s)" charge current.  So the
report gives the max of the moving average of P_regen over 0.02 (one sample), 0.1, 1, 10 s,
plus percentiles of the per-sample value and the share of time with net regeneration.

Limits: 50 Hz samples miss events shorter than 20 ms; falls / hard landings / emergency
stops are NOT in these measurements (command-box walking + pushes only).

  python3 tools/power/regen_current.py mujoco-sim/mjlab/analysis/out/v10afinal_fc.npz ... \
      [--vbatt 48] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ACTUATED = ["hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint", "crank_A_joint", "crank_B_joint"]
WINDOWS_S = [0.02, 0.1, 1.0, 10.0]


def movavg_max(x: np.ndarray, n: int) -> float:
    if n <= 1:
        return float(x.max())
    if len(x) < n:
        return float(x.mean())
    c = np.cumsum(np.concatenate([[0.0], x]))
    return float(((c[n:] - c[:-n]) / n).max())


def analyse(path: str, vbatt: float) -> dict:
    d = np.load(path)
    t = d["time"]
    dt = float(np.median(np.diff(t)))
    joints = [f"{s}_{j}" for s in ("L", "R") for j in ACTUATED]
    P = np.stack([d[f"Pmech_{j}"] for j in joints])          # (12, T)
    bus = P.sum(axis=0)                                       # net mechanical power [W]
    regen = np.clip(-bus, 0.0, None)                          # net regeneration [W]
    per_joint_regen_peak = {j: float(np.clip(-d[f"Pmech_{j}"], 0, None).max()) for j in joints}
    out = {
        "file": os.path.basename(path), "samples": int(len(t)), "duration_s": float(t[-1] - t[0]), "dt_s": dt,
        "vbatt_v": vbatt,
        "drive_power_w": {"mean": float(np.clip(bus, 0, None).mean()), "p99": float(np.percentile(bus, 99)), "max": float(bus.max())},
        "regen_time_share_pct": float(100.0 * (regen > 0).mean()),
        "regen_power_w": {"p50_when_regen": float(np.median(regen[regen > 0])) if (regen > 0).any() else 0.0,
                          "p99": float(np.percentile(regen, 99)), "p99_9": float(np.percentile(regen, 99.9)),
                          "max": float(regen.max())},
        "regen_power_window_max_w": {f"{w:g}s": movavg_max(regen, max(1, int(round(w / dt)))) for w in WINDOWS_S},
        "sum_of_joint_regen_peak_w": float(sum(per_joint_regen_peak.values())),
        "per_joint_regen_peak_w": per_joint_regen_peak,
    }
    out["regen_current_a"] = {
        "p99": out["regen_power_w"]["p99"] / vbatt, "p99_9": out["regen_power_w"]["p99_9"] / vbatt,
        "max_sample": out["regen_power_w"]["max"] / vbatt,
        **{f"window_{k}": v / vbatt for k, v in out["regen_power_window_max_w"].items()},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--vbatt", type=float, default=48.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = [analyse(p, a.vbatt) for p in a.npz]
    for r in res:
        c = r["regen_current_a"]
        print(f"{r['file']:24s} {r['duration_s']:7.0f}s  regen time {r['regen_time_share_pct']:5.1f}%  "
              f"P_regen p99 {r['regen_power_w']['p99']:6.1f} W  max {r['regen_power_w']['max']:6.1f} W  | "
              f"I @{a.vbatt:g}V: p99 {c['p99']:.2f}  p99.9 {c['p99_9']:.2f}  max {c['max_sample']:.2f}  "
              f"0.1s {c['window_0.1s']:.2f}  1s {c['window_1s']:.2f}  10s {c['window_10s']:.2f} A  "
              f"| drive max {r['drive_power_w']['max']:.0f} W")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
