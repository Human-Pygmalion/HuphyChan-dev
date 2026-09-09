#!/usr/bin/env python3
"""모터가 말하는 값과 실제 값을 칸마다 그린다. **이 컴퓨터에서 돌린다.**

2026-09-09. 오늘 하루에 세 칸이 따로따로 문제를 드러냈습니다. 발견될 때마다 놀라지 않으려면
한 장에 모아 두어야 합니다.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VMAX = 859.4
QUANT = 240.0 / 4095.0     # 토크 칸 한 계단 (±120 뉴턴미터를 12비트로)


def rd(p):
  with open(p) as f:
    return list(csv.DictReader(f))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--data", default="data")
  ap.add_argument("--out", default="../docs/img/motor_report_vs_truth.png")
  a = ap.parse_args(argv)

  fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.0))

  # --- 1) 속도 ------------------------------------------------------------
  sp = rd(os.path.join(a.data, "rvt_spin_4_hot.csv"))
  vt = [float(x["vel_true"]) for x in sp]
  vr = [float(x["vel_rep"]) for x in sp]
  axs[0].plot([0, max(vt) * 1.05], [0, max(vt) * 1.05], ":", color="#95a5a6",
              label="if it told the truth")
  axs[0].plot(vt, vr, "o-", color="#2c3e50", lw=1.8, ms=7, label="what it reports")
  axs[0].axhline(VMAX, color="#c0392b", ls="--", lw=1.4)
  for x, y in zip(vt, vr):
    if y < x * 0.98:
      axs[0].annotate(f"{y/x:.2f}x", (x, y), textcoords="offset points", xytext=(4, -13),
                      fontsize=9, color="#c0392b")
  axs[0].text(0.02, VMAX, " stops here: 15.0 rad/s", color="#c0392b", fontsize=9,
              va="bottom", transform=axs[0].get_yaxis_transform())
  axs[0].set_xlabel("true speed, from position [deg/s]")
  axs[0].set_ylabel("speed the motor reports [deg/s]")
  axs[0].set_title("SPEED\nexact below 859, then flat while the shaft keeps going")
  axs[0].legend(fontsize=9, loc="upper left"); axs[0].grid(alpha=0.3)

  # --- 2) 토크: 평균은 맞는가 ---------------------------------------------
  st = [x for x in rd(os.path.join(a.data, "rvt_still_4_hot.csv")) if x["still"] == "1"]
  cx = [float(x["cmd_nm"]) for x in st]
  cy = [float(x["rep_nm"]) for x in st]
  axs[1].plot([0, 1.3], [0, 1.3], ":", color="#95a5a6", label="if it told the truth")
  axs[1].plot([0, 1.3], [-QUANT, 1.3 - QUANT], "-", color="#8e44ad", lw=1.2,
              label=f"one step low ({QUANT:.3f} Nm)")
  axs[1].scatter(cx, cy, c="#2c3e50", s=40, zorder=3, label="held still (true = commanded)")
  tt = [float(x["tau_true"]) for x in sp]
  tr = [float(x["tau_rep"]) for x in sp]
  axs[1].scatter(tt, tr, c="#16a085", marker="s", s=45, zorder=3,
                 label="spinning (true = friction)")
  axs[1].set_xlabel("true torque [Nm]")
  axs[1].set_ylabel("torque the motor reports, averaged [Nm]")
  axs[1].set_title("TORQUE, averaged over ~1 s\nabout one quantisation step low, "
                   "otherwise straight")
  axs[1].legend(fontsize=8.5, loc="upper left"); axs[1].grid(alpha=0.3)
  axs[1].set_xlim(0, 1.3); axs[1].set_ylim(-0.1, 1.3)

  # --- 3) 토크: 한 점만 보면 --------------------------------------------
  sets = []
  for pat, lab, col, true_v in (
      ("fcurve_4_1.02_p_0_hot.csv", "commanded 1.02 Nm, spinning", "#16a085", None),
      ("coastmax_4_p_0_hot.csv", "commanded 0 Nm, coasting", "#c0392b", 0.0)):
    p = os.path.join(a.data, pat)
    if not os.path.exists(p):
      continue
    r = rd(p)
    t = [float(x["t_s"]) for x in r]
    tau = [float(x["tau_rep_nm"]) for x in r]
    if true_v is None:
      sel = [v for tt_, v in zip(t, tau) if tt_ > t[-1] - 3.0]
    else:
      vv = [abs(float(x["vel_deg_s"])) for x in r]
      v0 = max(vv[:20])
      sel = [v for v, s in zip(tau, vv) if 0.2 * v0 <= s <= 0.95 * v0]
    sets.append((sel, lab, col))
  for sel, lab, col in sets:
    axs[2].hist(sel, bins=45, alpha=0.55, color=col,
                label=f"{lab}\nmean {statistics.mean(sel):+.2f}, "
                      f"spread {statistics.pstdev(sel):.2f} Nm")
  axs[2].axvline(0, color="k", lw=1.0)
  axs[2].set_xlabel("torque the motor reports, single samples [Nm]")
  axs[2].set_ylabel("how many samples")
  axs[2].set_title("TORQUE, one sample at a time\n"
                   "swings +-0.6 Nm either way - a single reading says nothing")
  axs[2].legend(fontsize=8, loc="upper right"); axs[2].grid(alpha=0.3)

  fig.suptitle("RobStride RS04: what the motor says vs what is actually happening "
               "(bare shaft, 39-44 C)", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
