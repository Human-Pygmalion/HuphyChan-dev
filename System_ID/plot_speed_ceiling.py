#!/usr/bin/env python3
"""속도가 어디서 막히는지 그린다. **이 컴퓨터에서 돌린다.**

2026-09-09. "600 도/s 가 최고속도인가"라는 물음에서 시작해 두 개의 천장을 찾았습니다.

* **시키는 쪽 천장** — MIT 방식 명령의 속도 칸이 담을 수 있는 최대가 15.0 라디안/s
  = 859 도/s 입니다. 비례 제어의 처짐까지 빼면 실제로는 825 도/s 에서 멈춥니다.
* **읽는 쪽 천장** — 모터가 **보고하는** 속도도 같은 자리에서 잘립니다. 실제로 1362 도/s 로
  돌고 있어도 859 라고 답합니다.

두 번째가 더 위험합니다. 첫 번째는 "더 못 시킨다"이지만, 두 번째는 **"틀린 값을 준다"**
이기 때문입니다. 위치를 이어 붙여 미분하면 진짜 속도가 나옵니다.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VMAX_ENC = 859.4     # 15.0 라디안/s


def unwrap(raw):
  out = [0.0]
  for i in range(1, len(raw)):
    d = raw[i] - raw[i - 1]
    while d > 720.0:
      d -= 1440.0
    while d < -720.0:
      d += 1440.0
    out.append(out[-1] + d)
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--probe", default="data/speed_probe.csv")
  ap.add_argument("--warm", default="data/warmup_pi_maxspeed.csv")
  ap.add_argument("--out", default="../docs/img/motor_speed_ceiling.png")
  a = ap.parse_args(argv)

  fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))

  # --- 1) 속도 명령으로 올려 본 것 --------------------------------------
  with open(a.probe) as f:
    r = [(float(x["t_s"]), float(x["cmd_vel_deg_s"]), abs(float(x["vel_deg_s"])))
         for x in csv.DictReader(f)]
  cmds = sorted({x[1] for x in r})
  xs, ys = [], []
  for c in cmds:
    d = [x for x in r if x[1] == c]
    d = d[len(d) // 2:]                       # 자리잡은 뒤 절반만
    if len(d) > 100:
      xs.append(c); ys.append(sum(x[2] for x in d) / len(d))
  axs[0].plot([0, max(cmds) * 1.05], [0, max(cmds) * 1.05], ":", color="#95a5a6",
              label="if it followed exactly")
  axs[0].plot(xs, ys, "o-", color="#2c3e50", lw=1.8, ms=6, label="what it actually did")
  axs[0].axhline(VMAX_ENC, color="#c0392b", ls="--", lw=1.6)
  axs[0].text(0.02, VMAX_ENC, " command field runs out here: 15.0 rad/s = 859 deg/s",
              color="#c0392b", fontsize=9, va="bottom",
              transform=axs[0].get_yaxis_transform())
  axs[0].set_xlabel("commanded speed [deg/s]")
  axs[0].set_ylabel("speed actually reached [deg/s]")
  axs[0].set_title("Ceiling 1 - you cannot ask for more\n"
                   "flat from 900 on, while torque stayed at only 0.84 Nm")
  axs[0].legend(fontsize=9, loc="upper left"); axs[0].grid(alpha=0.3)

  # --- 2) 순수 토크로 밀었을 때 보고 속도 대 진짜 속도 --------------------
  with open(a.warm) as f:
    r = [(float(x["t_s"]), float(x["pos_deg"]), abs(float(x["vel_deg_s"])),
          abs(float(x["vel_true_deg_s"]))) for x in csv.DictReader(f)]
  r = [x for x in r if x[0] < 54.0]
  pos = unwrap([x[1] for x in r])
  t = [x[0] for x in r]
  # 0.3초 창으로 다시 계산 (기록된 값과 같은 방식, 그림용으로 다시 매끄럽게)
  true, i, k = [], 0, 0
  win = 0.3
  for i in range(len(t)):
    while k < len(t) and t[k] - t[i] < win:
      k += 1
    true.append((pos[min(k, len(t) - 1)] - pos[i]) / max(1e-6, t[min(k, len(t) - 1)] - t[i]))
  axs[1].plot(t, [x[2] for x in r], color="#c0392b", lw=1.2,
              label="what the motor reports")
  axs[1].plot(t, [abs(x) for x in true], color="#2c3e50", lw=1.4,
              label="true speed, from position")
  axs[1].axhline(VMAX_ENC, color="#c0392b", ls="--", lw=1.2, alpha=0.6)
  axs[1].annotate("reported value stops here\nwhile the motor keeps speeding up",
                  xy=(30, VMAX_ENC), xytext=(20, 450), fontsize=9, color="#c0392b",
                  arrowprops=dict(arrowstyle="->", color="#c0392b"))
  axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("speed [deg/s]")
  axs[1].set_title("Ceiling 2 - and it lies about it\n"
                   "pure torque 2.0 Nm; settles at 1362 deg/s = 23.8 rad/s\n"
                   "(run ended on the 44 C stop after 56 s)")
  axs[1].legend(fontsize=9, loc="lower right"); axs[1].grid(alpha=0.3)

  fig.suptitle("RobStride RS04: two separate speed ceilings, both in the message format",
               fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  fig.savefig(a.out, dpi=130)
  print(f"그림: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
