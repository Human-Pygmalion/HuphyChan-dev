#!/usr/bin/env python3
"""**+10도로 갈 때와 −10도로 갈 때가 같은가.** 한쪽씩 친 계단을 방향별로 갈라 견준다.

2026-09-10 · 사용자 지시: **"Step Input 실험 +10, -10 도로 여러번 해서 비교해 줘."**

왜 이걸 보는가
--------------
마찰은 **방향만 거스르는 힘**입니다. 그러니 이론상 **어느 쪽으로 가든 똑같이** 방해해야
합니다. 관성도 점성도 방향을 안 가립니다. 그래서 우리가 쓰는 모형이 맞다면 **+10 으로 가는
것과 −10 으로 가는 것이 거울처럼 같아야** 합니다.

다르다면 그 차이가 곧 **모형에 없는 무엇**입니다. 후보는 셋입니다.

* **중력** — 축에 매달린 것이 있으면 한쪽은 도와주고 한쪽은 방해합니다
* **코깅** — 자석과 철심이 만드는 톱니 같은 힘. 축의 자리에 따라 다릅니다
* **베어링 편심** — 한쪽으로 쏠려 끼는 것

네 가지를 따로 봅니다
---------------------
    0 → +10   플러스로 나감        +10 → 0   플러스에서 돌아옴
    0 → −10   마이너스로 나감      −10 → 0   마이너스에서 돌아옴

"나감"과 "돌아옴"을 나누는 이유는, 돌아올 때는 **직전까지 반대로 밀고 있던 마찰이 방향을
바꾸는** 순간이 끼기 때문입니다. 나갈 때와 조건이 다릅니다.

견주는 방법은 **부호를 뒤집어 겹치는 것**입니다. −10 쪽 곡선의 부호를 뒤집어 +10 쪽 위에
포개면, 남는 틈이 곧 방향에 따른 쏠림입니다.

    pd_updown.py --motor 4 --family RS04
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pd_2x2 import BEST, simulate  # noqa: E402

DEG = math.pi / 180.0
PRE, POST = 0.12, 1.35          # 모서리 앞뒤로 잘라 볼 시간 [초]

LEGS = [("out+", "0 to +A   (going out the plus way)", "#00a878", +1),
        ("out-", "0 to -A   (going out the minus way)", "#7b3fbf", -1),
        ("back+", "+A to 0   (coming back from plus)", "#e8443a", -1),
        ("back-", "-A to 0   (coming back from minus)", "#f2a007", +1)]


def load_runs(pattern):
  out = []
  for p in sorted(globmod.glob(pattern)):
    with open(p) as f:
      r = list(csv.DictReader(f))
    if len(r) < 200:
      continue
    off = float(r[0]["target_deg"])
    out.append((np.array([float(x["t_s"]) for x in r]),
                np.array([float(x["target_deg"]) for x in r]) - off,
                np.array([float(x["target_vel_deg_s"]) for x in r]),
                np.array([float(x["pos_deg"]) for x in r]) - off))
  return out


def grid(runs, dt=0.001):
  tmax = min(r[0][-1] for r in runs)
  ts = np.arange(0, tmax, dt)
  P, T, V = [], None, None
  for t, tgt, tgv, pos in runs:
    P.append(np.interp(ts, t, pos))
    if T is None:
      T, V = np.interp(ts, t, tgt), np.interp(ts, t, tgv)
  return ts, T, V, np.array(P)


def legs_of(ts, tgt, amp):
  """명령이 어느 층에서 어느 층으로 넘어갔는지로 네 가지를 가른다.

  층은 −A, 0, +A 셋뿐이므로, 표본마다 **가장 가까운 층**으로 이름표를 붙인 다음 이름표가
  바뀌는 자리를 모서리로 봅니다. 시각 격자에 다시 올릴 때 모서리가 번져도 안 흔들립니다."""
  lv = np.array([-amp, 0.0, amp])
  lab = np.abs(tgt[:, None] - lv[None, :]).argmin(axis=1)
  ed = np.nonzero(np.diff(lab))[0]
  out = {k: [] for k, _, _, _ in LEGS}
  for k in ed:
    a, b = lab[k], lab[min(k + 2, len(lab) - 1)]
    key = {(1, 2): "out+", (1, 0): "out-", (2, 1): "back+", (0, 1): "back-"}.get((a, b))
    if key:
      out[key].append((k, float(lv[a]), float(lv[b])))
  return out


def stack(ts, sig, edges, dt=0.001):
  """모서리마다 앞뒤를 잘라 **같은 자리에 포개** 놓는다. 돌려주는 것은 (시간, 조각들)."""
  n_pre, n_post = int(PRE / dt), int(POST / dt)
  rel = (np.arange(-n_pre, n_post) * dt)
  segs = []
  for k, lo, hi in edges:
    if k - n_pre < 0 or k + n_post > len(sig):
      continue
    segs.append((sig[k - n_pre:k + n_post] - lo) / (hi - lo))   # 0 에서 1 로 가도록 정규화
  return rel, (np.array(segs) if segs else np.zeros((0, len(rel))))


def metrics(rel, seg):
  """포개 놓은 조각에서 세 숫자. 값은 이미 0→1 로 정규화돼 있다."""
  if len(seg) == 0:
    return (float("nan"),) * 3
  ov, rise, ss = [], [], []
  for y in seg:
    after = rel > 0
    ov.append(max(0.0, float(np.max(y[after])) - 1.0) * 100)
    got = np.nonzero(after & (y >= 0.9))[0]
    rise.append((rel[got[0]] * 1e3) if len(got) else float("nan"))
    tail = rel > POST - 0.4
    ss.append(abs(float(np.median(y[tail])) - 1.0) * 100)
  f = lambda v: float(np.nanmean(v)) if len(v) else float("nan")   # noqa: E731
  return f(ov), f(rise), f(ss)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--amp", type=float, default=10.0)
  ap.add_argument("--tag", default="pm")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--out-dir", default="../docs/img")
  a = ap.parse_args(argv)

  pat = f"pd_pmstep_{a.motor}_kp{a.kp:g}_kd{a.kd:g}_a{a.amp:g}_r*_{a.tag}.csv"
  runs = load_runs(os.path.join(a.data, pat))
  if not runs:
    print(f"자료가 없습니다: {pat}")
    return 1
  ts, tgt, tgv, P = grid(runs)
  prm = BEST[a.family]
  sim = simulate(ts, tgt, tgv, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3, **prm)
  ed = legs_of(ts, tgt, a.amp)
  real_mean = P.mean(axis=0)

  print(f"{a.family} (모터 {a.motor}) · {len(runs)}회 · 이득 kp={a.kp} kd={a.kd} · "
        f"진폭 ±{a.amp:g}도")
  res = {}
  for key, label, col, sgn in LEGS:
    rel, segs_real = stack(ts, real_mean, ed[key])
    _, segs_sim = stack(ts, sim, ed[key])
    per_run = [stack(ts, p, ed[key])[1] for p in P]
    all_real = np.concatenate([x for x in per_run if len(x)], axis=0)
    res[key] = dict(rel=rel, real=segs_real.mean(axis=0) if len(segs_real) else None,
                    sim=segs_sim.mean(axis=0) if len(segs_sim) else None,
                    n=len(all_real), m_real=metrics(rel, all_real),
                    m_sim=metrics(rel, segs_sim))
    mr, msi = res[key]["m_real"], res[key]["m_sim"]
    print(f"  {label:38} 모서리 {res[key]['n']:3d}개 · 지나침 {mr[0]:5.1f} 대 {msi[0]:5.1f}% · "
          f"90%까지 {mr[1]:6.1f} 대 {msi[1]:6.1f} ms · 남은오차 {mr[2]:5.1f} 대 {msi[2]:5.1f}%")

  # ---------------------------------------------------------------- 그림
  fig, axs = plt.subplots(2, 2, figsize=(15.0, 9.2))
  pairs = (("out+", "out-", "going out from zero"), ("back+", "back-", "coming back to zero"))
  for ax, (ka, kb, title) in zip(axs[0], pairs):
    for k, lab, col in ((ka, "the plus way", "#00a878"), (kb, "the minus way (sign flipped)",
                                                          "#7b3fbf")):
      r = res[k]
      if r["real"] is None:
        continue
      ax.plot(r["rel"] * 1e3, r["real"] * a.amp, color=col, lw=2.0, alpha=0.85,
              label=f"real motor - {lab}")
      ax.plot(r["rel"] * 1e3, r["sim"] * a.amp, color=col, lw=1.3, ls=(0, (5, 2)), alpha=0.9,
              label=f"simulator - {lab}")
    ax.axhline(a.amp, color="#999", lw=0.9, ls=":")
    ax.set_title(f"{title}   (both drawn as 0 -> +{a.amp:g} deg)", fontsize=11)
    ax.set_xlabel("time from the step [ms]"); ax.set_ylabel("how far it went [deg]")
    ax.grid(alpha=0.28); ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

  ax = axs[1][0]
  for ka, kb, title in pairs:
    if res[ka]["real"] is None or res[kb]["real"] is None:
      continue
    d_real = (res[ka]["real"] - res[kb]["real"]) * a.amp
    d_sim = (res[ka]["sim"] - res[kb]["sim"]) * a.amp
    ax.plot(res[ka]["rel"] * 1e3, d_real, lw=2.0, alpha=0.85, label=f"real - {title}")
    ax.plot(res[ka]["rel"] * 1e3, d_sim, lw=1.3, ls=(0, (5, 2)), alpha=0.9,
            label=f"simulator - {title}")
  ax.axhline(0, color="#999", lw=0.9)
  ax.set_title("plus minus minus - what is left over is the lean to one side", fontsize=11)
  ax.set_xlabel("time from the step [ms]"); ax.set_ylabel("difference [deg]")
  ax.grid(alpha=0.28); ax.legend(fontsize=8.5, framealpha=0.9)

  ax = axs[1][1]
  txt = [f"{'':19}{'overshoot [%]':>16}{'time to 90% [ms]':>19}{'left over [%]':>16}", ""]
  for k, label, col, sgn in LEGS:
    mr, msi = res[k]["m_real"], res[k]["m_sim"]
    txt.append(f"{label.split('(')[0].strip():<12} real {mr[0]:15.1f}{mr[1]:19.1f}{mr[2]:16.1f}")
    txt.append(f"{'':<12} sim  {msi[0]:15.1f}{msi[1]:19.1f}{msi[2]:16.1f}")
    txt.append("")
  ax.axis("off")
  ax.text(0.0, 1.0, "\n".join(txt), family="monospace", fontsize=10.5, va="top")
  ax.set_title("the same three numbers, one row per direction", fontsize=11)

  fig.suptitle(
    f"RobStride {a.family} (motor {a.motor}) - is +{a.amp:g} deg the same as -{a.amp:g} deg?"
    f"   {len(runs)} runs, {res['out+']['n']} steps each way\n"
    f"friction only opposes motion, so the two should mirror each other - what does not "
    f"mirror is something the model does not have", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.91))
  os.makedirs(os.path.abspath(a.out_dir), exist_ok=True)
  out = os.path.join(a.out_dir, f"pd_updown_{a.family.lower()}.png")
  fig.savefig(out, dpi=125)
  print(f"그림: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
