#!/usr/bin/env python3
"""**코깅을 모형에 넣으면 응답이 얼마나 좋아지는가.** 넣고/빼고 같은 자료로 견준다.

2026-09-11 · 사용자 지시: **"코깅을 모형에 넣으면 응답이 얼마나 좋아지는지 비교해줘."**

무엇을 넣는가 — 맞춘 것이 아니라 **다른 실험에서 가져온 것**
------------------------------------------------------------
코깅힘은 **한 바퀴 훑기**(2026-09-10)에서 이미 쟀습니다. 자리마다 필요한 힘을 양방향으로 재서
더하면 마찰이 지워지고 코깅만 남습니다. 여기서는 **그 곡선을 그대로 가져다** 계단·삼각파
실험의 시뮬레이터에 얹습니다.

★ 중요한 점: **아무것도 맞추지 않습니다.** 크기도 위상도 다른 실험에서 온 값이고, 모터가
보고하는 각도는 같은 자리(전원을 끊은 적이 없음)이므로 그대로 대응됩니다. 그래서 이 시험은
**틀릴 수 있는 예측**입니다 — 맞으면 잔차가 줄고, 위상이나 부호가 틀렸으면 오히려 늘어납니다.

부호는 어떻게 정하는가
----------------------
느린 정속에서는 힘의 합이 0 입니다.

    제어기 토크 + 코깅힘 − 마찰 x 진행방향 = 0

그러므로 양방향 평균 `(갈 때 + 올 때)/2 = −코깅힘` 입니다. 즉 **코깅힘 = −(측정한 곡선)**.
이것도 고르는 것이 아니라 식에서 나옵니다.

세 가지를 견줍니다
------------------
    없음      지금 모형 (관성·마찰·점성만)
    사인 하나 으뜸 주기 하나로만 (mjlab 에 넣기 쉬운 형태)
    측정 곡선 훑기에서 나온 모양 그대로 (넣을 수 있는 최선)

    pd_cogging_effect.py --motor 4 --family RS04
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
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pd_2x2 import BEST, XML  # noqa: E402
from pd_cogging import bin_by_position, load as load_sweep  # noqa: E402

DEG = math.pi / 180.0
GEAR = 9.0
# 한 바퀴 훑기에서 나온 으뜸 주기 (출력축 한 바퀴에 몇 개). 189 = 21x9, 180 = 20x9.
N_OUT = {"RS04": 189.0, "RS03": 180.0}


def cogging_profile(data, motor, family, kp=20.0, kd=1.0, amp=180.0, tag="cog", step=0.25,
                    margin=8.0):
  """한 바퀴 훑기에서 **코깅힘(θ)** 을 만든다. θ 는 모터가 보고하는 각도 그대로(도)."""
  runs = load_sweep(os.path.join(
      data, f"pd_triangle_{motor}_kp{kp:g}_kd{kd:g}_a{amp:g}_r*_{tag}.csv"))
  if not runs:
    raise SystemExit("한 바퀴 훑기 자료가 없습니다")
  lo, hi = -amp + margin, amp - margin
  cen, f_fwd, n_fwd = bin_by_position(runs, lo, hi, step, +1)
  _, f_bwd, n_bwd = bin_by_position(runs, lo, hi, step, -1)
  ok = (n_fwd > 5) & (n_bwd > 5) & np.isfinite(f_fwd) & np.isfinite(f_bwd)
  cen, f_fwd, f_bwd = cen[ok], f_fwd[ok], f_bwd[ok]
  a_, b_ = f_fwd - f_fwd.mean(), f_bwd - f_bwd.mean()
  ms = int(round(3.0 / step))
  k = int(np.arange(-ms, ms + 1)[int(np.argmax(
      [np.corrcoef(a_[ms:-ms], np.roll(b_, s)[ms:-ms])[0, 1] for s in range(-ms, ms + 1)]))])
  half_sum = 0.5 * (f_fwd + np.roll(f_bwd, k))
  tau_cog = -(half_sum - half_sum.mean())      # 부호는 힘의 합에서 나온다 (위 설명)
  return cen, tau_cog


def fit_sinusoid(theta_deg, tau, n_out):
  """으뜸 주기 하나로만 나타낸다: tau = A sin(n_out x θ + φ). mjlab 에 넣기 쉬운 형태."""
  w = n_out * np.radians(theta_deg)
  M = np.vstack([np.sin(w), np.cos(w)]).T
  c, *_ = np.linalg.lstsq(M, tau, rcond=None)
  A = float(math.hypot(c[0], c[1]))
  phi = float(math.atan2(c[1], c[0]))
  pred = M @ c
  frac = 1.0 - np.sum((tau - pred) ** 2) / np.sum((tau - tau.mean()) ** 2)
  return A, phi, frac


def simulate(t, tgt, tgv, off, kp, kd, dt, tau_max, delay_s, cog=None, **prm):
  """cog(θ_abs_deg) -> 뉴턴미터 를 매 걸음 더한다. θ_abs 는 모터가 보고하는 각도."""
  m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, **prm))
  d = mujoco.MjData(m)
  n = int((t[-1] - t[0]) / dt)
  st = t[0] + np.arange(n) * dt
  qd = np.interp(st, t, tgt) * DEG
  vd = np.interp(st, t, tgv) * DEG
  lag = max(1, int(round(delay_s / dt)))
  hq, hv = [0.0] * lag, [0.0] * lag
  out = np.empty(n)
  for i in range(n):
    tau = kp * (qd[i] - hq[0]) + kd * (vd[i] - hv[0])
    tau = float(np.clip(tau, -tau_max, tau_max))
    if cog is not None:
      tau += float(cog(d.qpos[0] / DEG + off))     # 코깅은 상한 **밖**에서 더한다 (물리다)
    d.ctrl[0] = tau
    mujoco.mj_step(m, d)
    hq.append(d.qpos[0]); hq.pop(0)
    hv.append(d.qvel[0]); hv.pop(0)
    out[i] = d.qpos[0] / DEG
  return np.interp(t, st, out)


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
                np.array([float(x["pos_deg"]) for x in r]) - off, off))
  return out


def grid(runs, dt=0.001):
  tmax = min(r[0][-1] for r in runs)
  ts = np.arange(0, tmax, dt)
  P, T, V = [], None, None
  for t, tgt, tgv, pos, off in runs:
    P.append(np.interp(ts, t, pos))
    if T is None:
      T, V = np.interp(ts, t, tgt), np.interp(ts, t, tgv)
  return ts, T, V, np.array(P), float(np.mean([r[4] for r in runs]))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--out-dir", default="../docs/img")
  a = ap.parse_args(argv)

  th, tau_cog = cogging_profile(a.data, a.motor, a.family)
  n_out = N_OUT[a.family]
  A, phi, frac = fit_sinusoid(th, tau_cog, n_out)
  print(f"{a.family} (모터 {a.motor})")
  print(f"  훑기에서 온 코깅: 봉우리~골 {tau_cog.max()-tau_cog.min():.4f} 뉴턴미터")
  print(f"  사인 하나로 맞추면 크기 {A:.4f} 뉴턴미터 · 위상 {math.degrees(phi):+.1f}도 · "
        f"이 하나가 설명하는 몫 {frac*100:.0f}%")
  print(f"  → mjlab 에 넣을 형태:  tau -= {A:.4f} * sin({n_out:g} * q + {phi:.4f})   "
        f"(q 는 관절 각도 라디안)")

  lut = lambda x: np.interp(x, th, tau_cog, left=0.0, right=0.0)          # noqa: E731
  sinf = lambda x: A * math.sin(n_out * math.radians(x) + phi)            # noqa: E731
  variants = [("cogging left out (today's model)", None),
              ("one sine at the main period", sinf),
              ("the measured cogging curve", lut)]

  CASES = [("step +-10 one-sided", f"pd_pmstep_{a.motor}_kp20_kd1_a10_r*_pm.csv"),
           ("step +-5", f"pd_step_{a.motor}_kp20_kd1_a5_r*_s.csv"),
           ("step +-12", f"pd_step_{a.motor}_kp20_kd1_a12_r*_s.csv"),
           ("step +-30", f"pd_step_{a.motor}_kp20_kd1_a30_r*_s.csv"),
           ("triangle 30 deg", f"pd_triangle_{a.motor}_kp20_kd1_a30_r*_c.csv")]

  rows = []
  for label, pat in CASES:
    runs = load_runs(os.path.join(a.data, pat))
    if not runs:
      print(f"  {label:26} 자료 없음")
      continue
    ts, tgt, tgv, P, off = grid(runs)
    real = P.mean(axis=0)
    res = []
    for _, fn in variants:
      y = simulate(ts, tgt, tgv, off, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3,
                   cog=fn, **BEST[a.family])
      res.append(float(np.sqrt(np.mean((y - real) ** 2))))
    rows.append((label, len(runs), res))
    ch = [(r - res[0]) / res[0] * 100 for r in res]
    print(f"  {label:26} {len(runs):2d}회 · 잔차 없음 {res[0]:.4f} → 사인 {res[1]:.4f} "
          f"({ch[1]:+.0f}%) → 측정곡선 {res[2]:.4f} ({ch[2]:+.0f}%) 도")

  # ------------------------------------------------------------------ 멈추는 자리
  # 궤적 전체의 잔차는 계단을 오르내리는 큰 움직임이 지배합니다. 코깅이 실제로 바꾸는 것은
  # **멈추는 자리**이므로, 그것만 따로 잽니다 — 그리고 우리가 실물에서 본 **방향 쏠림**을
  # 모형이 만들어 내는지 봅니다.
  runs = load_runs(os.path.join(a.data, f"pd_pmstep_{a.motor}_kp20_kd1_a10_r*_pm.csv"))
  asym = None
  if runs:
    ts, tgt, tgv, P, off = grid(runs)
    real = P.mean(axis=0)
    amp = float(np.abs(tgt).max())
    lv = np.array([-amp, 0.0, amp])
    lab = np.abs(tgt[:, None] - lv[None, :]).argmin(axis=1)
    ed = np.nonzero(np.diff(lab))[0]

    def settle(sig):
      """0 → +A 와 0 → −A 로 나간 뒤, 다음 모서리 직전 0.5초에 어디에 서 있었나."""
      up, dn = [], []
      for i, k in enumerate(ed[:-1]):
        A0, B0 = lab[k], lab[min(k + 2, len(lab) - 1)]
        if (A0, B0) not in ((1, 2), (1, 0)):
          continue
        tn = ts[ed[i + 1]]
        m = (ts > tn - 0.5) & (ts < tn)
        if m.sum() < 5:
          continue
        v = float(np.median(sig[m] - lv[B0])) * (1 if B0 == 2 else -1)
        (up if B0 == 2 else dn).append(v)
      return float(np.mean(up)), float(np.mean(dn))

    print("\n  ★ 멈추는 자리 (목표를 가는 방향으로 지나친 양, 도) — 코깅이 실제로 바꾸는 곳")
    ru, rd = settle(real)
    print(f"    {'실물':<28} 플러스 {ru:+6.3f} · 마이너스 {rd:+6.3f} · "
          f"**쏠림 {ru - rd:+6.3f}**")
    asym = []
    for name, fn in variants:
      y = simulate(ts, tgt, tgv, off, a.kp, a.kd, a.dt, a.tau_max, a.delay_ms / 1e3,
                   cog=fn, **BEST[a.family])
      su, sd = settle(y)
      asym.append((name, su, sd, su - sd))
      print(f"    {name:<28} 플러스 {su:+6.3f} · 마이너스 {sd:+6.3f} · "
            f"**쏠림 {su - sd:+6.3f}**  (실물과 차 {abs((su-sd)-(ru-rd)):.3f})")

  if not rows:
    return 1
  fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.2))
  ax = axs[0]
  ax.plot(th, tau_cog, color="#7b3fbf", lw=0.9, alpha=0.8, label="measured over a full turn")
  ax.plot(th, A * np.sin(n_out * np.radians(th) + phi), color="#e8443a", lw=0.9, alpha=0.8,
          label=f"one sine, {n_out:g} per turn ({frac*100:.0f}% of it)")
  ax.set_xlim(-30, 30)
  ax.set_xlabel("shaft angle the motor reports [deg]"); ax.set_ylabel("cogging torque [N m]")
  ax.set_title("what is being added (zoomed to the step range)", fontsize=11)
  ax.grid(alpha=0.28); ax.legend(fontsize=9)

  ax = axs[1]
  x = np.arange(len(rows))
  w = 0.27
  cols = ["#8f8f8f", "#e8443a", "#00a878"]
  for j, (name, _) in enumerate(variants):
    ax.bar(x + (j - 1) * w, [r[2][j] for r in rows], w, color=cols[j], alpha=0.9, label=name)
  ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=18, ha="right",
                                       fontsize=9)
  ax.set_ylabel("how far the simulator is from the real motor\n(root-mean-square) [deg]",
                fontsize=10)
  ax.set_title("lower is better", fontsize=11)
  ax.grid(alpha=0.28, axis="y"); ax.legend(fontsize=9)
  fig.suptitle(f"RobStride {a.family} (motor {a.motor}) - does adding cogging help?\n"
               f"the cogging curve is taken from the full-turn sweep, not fitted here - "
               f"so it can make things worse", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.88))
  os.makedirs(os.path.abspath(a.out_dir), exist_ok=True)
  out = os.path.join(a.out_dir, f"pd_cogging_effect_{a.family.lower()}.png")
  fig.savefig(out, dpi=125); plt.close(fig)
  print(f"그림: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
