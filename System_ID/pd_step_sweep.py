#!/usr/bin/env python3
"""계단 명령을 **크기별로·자리별로** 훑어 놓고, 한 벌의 값이 그 전부를 맞히는지 본다.

2026-09-10 · 사용자 지시: **"Step Input 을 다양한 각도로 더 많이 넣어서 실험해줘.
삼각파는 줄여도 되."**

왜 크기를 훑으면 더 강한 시험이 되는가
--------------------------------------
계단 하나만 맞히는 것은 쉽습니다. 값 셋을 적당히 만지면 어느 한 크기는 맞출 수 있습니다.
**크기를 바꾸면 그 응답을 지배하는 물리가 바뀌기 때문에**, 여러 크기를 *한 벌의 값으로*
다 맞히는 것은 훨씬 어렵고, 그래서 훨씬 믿을 만합니다.

    작은 계단 (1.5~3도)   마찰의 죽은구간 안. **마찰**만 보인다.
                          자리잡은 뒤 남는 오차 = 마찰 ÷ 세기
    중간 계단 (5~12도)    토크가 아직 상한에 안 닿는 **선형 구간**.
                          지나친 양과 상승 시간이 **관성**과 **점성**을 같이 드러낸다.
    큰 계단 (20~45도)     토크가 상한에 붙어 버틴다. 그러면 응답이 **관성 하나로만** 정해진다.
                          가속도 = (토크 상한 − 마찰) ÷ 관성

견주는 값 세 벌
---------------
* **재측정값** — 이번에 벤치에서 잰 값 (`docs/131`)
* **바꾸기 전 시뮬레이터 값** — 2026-09-10 까지 코드에 들어 있던 값. 옮겨 적기 오류가
  섞여 있었습니다(`docs/131` 감사)
* **다른 연구원 값** — 같은 모터를 다른 방법으로 잰 값. 관성만 저희와 다릅니다

RS03 은 뒤의 두 벌이 **같은 값**이라 선이 둘만 나옵니다.

    pd_step_sweep.py --motor 4 --family RS04 --tag s
    pd_step_sweep.py --motor 4 --family RS04 --part position --tag p
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

from pd_2x2 import BEST, common_grid, load_runs, simulate  # noqa: E402

DEG = math.pi / 180.0

# 견줄 값 세 벌. 이름은 **뭘 뜻하는지 그대로** 적는다 (그림 범례에 그대로 나간다).
MODELS = {
  "RS04": [
    ("what we measured on the bench", BEST["RS04"], "#00a878", 1.9, 0.95),
    ("what the simulator used until today",
     dict(armature=0.016333, damping=0.009492, frictionloss=0.269456), "#f2a007", 1.4, 0.85),
    ("the other researcher's numbers",
     dict(armature=0.030605, damping=0.026008, frictionloss=0.526084), "#7b3fbf", 1.4, 0.85),
  ],
  "RS03": [
    ("what we measured on the bench", BEST["RS03"], "#00a878", 1.9, 0.95),
    ("what the simulator used until today (same as the other researcher's)",
     dict(armature=0.015265, damping=0.022342, frictionloss=0.285370), "#f2a007", 1.4, 0.85),
  ],
}

AMPS = [1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0]
TAU_MAX_USED = [8.0]      # one_case 가 쓰도록 main 에서 채운다
FAMILY_USED = ["RS04"]
HOMES = [-60.0, -30.0, 0.0, 30.0, 60.0]


def edges_of(tgt):
  """계단이 **위/아래 두 층 사이를 오간 자리**를 찾는다.

  ★ 2026-09-10 정정 — 앞의 판은 "값이 크게 뛴 곳"을 모서리로 봤습니다. 그런데 자료를 1
  밀리초 격자에 다시 올릴 때 계단 하나가 **두 칸에 걸쳐 번지면** 두 칸 모두 문턱을 넘어
  **모서리가 둘로 잡히고**, 계단 높이가 절반으로 계산되어 "지나친 양"이 최대 두 배로
  부풀려집니다. 번지느냐 마느냐는 모서리가 격자의 어디에 떨어지느냐에 달려 있어 회차마다
  달라지므로, **있지도 않은 차이**가 생깁니다(자리별 실험에서 53% 대 88% 로 벌어져 알아챘습니다).

  계단 명령은 **층이 둘뿐**이므로, 값이 가운데보다 위냐 아래냐로 나눈 뒤 그 구분이 바뀌는
  자리를 봅니다. 번짐에 영향을 받지 않습니다."""
  lo, hi = float(tgt.min()), float(tgt.max())
  if hi - lo < 1e-6:
    return np.array([], dtype=int), lo, hi
  up = tgt > 0.5 * (lo + hi)
  return np.where(np.diff(up.astype(int)) != 0)[0], lo, hi


def step_metrics(ts, tgt, pos):
  """계단 하나에서 뽑는 세 숫자. 셋이 각각 다른 값에 민감하다.

  * **남은 오차** — 자리잡은 뒤에도 못 간 만큼. 이론상 **마찰 ÷ 세기**
  * **지나친 양** — 목표를 가는 방향으로 얼마나 넘어갔나. **관성**이 클수록 커진다
  * **90%까지 걸린 시간** — 크게 뛸수록 토크 상한에 걸려 **관성**이 그대로 드러난다

  높이는 **명령의 두 층**(가장 낮은 값, 가장 높은 값)에서 가져옵니다. 모서리 바로 옆 칸의
  값을 쓰면 번진 중간값을 집을 수 있습니다."""
  ed, lo_lv, hi_lv = edges_of(tgt)
  mid = 0.5 * (lo_lv + hi_lv)
  e = pos - tgt
  ss, ov, rise = [], [], []
  for i, k in enumerate(ed[:-1]):
    te, tn = ts[k], ts[ed[i + 1]]
    going_up = tgt[min(k + 2, len(tgt) - 1)] > mid
    lo, hi = (lo_lv, hi_lv) if going_up else (hi_lv, lo_lv)
    d = hi - lo
    sgn = 1.0 if d > 0 else -1.0
    s1 = (ts > tn - 0.5) & (ts < tn)
    if s1.sum() > 5:
      ss.append(abs(float(np.median(e[s1]))))
    s2 = (ts > te) & (ts < te + 0.9)
    if s2.sum() > 5:
      ov.append(max(0.0, float(np.max((pos[s2] - hi) * sgn))) / abs(d) * 100)
      got = np.where((pos[s2] - (lo + 0.9 * d)) * sgn >= 0)[0]
      if len(got):
        rise.append((float(ts[s2][got[0]]) - te) * 1e3)
  f = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
  return f(ss), f(ov), f(rise)


def inertia_from_saturation(ts, tgt, pos, tau, tau_max, frictionloss):
  """토크가 **상한에 붙어 있는 동안**의 가속도로 관성을 곧바로 구한다.

  큰 계단을 치면 제어기가 곧장 토크 상한에 붙습니다. 그동안은 힘이 **일정**하므로

      관성 = (토크 상한 − 마찰) ÷ 가속도

  이 됩니다. 되먹임도 이득도 안 들어가고, 맞춤도 없습니다. 가속도는 위치를 2차식으로
  맞춰서 구합니다 — 두 번 미분하면 잡음이 커지기 때문입니다.

  이 값은 계단 크기가 클수록 잘 나옵니다. 작은 계단은 상한에 아예 안 붙습니다."""
  ed, _, _ = edges_of(tgt)
  out = []
  for k in ed[:-1]:
    te = ts[k]
    w = (ts >= te + 0.004) & (ts <= te + 0.075)      # 계단 직후 4~75 밀리초
    if w.sum() < 30:
      continue
    if np.mean(np.abs(tau[w])) < 0.98 * tau_max:     # 상한에 안 붙었으면 못 쓴다
      continue
    c = np.polyfit(ts[w] - te, pos[w], 2)
    acc = 2.0 * c[0] * DEG                            # 도/s^2 → 라디안/s^2
    if abs(acc) < 1e-6:
      continue
    out.append((tau_max - frictionloss) / abs(acc))
  return out


def one_case(data, motor, kp, kd, amp, tag, home, dt, tau_max, delay_ms, models):
  """한 조건(크기 하나 · 자리 하나)을 읽고, 실물 지표와 각 값의 시뮬레이터 지표를 낸다."""
  hs = "" if home == 0.0 else f"_h{home:+g}"
  pat = f"pd_step_{motor}_kp{kp:g}_kd{kd:g}_a{amp:g}{hs}_r*_{tag}.csv"
  runs = load_runs(os.path.join(data, pat))
  if not runs:
    return None
  ts, tgt, tgv, P = common_grid(runs)
  real = [step_metrics(ts, tgt, p) for p in P]
  # 토크는 common_grid 가 안 돌려주므로 파일에서 직접 읽는다
  Js = []
  for fp in sorted(globmod.glob(os.path.join(data, pat))):
    with open(fp) as f:
      r = list(csv.DictReader(f))
    if len(r) < 200:
      continue
    tt = np.array([float(x["t_s"]) for x in r])
    gg = np.array([float(x["target_deg"]) for x in r])
    pp = np.array([float(x["pos_deg"]) for x in r])
    qq = np.array([float(x["tau_cmd_nm"]) for x in r])
    Js += inertia_from_saturation(tt, gg, pp, qq, TAU_MAX_USED[0],
                                  BEST[FAMILY_USED[0]]["frictionloss"])
  sims = {}
  for name, prm, *_ in models:
    y = simulate(ts, tgt, tgv, kp, kd, dt, tau_max, delay_ms / 1e3, **prm)
    sims[name] = (step_metrics(ts, tgt, y), y)
  return dict(ts=ts, tgt=tgt, P=P, real=np.array(real, dtype=float), sims=sims,
              n=len(runs), pat=pat, J=np.array(Js, dtype=float))


def zoom_window(ts, tgt):
  """계단 하나가 통째로 보이는 창.

  **올라가는 모서리만** 고른다. 크기마다 아무 모서리나 고르면 어떤 칸은 올라가고 어떤 칸은
  내려가서, 여덟 칸을 나란히 놓아도 눈으로 못 견준다. 그리고 첫 모서리는 피한다 — 시작
  과도현상이 아직 남아 있어 회차마다 모양이 다르다."""
  ed, lo_lv, hi_lv = edges_of(tgt)
  mid = 0.5 * (lo_lv + hi_lv)
  up = [k for k in ed[:-1] if tgt[min(k + 2, len(tgt) - 1)] > mid]
  if not up:
    return (ts[0], ts[-1]) if len(ed) < 2 else (ts[ed[0]], ts[-1])
  k = up[min(1, len(up) - 1)]
  return max(ts[0], ts[k] - 0.18), min(ts[-1], ts[k] + 1.32)


def draw_panels(cases, amps, family, motor, kp, kd, out, xlabel):
  n = len(amps)
  cols = 4
  rows = int(math.ceil(n / cols))
  # 범례는 그림 아래에 한 줄로 깔린다. 칸이 한 줄뿐일 때도 글자가 x축 이름을 덮지 않도록,
  # **띄울 높이를 그림 높이에 맞춰** 계산한다(고정 비율로 두면 한 줄짜리에서 겹친다).
  fig_h = 3.35 * rows + 1.25
  fig, axs = plt.subplots(rows, cols, figsize=(5.0 * cols, fig_h), squeeze=False)
  for ax, key in zip(axs.ravel(), amps):
    c = cases.get(key)
    if c is None:
      ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
              color="#aaa", fontsize=9)
      ax.set_title(xlabel(key), fontsize=10)
      continue
    ts, tgt, P = c["ts"], c["tgt"], c["P"]
    lo, hi = zoom_window(ts, tgt)
    z = (ts >= lo) & (ts <= hi)
    ax.plot(ts[z], tgt[z], color="#8f8f8f", lw=1.1, ls=(0, (6, 3)), alpha=0.8,
            label="what we asked for")
    for k, pr in enumerate(P):
      ax.plot(ts[z], pr[z], color="#111111", lw=0.5, alpha=0.13,
              label=f"real motor, each of {len(P)} runs" if k == 0 else None)
    ax.plot(ts[z], P.mean(axis=0)[z], color="#111111", lw=1.6, alpha=0.50, zorder=6,
            label=f"real motor, average of {len(P)}")
    for name, prm, col, lw, al in MODELS[family]:
      ax.plot(ts[z], c["sims"][name][1][z], color=col, lw=lw, alpha=al, zorder=5,
              label="simulator - " + name)
    ax.set_title(xlabel(key), fontsize=10)
    ax.grid(alpha=0.28)
    ax.set_xlabel("time [s]", fontsize=9); ax.set_ylabel("angle [deg]", fontsize=9)
  for ax in axs.ravel()[len(amps):]:
    ax.axis("off")
  h, l = axs[0][0].get_legend_handles_labels()
  fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False,
             bbox_to_anchor=(0.5, 0.012))
  fig.suptitle(f"RobStride {family} (motor {motor}) - step commands, real bench against the "
               f"simulator\ngains kp {kp:g} / kd {kd:g}, torque limit the same on both sides",
               fontsize=12.5, y=0.995)
  fig.tight_layout(rect=(0, 0.80 / fig_h, 1, 1.0 - 0.72 / fig_h))
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=125); plt.close(fig)
  print(f"그림: {out}")


def draw_metrics(cases, keys, family, motor, kp, kd, out, xname, logx, extra_line=None):
  names = ["left-over error once it settles [deg]", "overshoot past the target [%]",
           "time to reach 90% of the step [ms]"]
  fig, axs = plt.subplots(2, 2, figsize=(13.6, 9.2))
  axs = axs.ravel()
  ks = [k for k in keys if cases.get(k) is not None]
  for j, (ax, nm) in enumerate(zip(axs[:3], names)):
    r_mu = [float(np.nanmean(cases[k]["real"][:, j])) for k in ks]
    r_sd = [float(np.nanstd(cases[k]["real"][:, j])) for k in ks]
    ax.errorbar(ks, r_mu, yerr=r_sd, color="#111111", lw=1.8, alpha=0.65, marker="o",
                ms=5, capsize=3, zorder=6, label="real motor (bar = spread over the runs)")
    for name, prm, col, lw, al in MODELS[family]:
      ys = [cases[k]["sims"][name][0][j] for k in ks]
      ax.plot(ks, ys, color=col, lw=lw + 0.3, alpha=al, marker="s", ms=4,
              label="simulator - " + name)
    if j == 0 and extra_line is not None:
      ax.axhline(extra_line, color="#c0392b", lw=1.1, ls=":",
                 label=f"friction / gain = {extra_line:.2f} deg (what theory says)")
    if j == 2 and any(math.isnan(v) for v in r_mu):
      # 이건 결측이 아니라 **결과**다. 계단이 마찰의 죽은구간보다 작으면 모터는 목표의
      # 90% 에 아예 도달하지 못하고 멈춘다.
      ax.text(0.02, 0.03,
              "missing points = the real motor never got to 90% of that step:\n"
              "it stopped inside the friction dead zone",
              transform=ax.transAxes, fontsize=8.4, va="bottom", color="#c0392b",
              bbox=dict(fc="white", ec="#e0b3ad", alpha=0.92))
    if logx:
      ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel(xname, fontsize=10); ax.set_ylabel(nm, fontsize=10)
    ax.set_title(nm, fontsize=10.5); ax.grid(alpha=0.28)
    ax.legend(fontsize=8, framealpha=0.9)

  # 네 번째 칸: 되먹임도 맞춤도 안 들어간 **직접 계산**
  ax = axs[3]
  xs, mu, sd, cnt = [], [], [], []
  for k in ks:
    J = cases[k].get("J")
    if J is None or len(J) < 3:
      continue
    xs.append(k); mu.append(float(np.mean(J))); sd.append(float(np.std(J))); cnt.append(len(J))
  if xs:
    ax.errorbar(xs, mu, yerr=sd, color="#111111", lw=1.8, alpha=0.7, marker="o", ms=5,
                capsize=3, zorder=6, label="from the bench, no fitting at all")
    for name, prm, col, lw, al in MODELS[family]:
      ax.axhline(prm["armature"], color=col, lw=lw + 0.3, alpha=al, ls="--",
                 label=name)
    for x, m_, c_ in zip(xs, mu, cnt):
      ax.annotate(f"{c_} edges", (x, m_), textcoords="offset points", xytext=(0, 9),
                  ha="center", fontsize=7.5, color="#666")
    if logx:
      ax.set_xscale("log")
      ax.set_xticks(xs); ax.set_xticklabels([f"{k:g}" for k in xs])
      ax.set_xticks([], minor=True)          # 눈금이 둘뿐일 때 "4 x 10^1" 이 끼어드는 것을 막는다
    lo = min(min(mu) - max(sd), min(p["armature"] for _, p, *_ in MODELS[family]))
    hi = max(max(mu) + max(sd), max(p["armature"] for _, p, *_ in MODELS[family]))
    ax.set_ylim(lo - 0.09 * (hi - lo), hi + 0.16 * (hi - lo))   # 글자 얹을 자리
  else:
    ax.text(0.5, 0.5, "no step here pushed the torque to its limit,\n"
                      "so this check needs the bigger steps",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888")
  ax.set_xlabel(xname, fontsize=10)
  ax.set_ylabel("rotor inertia [kg m2]", fontsize=10)
  ax.set_title("rotor inertia straight from the saturated acceleration\n"
               "inertia = (torque limit - friction) / acceleration", fontsize=10.5)
  ax.grid(alpha=0.28); ax.legend(fontsize=8, framealpha=0.9)

  fig.suptitle(f"RobStride {family} (motor {motor}) - the same three numbers across every "
               f"step, real against the simulator\ngains kp {kp:g} / kd {kd:g}",
               fontsize=12, y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.945))
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  fig.savefig(out, dpi=125); plt.close(fig)
  print(f"그림: {out}")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data", default="data")
  ap.add_argument("--motor", default="4")
  ap.add_argument("--family", default="RS04", choices=list(BEST))
  ap.add_argument("--part", default="sweep", choices=("sweep", "position"))
  ap.add_argument("--tag", default="s")
  ap.add_argument("--kp", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=1.0)
  ap.add_argument("--tau-max", type=float, default=8.0)
  ap.add_argument("--delay-ms", type=float, default=0.5)
  ap.add_argument("--dt", type=float, default=0.0002)
  ap.add_argument("--amp", type=float, default=10.0, help="자리별 실험에서 쓴 계단 크기")
  ap.add_argument("--out-dir", default="../docs/img")
  a = ap.parse_args(argv)

  fam, mot = a.family, a.motor
  models = MODELS[fam]
  TAU_MAX_USED[0], FAMILY_USED[0] = a.tau_max, fam
  print(f"{fam} (모터 {mot}) · 이득 kp={a.kp} kd={a.kd} · 태그 {a.tag}")

  if a.part == "sweep":
    cases = {}
    for A in AMPS:
      c = one_case(a.data, mot, a.kp, a.kd, A, a.tag, 0.0, a.dt, a.tau_max, a.delay_ms,
                   models)
      cases[A] = c
      if c is None:
        print(f"  계단 {A:5g}도 : 자료 없음")
        continue
      rm = np.nanmean(c["real"], axis=0)
      sm = c["sims"][models[0][0]][0]
      print(f"  계단 {A:5g}도 : {c['n']:2d}회 · 남은오차 실물 {rm[0]:5.2f} 대 "
            f"시뮬 {sm[0]:5.2f}도 · 지나침 {rm[1]:5.1f} 대 {sm[1]:5.1f}% · "
            f"90%까지 {rm[2]:6.1f} 대 {sm[2]:6.1f} ms")
    have = [A for A in AMPS if cases.get(A) is not None]
    if not have:
      print("자료가 하나도 없습니다 — 벤치 실험을 먼저 돌리세요", flush=True)
      return 1
    draw_panels(cases, have, fam, mot, a.kp, a.kd,
                os.path.join(a.out_dir, f"pd_step_sweep_{fam.lower()}.png"),
                lambda A: f"STEP  +-{A:g} deg")
    draw_metrics(cases, have, fam, mot, a.kp, a.kd,
                 os.path.join(a.out_dir, f"pd_step_metrics_{fam.lower()}.png"),
                 "step size [deg]", True,
                 extra_line=BEST[fam]["frictionloss"] / a.kp / DEG)
  else:
    cases = {}
    for H in HOMES:
      c = one_case(a.data, mot, a.kp, a.kd, a.amp, a.tag, H, a.dt, a.tau_max, a.delay_ms,
                   models)
      cases[H] = c
      if c is None:
        print(f"  자리 {H:+5g}도 : 자료 없음")
        continue
      rm = np.nanmean(c["real"], axis=0)
      print(f"  자리 {H:+5g}도 : {c['n']:2d}회 · 남은오차 {rm[0]:5.2f}도 · "
            f"지나침 {rm[1]:5.1f}% · 90%까지 {rm[2]:6.1f} ms")
    have = [H for H in HOMES if cases.get(H) is not None]
    if not have:
      print("자료가 하나도 없습니다 — 벤치 실험을 먼저 돌리세요", flush=True)
      return 1
    draw_panels(cases, have, fam, mot, a.kp, a.kd,
                os.path.join(a.out_dir, f"pd_step_position_{fam.lower()}.png"),
                lambda H: f"same +-{a.amp:g} deg step, taken at {H:+g} deg")
    draw_metrics(cases, have, fam, mot, a.kp, a.kd,
                 os.path.join(a.out_dir, f"pd_step_position_metrics_{fam.lower()}.png"),
                 "where on the shaft the step was taken [deg]", False,
                 extra_line=BEST[fam]["frictionloss"] / a.kp / DEG)
    sp = [float(np.nanmean(cases[H]["real"][:, 0])) for H in have]
    print(f"\n자리에 따른 남은 오차 폭: {min(sp):.3f} ~ {max(sp):.3f}도 "
          f"(퍼짐 {max(sp) - min(sp):.3f}도). 이 폭이 회차 흩어짐보다 크면 "
          f"마찰이 자리에 따라 다르다는 뜻입니다")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
