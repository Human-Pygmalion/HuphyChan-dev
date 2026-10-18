#!/usr/bin/env python3
"""Markdown tables for arm D (rough_gain_crank60_actdelay_v11a) next to baseline, arm A, arm B and arm C. Numbers only, no judgement.

Same inputs as gain_ab_tables_armC.py, plus the arm D files that gain_ab_eval.sh, crank_error_walk.py,
real_condition_sim.py and rough_load_probe.py left in docs/sim2real/data/gain_ab_eval/.
  python3 tools/sim2real/gain_ab_tables_armD.py tb|turn|stand|walk|rc|crank|probe|checks
Run from the repo root. `stand` and `walk` call analysis/*_verdict.py on the copied npz files (numpy only).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gain_ab_tables import turn_blocks, turn_summary  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
G = REPO / "docs/sim2real/data/gain_ab_eval"
MJ = REPO / "mujoco-sim/mjlab"
RUNS = [("기준선", "base", "rbmv11a_final"), ("팔 A", "armA", "gcrank60_final"), ("팔 B", "armB", "gmountain_final"), ("팔 C", "armC", "gaindr_final"), ("팔 D", "armD", "actdelay_final")]
TURN_RUNS = [("기준선", "base", "rbmv11a_final"), ("팔 A", "armA", "gcrank60_final"), ("팔 C", "armC", "gaindr_final"), ("팔 D (지연 끔)", "armD", "actdelay_final"), ("팔 D (지연 켬)", "armDon", "actdelay_on_final")]
RATED = {"hip_pitch": 40, "hip_roll": 40, "hip_yaw": 20, "knee": 40, "crank": 20}  # thermal_effort rated torque [N*m]
GROUPS = [("hip_pitch", "엉덩이 앞뒤"), ("hip_roll", "엉덩이 좌우"), ("hip_yaw", "엉덩이 비틀기"), ("knee", "무릎"), ("crank", "발목 크랭크")]


def verdict_lines(tag, kind, pattern, label="x"):
  out = subprocess.run(["python3", str(MJ / f"analysis/{kind}_verdict.py"), "--npz", str(MJ / f"analysis/out/m_{tag}_{pattern}.npz"), "--label", label],
                       capture_output=True, text=True).stdout
  return out


def tb():
  t = json.load(open(G / "tb_final_3runs.json"))
  b = json.load(open(G / "tb_final_armB.json"))["armB_resumed"]
  c = json.load(open(G / "tb_final_armC.json"))["armC"]
  d = json.load(open(G / "tb_final_armD.json"))["armD"]
  tr = json.load(open(G.parent / "2026-10-15_v11a_trial_tb_final200.json"))["trial"]
  names = [("Train/mean_reward", "평균 보상"), ("Train/mean_episode_length", "평균 에피소드 길이 (스텝)"),
           ("Episode_Termination/fell_over", "넘어짐 종료 (`fell_over`)"), ("Episode_Termination/low_base", "낮은 몸통 종료 (`low_base`)"),
           ("Episode_Termination/stagnation", "정체 종료 (`stagnation`)"), ("Episode_Reward/track_linear_velocity", "직선 속도 추종 보상"),
           ("Episode_Reward/track_angular_velocity", "회전 속도 추종 보상"), ("Metrics/twist/error_vel_xy_steady_moving", "움직일 때 속도 오차 [m/s]"),
           ("Curriculum/dr_levels/dr_factor", "무작위화 램프")]
  print("| 지표 | 기준선 | 팔 A | 팔 B | 팔 C | 묶음 시험 | 팔 D | 팔 D − 기준선 | 팔 D − 팔 A | 팔 D − 팔 C |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
  for k, n in names:
    if d[k] is None or (d[k]["final200"] != d[k]["final200"]):
      continue
    v = [t["base"][k]["final200"], t["armA"][k]["final200"], b[k]["final200"], c[k]["final200"], tr[k]["final200"], d[k]["final200"]]
    f = "{:.4f}" if "fell" in k else "{:.3f}" if abs(v[0]) < 10 else "{:.2f}"
    print(f"| {n} | " + " | ".join(f.format(x) for x in v) + f" | {f.format(v[5]-v[0])} | {f.format(v[5]-v[1])} | {f.format(v[5]-v[3])} |")


def turn():
  S = {}
  for lab, key, tag in TURN_RUNS:
    S[key] = turn_blocks(G / f"verdict_{tag}.txt")
  print("| 씨앗 | " + " | ".join(f"{l} 중앙값 | {l} 반대 방향" for l, _, _ in TURN_RUNS) + " |\n|---|" + "--:|--:|" * len(TURN_RUNS))
  for s in range(8):
    print(f"| {s} | " + " | ".join(f"{S[k][s]['median']:.1f} % | {S[k][s]['wrong']}/8" for _, k, _ in TURN_RUNS) + " |")
  cells = []
  for _, k, _ in TURN_RUNS:
    med, meds, wrong, tot = turn_summary(S[k])
    cells.append(f"**{med:.1f} %** | 합계 {wrong}/{tot}")
  print("| **여덟 씨앗의 중앙값** | " + " | ".join(cells) + " |")
  cmds = sorted({r[0] for x in S["base"].values() for r in x["rows"]})
  print("\n| 명령 [rad/s] | " + " | ".join(f"{c:+.2f}" for c in cmds) + " |\n|---|" + "--:|" * len(cmds))
  for lab, k, _ in TURN_RUNS:
    print(f"| {lab} | " + " | ".join(f"{np.median([r[2] for x in S[k].values() for r in x['rows'] if r[0]==c]):.1f}" for c in cmds) + " |")


def stand():
  rows = {}
  for lab, k, tag in RUNS:
    d, fo, le, hh = [], None, None, None
    for s in range(4):
      o = verdict_lines(tag, "standing", f"stand_s{s}")
      d.append(float(re.search(r"travelled \(m\)\s+([\d.]+)", o).group(1)))
    o = verdict_lines(tag, "standing", "stand_s*")
    pick = lambda pat: re.search(pat + r"\s+(.+)", o).group(1).strip()  # noqa: E731
    rows[k] = (d, pick(r"travelled \(m\)"), pick(r"foot off ground \(%\)"), pick(r"lift events"), pick(r"base height \(m\)"))
  print("| 지표 | " + " | ".join(l for l, _, _ in RUNS) + " |\n|---|--:|--:|--:|--:|--:|")
  print("| 흘러간 거리 평균 ± 표준편차 [m] | " + " | ".join(rows[k][1].replace("+-", "±") for _, k, _ in RUNS) + " |")
  print("| 씨앗별 흘러간 거리 [m] (0 / 1 / 2 / 3) | " + " | ".join(" / ".join(f"{x:.3f}" for x in rows[k][0]) for _, k, _ in RUNS) + " |")
  print("| 발이 땅에서 뜬 시간 비율 | " + " | ".join(rows[k][2].replace("+-", "±") for _, k, _ in RUNS) + " |")
  print("| 발 뜸 횟수 | " + " | ".join(rows[k][3].replace("+-", "±") for _, k, _ in RUNS) + " |")
  print("| 몸통 높이 [m] | " + " | ".join(rows[k][4].replace("+-", "±") for _, k, _ in RUNS) + " |")


def walk():
  P = {}
  for lab, k, tag in RUNS:
    o = verdict_lines(tag, "walking", "walk_s*")
    P[k] = {n: re.search(pat + r"\s+(.+)", o).group(1).strip().replace("+-", "±") for n, pat in
            [("speed", r"forward speed \(m/s\)"), ("gap", r"hip yaw L-R gap \(%\)"), ("drift", r"heading drift \(deg\)"),
             ("ratio", r"twist / step ratio"), ("step", r"stepping time \(%\)")]}
  for n, lab in [("speed", "실제 앞 속도 [m/s]"), ("gap", "엉덩이 비틀기 좌우 차이 [%]"), ("drift", "진행 방향 틀어짐 [도]"),
                 ("ratio", "비틀기 대 걸음 비율"), ("step", "걷는 시간 비율 [%]")]:
    print(f"| {lab} | " + " | ".join(P[k][n] for _, k, _ in RUNS) + " |")


def rc():
  def load(f):
    by = {}
    for r in json.load(open(f))["rows"]:
      by.setdefault(r["cond"], []).append(r)
    return by
  X = [load(G.parent / "2026-10-15_rc_rbmv11a_final_brief_dec0.json"), load(G / "rc_armA_brief_dec0.json"), load(G / "rc_armB_brief_dec0.json")]
  for c in X[0]:
    cells = []
    for by in X:
      rs = by[c]
      g = np.mean([r["gyro_rms_dps"] for r in rs])
      cells.append(f"{sum(not r['fell'] for r in rs)}/{len(rs)} | {np.mean([r['survived_s'] for r in rs]):.1f} | {np.mean([r['act_p99'] for r in rs]):.2f} | "
                   f"{np.mean([r['knee_std_deg'] for r in rs]):.1f} | {np.mean([r['knee_hz'] for r in rs]):.2f} | {g:.0f}")
    print(f"| {c} | " + " | ".join(cells) + " |")


def rc2():
  """Compact version of rc: survivors and mean survival of the three runs, detail columns for arm B only."""
  def load(f):
    by = {}
    for r in json.load(open(f))["rows"]:
      by.setdefault(r["cond"], []).append(r)
    return by
  X = [load(G.parent / "2026-10-15_rc_rbmv11a_final_brief_dec0.json"), load(G / "rc_armA_brief_dec0.json"), load(G / "rc_armB_brief_dec0.json"), load(G / "rc_armC_brief_dec0.json"), load(G / "rc_armD_brief_dec0.json")]
  for c in X[0]:
    cells = [f"{sum(not r['fell'] for r in by[c])}/{len(by[c])} | {np.mean([r['survived_s'] for r in by[c]]):.1f}" for by in X]
    rs = X[4][c]
    cells.append(f"{np.mean([r['act_p99'] for r in rs]):.2f} | {np.mean([r['knee_std_deg'] for r in rs]):.1f} | {np.mean([r['knee_hz'] for r in rs]):.2f} | "
                 f"{np.mean([np.mean(r['gyro_rms_dps']) for r in rs]):.0f}")
    print(f"| {c} | " + " | ".join(cells) + " |")


def crank():
  print("| 지표 | 기준선 | 팔 A | 팔 B | 팔 C | 팔 D |\n|---|--:|--:|--:|--:|--:|")
  J = [json.load(open(G / f"crank_{k}.json"))["pooled"] for k in ("base", "armA", "armB", "armC", "armD")]
  for key, lab in [("err_p50_deg", "오차 중앙값 [도]"), ("err_p99_deg", "오차 99 백분위수 [도]"), ("err_max_deg", "오차 최댓값 [도]"),
                   ("tau_p99_nm", "토크 절댓값 99 백분위수 [N·m]"), ("tau_rms_nm", "토크 제곱평균제곱근 [N·m]"), ("tau_max_nm", "토크 절댓값 최댓값 [N·m]")]:
    print(f"| {lab} | " + " | ".join(f"{j[key]:.1f}" for j in J) + " |")
  for k in ("base", "armA", "armB", "armC", "armD"):
    print(k, "per-seed err_p99:", " / ".join(f"{f['err_p99']:.1f}" for f in json.load(open(G / f"crank_{k}.json"))["files"]))


def _v(d, *keys, fmt="{:.1f}", scale=1.0):
  """Value at d[k1][k2]..., formatted; '—' when the run reached too few steps for the statistic to exist."""
  try:
    for k in keys:
      d = d[k]
    if d != d:  # nan
      return "—"
    return fmt.format(scale * d)
  except (KeyError, TypeError):
    return "—"


def probe():
  P = {}
  for suite, ks in (("slope", ("base", "armA", "armB", "armC", "armD")), ("stairs", ("base", "armA", "armB", "armC", "armD"))):
    for k in ks:
      d = json.load(open(G / f"{suite}_{k}.json"))
      for c in d["conditions"]:
        P[(c["cond"], k)] = c
  conds = [("slope_up_20_vx0.3", "오르막 20° / 0.3 m/s"), ("slope_up_20_vx0.6", "오르막 20° / 0.6 m/s"), ("stairs_up_15cm_vx0.3", "계단 오름 15 cm / 0.3 m/s")]
  print("[표 1] 넘어짐과 토크 99 백분위수")
  print("| 조건 | 정책 | 넘어짐 [%] (256 중) | 닿은 비율 [%] | 끝 위치 [m] | 분석 걸음 수 | 엉덩이 앞뒤 | 엉덩이 좌우 | 엉덩이 비틀기 | 무릎 | 발목 크랭크 | 크랭크 오차 99 백분위수 [도] |\n|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
  for c, cl in conds:
    for lab, k, _ in RUNS:
      x = P[(c, k)]
      a = x["all"]
      print(f"| {cl} | {lab} | {100*x['falls']:.1f} ({round(x['falls']*x['n_envs'])}) | {100*x['reached_feature']:.0f} | {x['end_x_mean_m']:.1f} | {x['valid_steps']:,} | "
            + " | ".join(_v(a, g, "tau", "p99") for g, _ in GROUPS) + " | " + _v(a, "crank", "err_deg", "p99", fmt="{:.0f}") + " |")
  print("\n[표 2] 목표 위치 오차 99 백분위수 [도] (|목표각 − 실제각|)")
  print("| 조건 | 정책 | " + " | ".join(n for _, n in GROUPS) + " |\n|---|---|" + "--:|" * 5)
  for c, cl in conds:
    for lab, k, _ in RUNS:
      a = P[(c, k)]["all"]
      print(f"| {cl} | {lab} | " + " | ".join(_v(a, g, "err_deg", "p99") for g, _ in GROUPS) + " |")
  print("\n[표 3] 토크 제곱평균제곱근 [N·m] (정격 대비 %)")
  print("| 조건 | 정책 | " + " | ".join(f"{n} (정격 {RATED[g]})" for g, n in GROUPS) + " |\n|---|---|" + "--:|" * 5)
  for c, cl in conds:
    for lab, k, _ in RUNS:
      a = P[(c, k)]["all"]
      print(f"| {cl} | {lab} | " + " | ".join((_v(a, g, "tau", "rms") + " (" + _v(a, g, "tau", "rms", fmt="{:.0f}", scale=100 / RATED[g]) + " %)") if _v(a, g, "tau", "rms") != "—" else "—" for g, _ in GROUPS) + " |")
  print("\n[표 4] 토크 99 백분위수 대 토크 상한 (포화 비율 %, 행동 한계 닿은 비율 %)")
  print("| 조건 | 정책 | " + " | ".join(n for _, n in GROUPS) + " |\n|---|---|" + "--:|" * 5)
  for c, cl in conds:
    for lab, k, _ in RUNS:
      a = P[(c, k)]["all"]
      print(f"| {cl} | {lab} | " + " | ".join((_v(a, g, "sat_frac", fmt="{:.3f}", scale=100) + " / " + _v(a, g, "clip_frac", fmt="{:.2f}", scale=100)) if _v(a, g, "sat_frac") != "—" else "—" for g, _ in GROUPS) + " |")
  print("\n넘어짐 원인:", {(c, k): P[(c, k)]["fall_causes"] for c, _ in conds for _, k, _ in RUNS})
  print("만들어진 게인 (probe json):", {k: json.load(open(G / f"slope_{k}.json"))["built_gains_first_env"] for k in ("armD",)})


if __name__ == "__main__":
  globals()[sys.argv[1]]()
