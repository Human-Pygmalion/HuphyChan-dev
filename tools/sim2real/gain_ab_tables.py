#!/usr/bin/env python3
"""Markdown tables for the finished-run evaluation of a gain A/B arm (numbers only, no judgement).

Reads what tools/sim2real/gain_ab_eval.sh, crank_error_walk.py, real_condition_sim.py and
tools/robot_gains/rough_load_probe.py left in docs/sim2real/data/gain_ab_eval/ and prints Korean
note tables on stdout. Run:  python3 tools/sim2real/gain_ab_tables.py <section>
sections: tb | turn | stand | rc | crank | probe | checks
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "docs/sim2real/data"
G = D / "gain_ab_eval"
ROUGH = Path(__file__).resolve().parents[2] / "tools/robot_gains/results/rough_load"


def turn_blocks(path):
  """-> {seed: dict(rows=[(cmd, ach, pct, wrong)], median, wrong)} from verdict logs (== m_<tag>_turn_s<N> == blocks)."""
  out, cur = {}, None
  for ln in Path(path).read_text().splitlines():
    m = re.match(r"=== m_.*_turn_s(\d+) ===", ln)
    if m:
      cur = int(m.group(1)); out[cur] = dict(rows=[], median=None, wrong=None); continue
    if cur is None:
      continue
    m = re.match(r"\s+(-?\d\.\d\d)\s+(-?\d\.\d+)\s+(-?\d+\.\d) %(\s+<- wrong way)?", ln)
    if m:
      out[cur]["rows"].append((float(m.group(1)), float(m.group(2)), float(m.group(3)), bool(m.group(4))))
    m = re.match(r"median (-?\d+\.\d) %\s+wrong way (\d)/8", ln)
    if m:
      out[cur]["median"], out[cur]["wrong"] = float(m.group(1)), int(m.group(2))
  return out


def turn_summary(blocks):
  meds = [b["median"] for _, b in sorted(blocks.items())]
  wrong = sum(b["wrong"] for b in blocks.values())
  return float(np.median(meds)), meds, wrong, 8 * len(blocks)


def main(sec):
  if sec == "tb":
    t = json.load(open(G / "tb_final_3runs.json"))
    names = [("Train/mean_reward", "평균 보상"), ("Train/mean_episode_length", "평균 에피소드 길이 (스텝)"),
             ("Episode_Termination/fell_over", "넘어짐 종료 (`fell_over`)"), ("Episode_Termination/low_base", "낮은 몸통 종료 (`low_base`)"),
             ("Episode_Termination/stagnation", "정체 종료 (`stagnation`)"), ("Episode_Reward/track_linear_velocity", "직선 속도 추종 보상"),
             ("Episode_Reward/track_angular_velocity", "회전 속도 추종 보상"), ("Metrics/twist/error_vel_xy_steady_moving", "움직일 때 속도 오차 [m/s]"),
             ("Curriculum/dr_levels/dr_factor", "무작위화 램프")]
    print("| 지표 | 기준선 | 묶음 시험 | 팔 A | 팔 A − 기준선 |\n|---|--:|--:|--:|--:|")
    for k, n in names:
      v = [t[r][k]["final200"] for r in ("base", "trial", "armA")]
      f = "{:.4f}" if "fell" in k else "{:.3f}" if abs(v[0]) < 10 else "{:.2f}"
      print(f"| {n} | " + " | ".join(f.format(x) for x in v) + f" | {f.format(v[2]-v[0])} |")
  elif sec == "turn":
    for lab, tag in (("기준선", "rbmv11a_final"), ("팔 A", "gcrank60_final")):
      b = turn_blocks(G / f"verdict_{tag}.txt")
      med, meds, wrong, tot = turn_summary(b)
      print(f"{lab}: 씨앗별 중앙값 {meds}, 중앙값 {med:.1f} %, 반대 방향 {wrong}/{tot}")
      cmds = sorted({r[0] for x in b.values() for r in x["rows"]})
      print("  명령별 중앙값 " + ", ".join(f"{c:+.2f}: {np.median([r[2] for x in b.values() for r in x['rows'] if r[0]==c]):.1f}" for c in cmds))
  elif sec == "rc":
    def load(f):
      by = {}
      for r in json.load(open(f))["rows"]:
        by.setdefault(r["cond"], []).append(r)
      return by
    B = load(D / "2026-10-15_rc_rbmv11a_final_brief_dec0.json")
    T = load(D / "2026-10-15_rc_trialv11a_final_brief_dec0.json")
    A = load(G / "rc_armA_brief_dec0.json")
    for c in B:
      row = []
      for X in (B, T, A):
        rs = X[c]; g = np.array([r["gyro_rms_dps"] for r in rs])
        row.append((sum(not r["fell"] for r in rs), len(rs), np.mean([r["survived_s"] for r in rs]), np.mean([r["act_p99"] for r in rs]),
                    np.mean([r["knee_std_deg"] for r in rs]), np.mean([r["knee_hz"] for r in rs]), float(g.mean())))
      print(f"| {c} | " + " | ".join(f"{x[0]}/{x[1]} | {x[2]:.1f} | {x[3]:.2f} | {x[4]:.1f} | {x[5]:.2f} | {x[6]:.0f}" for x in (row[0], row[2])) + f" | {row[1][0]}/{row[1][1]} |")


if __name__ == "__main__":
  main(sys.argv[1])
