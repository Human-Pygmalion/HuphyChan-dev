#!/usr/bin/env python3
"""**붙어 있다 떨어지는 힘(정지마찰)을 잰다.** 미끄러질 때의 마찰과 얼마나 다른가.

2026-09-11 · 사용자 지시: 벤치 A2.

왜 재는가
---------
시뮬레이터는 마찰을 **숫자 하나**로만 다룹니다(`frictionloss`). 그런데 실제 기계는 보통
**떨어지는 힘 > 미끄러지는 힘** 입니다. 그러면 실물만 들러붙고 시뮬레이터는 안 그럽니다 —
느리게 걷거나 서 있을 때 바로 드러납니다.

그리고 ±10도 실험에서 본 **방향 쏠림 0.4~0.5도**의 남은 원인 후보이기도 합니다(코깅으로는
설명이 안 됐습니다).

어떻게 재는가
-------------
멈춘 축에 토크를 **아주 천천히** 올려서, 움직이기 시작하는 순간의 토크를 적습니다. 천천히
올리는 것이 중요합니다 — 빨리 올리면 관성 때문에 실제보다 크게 나옵니다.

★ 코깅이 섞입니다. 코깅은 자리에 따라 ±0.23 뉴턴미터까지 밀고 당기므로, **한 자리에서 잰
값 하나는 못 믿습니다.** 그래서 축을 조금씩 옮겨 가며 **여러 자리에서, 양방향으로** 재고
평균을 냅니다. 자리마다의 흩어짐도 같이 적습니다 — 그게 곧 코깅의 크기입니다.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_profiles import home  # noqa: E402
from rs_direct_socket import Motor  # noqa: E402


def settle(m: Motor, still_deg: float = 0.03, window_s: float = 0.4, timeout_s: float = 4.0):
  """축이 **완전히 설 때까지** 기다린다. 돌려주는 값은 그때의 자리.

  ★ 2026-09-11 — 이것이 없어서 첫 측정이 망가졌습니다. 원점 복귀 직후에는 축이 미세하게
  흐르는데, 그 흐름이 '움직이기 시작했다'로 잡혀 **떨어지는 힘이 0** 으로 기록됐습니다.
  열두 자리 중 아홉 자리가 그랬습니다."""
  t0 = time.monotonic()
  buf = []
  while time.monotonic() - t0 < timeout_s:
    m.torque(0.0)
    st = m.recv(timeout_s=0.004)
    if st is None:
      continue
    buf.append((time.monotonic(), st.position_deg))
    while len(buf) > 2 and buf[-1][0] - buf[0][0] > window_s:
      buf.pop(0)
    if buf[-1][0] - buf[0][0] >= window_s:
      span = max(p for _, p in buf) - min(p for _, p in buf)
      if span < still_deg:
        return buf[-1][1]
  return buf[-1][1] if buf else float("nan")


def one_breakaway(m: Motor, sign: float, rate: float, move_deg: float, tau_cap: float):
  """토크를 천천히 올려 **움직이기 시작하는 순간**의 토크. 못 움직이면 None."""
  p0 = settle(m)
  t0 = time.monotonic()
  tau = 0.0
  while tau < tau_cap:
    tau = rate * (time.monotonic() - t0)
    m.torque(sign * tau)
    st = m.recv(timeout_s=0.004)
    if st is None:
      continue
    if abs(st.position_deg - p0) > move_deg:
      for _ in range(30):
        m.torque(0.0); m.recv(timeout_s=0.002)
      return tau
  for _ in range(30):
    m.torque(0.0); m.recv(timeout_s=0.002)
  return None


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--places", type=int, default=16, help="몇 자리에서 잴까 (한 바퀴에 고루)")
  ap.add_argument("--rate", type=float, default=0.6, help="토크를 초당 몇 뉴턴미터씩 올릴까")
  ap.add_argument("--move-deg", type=float, default=0.20, help="이만큼 움직이면 '떨어졌다'")
  ap.add_argument("--tau-cap", type=float, default=4.0)
  ap.add_argument("--out", default=os.path.expanduser("~/System_ID/data/stiction.csv"))
  a = ap.parse_args(argv)

  rows = []
  with Motor(motor_id=a.motor_id, model=a.model) as m:
    s = m.read()
    if s is None:
      print("모터가 응답하지 않습니다", file=sys.stderr); return 1
    print(f"모터 {a.motor_id}({a.model}) · 시작 {s.position_deg:.2f}도 · {s.temp_c:.0f}도")
    print(f"토크를 초당 {a.rate} 뉴턴미터씩 올려 {a.move_deg}도 움직이는 순간을 적습니다 · "
          f"{a.places}자리 x 양방향\n")
    m.clear_fault(); time.sleep(0.1); m.enable(); time.sleep(0.1)
    base = m.read(tries=10).position_deg
    step = 360.0 / a.places
    print(f"  {'자리':>8}{'+ 방향':>10}{'− 방향':>10}{'평균':>9}{'온도':>7}")
    for i in range(a.places):
      tgt = base + i * step
      home(m, target_deg=tgt, tol_deg=6.0, runaway_deg=500.0, timeout_s=25.0)
      m.clear_fault(); time.sleep(0.05); m.enable(); time.sleep(0.05)
      out = {}
      # 순서를 자리마다 번갈아 둔다. 늘 + 를 먼저 재면, − 는 방금 + 가 움직여 놓은 자리에서
      # 시작하게 되어 한쪽으로 치우친다.
      order = ((+1.0, "p"), (-1.0, "n")) if i % 2 == 0 else ((-1.0, "n"), (+1.0, "p"))
      for sgn, nm in order:
        v = one_breakaway(m, sgn, a.rate, a.move_deg, a.tau_cap)
        out[nm] = v
        time.sleep(0.5)
      st = m.read(tries=8)
      both = [v for v in out.values() if v is not None]
      print(f"  {tgt:8.0f}{(out['p'] if out['p'] else float('nan')):10.3f}"
            f"{(out['n'] if out['n'] else float('nan')):10.3f}"
            f"{(statistics.mean(both) if both else float('nan')):9.3f}{st.temp_c:7.0f}")
      rows.append((tgt, st.position_deg, out["p"], out["n"], st.temp_c))
      if st.temp_c > 45:
        print("  ★ 온도 45도 — 멈춥니다"); break
    home(m, target_deg=base, tol_deg=8.0, runaway_deg=500.0)

  ps = [r[2] for r in rows if r[2] is not None]
  ns = [r[3] for r in rows if r[3] is not None]
  al = ps + ns
  print(f"\n  + 방향 {statistics.mean(ps):.3f} ± {statistics.pstdev(ps):.3f} (n={len(ps)})")
  print(f"  − 방향 {statistics.mean(ns):.3f} ± {statistics.pstdev(ns):.3f} (n={len(ns)})")
  print(f"  ★ 떨어지는 힘 전체 평균 **{statistics.mean(al):.3f}** ± "
        f"{statistics.pstdev(al):.3f} 뉴턴미터")
  with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["target_deg", "pos_deg", "breakaway_plus", "breakaway_minus",
                                   "temp_c"])
    w.writerows(rows)
  print(f"  기록: {a.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
