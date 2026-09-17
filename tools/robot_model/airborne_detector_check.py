#!/usr/bin/env python3
"""**공중 감지기가 정말 갈리는지 확인한다.** 안 갈리면 그렇게 보고한다.

2026-09-17 · 사용자가 고른 방법(4): 토크로 공중을 감지해 자세를 고정.

`pygviewer/airborne.py` 의 생각은 이렇습니다 — 숫자를 미리 박지 않고, **지금 자세에서
모형에게** "허공에 매달렸다면 모터가 내야 할 힘" 을 물어본 뒤 실측과 견줍니다.
매달려 있으면 둘이 거의 같고, 딛고 있으면 바닥이 받쳐 주는 만큼 다릅니다.

이 스크립트는 그 생각이 **우리 로봇에서 실제로 통하는지** 봅니다. `airborne_signature.py`
가 남긴 기록(땅/공중 각각의 관절각과 토크)을 그대로 넣어 봅니다.

판정: 두 무리가 **안 겹쳐야** 씁니다. 겹치면 쓰면 안 됩니다 — 잘못 켜지면 **서 있는
로봇의 다리를 얼려서 넘어뜨립니다.** 못 쓴다는 결과도 결과입니다.

    airborne_detector_check.py --npz docs/data/airborne_v7b.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/pygviewer"))

from pygviewer.airborne import HangingTorqueModel                 # noqa: E402

XML = (REPO / "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls"
       / "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml")


def residuals(h: HangingTorqueModel, q_all, tau_all, jnames, anames) -> np.ndarray:
  """표본마다 `mean |tau_실측 - tau_매달림(q)|` 을 낸다."""
  ai = {n: i for i, n in enumerate(anames)}
  out = []
  for k in range(len(q_all)):
    q = {n: float(q_all[k][i]) for i, n in enumerate(jnames)}
    want = h.tau(q)
    got = np.array([tau_all[k][ai[n]] for n in h.act_names], dtype=float)
    out.append(float(np.abs(got - want).mean()))
  return np.array(out)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--npz", default=str(REPO / "docs/data/airborne_v7b.npz"))
  ap.add_argument("--xml", default=str(XML))
  ap.add_argument("--stride", type=int, default=5, help="표본 건너뛰기 (역동역학이 느림)")
  n = ap.parse_args(argv)

  import mujoco

  d = np.load(n.npz, allow_pickle=True)
  anames = [str(x) for x in d["act_names"]]
  jnames = [str(x) for x in d["joint_names"]]
  m = mujoco.MjModel.from_xml_path(n.xml)
  # 모형이 아는 이름으로 맞춘다 (`robot/` 접두어가 붙는 판이 있다)
  h = HangingTorqueModel(m, jnames)

  s = max(1, n.stride)
  rg = residuals(h, d["ground_q"][::s], d["ground_tau"][::s], jnames, anames)
  ra = residuals(h, d["air_q"][::s], d["air_tau"][::s], jnames, anames)

  print(f"\n표본 {len(rg)} + {len(ra)}   (관절 {len(jnames)}개, 모형 {Path(n.xml).name})")
  print(f"{'상황':<10}{'최소':>10}{'평균':>10}{'최대':>10}{'표준편차':>10}")
  print(f"{'땅':<10}{rg.min():>10.2f}{rg.mean():>10.2f}{rg.max():>10.2f}{rg.std():>10.2f}")
  print(f"{'공중':<10}{ra.min():>10.2f}{ra.mean():>10.2f}{ra.max():>10.2f}{ra.std():>10.2f}")

  gap = float(rg.min() - ra.max())
  ok = gap > 0
  print(f"\n틈(땅 최소 - 공중 최대) = {gap:+.2f} N*m  ->  "
        + ("**갈립니다.** 그 사이 어디에 선을 그어도 두 상황이 나뉩니다."
           if ok else "**겹칩니다. 쓰면 안 됩니다** — 서 있는데 켜질 수 있습니다."))
  if ok:
    thr = 0.5 * (rg.min() + ra.max())
    margin = min(rg.min() - thr, thr - ra.max())
    print(f"권하는 문턱값: enter={thr:.2f} (양쪽 여유 {margin:.2f} N*m)")
  out = dict(npz=n.npz, ground=dict(min=rg.min(), mean=rg.mean(), max=rg.max()),
             air=dict(min=ra.min(), mean=ra.mean(), max=ra.max()),
             gap=gap, usable=bool(ok))
  p = REPO / "docs/data/airborne_detector_check.json"
  p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
               encoding="utf-8")
  print(f"기록: {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
