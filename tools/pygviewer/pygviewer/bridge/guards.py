#!/usr/bin/env python3
"""HUPHY 에 **없는** 안전장치들. 제어 루프의 `Motion` 을 감싸는 방식으로 씁니다.

2026-09-13 · `System_ID/huphy_live.py` 에서 옮겨왔습니다. 옮긴 이유는 그 파일이
`bridge/huphy_remote_motion.py` 와 같은 일을 중복해서 하고 있었기 때문입니다
(`docs/137`, `docs/138`). 목표를 받고 HUPHY 로 넣는 일은 브리지가 하고, **HUPHY 에
없어서 우리가 만든 것만** 여기 남습니다.

    AnklePairGuard   발목 A·B 를 **조합으로** 자름. HUPHY 는 모터 공간(AB)에서
                     두 각을 따로만 자르고 조합은 아무도 안 봄
    ThermalCutoff    권선이 뜨거우면 그 관절 목표를 지금 자리로 묶음.
                     **HUPHY 는 온도로 아무것도 막지 않음**
    GainControl      관절별 세기·감쇠를 돌면서 바꿈. 브리지에는 상한만 있음

감싸는 순서는 바깥에서 안으로 `AnklePairGuard(ThermalCutoff(목표보유))` 입니다 --
온도로 묶인 목표도 발목 조합 검사를 거쳐야 하기 때문입니다.

**HUPHY 를 모듈 수준에서 불러오지 않습니다.** 브리지가 그렇듯, 이 파일도 하드웨어가
없는 개발 기계에서 그대로 불러와 시험할 수 있어야 합니다. `GainControl` 만 실제로
게인을 갈아 끼울 때 `huphy.config.schema` 를 늦게 부릅니다.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("guards")


def limbs_of(machine):
  """팔다리 목록. 다리 하나짜리면 빈 목록.

  ★ 속성 이름은 `parts` 입니다 (`robots/biped.py:122`). 제가 `limbs` 로 짐작해
  써 놨다가 2026-09-12 에 양다리 구동이 통째로 실패했습니다 — 접두사가 안 붙어
  `ValueError: 모르는 관절 [...]. 이름 앞에 팔다리를 붙일 것` 으로 루프가 섰습니다.
  짐작한 이름을 여러 군데 흩어 놓았던 것이 문제라, 이제 여기 한 곳에만 둡니다.
  """
  return list(getattr(machine, "parts", ()) or ())


def joint_prefix(machine, part) -> str:
  """양다리면 `왼다리/`, 다리 하나면 빈 문자열.

  접두사는 **팔다리의 id** 입니다 (`biped.py:84 join_name`). 설정의 `name` 과
  같을 수도 다를 수도 있으니 id 를 씁니다.
  """
  if not limbs_of(machine):
    return ""
  return f"{getattr(part, 'id', getattr(part.config, 'name', ''))}/"


# ===========================================================================
# HUPHY 가 안 해 주는 것 (2) — 발목 두 모터를 **한 쌍으로** 자르기
# ===========================================================================
# 과열 차단은 **여기 없습니다** — 브리지에 이미 있습니다
# ===========================================================================
#
# 2026-09-13. `System_ID/huphy_live.py` 의 `ThermalCutoff` 를 여기로 옮겼다가
# **바로 되물렸습니다.** `bridge/motor_fault.py` 에 같은 것이 이미 있고, 그쪽이 더
# 강합니다 -- 50도에서 끊고 45도에서 되살리는 이력(履歷)이 있고, 목표를 지금 자리로
# 묶는 데서 그치지 않고 **게인까지 0 으로** 내려 토크를 확실히 없앱니다.
# `huphy_remote_motion._update_thermal_cutoff` 가 매 주기 그것을 돌리고,
# `{관절}/temp_valid` · `{관절}/cutoff` 를 관측으로 내보냅니다.
#
# 옮기기 전에 목적지를 안 봤습니다. 같은 실수를 문서로 남겨 둡니다 (`docs/138` 정정).

# ===========================================================================
# 로봇 한 마리에서 팔다리를 꺼내는 자리 — **이름을 여기 한 곳에만 적는다**
# ===========================================================================

ANKLE_AXIS_DEG = 22.75
"""발목 크랭크 한 축이 혼자 갈 수 있는 한계.

**실측으로 고른 값입니다.** HUPHY 의 발목 기구학으로 A·B 격자를 훑어, 네 귀퉁이가
전부 발판 자세 한계 안에 드는 가장 큰 정사각형을 찾았습니다. 아래 `ANKLE_ROM_*`
기준으로 잰 값이고, 값을 바꾸면 이것도 다시 재야 합니다 —
`tools/huphy_audit/e2e_live.py` 의 E27 이 그 일치를 검사합니다."""

ANKLE_ROM_PITCH = (-50.0, 30.0)
ANKLE_ROM_ROLL = (-20.0, 20.0)
"""발판이 실제로 가도 되는 자세. **시뮬레이터 모델에서 그대로 읽은 값입니다.**

사용자 지시(2026-09-12) *"rp 모드 mjcf / urdf 확인해봐"* 를 따라 확인했습니다.
`mjlab/.../pygmalion/xmls/*.xml` 의 모든 판이 같은 값을 갖고 있고 양발이 같습니다.

    <joint name="L_ankle_pitch_joint" range="-0.872665 0.523599"/>   -> -50.00 ~ +30.00도
    <joint name="L_ankle_roll_joint"  range="-0.349066 0.349066"/>   -> -20.00 ~ +20.00도

★ 제가 처음에 방향을 몰라 "두 해석의 교집합인 ±30" 으로 잡아 두었던 것을 **실제
값으로 바꿨습니다.** 숙임은 한쪽으로 치우쳐 있습니다 — 한쪽 50도, 다른 쪽 30도.

HUPHY 자신의 `AnkleEnvelope` 기본값(숙임 ±40 · 기울임 ±25,
`kinematics/ankle.py:148`)은 **다른 값**이고, 그 코드가 스스로 "소프트웨어 시험
범위이지 링키지가 닿는 범위가 아니다" 라고 적어 두었습니다. 우리는 학습이 쓰는
실제 가동범위를 씁니다."""


class AnklePairGuard:
  """발목 A·B 를 **두 겹으로** 자른다. 축마다 한 번, 조합으로 한 번.

  ## 왜 필요한가

  발목은 모터 두 개가 로드 두 개로 발판 하나를 미는 **닫힌 기구**입니다. 그래서
  두 각도를 따로 자르면 모자랍니다 — 각각은 범위 안이어도 **그 조합이 발판을
  한계 밖으로** 보낼 수 있습니다.

  HUPHY 는 발판 공간(`ankle_pitch`/`ankle_roll`)으로 줄 때만 기구학으로 조합을
  봅니다. 우리는 **모터 공간(`ankle_a`/`ankle_b`)** 으로 줍니다 — 지금 구현체도
  학습된 정책도 그 공간으로 돌기 때문입니다(사용자 결정 2026-09-12). 그 공간에서는
  HUPHY 가 A·B 를 **따로따로** 자르고 조합은 아무도 안 봅니다.

  실측한 구멍: A와 B를 각자 ±53.75도까지 열면 A+53.8/B-53.8 에서 **발판이 67.4도
  숙고**, A+53.8/B+53.8 에서 **45.2도 기웁니다.**

  ## 어떻게 자르는가 — **둘 다 만족해야 한다** (사용자 지시: "and 문으로 클리핑")

      1) 축마다      |A| <= ANKLE_AXIS_DEG,  |B| <= ANKLE_AXIS_DEG
      2) 조합으로    solve_fk(A, B) 가 ROM 안

  둘 중 하나라도 어긋나면 **직전에 받아들인 자리 쪽으로 물러납니다.** 거부하지 않고
  자르는 이유는, 한쪽 모터만 거부하면 두 로드가 서로 다른 자세를 요구해 기구가
  비틀리기 때문입니다 — 둘은 같이 움직이거나 같이 물러나야 합니다.

  물러나는 방법은 **반씩 좁히기**입니다. 직전 자리는 정의상 안전하므로, 그 사이
  어딘가에 경계가 있습니다.
  """

  def __init__(self, inner, robot, *, axis_deg: float = ANKLE_AXIS_DEG,
               rom_pitch=ANKLE_ROM_PITCH, rom_roll=ANKLE_ROM_ROLL, steps: int = 12):
    self.inner = inner
    self.axis = abs(float(axis_deg))
    self.rom_pitch = tuple(rom_pitch)
    self.rom_roll = tuple(rom_roll)
    self.steps = int(steps)
    self.axis_clipped = 0
    self.pair_clipped = 0
    self.no_kinematics = 0
    # 다리별 (발목A 이름, 발목B 이름, 기구학) 과 직전에 받아들인 자리
    self.legs = []
    for part in limbs_of(robot) or [robot]:
      kin = getattr(part, "kinematics", None)
      cfg = getattr(part, "config", None)
      if kin is None or cfg is None:
        continue
      prefix = joint_prefix(robot, part)
      self.legs.append(dict(a=prefix + "ankle_a", b=prefix + "ankle_b",
                            kin=kin, last=(0.0, 0.0)))

  def _inside(self, kin, a: float, b: float) -> bool:
    """이 조합이 발판을 ROM 안에 두는가. 기구학이 못 풀면 **안 되는 것으로 본다.**"""
    try:
      pitch, roll = kin.solve_fk(a, b)
    except Exception:                                     # noqa: BLE001
      return False
    return (self.rom_pitch[0] <= pitch <= self.rom_pitch[1]
            and self.rom_roll[0] <= roll <= self.rom_roll[1])

  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    action = dict(self.inner(t, observation))
    for leg in self.legs:
      na, nb = leg["a"], leg["b"]
      if na not in action or nb not in action:
        continue
      # (1) 축마다
      a = max(-self.axis, min(self.axis, float(action[na])))
      b = max(-self.axis, min(self.axis, float(action[nb])))
      if a != action[na] or b != action[nb]:
        self.axis_clipped += 1
      # (2) 조합으로 -- 둘 다 만족해야 통과
      if self._inside(leg["kin"], a, b):
        leg["last"] = (a, b)
      else:
        la, lb = leg["last"]
        if not self._inside(leg["kin"], la, lb):
          # 직전 자리마저 밖이면 (첫 주기 등) 원점으로. 원점은 늘 안전하다.
          la = lb = 0.0
        lo, hi = 0.0, 1.0            # 직전 자리(0) 에서 원하는 자리(1) 사이
        for _ in range(self.steps):
          mid = 0.5 * (lo + hi)
          if self._inside(leg["kin"], la + (a - la) * mid, lb + (b - lb) * mid):
            lo = mid
          else:
            hi = mid
        a, b = la + (a - la) * lo, lb + (b - lb) * lo
        leg["last"] = (a, b)
        self.pair_clipped += 1
      action[na], action[nb] = a, b
    return action


# ===========================================================================
# 상태를 내보내는 통로 — **버스를 읽는 곳은 여기 하나뿐이다**
# ===========================================================================

class GainControl:
  """관절마다 세기·감쇠를 바꾼다. **돌아가는 중에도 바뀐다.**

  로봇 프로그램은 매 주기 `motor.gains.kp` 를 읽어 프레임에 싣습니다
  (`robots/leg.py:562`). 그래서 설정의 `Motor` 를 갈아 끼우면 **다음 주기부터**
  적용됩니다. 루프를 세웠다 다시 띄울 필요가 없습니다.

  `Motor` 는 얼린 자료형이라 `motor.gains = ...` 로는 못 바꿉니다 — 통째로 새
  `Motor` 를 만들어 사전의 그 자리에 넣습니다. 사전 항목 하나를 바꾸는 것은
  가리키는 곳을 옮기는 한 번의 동작이라, 제어 루프가 그 사이에 반쪽짜리 값을
  읽지 않습니다.

  ## 상한을 두는 이유

  세기를 크게 주면 **같은 오차에 토크가 그만큼 커집니다.** 세기 10에 오차 6도면
  약 1뉴턴미터인데, 100이면 10뉴턴미터입니다. 숫자를 잘못 한 자리 더 치는 것으로
  그렇게 됩니다. 그래서 상한을 두고, 넘으면 **상한으로 자르고 그 사실을 알립니다.**

  ★ 프로토콜 눈금도 같이 봅니다. 세기는 12비트로 0~5000 을 나눠 담으므로 눈금이
  1.2210 입니다. 그보다 작은 값은 **0으로 나가** 힘이 아예 안 들어갑니다 —
  "약하게 해 뒀다" 고 믿는데 실은 0인 반대 방향의 위험이라, 여기서 걸러 알립니다.
  """

  QUANTUM = 5000.0 / 4095.0            # 세기 눈금, 약 1.2210
  KD_QUANTUM = 100.0 / 4095.0          # 감쇠 눈금, 약 0.02442

  def __init__(self, machine, *, kp_cap: float, kd_cap: float):
    self.machine = machine
    self.kp_cap = float(kp_cap)
    self.kd_cap = float(kd_cap)
    self._lock = threading.Lock()
    self.changes = 0
    self.clipped = 0
    self.note = ""
    # 관절 이름 -> (팔다리, 모터 이름)
    self._where = {}
    for part in limbs_of(machine) or [machine]:
      cfg = getattr(part, "config", None)
      if cfg is None:
        continue
      for joint in cfg.motors:
        self._where[joint_prefix(machine, part) + joint] = (part, joint)

  def current(self) -> Dict[str, Dict[str, float]]:
    out = {}
    for full, (part, joint) in self._where.items():
      g = part.config.motors[joint].gains
      out[full] = dict(kp=round(float(g.kp), 3), kd=round(float(g.kd), 3))
    return out

  def set(self, joint: str, kp=None, kd=None) -> Dict[str, Any]:
    """한 관절의 세기·감쇠를 바꾼다. 상한 밖이면 잘라서 넣고 알린다."""
    # ★ HUPHY 를 부르지 않습니다. 지금 걸려 있는 게인 객체를 그대로 복사해
    # 숫자만 갈아 끼웁니다 -- 하드웨어가 없는 기계에서도 이 파일이 돌아야 합니다.
    from dataclasses import replace
    where = self._where.get(joint)
    if where is None:
      tail = joint.split("/")[-1]
      hits = [k for k in self._where if k.split("/")[-1] == tail]
      if len(hits) != 1:
        return dict(ok=False, why=f"모르거나 모호한 관절입니다: {joint}")
      where = self._where[hits[0]]
      joint = hits[0]
    part, name = where
    with self._lock:
      old = part.config.motors[name].gains
      want_kp = float(old.kp if kp is None else kp)
      want_kd = float(old.kd if kd is None else kd)
      if not (math.isfinite(want_kp) and math.isfinite(want_kd)):
        return dict(ok=False, why="숫자가 아닙니다")
      new_kp = max(0.0, min(self.kp_cap, want_kp))
      new_kd = max(0.0, min(self.kd_cap, want_kd))
      notes = []
      if new_kp != want_kp or new_kd != want_kd:
        self.clipped += 1
        notes.append(f"상한({self.kp_cap:g}/{self.kd_cap:g})으로 자름")
      if 0.0 < new_kp < self.QUANTUM:
        notes.append(f"세기 {new_kp:.3f} 는 눈금 {self.QUANTUM:.4f} 보다 작아 "
                     f"**0으로 나갑니다** — 힘이 아예 안 들어갑니다")
      part.config.motors[name] = replace(
        part.config.motors[name], gains=replace(old, kp=new_kp, kd=new_kd))
      self.changes += 1
      # ★ 통신선에 실제로 실리는 값은 **눈금으로 잘립니다.** 세기를 15 로 넣으면
      # 모터는 14.65 를 받습니다. 사람이 넣은 숫자와 다르므로 같이 알립니다.
      eff_kp = math.floor(new_kp / self.QUANTUM) * self.QUANTUM
      eff_kd = math.floor(new_kd / self.KD_QUANTUM) * self.KD_QUANTUM
      if abs(eff_kp - new_kp) > 1e-6:
        notes.append(f"통신선에는 {eff_kp:.2f} 로 나갑니다 (눈금 {self.QUANTUM:.4f})")
      self.note = f"{joint} 세기 {new_kp:g} 감쇠 {new_kd:g}" + (
        " (" + " · ".join(notes) + ")" if notes else "")
    logger.info("%s", self.note)
    return dict(ok=True, joint=joint, kp=new_kp, kd=new_kd,
                kp_on_wire=round(eff_kp, 3), kd_on_wire=round(eff_kd, 3),
                notes=notes)


# ===========================================================================
# 명령을 받는 통로
# ===========================================================================
