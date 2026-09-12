#!/usr/bin/env python3
"""발목 크랭크 명령을 자르는 **여러 방식**을 같은 입출력으로 맞춰 놓은 곳.

2026-09-13 · 사용자 지시: "그쪽 로직들도 구현해서, 이전 만든 분석기에서 비교가능하게
드롭다운으로 추가."

전부 `ankle_clip.ClipResult` 를 냅니다 -- 받는 것은 명령 크랭크각 두 개, 내는 것은
나갈 크랭크각 두 개 + 그때의 발판 자세 + 어느 블럭이 반응했는지. 그래야 화면과 검사가
방식을 바꿔 끼우기만 하면 됩니다.

## 들어 있는 네 가지

    none            안 자름. 막는 것과 비교해 보려고 둡니다
    ours_pocs       우리 방식. 두 상자(모터/ROM)를 번갈아 투영 (docs/139)
    unitree_pr      Unitree G1 방식. 발판 자세 공간에서 한 번만 자르고 되돌림
    booster_torque  Booster T1 방식. 각도는 안 자르고 **힘**을 자름 (kp=0 + 토크 상한)

## 남의 코드에서 읽어 온 근거 (docs/142 에 자세히)

* Unitree `unitree_rl_gym` `deploy/deploy_real/deploy_real.py` 에는 **자르는 코드가
  한 줄도 없습니다.** 대신 `cmd.mode_pr = MotorMode.PR` 로 두는데, 주석이
  `PR = 0  # Series Control for Pitch/Roll Joints` / `AB = 1  # Parallel Control for
  A/B Joints` 입니다. 즉 **정책은 발판의 숙임·기울임을 명령하고, 크랭크로 바꾸는 일과
  한계 처리는 로봇 펌웨어가 합니다.** 우리 로봇에는 그 펌웨어 층이 없으므로, 같은
  생각을 우리 쪽에서 하면 "자세 공간에서 네모로 자르고 되돌린다" 가 됩니다.
* Booster `booster_gym` `deploy/deploy.py` 는 평행기구 관절
  (`cfg["mech"]["parallel_mech_indexes"]`, T1 은 [15, 16, 21, 22] = 발목 두 쌍)만
  따로 다룹니다. 위치 목표를 **안 보내고**, 호스트에서 토크를 만들어 잘라 보냅니다:

      motor_cmd[i].q  = dof_pos_latest[i]        # 목표 = 지금 위치
      motor_cmd[i].tau = clip((filtered_target[i] - dof_pos_latest[i]) * stiffness[i],
                              -torque_limit[i], +torque_limit[i])
      motor_cmd[i].kp = 0.0                       # 모터의 위치 고리를 끔

  **각도가 아니라 힘을 자릅니다.** T1 발목의 stiffness 50, torque_limit 24/15 N·m.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

from ankle_clip import AnkleClipper, ClipResult, _clamp  # noqa: E402

DEG = math.pi / 180.0

#: 심의 크랭크 세기 [N·m/rad]. `mujoco-sim` 의 액추에이터 값 그대로.
SIM_CRANK_KP = 22.265625
#: 크랭크 모터(RS03) 정격 토크 [N·m]. Booster 방식의 상한 기본값.
CRANK_TORQUE_LIMIT = 20.0

NAMES: Dict[str, str] = {
  "none": "안 자름 (막는 것을 눈으로 보려고)",
  "ours_pocs": "우리 방식 · 두 상자 번갈아 자르기",
  "unitree_pr": "Unitree G1 방식 · 발판 자세로 한 번만 자름",
  "booster_torque": "Booster T1 방식 · 각도 대신 힘을 자름",
}
ORDER = ["ours_pocs", "unitree_pr", "booster_torque", "none"]


class Strategy:
  """방식 하나. `clip(a, b)` 만 있으면 됩니다."""

  key = "none"

  def __init__(self, clipper: AnkleClipper):
    self.c = clipper

  # 방식마다 판정 기준이 같아야 비교가 됩니다. 결과가 ROM 안인지 여부는 **언제나**
  # 우리 기구학으로 다시 확인합니다 (방식이 스스로 "안전하다" 고 말하게 두지 않음).
  def _verdict(self, res: ClipResult) -> ClipResult:
    f = self.c.fk(res.a_out, res.b_out)
    if f is None:
      res.inside = False
      res.notes.append("최종 자세를 못 구함")
    else:
      res.pitch, res.roll = f
      res.inside = (self.c.rom_pitch[0] - 1e-3 <= res.pitch <= self.c.rom_pitch[1] + 1e-3
                    and self.c.rom_roll[0] - 1e-3 <= res.roll <= self.c.rom_roll[1] + 1e-3)
    if not res.inside:
      res.notes.append("★ 못 막았습니다 — 결과가 여전히 허용 영역 밖입니다")
    res.stage = res.stages[-1] if res.stages else "none"
    return res

  def clip(self, a: float, b: float) -> ClipResult:
    raise NotImplementedError

  def reset(self) -> None:
    pass


class NoClip(Strategy):
  """아무것도 안 함. 다른 방식이 무엇을 막고 있는지 보이게 하려고 둡니다."""

  key = "none"

  def clip(self, a: float, b: float) -> ClipResult:
    return self._verdict(ClipResult(a_in=a, b_in=b, a_out=a, b_out=b))


class OursPOCS(Strategy):
  """우리 방식. `ankle_clip.AnkleClipper.clip` 을 그대로 부릅니다."""

  key = "ours_pocs"

  def clip(self, a: float, b: float) -> ClipResult:
    return self.c.clip(a, b)

  def reset(self) -> None:
    self.c._last_ok = None
    self.c._guess = (0.0, 0.0)


class UnitreePR(Strategy):
  """Unitree G1 방식을 우리 로봇에 옮긴 것.

  G1 에서는 정책이 **발판의 숙임·기울임**을 명령하고 펌웨어가 크랭크로 바꿉니다
  (`mode_pr = PR`). 우리 화면은 크랭크 명령을 받으므로 같은 뜻이 되려면

      크랭크 명령 --(정기구학)--> 발판 자세 --(네모로 자름)--> --(역기구학)--> 크랭크

  **한 번만** 합니다. 되풀이도, 표도, "그 A 로는 어떤 B 도 못 간다" 판정도 없습니다 --
  G1 쪽에 그런 것이 없기 때문입니다. 그래서 이 방식은 우리 방식보다 **덜 막습니다**.
  얼마나 덜 막는지가 이 화면으로 보고 싶은 것입니다.
  """

  key = "unitree_pr"

  def clip(self, a: float, b: float) -> ClipResult:
    res = ClipResult(a_in=a, b_in=b, a_out=a, b_out=b)

    # 모터 한계는 G1 도 (펌웨어에서) 겁니다.
    a2 = _clamp(a, -self.c.limit, self.c.limit)
    b2 = _clamp(b, -self.c.limit, self.c.limit)
    if (a2, b2) != (a, b):
      res.stages.append("motor_box")
      res.notes.append(f"모터 한계 ±{self.c.limit:g}도로 자름")
    res.a_out, res.b_out = a2, b2

    f = self.c.fk(res.a_out, res.b_out)
    if f is None:
      res.stages.append("fk_failed")
      res.notes.append("정기구학이 안 풀림 — G1 방식에는 대비책이 없습니다")
      return self._verdict(res)

    p, r = f
    pc = _clamp(p, *self.c.rom_pitch)
    rc = _clamp(r, *self.c.rom_roll)
    res.pitch, res.roll = p, r
    if (pc, rc) == (p, r):
      return self._verdict(res)

    s = self.c.ik(pc, rc)
    if s is None:
      res.stages.append("ik_failed")
      res.notes.append("역기구학이 안 풀림 — G1 방식에는 대비책이 없습니다")
      return self._verdict(res)

    res.stages.append("rp_box")
    res.notes.append(f"발판 {p:+.1f}/{r:+.1f}도를 {pc:+.1f}/{rc:+.1f}도로 자르고 되돌림 (한 번)")
    res.a_out = _clamp(s[0], -self.c.limit, self.c.limit)
    res.b_out = _clamp(s[1], -self.c.limit, self.c.limit)
    res.rounds = 1
    return self._verdict(res)


class BoosterTorque(Strategy):
  """Booster T1 방식.

  각도를 **안 자릅니다.** 대신 지금 위치와 목표의 차이를 심과 같은 세기로 토크로 바꾸고,
  그 **토크를 자릅니다.** 모터에는 위치 게인 0 으로 순수 토크만 보냅니다.

  화면에서 "나가는 각도" 를 보여 주려면 그 토크에 해당하는 유효 목표로 되돌려 적어야
  합니다:  나가는 각 = 지금 위치 + 잘린 토크 / 세기.

  **이 방식은 발판을 ROM 안에 넣어 주지 않습니다.** 힘의 상한만 정합니다. 그것이
  Booster 의 선택이고, 화면에서 빨간 영역이 남는 것이 **버그가 아니라 그 방식의
  성질**입니다.
  """

  key = "booster_torque"

  def __init__(self, clipper: AnkleClipper, *, kp: float = SIM_CRANK_KP,
               torque_limit: float = CRANK_TORQUE_LIMIT):
    super().__init__(clipper)
    self.kp = float(kp)
    self.tmax = float(torque_limit)
    self.measured: Tuple[float, float] = (0.0, 0.0)

    # Booster 는 목표에 **1차 저역통과**도 겁니다 (deploy.py:
    # `filtered = filtered*0.8 + target*0.2`). 같은 필터가 학습 쪽에도 있어서
    # 심과 실물이 어긋나지 않습니다. 여기서도 같은 계수를 씁니다.
    self.alpha = 0.2
    self.filtered: Optional[Tuple[float, float]] = None

  def reset(self) -> None:
    self.measured = (0.0, 0.0)
    self.filtered = None

  def clip(self, a: float, b: float) -> ClipResult:
    res = ClipResult(a_in=a, b_in=b, a_out=a, b_out=b)

    if self.filtered is None:
      self.filtered = (a, b)
    else:
      self.filtered = (self.filtered[0] * (1 - self.alpha) + a * self.alpha,
                       self.filtered[1] * (1 - self.alpha) + b * self.alpha)
    fa, fb = self.filtered

    out = []
    hit = False
    for tgt, cur in ((fa, self.measured[0]), (fb, self.measured[1])):
      tau = (tgt - cur) * DEG * self.kp
      tau_c = _clamp(tau, -self.tmax, self.tmax)
      if abs(tau_c - tau) > 1e-9:
        hit = True
      out.append(cur + tau_c / (self.kp * DEG))
    res.a_out, res.b_out = out

    if abs(fa - a) > 1e-6 or abs(fb - b) > 1e-6:
      res.stages.append("lowpass")
      res.notes.append(f"1차 저역통과(계수 {self.alpha}) → A{fa:+.1f} B{fb:+.1f}")
    if hit:
      res.stages.append("torque_cap")
      res.notes.append(
        f"토크를 ±{self.tmax:g} N·m 로 자름 (세기 {self.kp:g} N·m/rad). "
        f"각도는 안 잘랐습니다 — 발판이 ROM 밖일 수 있습니다")
    return self._verdict(res)

  def accept(self, a: float, b: float) -> None:
    """모터가 실제로 그 자리에 갔다고 알려 준다 (다음 주기의 '지금 위치')."""
    self.measured = (a, b)


BUILDERS: Dict[str, Callable[[AnkleClipper], Strategy]] = {
  "none": NoClip,
  "ours_pocs": OursPOCS,
  "unitree_pr": UnitreePR,
  "booster_torque": BoosterTorque,
}


def make(key: str, clipper: AnkleClipper) -> Strategy:
  if key not in BUILDERS:
    raise KeyError(f"모르는 방식 {key!r}. 있는 것: {list(BUILDERS)}")
  return BUILDERS[key](clipper)
