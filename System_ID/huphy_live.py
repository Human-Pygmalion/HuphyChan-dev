#!/usr/bin/env python3
"""화면에서 준 목표를 **로봇 프로그램(HUPHY) 의 제어 루프에** 그대로 태운다.

2026-09-12 사용자 지시: *"애초에 내가 HUPHY 레포의 명령 넣는 코드 활용해서 개발하라
했는데 왜 안 했지? 당장 해."*

맞는 지시입니다. 그전까지는 `cmd_guard.py` 가 한계 클리핑·속도 클리핑·발목 쌍
검사·MIT 프레임 인코딩을 **전부 다시 만들어** 쓰고 있었습니다. HUPHY 에 이미 다 있고,
실물에서 검증된 것도 그쪽입니다.


## 어디에 끼워 넣는가

HUPHY 의 제어 루프가 매 주기 부르는 것은 **딱 한 줄**입니다.

    action = motion(t, observation)        # control/loop.py:270

계약은 이렇습니다 (`control/loop.py:155`, `robots/base.py:47`).

    Motion = Callable[[float, Dict[str, Any]], Optional[Action]]
    Action = Mapping[str, float]     # 관절 이름 -> 목표 각도 (cal 공간, **도**)

그러므로 우리가 만들 것은 **그 모양의 물건 하나**입니다. 그 아래(한계·속도·프레임·
전송·수거)는 전부 HUPHY 가 합니다.


## 이 파일이 물려받는 안전장치 (HUPHY 가 적용하는 순서대로)

    1  발목 도달성        kinematics/ankle.py:296  solve_ik 실패 시 두 모터 통째 버림
    2  발목 공간 혼합 거부  robots/leg.py:443
    3  모르는 관절 거부     robots/leg.py:485       ValueError -> 루프 정지
    4a 유한값 검사         safety/guards.py:75     NaN/Inf 면 그 모터에 아무것도 안 보냄
    4b 상태 미확보 거부     safety/guards.py:107    측정값이 한 번도 없으면 안 보냄
    4c 위치 한계 클리핑     safety/limits.py:66     하드스톱 - command_margin_deg
    4d 한 주기 이동량 클리핑 safety/guards.py:85     **직전 측정 위치** 기준 max_delta_deg
    5  cal -> raw 변환     robots/leg.py:380
    6  MIT 프레임 인코딩    motors/robstride/codec/mit.py:105
    7  통신 두절 감시       safety/link.py          연속 무응답이면 루프 정지
    8  정지 완충           control/loop.py:377     자세를 붙잡은 5주기 뒤 토크 차단

HUPHY 가 **안 해 주는 것**은 이 파일이 합니다 — 권선 과열 차단 하나입니다.

~~그리고 켜기 명령이 실제로 먹혔는지 확인하는 것~~ **철회 (2026-09-12)**: 모터가
보고하는 동작모드 비트는 **모델에 따라 안 올라옵니다**. 그걸 근거로 삼았다가 멀쩡한
모터 일곱 개를 고장으로 몰았습니다. 아래 `MODE_IS_REPORTED` 참조.


## 목표를 부분만 갱신하는 이유

`Leg._motor_targets` 는 **넘어온 키만** 순회합니다 (`robots/leg.py:475`). 어떤 관절
이름이 이번 주기 dict 에 없으면 그 모터에는 **아무 명령도 안 나가고**, 모터는 펌웨어에
남아 있는 **직전 명령을 계속 실행**합니다. "지금 자리를 유지" 가 아닙니다.

그래서 `LiveTargets` 는 **항상 열두 관절 전부를 들고 있고**, 화면에서 온 값으로
`dict.update` 만 합니다. 통째로 갈아끼우지 않습니다.


## 응답을 기다리는 창 — 3밀리초

토크 없이(`--gain-scale 0`) 창만 바꿔 가며 쟀습니다. **짧게 재면 0이 최선처럼
보이는데, 오래 재면 뒤집힙니다.**

    창        10초 (양다리)        90초 (양다리)
    0 ms      85Hz · 무응답 0      **20초 만에 통신두절로 죽음**
    2 ms      66Hz                 66.2Hz · 버팀
    **3 ms**  68Hz                 **68.0Hz · 버팀**
    5 ms      57Hz                 61.1Hz · 버팀

**창 0은 빠른 대신 죽습니다.** 기다리지 않으면 그 주기에 "응답 없음" 이 자주 찍히고,
로봇 프로그램은 연속 다섯 번이면 통신두절로 봅니다(`safety/link.py:44`). 20초를 못
넘겼습니다.

☆ **철회 2번 (2026-09-12)**: 같은 날 ~~"창 5가 낫다"~~ 고 했다가(한 다리만 보고),
~~"창 0이 최선"~~ 이라고 뒤집었는데(10초만 재고), 90초로 재니 **창 0이 죽습니다.**
짧게 재고 결론 낸 것이 두 번 다 원인이었습니다.

★ 참고: 벤치에서 모터 한두 개로 연속 제어가 잘 되던 것은 **왕복이 0.53밀리초**라
5밀리초 안에 여유롭게 들어왔기 때문입니다. 12개는 한 주기에 6.28밀리초가 들어
사정이 다릅니다. 벤치에서 통하던 감각을 그대로 옮기면 안 됩니다.

무응답이 여전히 많은 것(90초에 2472~5793주기)은 **대기창으로 풀 문제가 아닙니다** —
"응답이 왔다" 의 정의가 위상에 좌우되기 때문입니다. 밀린 개수로 세는 설계안이
`docs/132_link_health_design.md` 에 있습니다.

원본: `docs/experiments/2026-09-12_loop_timing.md`


## 돌리는 법

    huphy_live.py --config ~/HUPHY/config/robot_v1.0.yaml --robot
    huphy_live.py --config ~/HUPHY/config/robot_v1.0.yaml --limb left_leg
    huphy_live.py ... --dry-run        # 계산만 하고 CAN 으로 안 보냄

`--listen` 으로 받는 목표는 `{"name": "<관절>", "deg": <도>}` 한 줄짜리 JSON 입니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

logger = logging.getLogger("huphy_live")


# ===========================================================================
# 목표를 들고 있는 물건 — 이것이 HUPHY 의 Motion 계약을 만족한다
# ===========================================================================
class LiveTargets:
  """화면이 갱신하고 제어 루프가 읽는 목표 표. **HUPHY 의 `Motion` 그 자체다.**

  화면은 초당 10번쯤 갱신하고 제어 루프는 초당 100번 읽습니다. 그래서 같은 값이
  여러 주기 되풀이되는데, 그것이 만드는 계단은 HUPHY 의 한 주기 이동량 클리핑
  (`safety/guards.py:85`)이 완충합니다.

  ★ 살아 움직이는 값을 넘겨도 되는가: 됩니다. HUPHY 자신이 그렇게 씁니다 —
  `scripts/run.py:152 staged()` 가 클로저 안 가변 상태를 **바깥 스레드**에서
  바꿉니다(Enter 를 기다리는 스레드). 루프는 반환값이 시간만의 함수인지 검사하지
  않습니다.
  """

  def __init__(self, joint_names, initial: Optional[Dict[str, float]] = None):
    self.joint_names = tuple(joint_names)
    self._lock = threading.Lock()
    # **항상 전부 들고 있는다.** 위 "부분만 갱신하는 이유" 참조.
    self._targets: Dict[str, float] = {n: 0.0 for n in self.joint_names}
    if initial:
      self.update(initial)
    self.rx = 0
    self.rejected_unknown = 0
    self.rejected_not_finite = 0
    self.note = "아직 명령 없음"
    self.last_rx_mono: Optional[float] = None
    self._measured: Dict[str, float] = {}
    self._seeded = False
    self._fresh = False
    self._blocked = 0

  # ---- 화면 쪽에서 부르는 것 ---------------------------------------------
  def update(self, partial: Dict[str, float]) -> Dict[str, Any]:
    """**부분 갱신만.** 통째로 갈아끼우지 않는다.

    모르는 이름은 **여기서** 거른다. HUPHY 쪽까지 흘려보내면 `leg.py:485` 가
    `ValueError` 를 던지고 **루프 전체가 멈춥니다** — 오타 하나로 로봇이 서는 것은
    과합니다. 대신 세어 두고 화면에 알립니다.
    """
    bad, notfinite, taken = [], [], {}
    for name, value in partial.items():
      name = self._resolve(name)
      if name is None or name not in self._targets:
        bad.append(name if name is not None else "(모호함)")
        continue
      try:
        v = float(value)
      except (TypeError, ValueError):
        notfinite.append(name)
        continue
      if not math.isfinite(v):
        notfinite.append(name)
        continue
      taken[name] = v
    with self._lock:
      self._targets.update(taken)
      self.rx += len(taken)
      self.rejected_unknown += len(bad)
      self.rejected_not_finite += len(notfinite)
      if taken:
        self.last_rx_mono = time.monotonic()
        one = next(iter(taken))
        self.note = f"{one} 목표 {taken[one]:+.2f}도" + (
          f" 외 {len(taken)-1}개" if len(taken) > 1 else "")
    return dict(ok=not (bad or notfinite), taken=sorted(taken),
                unknown=bad, not_finite=notfinite)

  def _resolve(self, name: str) -> Optional[str]:
    """보내는 쪽 표기를 우리 표기로 맞춘다. 모호하면 `None`.

    제어 루프가 양다리면 이름에 팔다리가 붙고(`left_leg/hip_yaw`), 다리 하나면
    안 붙습니다(`hip_yaw`). 보내는 쪽(화면)은 어느 쪽으로 돌고 있는지 모르는
    편이 낫습니다 — 여기서 받아 줍니다.

    **모호하면 받지 않습니다.** 양다리인데 `hip_yaw` 만 오면 어느 다리인지
    정할 근거가 없고, 조용히 한쪽을 고르면 반대쪽이 안 움직이는 줄 모른 채
    돕니다.
    """
    if name in self._targets:
      return name
    tail = str(name).split("/")[-1]
    hits = [n for n in self._targets if n.split("/")[-1] == tail]
    if len(hits) == 1:
      return hits[0]
    if len(hits) > 1:
      return None                            # 모호함 — 팔다리를 붙여 줘야 함
    return name                              # 정말 모르는 이름

  def freeze(self, name: str) -> Dict[str, Any]:
    """그 관절의 목표를 **지금 측정 위치**로 옮긴다. 더 움직이지 않게.

    `observe()` 가 매 주기 넣어 주는 최신 측정값을 씁니다. 아직 한 번도 못 읽었으면
    그대로 둡니다 — 모르는 값으로 목표를 덮어쓰는 것이 더 위험합니다.
    """
    with self._lock:
      name = self._resolve(name) or name
      if name not in self._targets:
        return dict(ok=False, why=f"모르는 관절입니다: {name}")
      pos = self._measured.get(name)
      if pos is None:
        return dict(ok=False, why="아직 그 관절의 측정값을 못 받았습니다")
      self._targets[name] = float(pos)
      self.note = f"{name} 지금 자리({pos:+.2f}도)에 묶음"
    return dict(ok=True, deg=pos)

  def observe(self, observation: Dict[str, Any]) -> None:
    """제어 루프가 준 관찰에서 측정 위치를 갈무리한다. `freeze` 와 씨뿌리기가 쓴다.

    ★ **응답을 한 번도 못 받았으면 그 값은 각도가 아니라 빈칸(0.0)입니다.**
    `MotorState` 의 기본값이 0이고, 응답이 오기 전에는 `is_valid` 가 False 입니다
    (`motors/base.py:138`). 관찰에는 그 0이 그냥 숫자로 실려 옵니다 — 키가 있는지만
    봐서는 못 가립니다.

    2026-09-12 에 이것에 걸렸습니다. 응답 대기창을 0으로 바꾸자 **첫 주기에 응답이
    아예 없어서**, 관절이 0.03·3.59·3.40·9.46도에 있는데 시작 자세를 전부 `+0.00도`
    로 잡고 열두 관절에 0도를 명령했습니다. 무릎은 여유 3도가 막아 줬지만 막아 준
    것과 안 시킨 것은 다릅니다.

    그래서 `stale_motors` 로 **응답을 받았는지** 먼저 봅니다. 팔다리마다 나오므로
    (`robots/leg.py:406`, 양다리는 `왼다리/stale_motors`) 전부 0이어야 합니다.
    """
    stale = [v for k, v in observation.items() if k.endswith("stale_motors")]
    with self._lock:
      self._fresh = bool(stale) and all(int(v) == 0 for v in stale)
      if not self._fresh:
        return                               # 빈칸을 각도로 갈무리하지 않는다
      for name in self._targets:
        pos = observation.get(f"{name}.pos")
        if pos is not None:
          self._measured[name] = float(pos)

  def snapshot(self) -> Dict[str, float]:
    with self._lock:
      return dict(self._targets)

  # ---- HUPHY 가 부르는 것 (Motion 계약) -----------------------------------
  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    del t                                   # 시간에는 기대지 않는다
    self.observe(observation)               # '힘 빼기' 가 쓸 최신 측정값만 갈무리
    self._seed_once()
    with self._lock:
      if not self._seeded:
        # ★ **자세를 모르면 아무것도 안 보낸다.** 빈 사전을 주면 HUPHY 가
        # `if action:` 에서 걸러 전송을 건너뜁니다 (`control/loop.py:271`).
        # 모터는 꺼진 채 남고, 수거는 계속 돌아 다음 주기에 자세를 알게 됩니다.
        #
        # 그전에는 못 뿌렸을 때 목표의 초기값인 **0도가 그대로 나갔습니다.**
        # 2026-09-12 에 실제로 그랬습니다 — 응답 대기창을 0으로 바꾼 뒤 씨뿌리기가
        # 한 번도 안 됐는데, 열두 관절에 20초 동안 0도가 나갔습니다. 아무것도 안
        # 움직인 것은 정지마찰 덕이지 막아서가 아닙니다.
        self._blocked += 1
        return {}
      return dict(self._targets)

  def _seed_once(self) -> None:
    """첫 주기에 목표를 **지금 있는 자리**로 맞춘다. 명령을 받기 전까지만.

    ★ 이게 없으면 제어 루프가 켜지는 순간 모든 관절에 **0도를 그대로** 명령합니다.
    관절이 0도에 있지 않으면 그만큼이 계단이 되어 한 번에 튑니다 — 2026-09-12 에
    왼 엉덩이회전이 +11.26도에 있었으니 11도짜리 계단이었을 것입니다. 세기 10에서
    그 오차는 약 2뉴턴미터라, 매달린 관절이 홱 돕니다.

    HUPHY 자신도 같은 생각을 합니다 — `scripts/selftest.py` 의 `approach` 가 **지금
    자세를 읽어** 거기서부터 목표까지 끌고 갑니다. 여기서는 화면이 목표를 주므로
    끌고 갈 곳이 없고, 대신 **가만히 있는 것**에서 시작합니다. 움직임은 전부 사람이
    시켜야 생깁니다.

    한 번만 합니다. 사람이 목표를 준 뒤에는 측정값이 목표를 덮어쓰면 안 됩니다 —
    그러면 관절이 처질 때 목표가 따라 내려가 영영 못 올라갑니다.
    """
    if self._seeded:
      return
    with self._lock:
      if self.rx:                            # 이미 사람이 목표를 줬다
        self._seeded = True
        return
      if not self._fresh:
        return                               # 아직 응답을 한 번도 못 받았다
      missing = [n for n in self._targets if n not in self._measured]
      if missing:
        return                               # 아직 다 못 읽음. 다음 주기에
      self._targets.update(self._measured)
      self._seeded = True
      self.note = "지금 자세를 그대로 잡고 시작"
    logger.info("시작 자세를 그대로 잡음: %s",
                ", ".join(f"{n} {v:+.2f}도" for n, v in sorted(self._measured.items())))


# ===========================================================================
# HUPHY 가 안 해 주는 것 — 과열 차단
# ===========================================================================
class ThermalCutoff:
  """권선이 뜨거우면 그 관절 목표를 **지금 위치로 묶어** 힘을 쓰지 않게 한다.

  HUPHY 에는 온도로 무언가를 막는 코드가 없습니다 — `MotorState.temp_c` 를 관찰로
  내보내기만 합니다(`robots/leg.py:287`). 2026-09-11 전수 검사에서 확인한 구멍이고,
  여기서 메웁니다.

  **토크를 끊지 않고 목표를 묶는 이유**: 목표 위치 제어에서 토크를 끊으면 중력이
  관절을 끌어내려 화면과 실물이 어긋납니다(2026-09-12 사용자 지적). 그래서 뜨거우면
  "더 일하지 마라"로 만들되 붙잡고는 있게 합니다. 그래도 계속 오르면 사람이
  `--max-temp-c` 를 낮추고 힘을 빼야 합니다.
  """

  def __init__(self, inner, max_temp_c: float, release_margin_c: float = 5.0):
    self.inner = inner
    self.max_temp_c = float(max_temp_c)
    self.release_margin_c = float(release_margin_c)
    self.hot: Dict[str, bool] = {}
    self.events = 0

  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    action = dict(self.inner(t, observation))
    for name in list(action):
      # 관찰은 **모터 이름**으로 옵니다 (hip_yaw, ankle_a, ...). 발판 공간 이름
      # (ankle_pitch/ankle_roll)으로 목표를 주면 그 둘은 온도 키가 없으므로
      # 아래 `temp is None` 에서 건너뜁니다 — 그 경우 발목 과열은 못 막습니다.
      # 우리는 모터 공간(ankle_a/ankle_b)으로 주므로 해당되지 않습니다.
      # ★ 키 이름은 `.temp` 다. `.temp_c` 가 아니다 (robots/leg.py:397).
      # 2026-09-12 E2E 검사(E11)가 이 오타를 잡았다 — 그전에는 온도 차단이
      # **영원히 안 걸렸다.** 값이 늘 None 이라 조용히 건너뛰었기 때문이다.
      temp = observation.get(f"{name}.temp")
      pos = observation.get(f"{name}.pos")
      if temp is None:
        continue
      was = self.hot.get(name, False)
      if temp >= self.max_temp_c:
        if not was:
          self.events += 1
          logger.warning("%s 권선 %.1f도 — 목표를 지금 자리로 묶음", name, temp)
        self.hot[name] = True
      elif was and temp <= self.max_temp_c - self.release_margin_c:
        self.hot[name] = False
        logger.info("%s 권선 %.1f도로 내려감 — 다시 받음", name, temp)
      if self.hot.get(name) and pos is not None:
        action[name] = float(pos)
    return action


# ===========================================================================
# 로봇 한 마리에서 팔다리를 꺼내는 자리 — **이름을 여기 한 곳에만 적는다**
# ===========================================================================
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
class LinkProbe:
  """새 두절 판정을 **실물 위에서 흉내만 내 본다.** 아무것도 멈추지 않는다.

  ## 왜 이게 필요한가

  두절 판정 방식을 바꿨습니다 (`docs/132`). 예전에는 **연속** 무응답 주기를 셌고,
  새 방식은 **최근 열 번의 명령 중 몇 번 답이 없었나**를 셉니다. 책상 위 가짜
  모터로는 다 통과했지만, 실물 배선이 얼마나 자주 답을 빠뜨리는지는 **재 보기
  전에는 모릅니다.**

  새 규칙이 지금 배선에서 헛되이 걸린다면, 그것을 로봇에 올리는 순간 멀쩡한
  로봇이 몇 분마다 섭니다. 그래서 **먼저 흉내만 내 봅니다** — 로봇 프로그램은
  예전 규칙 그대로 돌고, 여기서는 "새 규칙이었다면 걸렸을까" 만 세어 기록합니다.

  ## 무엇을 세는가

  매 주기 `link_status()` 를 읽습니다. `ack` 가 1 이면 답한 것, 0 이면 이번 주기에
  명령했는데 답이 없는 것, -1 이면 명령하지 않은 것(세지 않음)입니다.

      lost      최근 `window` 번 명령 중 답이 없었던 횟수  <- 새 규칙이 보는 값
      lost_max  그 값이 이번 운용에서 도달한 최대
      miss_max  연속 무응답이 도달한 최대                  <- 예전 규칙이 보는 값
      asked     명령한 주기 수
      missed    그중 답이 없던 주기 수

  `lost_max` 가 문턱(기본 5)에 가까우면 **올리면 안 됩니다.**
  """

  WINDOW = 20
  """연속 문턱의 네 배. `Leg._link_window` 와 같은 값이어야 뜻이 같습니다."""

  THRESHOLD = 15
  """창 안에서 이만큼 잃으면 "샘" 으로 봅니다 = 75%.

  실측으로 고른 값입니다 (2026-09-12, 12개 모터 60초 6001주기). 지금 배선은
  모터마다 응답의 17.8~27.5% 를 잃고, 창 20 안에서 최악 12번입니다. 절반(10번)
  기준이면 전체 주기의 15.5% 에서 두절로 판정해 **로봇이 못 돕니다.**
  자세한 표는 HUPHY `safety/link.py` 문서에 있습니다."""

  def __init__(self, inner, robot, *, window: int = WINDOW):
    self.inner = inner
    self.robot = robot
    self.window = int(window)
    self.recent: Dict[str, deque] = {}
    self.lost_max: Dict[str, int] = {}
    self.miss_max: Dict[str, int] = {}
    self.asked: Dict[str, int] = {}
    self.missed: Dict[str, int] = {}
    self.cycles = 0
    self.would_trip = 0
    """새 규칙이었다면 두절로 판정했을 주기 수. **0 이어야 올릴 수 있습니다.**"""

  def _status(self) -> Dict[str, Any]:
    getter = getattr(self.robot, "link_status", None)
    if getter is None:
      return {}
    try:
      return getter() or {}
    except Exception:                                       # noqa: BLE001
      return {}

  def sample(self) -> None:
    self.cycles += 1
    tripped = False
    for name, row in self._status().items():
      ack = float(row.get("ack", -1.0))
      if ack < 0.0:
        continue                                   # 명령 안 한 모터는 안 셈
      silent = ack == 0.0
      buf = self.recent.setdefault(name, deque(maxlen=self.window))
      buf.append(silent)
      self.asked[name] = self.asked.get(name, 0) + 1
      if silent:
        self.missed[name] = self.missed.get(name, 0) + 1
      # 0 이어도 **자리를 만들어 둡니다.** 안 그러면 "한 번도 안 샜다" 와
      # "그 모터를 아예 못 봤다" 가 둘 다 빈칸이라 구분이 안 됩니다.
      lost = sum(buf)
      self.lost_max[name] = max(self.lost_max.get(name, 0), lost)
      self.miss_max[name] = max(self.miss_max.get(name, 0), int(row.get("miss", 0)))
      if lost >= self.THRESHOLD:
        tripped = True
    if tripped:
      self.would_trip += 1

  def rows(self) -> Dict[str, Dict[str, int]]:
    """관절 이름 -> 지금 값. 상태 전송에 실어 화면·기록으로 보냅니다."""
    return {
      name: dict(lost=int(sum(buf)), lost_max=int(self.lost_max.get(name, 0)),
                 miss_max=int(self.miss_max.get(name, 0)),
                 asked=int(self.asked.get(name, 0)),
                 missed=int(self.missed.get(name, 0)))
      for name, buf in self.recent.items()
    }

  def report(self) -> str:
    if not self.recent:
      return "  통신 흉내내기: 잰 것이 없습니다 (link_status 를 못 읽었습니다)"
    lines = [f"  '샘' 규칙 흉내내기 — {self.cycles}주기 "
             f"(최근 {self.window}번 중 {self.THRESHOLD}번 무응답이면 두절)"]
    lines.append(f"    {'관절':<22}{'명령':>8}{'무응답':>8}{'손실률':>8}"
                 f"{'창최대':>8}{'연속최대':>9}")
    for name in sorted(self.recent):
      asked = max(1, self.asked.get(name, 0))
      lines.append(f"    {name:<22}{self.asked.get(name, 0):>8}"
                   f"{self.missed.get(name, 0):>8}"
                   f"{100.0 * self.missed.get(name, 0) / asked:>7.1f}%"
                   f"{self.lost_max.get(name, 0):>8}{self.miss_max.get(name, 0):>9}")
    worst = max(self.lost_max.values()) if self.lost_max else 0
    if self.would_trip:
      lines.append(f"    ★ '샘' 규칙이었다면 {self.would_trip}주기에서 두절로 "
                   f"판정했습니다 (창 최대 {worst}). **올리면 안 됩니다.**")
    else:
      lines.append(f"    '샘' 규칙으로는 한 번도 안 걸렸습니다 (창 최대 {worst} "
                   f"< 문턱 {self.THRESHOLD}). 여유 {self.THRESHOLD - worst}번.")
    return "\n".join(lines)

  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    self.sample()
    return self.inner(t, observation)


class StateSender:
  """모터 상태를 JSON 한 줄로 UDP 에 뿌린다. 받지는 않는다.

  ★ **왜 화면이 직접 CAN 을 안 읽는가.** 통신선 하나를 두 프로그램이 각자 읽으면
  서로의 응답을 가로채 엉뚱한 값을 집습니다. 그래서 버스를 만지는 것은 이
  프로그램 하나이고, 화면은 여기서 나온 것을 **받아서 그리기만** 합니다.

  같은 내용을 여러 곳에 보낼 수 있습니다 — 화면(관절 페이지)과 3차원 뷰어.
  """

  def __init__(self, robot, targets, targets_to, hz: float = 20.0, gains=None,
               probe=None):
    self.robot = robot
    self.targets = targets
    self.gains = gains
    self.probe = probe
    """새 두절 규칙 흉내내기(`LinkProbe`). 값이 기록에 남아야 나중에 따집니다."""
    """화면이 **지금 걸린 값**을 보여 줘야 합니다. 화면이 따로 기억하면 둘이
    어긋나고, 사람은 자기가 넣은 숫자가 걸렸다고 믿습니다."""
    self.addrs = []
    for one in targets_to:
      host, _, port = str(one).partition(":")
      if host:
        self.addrs.append((host, int(port or 9873)))
    self.period = 1.0 / float(hz)
    self.sent = 0
    self._stop = threading.Event()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  def start(self):
    if self.addrs:
      threading.Thread(target=self._run, daemon=True).start()
    return self

  def stop(self):
    self._stop.set()

  def rows(self) -> Dict[str, Any]:
    """모터 번호 -> 사람이 읽는 값. 관절 화면이 쓰던 것과 같은 모양."""
    out: Dict[str, Any] = {}
    now = time.monotonic()
    wanted = self.targets.snapshot()
    for part in limbs_of(self.robot) or [self.robot]:
      cfg = getattr(part, "config", None)
      bus = getattr(part, "bus", None)
      if cfg is None or bus is None:
        continue
      prefix = joint_prefix(self.robot, part)
      for joint, motor in cfg.motors.items():
        try:
          st = bus.state(motor.id)
        except Exception:                                   # noqa: BLE001
          continue
        full = prefix + joint
        # 화면이 슬라이더를 만들려면 **갈 수 있는 구간**을 알아야 합니다.
        # 한계는 가드가 실제로 보는 자리에서 읽고(`config.motors[…].limits_deg`),
        # 여유를 문 값을 줍니다 — 화면이 보여 주는 끝이 실제로 닿는 끝이어야
        # 사람이 "왜 거기까지 안 가지" 하고 헤매지 않습니다.
        span = getattr(motor, "limits_deg", None)
        margin = float(getattr(getattr(part, "safety", None),
                               "command_margin_deg", 0.0) or 0.0)
        if span:
          lo, hi = float(span[0]) + margin, float(span[1]) - margin
          if hi < lo:
            lo = hi = 0.5 * (lo + hi)
        else:
          lo = hi = None
        g = getattr(motor, "gains", None)
        out[str(motor.id)] = dict(
          joint=full, model=getattr(motor, "model", ""),
          kp=None if g is None else round(float(g.kp), 3),
          kd=None if g is None else round(float(g.kd), 3),
          # ★ 12비트 눈금으로 잘린 **실제로 나가는 값**. 15 를 넣어도 14.65 가
          # 나가고, 눈금(1.2210)보다 작으면 **0** 이 나갑니다 — 화면이 "약하게
          # 해 뒀다" 와 "힘이 아예 안 들어간다" 를 구분해 보여 줘야 합니다.
          kp_wire=None if g is None else round(
            math.floor(float(g.kp) / GainControl.QUANTUM) * GainControl.QUANTUM, 2),
          lo=None if lo is None else round(lo, 2),
          hi=None if hi is None else round(hi, 2),
          deg=round(float(getattr(st, "position_deg", 0.0)), 2),
          vel=round(float(getattr(st, "velocity_deg_s", 0.0)), 1),
          nm=round(float(getattr(st, "torque_nm", 0.0)), 2),
          temp=round(float(getattr(st, "temp_c", 0.0)), 1),
          # ★ 동작모드는 **판단에 쓰지 않는다** (위 MODE_IS_REPORTED 참조).
          # 그대로 실어 보내되, 화면이 "꺼짐" 으로 읽지 않도록 모델도 같이 보낸다.
          mode=getattr(st, "mode", None),
          mode_reported=MODE_IS_REPORTED.get(getattr(motor, "model", ""), None),
          tgt=round(float(wanted.get(full, 0.0)), 2),
          age_ms=round((now - float(getattr(st, "stamp", 0.0) or now)) * 1000.0, 1),
        )
        if self.probe is not None:
          out[str(motor.id)].update(self.probe.rows().get(full, {}))
    return out

  def _run(self):
    while not self._stop.is_set():
      t0 = time.monotonic()
      pkt = json.dumps(dict(t=round(t0, 3), motors=self.rows(),
                            note=self.targets.note,
                            rx=self.targets.rx)).encode()
      for addr in self.addrs:
        try:
          self.sock.sendto(pkt, addr)
          self.sent += 1
        except OSError:
          pass
      time.sleep(max(0.0, self.period - (time.monotonic() - t0)))


# ===========================================================================
# 관절마다 세기(kp)와 감쇠(kd)를 숫자로
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
    from dataclasses import replace
    from huphy.motors.base import Gains
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
        part.config.motors[name], gains=Gains(kp=new_kp, kd=new_kd))
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
class TargetServer:
  """한 줄 JSON 을 UDP 로 받아 `LiveTargets` 에 넣는다. 보내지는 않는다."""

  def __init__(self, targets: LiveTargets, port: int, token: str = "", gains=None):
    self.targets, self.port, self.token = targets, int(port), str(token)
    self.gains = gains
    """세기·감쇠를 바꾸는 길. 목표와 **같은 통로**로 받습니다 — 화면이 통로를
    둘 들고 있으면 하나만 살아 있는 상태를 못 알아챕니다."""
    self._stop = threading.Event()
    self.bad_token = 0

  def start(self):
    threading.Thread(target=self._run, daemon=True).start()
    return self

  def stop(self):
    self._stop.set()

  def _run(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", self.port))
    sock.settimeout(0.3)
    while not self._stop.is_set():
      try:
        raw, _ = sock.recvfrom(65535)
      except (TimeoutError, socket.timeout):
        continue
      except OSError:
        break
      try:
        msg = json.loads(raw)
      except (ValueError, TypeError):
        continue
      if self.token and str(msg.get("token", "")) != self.token:
        self.bad_token += 1
        continue
      if "gains" in msg and isinstance(msg["gains"], dict) and self.gains:
        for joint, g in msg["gains"].items():
          if isinstance(g, dict):
            self.gains.set(str(joint), kp=g.get("kp"), kd=g.get("kd"))
      elif "release" in msg:
        # 화면의 '힘 빼기'. 목표를 **지금 자리로** 옮겨 더 일하지 않게 한다.
        # 토크를 끊지 않는 이유는 ThermalCutoff 머리말과 같다 — 목표 위치 제어에서
        # 힘을 끊으면 중력이 관절을 끌어내려 화면과 실물이 어긋난다.
        self.targets.freeze(str(msg["release"]))
      elif "targets" in msg and isinstance(msg["targets"], dict):
        self.targets.update(msg["targets"])
      elif "name" in msg:
        self.targets.update({str(msg["name"]): msg.get("deg", 0.0)})
    sock.close()


# ===========================================================================
# 켜기가 실제로 먹혔는지 확인 — HUPHY 에 대응물 없음
# ===========================================================================
MODE_IS_REPORTED = {"RS04": True, "RS03": False}
"""상태 프레임의 동작모드 비트를 그 모델이 **실제로 올려 주는가.**

2026-09-12 실측. RS04 는 힘이 들어가면 곧바로 2로 바뀝니다. **RS03 은 명령을
따라 움직이는 동안에도 끝까지 0입니다** — 3번(RS03)을 10도 몰았더니 실제 각도가
10.40도 움직였는데 6,543주기 전부 0이었고, 같은 시험에서 4번(RS04)은 5,811주기
전부 2였습니다.
"""


def report_modes(robot) -> Dict[int, Any]:
    """모터가 보고한 동작모드를 **판단 없이** 모아 돌려준다.

    ★ 이 값으로 "켜졌는가" 를 판단하지 마십시오. 위 `MODE_IS_REPORTED` 참조 —
    모델에 따라 안 올라옵니다. 2026-09-12 에 그걸 판단 근거로 썼다가 멀쩡한 모터
    일곱 개를 고장으로 몰았습니다.

    켜졌는지 정말 알고 싶으면 **관절이 명령을 따라가는지** 보십시오 — 목표와 측정
    위치의 차이가 줄어들면 힘이 들어간 것입니다.
    """
    out: Dict[int, Any] = {}
    for part in limbs_of(robot) or [robot]:
        bus = getattr(part, "bus", None)
        if bus is None:
            continue
        for motor_id, state in getattr(bus, "_states", {}).items():
            out[motor_id] = getattr(state, "mode", None)
    return out


# ===========================================================================
# 진입점
# ===========================================================================
def trim_limits(machine, trim_deg: float) -> Dict[str, Any]:
  """실측 한계를 **양쪽으로** `trim_deg` 만큼 좁힌다. 설정 파일은 안 건드린다.

  사용자 지시(2026-09-12): *"최대/최소 범위는 ±15도 줄이고"* -> "한계에서 양쪽으로
  15도씩 안으로". 시험하는 동안만 좁히는 것이라 파일을 고치지 않고 여기서 바꿉니다
  — 파일을 고치면 되돌리는 것을 잊습니다.

  ★ **`calibration` 만 고치면 아무 일도 안 일어납니다.** 한계를 자르는 가드는
  `calibration` 이 아니라 **`config.motors[이름].limits_deg`** 를 봅니다
  (`robots/leg.py:241`, `:546`). `Leg._set_calibration` 이 둘을 잇는 자리라,
  여기서도 그것을 다시 불러 값을 내려보냅니다.

  2026-09-12 에 `calibration` 만 고쳤다가 좁히기 전 한계(45도)가 그대로 나가는 것을
  검사 E33 이 잡았습니다 — 설정이 중간에서 증발하는 전형적인 모양입니다.
  """
  from dataclasses import replace
  out = {}
  for part in limbs_of(machine) or [machine]:
    cal = getattr(part, "calibration", None)
    if not cal:
      continue
    narrowed = {}
    for name, entry in cal.items():
      span = getattr(entry, "limits_deg", None)
      if not span:
        narrowed[name] = entry
        continue
      lo, hi = float(span[0]), float(span[1])
      if hi - lo <= 2.0 * trim_deg:
        mid = 0.5 * (lo + hi)
        new_span = (mid, mid)
      else:
        new_span = (lo + trim_deg, hi - trim_deg)
      narrowed[name] = replace(entry, limits_deg=new_span)
      out[joint_prefix(machine, part) + name] = new_span
    setter = getattr(part, "_set_calibration", None)
    if setter is None:
      raise SystemExit("이 판의 HUPHY 에는 _set_calibration 이 없습니다 — "
                       "한계를 좁히는 길이 달라졌을 수 있으니 확인하십시오")
    setter(narrowed)
  return out


def pick_joints(machine, wanted) -> list:
  """명령할 관절만 고른다. **고르지 않은 관절에는 아무것도 안 나간다.**

  ★ 왜 중요한가: `Leg._motor_targets` 는 **넘어온 키만** 순회합니다
  (`robots/leg.py:475`). 목표 표에 없는 관절은 그 주기에 명령을 안 받고, 한 번도
  힘을 넣은 적이 없으면 **꺼진 채로 남습니다.** 그래서 여는 관절을 고르는 일이
  곧 안전장치입니다.

  이게 없으면 다리의 관절 **전부**가 매 주기 0도를 받습니다 — 무릎은 0도가 곧
  하드스톱이라 그대로 밀어붙이게 됩니다.
  """
  # ★ `joint_names` 로는 부족합니다. 그것은 **발판 공간**(ankle_pitch/ankle_roll)만
  # 내는데, 우리는 발목을 **모터 공간**(ankle_a/ankle_b)으로 줍니다 — 지금 구현체도
  # 학습된 정책도 그 공간이기 때문입니다(사용자 결정 2026-09-12). 받을 수 있는
  # 이름 전부는 `robots/leg.py:87 COMMANDABLE` 입니다.
  from huphy.robots.leg import COMMANDABLE
  names = []
  for part in limbs_of(machine) or [machine]:
    names.extend(joint_prefix(machine, part) + n for n in COMMANDABLE)
  if not wanted:
    # 안 고르면 **발판 공간 기본값**을 씁니다. 여덟 개를 통째로 주면 HUPHY 가
    # "두 공간이 섞임" 으로 거부합니다 (`leg.py:443`).
    return [n for n in names if not n.split("/")[-1] in ("ankle_a", "ankle_b")]
  chosen = [n for n in names if n in wanted or n.split("/")[-1] in wanted]
  missing = [w for w in wanted
             if not any(w == n or w == n.split("/")[-1] for n in names)]
  if missing:
    raise SystemExit(f"그런 관절이 없습니다: {missing}. 가능한 것: {sorted(set(names))}")
  # 두 발목 공간을 섞어 고르는 것도 여기서 막습니다 -- 제어 루프가 도는 중에
  # ValueError 로 서는 것보다 시작 전에 안 뜨는 편이 낫습니다.
  tails = {n.split("/")[-1] for n in chosen}
  if tails & {"ankle_pitch", "ankle_roll"} and tails & {"ankle_a", "ankle_b"}:
    raise SystemExit("발목을 두 공간으로 섞어 고를 수 없습니다 — "
                     "ankle_pitch+ankle_roll 이거나 ankle_a+ankle_b 중 한 쌍만")
  for pair in (("ankle_pitch", "ankle_roll"), ("ankle_a", "ankle_b")):
    if tails & set(pair) and not set(pair) <= tails:
      raise SystemExit(f"발목은 {pair[0]} 와 {pair[1]} 를 **함께** 골라야 합니다 — "
                       f"모터 두 개가 두 자유도를 같이 만들기 때문입니다")
  return chosen


def build(args):
  """설정을 읽어 로봇과 제어 루프를 세운다. **HUPHY 의 도구만 쓴다.**"""
  from huphy.config import load_robot
  from huphy.control import ControlLoop, Mode
  from huphy.scripts.bringup import build_biped, build_leg
  # `build_leg` 은 이름이 아니라 **설정 객체**를 받습니다. 고르는 일은 HUPHY 자신의
  # 함수에 맡깁니다 — 오타를 스택 트레이스 대신 사람 말로 알려 주고, "팔다리마다
  # CAN 채널이 달라 잘못 고르면 엉뚱한 쪽이 움직인다" 는 경고도 그쪽에 있습니다.
  from huphy.scripts.commission import _pick_limb

  robot_cfg = load_robot(args.config)
  drain_s = None if args.drain_ms is None else args.drain_ms / 1000.0
  if args.robot:
    machine = build_biped(robot_cfg, allow_uncalibrated=args.allow_uncalibrated,
                          drain_s=drain_s, gain_scale=args.gain_scale,
                          reader=args.reader)
  else:
    machine = build_leg(robot_cfg, _pick_limb(robot_cfg, args.limb),
                        allow_uncalibrated=args.allow_uncalibrated,
                        drain_s=drain_s, gain_scale=args.gain_scale,
                        reader=args.reader)
  hz = args.hz or getattr(machine, "control_hz", 100.0)
  loop = ControlLoop(machine, hz=hz, mode=Mode.CONTROL)
  return machine, loop, hz


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("--robot", action="store_true", help="양다리")
  g.add_argument("--limb", help="다리 하나 (예: left_leg)")
  ap.add_argument("--hz", type=float, default=None)
  ap.add_argument("--listen", type=int, default=9872, help="목표를 받는 포트")
  ap.add_argument("--token", default="", help="맞는 것만 받음. 비우면 검사 안 함")
  ap.add_argument("--max-temp-c", type=float, default=50.0)
  ap.add_argument("--allow-uncalibrated", action="store_true")
  ap.add_argument("--duration", type=float, default=None)
  ap.add_argument("--state-to", nargs="*", default=[],
                  help="상태를 보낼 자리. 예: 127.0.0.1:9873 10.8.0.16:9870")
  ap.add_argument("--state-hz", type=float, default=20.0)
  ap.add_argument("--joints", nargs="*", default=[],
                  help="명령할 관절만. 안 주면 **전부**. 고르지 않은 관절에는 "
                       "아무것도 안 나갑니다 (예: hip_yaw ankle_a ankle_b)")
  ap.add_argument("--trim-deg", type=float, default=0.0,
                  help="한계를 양쪽으로 이만큼 안으로 좁힘. 설정 파일은 안 건드림")
  ap.add_argument("--link-probe", action="store_true",
                  help="'샘' 판정(최근 20번 중 15번 무응답)을 **흉내만** 내 "
                       "보고 끝에 표로 냅니다. 아무것도 멈추지 않습니다")
  ap.add_argument("--kp-cap", type=float, default=15.0,
                  help="세기 상한. 사용자 지시(2026-09-12)로 15. 무릎을 정지마찰에서 "
                       "풀려면 벤치 실측 20 이 필요하므로, 올릴 때는 **일부러** 올릴 것")
  ap.add_argument("--kd-cap", type=float, default=5.0, help="감쇠 상한")
  ap.add_argument("--reader", dest="reader", action="store_true", default=True,
                  help="채널마다 수신 스레드를 붙임 (기본 켬)")
  ap.add_argument("--no-reader", dest="reader", action="store_false")
  ap.add_argument("--gain-scale", type=float, default=1.0,
                  help="설정의 세기에 곱함. **0 이면 계산은 다 하고 힘은 안 나감** — "
                       "주기를 재거나 길이 뚫렸는지 볼 때 씁니다")
  ap.add_argument("--drain-ms", type=float, default=5.0,
                  help="응답을 기다리는 창. **0 으로 두면 20초 만에 통신두절로 "
                       "죽습니다.** 기본 5.0 은 실측으로 고른 값입니다 "
                       "(2026-09-12, 힘이 들어간 채 120초씩): 3 -> 5 로 늘리면 "
                       "응답 손실이 평균 16.8% -> 7.8% 로 줄고, 제어 주기는 "
                       "초당 87.4 -> 80.7회로 8% 느려집니다. docs/133 참조")
  a = ap.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  machine, loop, hz = build(a)

  if a.trim_deg > 0:
    narrowed = trim_limits(machine, a.trim_deg)
  else:
    narrowed = {}
  names = pick_joints(machine, set(a.joints))
  live = LiveTargets(names)
  motion = AnklePairGuard(ThermalCutoff(live, a.max_temp_c), machine)
  probe = LinkProbe(motion, machine) if a.link_probe else None
  if probe is not None:
    motion = probe                      # 흉내만 냅니다. 아무것도 안 멈춥니다.
  gains = GainControl(machine, kp_cap=a.kp_cap, kd_cap=a.kd_cap)
  server = TargetServer(live, a.listen, a.token, gains=gains).start()
  state = StateSender(machine, live, a.state_to, hz=a.state_hz, gains=gains,
                      probe=probe).start()

  all_names = list(getattr(machine, "joint_names", ()))
  print(f"\n{getattr(machine, 'id', '로봇')} · 초당 {hz:.0f}회")
  print(f"  ** 명령할 관절 {len(names)}/{len(all_names)}개: {names} **")
  held_out = [n for n in all_names if n not in names]
  if held_out:
    print(f"     여는 게 아닌 관절 {len(held_out)}개에는 **아무것도 안 나갑니다**: "
          f"{held_out}")
  if narrowed:
    print(f"  한계를 양쪽 {a.trim_deg:g}도 안으로 좁힘:")
    for n in names:
      span = narrowed.get(n)
      if span:
        print(f"     {n:24} {span[0]:+8.2f} ~ {span[1]:+8.2f}도")
  print(f"  목표를 받는 자리 udp:{a.listen}"
        f"{' (암호 있음)' if a.token else ' (암호 없음 — 망 안에서만 쓸 것)'}")
  if a.gain_scale == 0.0:
    print("  ** 세기 0 — 계산은 다 하지만 힘이 나가지 않습니다 (주기 재기용) **")
  elif a.gain_scale != 1.0:
    print(f"  세기에 {a.gain_scale:g} 를 곱합니다")
  print(f"  세기 상한 {a.kp_cap:g} · 감쇠 상한 {a.kd_cap:g} "
        f"(관절마다 화면에서 숫자로 바꿀 수 있습니다)")
  print(f"  권선 {a.max_temp_c:g}도를 넘으면 그 관절 목표를 지금 자리로 묶음")
  print(f"  발목은 축마다 \u00b1{ANKLE_AXIS_DEG:g}도 **그리고** 조합이 "
        f"숙임 {ANKLE_ROM_PITCH} 기울임 {ANKLE_ROM_ROLL} 안일 때만 나갑니다")
  print("  한계·속도·프레임·전송·수거는 전부 로봇 프로그램(HUPHY)이 합니다.")
  if a.state_to:
    print(f"  상태를 내보내는 자리 {', '.join(a.state_to)} (초당 {a.state_hz:g}번, 보내기만)")
    print("  ** 통신선을 읽는 것은 이 프로그램 하나입니다 — 화면은 여기서 받아 그립니다. **")
  print("  멈추려면 Ctrl-C.\n")

  try:
    stats = loop.run(motion, duration_s=a.duration)
  except KeyboardInterrupt:
    stats = None
    print("\n  사람이 멈춤")
  finally:
    server.stop()
    state.stop()

  modes = report_modes(machine)
  if modes:
    print(f"\n  모터가 보고한 동작모드: {modes}")
    print("  (RS03 은 힘이 들어가도 0으로 보고합니다 — 이 값으로 판단하지 마십시오)")
  if not live._seeded:
    print("\n  ** 시작 자세를 한 번도 못 읽어 아무 명령도 안 보냈습니다 "
          f"({live._blocked}주기). 통신을 확인하십시오. **")
  elif live._blocked:
    print(f"  시작 자세를 읽기까지 {live._blocked}주기 동안 아무것도 안 보냈습니다")
  # 감싼 순서: LinkProbe(있으면) -> AnklePairGuard -> ThermalCutoff -> LiveTargets
  ankle = motion.inner if isinstance(motion, LinkProbe) else motion
  print(f"  받은 목표 {live.rx}개 · 모르는 이름 {live.rejected_unknown}개 "
        f"· 숫자가 아님 {live.rejected_not_finite}개 "
        f"· 과열로 묶음 {ankle.inner.events}회")
  print(f"  발목: 축으로 자름 {ankle.axis_clipped}회 · 조합으로 물러남 "
        f"{ankle.pair_clipped}회")
  print(f"  세기 바꿈 {gains.changes}회 · 상한으로 자름 {gains.clipped}회")
  if probe is not None:
    print(probe.report())
  if stats is not None and getattr(stats, "link_loss", None):
    print(f"\n  ** 통신 두절로 멈춤: {stats.link_loss} **")
    return 1
  return 0


if __name__ == "__main__":
  sys.path.insert(0, os.path.expanduser("~/HUPHY/src"))
  raise SystemExit(main())
