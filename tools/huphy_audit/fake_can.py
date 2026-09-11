#!/usr/bin/env python3
"""로봇 프로그램(HUPHY)을 하드웨어 없이 돌리기 위한 **가짜 통신선**.

진짜 모터 대신 이 파일이 답을 합니다. 그래야 "선을 뽑았을 때", "모터가 너무
뜨거울 때" 처럼 **일부러 만들면 위험한 상황**을 책상 위에서 만들어 볼 수 있습니다.

무엇을 흉내내는가

* 명령을 받으면 상태(각도·속도·힘·온도·고장표시)로 답합니다. 진짜 모터가 그럽니다.
* `answer = False` 로 두면 **답을 안 합니다** — 선이 끊긴 상황.
* `follow = False` 로 두면 명령을 받아도 **안 움직입니다** — 관절이 걸린 상황.
* `temp_c`, `fault`, `warning` 을 아무 값으로나 둘 수 있습니다 — 과열·고장 상황.

한 가지 중요한 규칙: 힘(kp)이 0이면 움직이지 않습니다. 진짜 모터가 그렇습니다.
"""

from __future__ import annotations

import math
import sys
import types
from collections import deque
from typing import Dict

from huphy.motors.robstride import tables as T
from huphy.motors.robstride.codec import mit


class FakeMessage:
  def __init__(self, arbitration_id, data, is_extended_id=False):
    self.arbitration_id = arbitration_id
    self.data = data
    self.is_extended_id = is_extended_id


class FakeMotors:
  """이 책상 위 '모터들'의 상태. 시나리오가 여기를 만져 상황을 만듭니다."""

  def __init__(self, models: Dict[int, T.Model]):
    self.models = dict(models)
    self.position: Dict[int, float] = {i: 0.0 for i in models}
    self.velocity: Dict[int, float] = {i: 0.0 for i in models}
    self.torque: Dict[int, float] = {i: 0.0 for i in models}
    self.temp_c: Dict[int, float] = {i: 30.0 for i in models}
    self.fault: Dict[int, bool] = {i: False for i in models}
    self.warning: Dict[int, bool] = {i: False for i in models}
    self.mode: Dict[int, int] = {i: 2 for i in models}   # 2 = 힘이 켜진 상태
    self.answer = True          # False 면 아무 답도 안 함 (선이 끊김)
    self.follow = True          # False 면 명령대로 안 움직임 (관절이 걸림)
    self.sent = []              # 실제로 선에 나간 프레임 전부


class FakeCanBus:
  """python-can 의 `Bus` 자리에 들어가는 것. 한 번에 하나만 씁니다."""

  motors: FakeMotors = None     # type: ignore[assignment]

  def __init__(self, **kwargs):
    self.rx = deque()

  def send(self, msg):
    m = FakeCanBus.motors
    m.sent.append(msg)
    mid = msg.arbitration_id
    d = bytes(msg.data)
    if mid not in m.models:
      return
    enc = T.encoding_for(m.models[mid])

    is_control = d[0] == 0xFF          # 켜기/끄기/영점 같은 관리 명령
    if not is_control and m.follow:
      kp_u = ((d[3] & 0x0F) << 8) | d[4]
      if kp_u > 0:                      # 힘이 0이면 안 움직임
        target = math.degrees(mit.uint_to_float(
          (d[0] << 8) | d[1], -enc.pmax_rad, enc.pmax_rad, 16))
        m.position[mid] = target

    if not m.answer:
      return
    self.rx.append(FakeMessage(mid, _state_frame(m, mid, enc)))

  def recv(self, timeout=None):
    return self.rx.popleft() if self.rx else None

  def shutdown(self):
    pass


def _state_frame(m: FakeMotors, mid: int, enc) -> bytes:
  """모터가 돌려보내는 8바이트. 온도는 12비트, 위 4비트는 표시등입니다."""
  q = mit.float_to_uint(math.radians(m.position[mid]), -enc.pmax_rad, enc.pmax_rad, 16)
  v = mit.float_to_uint(math.radians(m.velocity[mid]), -enc.vmax_rad_s, enc.vmax_rad_s, 12)
  tau = mit.float_to_uint(m.torque[mid], -enc.tmax_nm, enc.tmax_nm, 12)
  temp_u = max(0, min(0xFFF, int(round(m.temp_c[mid] * 10.0))))
  flags = ((m.mode[mid] & 0x03) << 6) | (0x20 if m.fault[mid] else 0) \
      | (0x10 if m.warning[mid] else 0)
  return bytes([
    mid,
    (q >> 8) & 0xFF, q & 0xFF,
    (v >> 4) & 0xFF,
    ((v & 0x0F) << 4) | ((tau >> 8) & 0x0F), tau & 0xFF,
    flags | ((temp_u >> 8) & 0x0F),
    temp_u & 0xFF,
  ])


def install(models: Dict[int, T.Model]) -> FakeMotors:
  """가짜 통신선을 끼워 넣고, 만질 수 있는 '모터들'을 돌려줍니다."""
  motors = FakeMotors(models)
  FakeCanBus.motors = motors
  mod = types.ModuleType("can")
  mod.Message = FakeMessage
  mod.interface = types.SimpleNamespace(Bus=FakeCanBus)
  sys.modules["can"] = mod
  return motors
