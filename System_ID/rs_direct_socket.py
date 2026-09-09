#!/usr/bin/env python3
"""RobStride 모터에 직접 명령한다. **파이썬 기본 기능만** 쓴다. 라즈베리파이용.

2026-09-09. `rs_direct.py` 와 하는 일이 같고, 통신 부분만 다릅니다.

* `rs_direct.py` — `python-can` 꾸러미를 씁니다. 로봇 가상머신에는 이미 깔려 있습니다.
* 이 파일 — 파이썬에 **기본으로 들어 있는** ``socket`` 만 씁니다. 새 기계에 아무것도 설치하지
  않고 바로 돌리려고 그렇게 했습니다. 리눅스의 CAN 통신은 그 자체가 소켓이라, 꾸러미 없이도
  프레임을 그대로 주고받을 수 있습니다.

프레임을 만들고 푸는 일은 **HUPHY 의 코덱을 그대로** 씁니다(`rscodec/`). 그건 안전장치가
아니라 모터가 알아듣는 언어라서, 손으로 다시 유도하면 조용히 틀립니다. 특히 게인 범위가
모델마다 달라서(RS03·RS04 는 0~5000/0~100, RS00·RS02 는 0~500/0~5) 표를 잘못 쓰면 같은
숫자가 10배로 들어갑니다.

    from rs_direct_socket import Motor
    with Motor(channel="can0", motor_id=4, model="RS04") as m:
        st = m.read()          # 힘을 전혀 안 쓰고 상태만
        m.enable()
        m.torque(0.3)          # 순수 토크 (kp=kd=0)
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rscodec import mit, tables  # noqa: E402

CAN_FRAME_FMT = "=IB3x8s"
"""리눅스의 ``struct can_frame``: 32비트 번호, 길이 1바이트, 빈칸 3바이트, 자료 8바이트."""
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)


@dataclass
class State:
  """모터가 한 응답에 실어 보낸 것 전부. 각도는 모터 자신의 raw 공간이다."""

  motor_id: int
  position_deg: float
  velocity_deg_s: float
  torque_nm: float
  temp_c: float
  stamp: float
  mode: int
  """0 Reset / 1 Cali / **2 Motor**. 토크가 정말 켜졌는지는 이 값으로 본다."""
  fault: bool
  warn: bool


class Motor:
  def __init__(self, channel: str = "can0", motor_id: int = 4, model: str = "RS04"):
    self.channel = channel
    self.motor_id = int(motor_id)
    self.model = model
    self.enc = tables.MIT_ENCODING[tables.Model(model)]
    self._s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    self._s.bind((channel,))
    self._s.settimeout(0.05)
    self._enabled = False

  # ----------------------------------------------------------------- 열고 닫기
  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
    return False

  def close(self) -> None:
    """**힘을 먼저 빼고** 닫는다. 순서가 반대면 정지 명령을 보낼 길이 사라지고, 모터는
    마지막 명령을 계속 붙들고 있는다."""
    try:
      for _ in range(5):
        self.disable()
        time.sleep(0.01)
    except Exception as exc:
      print(f"[rs_direct_socket] 토크 끄기 실패: {exc} — 전원을 확인하세요", flush=True)
    finally:
      self._s.close()

  # ----------------------------------------------------------------- 보내기
  def _send(self, data: bytes, tries: int = 40) -> bool:
    """보낸다. 보낼 자리가 없으면 비기를 기다렸다 다시 보낸다.

    토크를 끄는 명령만은 절대 예외가 되면 안 된다 — 2026-09-09 에 다른 기계에서 송신 큐가
    넘쳐 그 예외가 토크 끄는 경로 안에서 났고, 토크가 켜진 채 끝날 뻔했다."""
    frame = struct.pack(CAN_FRAME_FMT, self.motor_id, len(data), data)
    for _ in range(tries):
      try:
        self._s.send(frame)
        return True
      except OSError:
        time.sleep(0.002)
    return False

  def _cmd(self, command: int, f_cmd: int = tables.F_CMD_DEFAULT) -> None:
    self._send(bytes([0xFF] * 6 + [f_cmd & 0xFF, command & 0xFF]))

  def enable(self) -> None:
    self._cmd(tables.CMD_ENABLE)
    self._enabled = True

  def disable(self) -> None:
    self._cmd(tables.CMD_STOP)
    self._enabled = False

  def clear_fault(self) -> None:
    self._cmd(tables.CMD_FAULT)

  # ----------------------------------------------------------------- MIT 명령
  def send(self, position_deg: float = 0.0, velocity_deg_s: float = 0.0,
           kp: float = 0.0, kd: float = 0.0, torque_nm: float = 0.0) -> None:
    """모터 펌웨어가 계산하는 식:

        tau = kp*(목표각 - 현재각) + kd*(목표속도 - 현재속도) + torque_nm

    kp=kd=0 이면 목표각이 무엇이든 무시되고 torque_nm 만 남는다.
    """
    self._send(mit.pack_command(position_deg=position_deg, velocity_deg_s=velocity_deg_s,
                                kp=kp, kd=kd, torque_nm=torque_nm, enc=self.enc))

  def torque(self, tau_nm: float) -> None:
    """순수 토크. 위치 되먹임 없음."""
    self.send(torque_nm=tau_nm)

  def hold(self, position_deg: float, kp: float = 8.0, kd: float = 0.6) -> None:
    """위치 유지. 실험을 **끝낸 뒤** 내려오는 데만 쓴다 — 실험 중에 켜면 내부 PD 가 우리가
    재려는 물리를 가린다."""
    self.send(position_deg=position_deg, kp=kp, kd=kd)

  # ----------------------------------------------------------------- 응답
  def recv(self, timeout_s: float = 0.005) -> State | None:
    deadline = time.monotonic() + timeout_s
    while True:
      left = deadline - time.monotonic()
      if left <= 0:
        return None
      self._s.settimeout(left)
      try:
        raw = self._s.recv(CAN_FRAME_SIZE)
      except (socket.timeout, OSError):
        return None
      _cid, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
      try:
        mid, pos, vel, tau, temp = mit.decode_state(data[:dlc], enc=self.enc)
      except (ValueError, KeyError):
        continue
      if mid != self.motor_id:
        continue
      fl = mit.decode_flags(data[:dlc])
      return State(motor_id=mid, position_deg=pos, velocity_deg_s=vel, torque_nm=tau,
                   temp_c=temp, stamp=time.monotonic(),
                   mode=fl.mode, fault=bool(fl.fault), warn=bool(fl.warning))

  def read(self, tries: int = 20) -> State | None:
    """힘을 전혀 쓰지 않고 상태만 얻는다.

    MIT 방식에는 읽기 전용 명령이 없다 — 모터는 **명령을 받아야** 답한다. 그래서 이득도
    토크도 0인 명령을 보내고 그 응답을 받는다. 아무 일도 일어나지 않는다."""
    for _ in range(tries):
      self.send()
      st = self.recv(timeout_s=0.01)
      if st is not None:
        return st
    return None
