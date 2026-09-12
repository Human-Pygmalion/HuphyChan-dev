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


## 응답을 기다리는 창을 5밀리초로 둔 이유

로봇 프로그램 기본값은 **2밀리초**입니다. 2026-09-12 에 토크 없이 실측해 보니
그 값에서는 한 모터가 **연속 마흔 주기**까지 응답을 놓칩니다. 로봇 프로그램이
멈추는 조건이 "한 모터가 연속 다섯 주기 무응답" 이라, 실제로 그것에 걸려 영자세
유지가 296.5초에서 끊겼습니다.

    창      실제 초당   전부 받은 주기   최악 연속 무응답   다섯을 친 모터
    2 ms      200.0          0.3%             40           대부분
    4 ms      200.0         32.8%              5           둘
    **5 ms**  **199.9**     **55.2%**         **4**        **없음**
    8 ms      170.6         62.4%              3           없음

5밀리초가 분기점입니다 — 초당 200번을 그대로 지키면서 다섯을 치는 모터가 하나도
없습니다. 6밀리초 이상은 속도를 잃으면서 얻는 것이 거의 없습니다.

★ **이 숫자는 제어 계산이 없는 상태**입니다. 로봇 프로그램은 매 주기 계산을 하므로
여유가 이보다 좁습니다. 그리고 5밀리초에서도 주기의 45퍼센트는 모터 하나가 한 주기
묵은 값을 씁니다 — "패킷로스 없이" 라는 요구를 만족하지는 못합니다. 진짜 해법은
통신 어댑터 펌웨어 교체이고, 그건 부트로더에 사람 손이 필요해 보류 중입니다
(`System_ID/FIRMWARE_RUNBOOK.md`).

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

  # ---- 화면 쪽에서 부르는 것 ---------------------------------------------
  def update(self, partial: Dict[str, float]) -> Dict[str, Any]:
    """**부분 갱신만.** 통째로 갈아끼우지 않는다.

    모르는 이름은 **여기서** 거른다. HUPHY 쪽까지 흘려보내면 `leg.py:485` 가
    `ValueError` 를 던지고 **루프 전체가 멈춥니다** — 오타 하나로 로봇이 서는 것은
    과합니다. 대신 세어 두고 화면에 알립니다.
    """
    bad, notfinite, taken = [], [], {}
    for name, value in partial.items():
      if name not in self._targets:
        bad.append(name)
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

  def freeze(self, name: str) -> Dict[str, Any]:
    """그 관절의 목표를 **지금 측정 위치**로 옮긴다. 더 움직이지 않게.

    `observe()` 가 매 주기 넣어 주는 최신 측정값을 씁니다. 아직 한 번도 못 읽었으면
    그대로 둡니다 — 모르는 값으로 목표를 덮어쓰는 것이 더 위험합니다.
    """
    with self._lock:
      if name not in self._targets:
        return dict(ok=False, why=f"모르는 관절입니다: {name}")
      pos = self._measured.get(name)
      if pos is None:
        return dict(ok=False, why="아직 그 관절의 측정값을 못 받았습니다")
      self._targets[name] = float(pos)
      self.note = f"{name} 지금 자리({pos:+.2f}도)에 묶음"
    return dict(ok=True, deg=pos)

  def observe(self, observation: Dict[str, Any]) -> None:
    """제어 루프가 준 관찰에서 측정 위치를 갈무리한다. `freeze` 가 쓴다."""
    with self._lock:
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
    with self._lock:
      return dict(self._targets)


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
# 상태를 내보내는 통로 — **버스를 읽는 곳은 여기 하나뿐이다**
# ===========================================================================
class StateSender:
  """모터 상태를 JSON 한 줄로 UDP 에 뿌린다. 받지는 않는다.

  ★ **왜 화면이 직접 CAN 을 안 읽는가.** 통신선 하나를 두 프로그램이 각자 읽으면
  서로의 응답을 가로채 엉뚱한 값을 집습니다. 그래서 버스를 만지는 것은 이
  프로그램 하나이고, 화면은 여기서 나온 것을 **받아서 그리기만** 합니다.

  같은 내용을 여러 곳에 보낼 수 있습니다 — 화면(관절 페이지)과 3차원 뷰어.
  """

  def __init__(self, robot, targets, targets_to, hz: float = 20.0):
    self.robot = robot
    self.targets = targets
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
    for part in getattr(self.robot, "limbs", None) or [self.robot]:
      cfg = getattr(part, "config", None)
      bus = getattr(part, "bus", None)
      if cfg is None or bus is None:
        continue
      prefix = f"{cfg.name}/" if getattr(self.robot, "limbs", None) else ""
      for joint, motor in cfg.motors.items():
        try:
          st = bus.state(motor.id)
        except Exception:                                   # noqa: BLE001
          continue
        full = prefix + joint
        out[str(motor.id)] = dict(
          joint=full, model=getattr(motor, "model", ""),
          deg=round(float(getattr(st, "position_deg", 0.0)), 2),
          vel=round(float(getattr(st, "velocity_deg_s", 0.0)), 1),
          nm=round(float(getattr(st, "torque_nm", 0.0)), 2),
          temp=round(float(getattr(st, "temp_c", 0.0)), 1),
          # ★ 동작모드는 **판단에 쓰지 않는다** (위 MODE_IS_REPORTED 참조).
          # 그대로 실어 보내되, 화면이 "꺼짐" 으로 읽지 않도록 모델도 같이 보낸다.
          mode=getattr(st, "mode", None),
          mode_reported=MODE_IS_REPORTED.get(getattr(motor, "model", ""), None),
          tgt=round(float(wanted.get(full, 0.0)), 2),
          age_s=round(now - float(getattr(st, "stamp", 0.0) or now), 2),
        )
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
# 명령을 받는 통로
# ===========================================================================
class TargetServer:
  """한 줄 JSON 을 UDP 로 받아 `LiveTargets` 에 넣는다. 보내지는 않는다."""

  def __init__(self, targets: LiveTargets, port: int, token: str = ""):
    self.targets, self.port, self.token = targets, int(port), str(token)
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
      if "release" in msg:
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
    for part in getattr(robot, "limbs", None) or [robot]:
        bus = getattr(part, "bus", None)
        if bus is None:
            continue
        for motor_id, state in getattr(bus, "_states", {}).items():
            out[motor_id] = getattr(state, "mode", None)
    return out


# ===========================================================================
# 진입점
# ===========================================================================
def build(args):
  """설정을 읽어 로봇과 제어 루프를 세운다. **HUPHY 의 도구만 쓴다.**"""
  from huphy.config import load_robot
  from huphy.control import ControlLoop, Mode
  from huphy.scripts.bringup import build_biped, build_leg

  robot_cfg = load_robot(args.config)
  drain_s = None if args.drain_ms is None else args.drain_ms / 1000.0
  if args.robot:
    machine = build_biped(robot_cfg, allow_uncalibrated=args.allow_uncalibrated,
                          drain_s=drain_s)
  else:
    machine = build_leg(robot_cfg, args.limb,
                        allow_uncalibrated=args.allow_uncalibrated,
                        drain_s=drain_s)
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
  ap.add_argument("--drain-ms", type=float, default=5.0,
                  help="응답을 기다리는 창. 로봇 프로그램 기본은 2.0 인데 "
                       "실측으로 5.0 이 나았습니다 — 아래 주석 참조")
  a = ap.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  machine, loop, hz = build(a)

  names = list(getattr(machine, "joint_names", ()))
  live = LiveTargets(names)
  motion = ThermalCutoff(live, a.max_temp_c)
  server = TargetServer(live, a.listen, a.token).start()
  state = StateSender(machine, live, a.state_to, hz=a.state_hz).start()

  print(f"\n{getattr(machine, 'id', '로봇')} · 초당 {hz:.0f}회 · 관절 {len(names)}개")
  print(f"  목표를 받는 자리 udp:{a.listen}"
        f"{' (암호 있음)' if a.token else ' (암호 없음 — 망 안에서만 쓸 것)'}")
  print(f"  권선 {a.max_temp_c:g}도를 넘으면 그 관절 목표를 지금 자리로 묶음")
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
  print(f"  받은 목표 {live.rx}개 · 모르는 이름 {live.rejected_unknown}개 "
        f"· 숫자가 아님 {live.rejected_not_finite}개 · 과열로 묶음 {motion.events}회")
  if stats is not None and getattr(stats, "link_loss", None):
    print(f"\n  ** 통신 두절로 멈춤: {stats.link_loss} **")
    return 1
  return 0


if __name__ == "__main__":
  sys.path.insert(0, os.path.expanduser("~/HUPHY/src"))
  raise SystemExit(main())
