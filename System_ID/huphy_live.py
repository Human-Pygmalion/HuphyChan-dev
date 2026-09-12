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

HUPHY 가 **안 해 주는 것**은 이 파일이 합니다 — 권선 과열 차단, 그리고 켜기 명령이
실제로 먹혔는지 확인하는 것. 둘 다 HUPHY 에 대응물이 없습니다.


## 목표를 부분만 갱신하는 이유

`Leg._motor_targets` 는 **넘어온 키만** 순회합니다 (`robots/leg.py:475`). 어떤 관절
이름이 이번 주기 dict 에 없으면 그 모터에는 **아무 명령도 안 나가고**, 모터는 펌웨어에
남아 있는 **직전 명령을 계속 실행**합니다. "지금 자리를 유지" 가 아닙니다.

그래서 `LiveTargets` 는 **항상 열두 관절 전부를 들고 있고**, 화면에서 온 값으로
`dict.update` 만 합니다. 통째로 갈아끼우지 않습니다.


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

  def snapshot(self) -> Dict[str, float]:
    with self._lock:
      return dict(self._targets)

  # ---- HUPHY 가 부르는 것 (Motion 계약) -----------------------------------
  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    del t, observation                      # 시간에도 관찰에도 기대지 않는다
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
      if "targets" in msg and isinstance(msg["targets"], dict):
        self.targets.update(msg["targets"])
      elif "name" in msg:
        self.targets.update({str(msg["name"]): msg.get("deg", 0.0)})
    sock.close()


# ===========================================================================
# 켜기가 실제로 먹혔는지 확인 — HUPHY 에 대응물 없음
# ===========================================================================
def report_enable(robot) -> Dict[str, Any]:
  """모터가 **스스로 보고한 동작 모드**를 모아 돌려준다.

  HUPHY 의 `enable_torque()` 는 켜기를 보내고 `_torque_on[mid] = True` 로
  **믿기만** 합니다 (`motors/robstride/bus.py:175-180`). 2026-09-12 에 열두 중
  일곱이 켜기를 거부하는데 아무도 모르는 상태를 만들었습니다. 여기서 대조합니다.
  """
  out: Dict[str, Any] = {}
  for part in getattr(robot, "limbs", None) or [robot]:
    bus = getattr(part, "bus", None)
    if bus is None:
      continue
    for motor_id, state in getattr(bus, "_states", {}).items():
      mode = getattr(state, "mode", None)
      out[motor_id] = dict(mode=mode, on=(mode == 2))
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
  if args.robot:
    machine = build_biped(robot_cfg, allow_uncalibrated=args.allow_uncalibrated)
  else:
    machine = build_leg(robot_cfg, args.limb,
                        allow_uncalibrated=args.allow_uncalibrated)
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
  a = ap.parse_args(argv)

  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  machine, loop, hz = build(a)

  names = list(getattr(machine, "joint_names", ()))
  live = LiveTargets(names)
  motion = ThermalCutoff(live, a.max_temp_c)
  server = TargetServer(live, a.listen, a.token).start()

  print(f"\n{getattr(machine, 'id', '로봇')} · 초당 {hz:.0f}회 · 관절 {len(names)}개")
  print(f"  목표를 받는 자리 udp:{a.listen}"
        f"{' (암호 있음)' if a.token else ' (암호 없음 — 망 안에서만 쓸 것)'}")
  print(f"  권선 {a.max_temp_c:g}도를 넘으면 그 관절 목표를 지금 자리로 묶음")
  print("  한계·속도·프레임·전송·수거는 전부 로봇 프로그램(HUPHY)이 합니다.")
  print("  멈추려면 Ctrl-C.\n")

  try:
    stats = loop.run(motion, duration_s=a.duration)
  except KeyboardInterrupt:
    stats = None
    print("\n  사람이 멈춤")
  finally:
    server.stop()

  modes = report_enable(machine)
  off = [m for m, r in modes.items() if not r["on"]]
  if modes:
    print(f"\n  모터가 보고한 상태: 켜짐 {len(modes)-len(off)}개"
          + (f" · **안 켜짐 {off}**" if off else ""))
  print(f"  받은 목표 {live.rx}개 · 모르는 이름 {live.rejected_unknown}개 "
        f"· 숫자가 아님 {live.rejected_not_finite}개 · 과열로 묶음 {motion.events}회")
  if stats is not None and getattr(stats, "link_loss", None):
    print(f"\n  ** 통신 두절로 멈춤: {stats.link_loss} **")
    return 1
  return 0


if __name__ == "__main__":
  sys.path.insert(0, os.path.expanduser("~/HUPHY/src"))
  raise SystemExit(main())
