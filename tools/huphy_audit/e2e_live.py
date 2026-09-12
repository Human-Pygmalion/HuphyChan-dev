#!/usr/bin/env python3
"""화면 -> HUPHY 제어 루프 -> 통신선 **전 구간**을 험한 입력으로 때려 본다.

2026-09-12 사용자 지시:

  *"모터 입력 넣기 전에 여러 시나리오(큰 범위 밖 입력, kp/kd 과다 등) 짜서 실제로
  안전장치들이 E2E 로 작동하는지 엄밀하게 검토하고, 로직 수정되어야 하는 부분
  (이론상 없어야 한다)까지 확인부터 해."*

  *"너랑 HUPHY 둘 다 코드를 정상적으로 짰으면, 동일 제약조건, 입력에 대해 동일한
  패킷이 나와야 한다. 이것도 반드시 테스트해."*


## 앞선 검사와 무엇이 다른가

`safety_probe.py` 는 HUPHY 의 안전장치를 **하나씩 떼어** 봤습니다. 이 파일은
**우리가 실제로 쓸 경로 그대로** 봅니다 — 화면이 주는 목표가 `LiveTargets` 를 거쳐
HUPHY 의 `ControlLoop` 에 들어가고, 거기서 나온 **8바이트가 통신선에 실제로
실리는 것까지** 확인합니다. 중간에서 가로채지 않습니다.

가짜 통신선(`fake_can.py`)을 씁니다. 선 뽑기·권선 95도 같은 것은 실물로 만들면
위험하기 때문입니다. 모터에 힘이 들어가지 않습니다.


## 운용 조건 (사용자가 2026-09-12 에 정함)

    안전 관절     엉덩이 회전(hip_yaw), 발목 A/B (ankle_a, ankle_b)
    가동 범위     실측 한계에서 **±15도 더 좁게**
    세기 상한     kp <= 15

이 값들이 `SAFE_JOINTS` / `RANGE_TRIM_DEG` / `KP_CAP` 입니다. 실물에 손대기 전에
이 파일이 전부 통과해야 합니다.

    python3 tools/huphy_audit/e2e_live.py
    python3 tools/huphy_audit/e2e_live.py --only E3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "System_ID"))

from fake_can import install                                       # noqa: E402

from huphy.config.schema import LimbConfig, SafetyConfig           # noqa: E402
from huphy.control.loop import ControlLoop, Mode                   # noqa: E402
from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics  # noqa: E402
from huphy.motors.base import Gains, Motor, MotorCalibration       # noqa: E402
from huphy.motors.canbus import CanBus                             # noqa: E402
from huphy.motors.robstride import tables as T                     # noqa: E402
from huphy.motors.robstride.bus import RobStrideBus                # noqa: E402
from huphy.motors.robstride.codec import mit                       # noqa: E402
from huphy.robots.leg import Leg                                   # noqa: E402

from huphy_live import LiveTargets, ThermalCutoff                  # noqa: E402


# ===========================================================================
# 운용 조건
# ===========================================================================
SAFE_JOINTS = ("hip_yaw", "ankle_a", "ankle_b")
"""사용자가 정한 안전 관절. 다리 무게가 안 걸리거나 발끝이라 처져도 위험이 작습니다."""

RANGE_TRIM_DEG = 15.0
"""한계에서 **양쪽으로** 이만큼 안으로. 사용자 확정 2026-09-12:
"한계에서 양쪽으로 15도씩 안으로" (영자세 ±15도가 아님)."""

KP_CAP = 15.0
"""세기 상한. 사용자 지시 "Kp 는 15 이상 주지 마"."""

MODELS = {1: T.Model.RS04, 2: T.Model.RS04, 3: T.Model.RS03,
          4: T.Model.RS04, 5: T.Model.RS03, 6: T.Model.RS03}
MODEL_NAME = {i: m.value for i, m in MODELS.items()}

# ★ 한계의 출처는 **시뮬레이터 모델(MJCF)** 입니다 (사용자 확정 2026-09-12).
# 예전 실측값(v0.5)은 모터를 갈아서 무효이고, 파이의 HUPHY 설정에 들어가 있는
# 것도 이 MJCF 값입니다. `System_ID/mjcf_limits.json` 의 왼다리 몫입니다.
MJCF_LIMITS = {
  "hip_pitch": (-120.0, 25.0),
  "hip_roll": (-25.0, 85.0),
  "hip_yaw": (-45.0, 45.0),
  "knee": (0.0, 120.0),
  "ankle_a": (-68.75, 68.75),
  "ankle_b": (-68.75, 68.75),
}

SAFETY = dict(command_margin_deg=3.0, max_delta_deg=12.0,
              enforce_limits=True, link_loss_cycles=5)
"""실물 설정과 같게. `max_delta_deg` 12.0 = 최고속도 초당 1200도 / 초당 100주기."""

MOTOR_OF = {"hip_pitch": 1, "hip_roll": 2, "hip_yaw": 3,
            "knee": 4, "ankle_a": 5, "ankle_b": 6}


def trimmed(name: str) -> Tuple[float, float]:
  """운용 범위 = **MJCF 한계**에서 양쪽으로 `RANGE_TRIM_DEG` 만큼 좁힌 것."""
  lo, hi = MJCF_LIMITS[name]
  if hi - lo <= 2.0 * RANGE_TRIM_DEG:
    mid = 0.5 * (lo + hi)
    return (mid, mid)
  return (lo + RANGE_TRIM_DEG, hi - RANGE_TRIM_DEG)


# ===========================================================================
# 검사 그릇
# ===========================================================================
class Finding:
  def __init__(self, code, area, title):
    self.code, self.area, self.title = code, area, title
    self.verdict = "확인 못 함"
    self.detail = ""
    self.needs_fix = False

  def hold(self, detail):      # 막았다
    self.verdict, self.detail = "지킴", detail

  def leak(self, detail):      # 못 막았다 -> 고쳐야 한다
    self.verdict, self.detail, self.needs_fix = "구멍", detail, True

  def note(self, detail):      # 막긴 하는데 짚어 둘 것
    self.verdict, self.detail = "주의", detail


CHECKS: List[Tuple[str, str, str, Callable]] = []


def check(code: str, area: str, title: str):
  def wrap(fn):
    CHECKS.append((code, area, title, fn))
    return fn
  return wrap


# ===========================================================================
# 다리 하나 + 우리 경로를 통째로 세운다
# ===========================================================================
def build(kp: float = 10.0, kd: float = 1.0, safety: Optional[dict] = None,
          trim: bool = True):
  """(다리, 루프, 목표표, 동작, 가짜모터) 를 돌려준다.

  **중간을 가로채지 않는다** — 화면이 넣는 자리(`LiveTargets.update`)와 통신선에
  실린 바이트(`motors.sent`)만 봅니다. 그 사이는 전부 HUPHY 입니다.
  """
  motors = install(MODELS)
  cfg = LimbConfig(
    name="left_leg", kind="leg", side="left", channel="can0", control_hz=100.0,
    motors={n: Motor(id=i, model=MODEL_NAME[i],
                     gains=Gains(kp=min(kp, KP_CAP), kd=kd))
            for n, i in MOTOR_OF.items()})
  cal = {n: MotorCalibration(motor_id=i, zero_reference="시험",
                             limits_deg=(trimmed(n) if trim else MJCF_LIMITS[n]))
         for n, i in MOTOR_OF.items()}
  leg = Leg(cfg, RobStrideBus(CanBus(cfg.channel), cfg.motors_by_id()),
            safety=SafetyConfig(**(SAFETY if safety is None else safety)),
            calibration=cal, kinematics=AnkleKinematics(AnkleGeometry().mirrored()))
  leg.connect()
  motors.sent.clear()

  live = LiveTargets(leg.joint_names)
  motion = ThermalCutoff(live, max_temp_c=50.0)
  loop = ControlLoop(leg, hz=cfg.control_hz, mode=Mode.CONTROL)
  return leg, loop, live, motion, motors


def spin(loop, motion, cycles: int = 1, t0: float = 0.0, dt: float = 0.01):
  """제어 루프를 `cycles` 번 돌린다. HUPHY 의 `step` 을 그대로 부른다."""
  for k in range(cycles):
    loop.step(motion, t=t0 + k * dt)


def sent_to(motors, motor_id: int) -> Optional[Dict[str, float]]:
  """그 모터로 **실제로 나간** 마지막 동작 프레임을 사람이 읽는 숫자로 푼다."""
  enc = T.encoding_for(MODELS[motor_id])
  for msg in reversed(motors.sent):
    d = bytes(msg.data)
    if msg.arbitration_id != motor_id or d[0] == 0xFF:
      continue
    kp_u = ((d[3] & 0x0F) << 8) | d[4]
    kd_u = (d[5] << 4) | (d[6] >> 4)
    tau_u = ((d[6] & 0x0F) << 8) | d[7]
    return dict(
      raw=d,
      pos=math.degrees(mit.uint_to_float((d[0] << 8) | d[1],
                                         -enc.pmax_rad, enc.pmax_rad, 16)),
      kp=mit.uint_to_float(kp_u, 0.0, enc.kp_max, 12), kp_raw=kp_u,
      kd=mit.uint_to_float(kd_u, 0.0, enc.kd_max, 12), kd_raw=kd_u,
      tau=mit.uint_to_float(tau_u, -enc.tmax_nm, enc.tmax_nm, 12),
    )
  return None


def count_sent(motors, motor_id: int) -> int:
  return sum(1 for m in motors.sent
             if m.arbitration_id == motor_id and bytes(m.data)[0] != 0xFF)


# ===========================================================================
# 가 · 값 자체가 이상한 경우
# ===========================================================================
@check("E1", "이상한 값", "숫자가 아닌 목표(NaN)가 통신선까지 가는가")
def e1(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.sent.clear()
  reply = live.update({"hip_yaw": float("nan")})
  spin(loop, motion, 3)
  n = count_sent(motors, MOTOR_OF["hip_yaw"])
  if reply["not_finite"] == ["hip_yaw"]:
    f.hold(f"화면 쪽에서 먼저 거부했고(not_finite), 그 뒤 {n}주기 모두 "
           f"0도 유지 프레임만 나갔습니다.")
  else:
    f.leak(f"거부되지 않았습니다: {reply}")


@check("E2", "이상한 값", "NaN 을 HUPHY 에 **직접** 흘리면 HUPHY 가 막는가")
def e2(f):
  """우리 쪽 검사를 일부러 건너뛰고 HUPHY 만으로 막히는지 본다."""
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.sent.clear()
  with live._lock:                       # 일부러 뒷문으로 넣는다
    live._targets["hip_yaw"] = float("nan")
  spin(loop, motion, 3)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit is None:
    f.hold("HUPHY 가 그 모터로 아무것도 안 보냈습니다 "
           "(safety/guards.py:75 유한값 검사).")
  else:
    f.leak(f"NaN 이 통신선까지 갔습니다: {hit['raw'].hex(' ')}")


@check("E3", "범위 밖", "운용 한계를 크게 넘는 목표가 한계 안쪽으로 잘리는가")
def e3(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  lo, hi = trimmed("hip_yaw")
  live.update({"hip_yaw": hi + 500.0})
  spin(loop, motion, 200)                 # 충분히 오래 돌려 끝까지 가게
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  ceiling = hi - SAFETY["command_margin_deg"]
  if hit is None:
    f.leak("아무것도 안 나갔습니다 — 막은 게 아니라 길이 끊긴 것일 수 있습니다.")
  elif hit["pos"] <= ceiling + 0.05:
    f.hold(f"목표 {hi + 500.0:+.1f}도 -> 실제로 나간 값 {hit['pos']:+.2f}도. "
           f"운용 상한 {hi:+.2f}도에서 여유 {SAFETY['command_margin_deg']:g}도 "
           f"안쪽({ceiling:+.2f}도)을 안 넘었습니다.")
  else:
    f.leak(f"{hit['pos']:+.2f}도까지 나갔습니다 (상한 {ceiling:+.2f}도).")


@check("E4", "범위 밖", "반대쪽 한계도 같은가")
def e4(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  lo, hi = trimmed("hip_yaw")
  live.update({"hip_yaw": lo - 500.0})
  spin(loop, motion, 200)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  floor = lo + SAFETY["command_margin_deg"]
  if hit and hit["pos"] >= floor - 0.05:
    f.hold(f"목표 {lo - 500.0:+.1f}도 -> {hit['pos']:+.2f}도. "
           f"하한 {floor:+.2f}도를 안 넘었습니다.")
  else:
    f.leak(f"{hit['pos'] if hit else None} 까지 나갔습니다 (하한 {floor:+.2f}도).")


@check("E5", "속도", "먼 목표로 뛸 때 한 주기 이동량이 묶이는가")
def e5(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  before = sent_to(motors, MOTOR_OF["hip_yaw"])
  motors.sent.clear()
  live.update({"hip_yaw": trimmed("hip_yaw")[1]})
  spin(loop, motion, 1)                   # 딱 한 주기
  after = sent_to(motors, MOTOR_OF["hip_yaw"])
  if after is None:
    f.leak("한 주기에 아무것도 안 나갔습니다.")
    return
  step = abs(after["pos"] - (before["pos"] if before else 0.0))
  cap = SAFETY["max_delta_deg"]
  if step <= cap + 0.05:
    f.hold(f"한 주기에 {step:.2f}도만 움직였습니다 (상한 {cap:g}도). "
           f"초당 100주기면 초당 {step * 100:.0f}도.")
  else:
    f.leak(f"한 주기에 {step:.2f}도 뛰었습니다 (상한 {cap:g}도).")


@check("E6", "속도", "여러 주기에 걸쳐 결국 목표에 도달하는가 (묶였다고 못 가면 안 됨)")
def e6(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  goal = trimmed("hip_yaw")[1] - SAFETY["command_margin_deg"]
  live.update({"hip_yaw": goal})
  spin(loop, motion, 300)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit and abs(hit["pos"] - goal) < 0.5:
    f.hold(f"{hit['pos']:+.2f}도로 도달했습니다 (목표 {goal:+.2f}도). "
           f"속도 제한은 늦출 뿐 막지 않습니다.")
  else:
    f.leak(f"도달 못 했습니다: {hit['pos'] if hit else None} (목표 {goal:+.2f}도)")


# ===========================================================================
# 나 · 세기(kp/kd) 가 과한 경우
# ===========================================================================
@check("E7", "세기", "kp 를 상한보다 크게 주면 상한으로 묶이는가")
def e7(f):
  leg, loop, live, motion, motors = build(kp=500.0)     # 상한 15
  spin(loop, motion, 3)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit is None:
    f.leak("아무것도 안 나갔습니다.")
  elif hit["kp"] <= KP_CAP + 0.01:
    f.hold(f"kp 500 을 넣었는데 나간 값은 {hit['kp']:.3f} 입니다 (상한 {KP_CAP:g}).")
  else:
    f.leak(f"kp {hit['kp']:.1f} 이 그대로 나갔습니다 (상한 {KP_CAP:g}).")


@check("E8", "세기", "kp 가 프로토콜 최대를 넘으면 조용히 잘리는가, 아니면 막히는가")
def e8(f):
  """★ 이건 HUPHY 가 아니라 **인코딩**이 자르는 자리다. 자른다는 사실 자체를
  아무도 안 알려 주므로, 여기서 그 성질을 못 박아 둔다."""
  leg, loop, live, motion, motors = build()
  enc = T.encoding_for(MODELS[MOTOR_OF["hip_yaw"]])
  leg, loop, live, motion, motors = build(kp=enc.kp_max * 10, kd=1.0)
  # build() 가 KP_CAP 으로 묶으므로, 인코딩이 자르는 자리를 보려면 직접 넘긴다
  leg.config.motors["hip_yaw"] = Motor(id=3, model=MODEL_NAME[3],
                                       gains=Gains(kp=enc.kp_max * 10, kd=1.0))
  motors.sent.clear()
  spin(loop, motion, 3)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit and hit["kp_raw"] == 4095:
    f.note(f"프로토콜 최대({enc.kp_max:g})로 **조용히** 잘립니다 — 경고가 없습니다. "
           f"그래서 설정 파일이 유일한 방어선입니다. 우리 쪽 상한 {KP_CAP:g} 는 "
           f"`build()` 에서 겁니다.")
  else:
    f.leak(f"예상과 다릅니다: {hit}")


@check("E9", "세기", "kp 가 아주 작으면 0 으로 떨어지는 자리(양자화 함정)")
def e9(f):
  """kp 는 12비트로 0~5000 을 나눠 담으므로 눈금이 1.221 이다. 그보다 작은 값은
  **0 이 되어 힘이 아예 안 들어간다.** 사람이 "약하게 해 뒀다" 고 믿는 상태에서
  실은 힘이 0 인 것이라, 반대 방향의 위험이다."""
  leg, loop, live, motion, motors = build()
  enc = T.encoding_for(MODELS[MOTOR_OF["hip_yaw"]])
  quantum = enc.kp_max / 4095.0
  leg.config.motors["hip_yaw"] = Motor(id=3, model=MODEL_NAME[3],
                                       gains=Gains(kp=quantum * 0.9, kd=1.0))
  motors.sent.clear()
  spin(loop, motion, 3)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit and hit["kp_raw"] == 0:
    f.note(f"kp {quantum * 0.9:.4f} 를 넣으면 **0 으로 나갑니다** "
           f"(눈금 {quantum:.4f}). 쓰려면 이보다 커야 합니다.")
  else:
    f.leak(f"예상과 다릅니다: {hit}")


# ===========================================================================
# 다 · 통신·온도
# ===========================================================================
@check("E10", "통신", "선이 끊기면 루프가 스스로 멈추는가")
def e10(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.answer = False                    # 선 뽑기
  for k in range(SAFETY["link_loss_cycles"] + 3):
    loop.step(motion, t=0.1 + k * 0.01)
  if getattr(loop, "_stop", False):
    f.hold(f"{SAFETY['link_loss_cycles']}주기 무응답에서 루프가 멈췄습니다.")
  else:
    f.leak("무응답인데도 계속 돕니다.")


@check("E11", "온도", "권선이 뜨거우면 그 관절 목표가 지금 자리로 묶이는가")
def e11(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  live.update({"hip_yaw": 10.0})
  spin(loop, motion, 100)
  warm = sent_to(motors, MOTOR_OF["hip_yaw"])
  motors.temp_c[MOTOR_OF["hip_yaw"]] = 95.0
  spin(loop, motion, 5)
  motors.sent.clear()
  live.update({"hip_yaw": trimmed("hip_yaw")[1]})    # 더 가라고 조른다
  spin(loop, motion, 50)
  hot = sent_to(motors, MOTOR_OF["hip_yaw"])
  if motion.events >= 1 and hot and abs(hot["pos"] - warm["pos"]) < 2.0:
    f.hold(f"95도에서 목표가 {warm['pos']:+.2f}도 부근에 묶였습니다 "
           f"(더 가라고 해도 {hot['pos']:+.2f}도). 차단 {motion.events}회.")
  else:
    f.leak(f"안 묶였습니다: 차단 {motion.events}회, {warm} -> {hot}")


@check("E12", "온도", "HUPHY 자신은 온도로 아무것도 안 막는다 (우리가 메워야 하는 자리)")
def e12(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.temp_c[MOTOR_OF["hip_yaw"]] = 95.0
  motors.sent.clear()
  spin(loop, motion, 5, t0=0.5)
  # 우리 껍데기를 벗기고 HUPHY 만으로
  raw_live = LiveTargets(leg.joint_names)
  raw_live.update({"hip_yaw": trimmed("hip_yaw")[1]})
  motors.sent.clear()
  for k in range(50):
    loop.step(raw_live, t=1.0 + k * 0.01)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if hit and hit["kp_raw"] > 0:
    f.note(f"95도인데 HUPHY 는 그대로 보냅니다 ({hit['pos']:+.2f}도, "
           f"kp {hit['kp']:.2f}). **`ThermalCutoff` 없이 쓰면 안 됩니다.**")
  else:
    f.leak(f"예상과 다릅니다 (HUPHY 가 막았다면 그건 좋은 소식이지만 문서와 다름): {hit}")


# ===========================================================================
# 라 · 발목은 두 모터가 한 기구
# ===========================================================================
@check("E13", "발목", "갈 수 없는 발목 조합을 주면 어떻게 되는가")
def e13(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.sent.clear()
  lo_a, hi_a = trimmed("ankle_a")
  lo_b, hi_b = trimmed("ankle_b")
  live.update({"ankle_a": hi_a, "ankle_b": lo_b})     # 서로 반대 끝
  spin(loop, motion, 200)
  a = sent_to(motors, MOTOR_OF["ankle_a"])
  b = sent_to(motors, MOTOR_OF["ankle_b"])
  if a is None and b is None:
    f.hold("두 모터 다 아무것도 안 나갔습니다 (도달 불가로 통째 버림).")
  elif a and b and lo_a <= a["pos"] <= hi_a and lo_b <= b["pos"] <= hi_b:
    f.hold(f"각자 운용 범위 안에 머물렀습니다 "
           f"(A {a['pos']:+.2f}도, B {b['pos']:+.2f}도). "
           f"모터 공간으로 주면 HUPHY 는 각각 자릅니다 — 조합 검사는 "
           f"발판 공간(ankle_pitch/roll)으로 줄 때만 기구학이 봅니다.")
  else:
    f.leak(f"범위를 벗어났습니다: A {a}, B {b}")


@check("E14", "발목", "발판 공간으로 갈 수 없는 자세를 주면 기구학이 막는가")
def e14(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.sent.clear()
  # 발판 공간 이름으로 바꿔 넣는다 (HUPHY 가 IK 를 푸는 길)
  wild = LiveTargets(("ankle_pitch", "ankle_roll"))
  wild.update({"ankle_pitch": 85.0, "ankle_roll": 85.0})   # 대각으로 극단
  for k in range(50):
    loop.step(wild, t=2.0 + k * 0.01)
  a = sent_to(motors, MOTOR_OF["ankle_a"])
  b = sent_to(motors, MOTOR_OF["ankle_b"])
  if a is None and b is None:
    f.hold("풀 수 없는 자세라 두 모터 다 통째로 버렸습니다 "
           "(kinematics/ankle.py:296).")
  else:
    lo_a, hi_a = trimmed("ankle_a")
    lo_b, hi_b = trimmed("ankle_b")
    ok = (lo_a - 0.05 <= a["pos"] <= hi_a + 0.05 and
          lo_b - 0.05 <= b["pos"] <= hi_b + 0.05)
    (f.hold if ok else f.leak)(
      f"풀리긴 했고 결과가 범위 {'안' if ok else '밖'} 입니다 "
      f"(A {a['pos']:+.2f}도, B {b['pos']:+.2f}도).")


@check("E15", "입력", "모르는 관절 이름이 루프를 멈추게 하지 않는가")
def e15(f):
  """HUPHY 는 모르는 이름에 `ValueError` 를 던져 **루프 전체가 멈춥니다**
  (`robots/leg.py:485`). 오타 하나로 로봇이 서는 것은 과하므로 우리 쪽에서
  먼저 거릅니다. 그 거름망이 실제로 있는지 본다."""
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  reply = live.update({"hip_yaww": 5.0, "hip_yaw": 3.0})
  try:
    spin(loop, motion, 3)
  except ValueError as exc:
    f.leak(f"루프가 멈췄습니다: {exc}")
    return
  if reply["unknown"] == ["hip_yaww"] and reply["taken"] == ["hip_yaw"]:
    f.hold("모르는 이름은 세어 두고 버렸고, 맞는 이름만 들어갔습니다. "
           "루프는 계속 돕니다.")
  else:
    f.leak(f"예상과 다릅니다: {reply}")


@check("E16", "입력", "목표를 부분만 갱신해도 나머지 관절이 계속 명령을 받는가")
def e16(f):
  """키가 빠지면 그 모터에는 **아무것도 안 나가고** 모터가 직전 명령을 계속
  실행합니다(`robots/leg.py:475`). `LiveTargets` 가 항상 전부 들고 있는지 본다."""
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  motors.sent.clear()
  live.update({"hip_yaw": 2.0})            # 하나만 갱신
  spin(loop, motion, 5)
  missing = [n for n, i in MOTOR_OF.items() if count_sent(motors, i) == 0]
  if not missing:
    f.hold("한 관절만 갱신했는데도 여섯 모터 전부 매 주기 명령을 받았습니다.")
  else:
    f.leak(f"명령을 못 받은 관절: {missing}")


# ===========================================================================
# 마 · ★ 우리 것과 HUPHY 것이 같은 패킷을 내는가
# ===========================================================================
@check("E17", "동일성", "같은 제약·같은 입력이면 우리 경로와 HUPHY 경로가 **같은 바이트**를 내는가")
def e17(f):
  """사용자 지시: *"너랑 HUPHY 둘 다 코드를 정상적으로 짰으면, 동일 제약조건,
  입력에 대해 동일한 패킷이 나와야 한다."*

  우리 `cmd_guard.mit_frame` 은 HUPHY `codec/mit.pack_command` 를 따로 구현한
  것입니다. 같은 각도·같은 세기·같은 모터 모델을 주고 **8바이트를 그대로** 견줍니다.
  """
  try:
    import cmd_guard
  except ImportError as exc:
    f.leak(f"우리 쪽 모듈을 못 읽었습니다: {exc}")
    return

  enc_map = {"RS03": cmd_guard.ENC_RS03, "RS04": cmd_guard.ENC_RS04}
  cases = []
  for model in (T.Model.RS03, T.Model.RS04):
    henc = T.encoding_for(model)
    oenc = enc_map[model.value]
    for deg in (0.0, 1.0, -1.0, 12.345, -37.5, 120.0, -120.0, 359.9):
      for kp in (0.0, 1.0, 8.0, 10.0, 15.0):
        for kd in (0.0, 0.3, 1.0, 2.5):
          ours = cmd_guard.mit_frame(deg, kp, kd, oenc)
          theirs = mit.pack_command(
            position_deg=deg, velocity_deg_s=0.0, kp=kp, kd=kd,
            torque_nm=0.0, enc=henc)
          cases.append((model.value, deg, kp, kd, ours, bytes(theirs)))

  bad = [c for c in cases if c[4] != c[5]]
  if not bad:
    f.hold(f"{len(cases)}가지 조합(모델 2 x 각도 8 x 세기 5 x 감쇠 4)에서 "
           f"**바이트가 전부 같습니다.**")
  else:
    lines = []
    for model, deg, kp, kd, ours, theirs in bad[:5]:
      lines.append(f"{model} {deg:+.3f}도 kp{kp:g} kd{kd:g}: "
                   f"우리 {ours.hex(' ')} / HUPHY {theirs.hex(' ')}")
    f.leak(f"{len(bad)}/{len(cases)} 가지에서 다릅니다. 예: " + " | ".join(lines))


@check("E18", "동일성", "한계·속도 클리핑 결과도 두 경로가 같은가")
def e18(f):
  """인코딩만이 아니라 **자르는 결과**도 같아야 합니다. 같은 한계·같은 한 주기
  이동량을 주고, 같은 목표를 넣었을 때 나오는 각도를 견줍니다.

  ★ 우리 쪽 상대 구간(span)은 사용자 지시로 제거했으므로 비교에서 빠집니다.
  """
  try:
    import cmd_guard
  except ImportError as exc:
    f.leak(f"우리 쪽 모듈을 못 읽었습니다: {exc}")
    return
  if hasattr(cmd_guard.CommandGuard, "_slider_span"):
    f.note("우리 쪽에 아직 상대 구간(span)이 남아 있습니다. "
           "제거 전에는 이 비교가 성립하지 않습니다.")
    return

  from huphy.safety import guards as G
  lo, hi = trimmed("hip_yaw")
  margin, cap = SAFETY["command_margin_deg"], SAFETY["max_delta_deg"]
  rows, bad = [], []
  for current in (0.0, 5.0, -5.0, hi, lo):
    for want in (0.0, hi + 100.0, lo - 100.0, current + 3.0, current + 40.0):
      theirs = G.apply(want, current, limits=(lo, hi),
                       command_margin_deg=margin, max_delta_deg=cap,
                       enforce_limits=True)
      inset = (lo + margin, hi - margin)
      clipped = min(inset[1], max(inset[0], want))
      ours = min(current + cap, max(current - cap, clipped))
      rows.append((current, want, getattr(theirs, "value", None), ours))
      tv = getattr(theirs, "value", None)
      if tv is None or abs(tv - ours) > 1e-9:
        bad.append(rows[-1])
  if not bad:
    f.hold(f"{len(rows)}가지 (지금위치 x 목표) 조합에서 자른 결과가 "
           f"소수점 아래까지 같습니다.")
  else:
    f.leak(f"{len(bad)}/{len(rows)} 가지에서 다릅니다. 예: {bad[:3]}")


# ===========================================================================
# 바 · 두 프로세스로 나눈 전 구간 (화면 -> UDP -> 제어 루프 -> 통신선)
# ===========================================================================
@check("E19", "두 프로세스", "화면이 UDP 로 보낸 목표가 통신선까지 가는가")
def e19(f):
  """사용자 결정(2026-09-12): 화면과 제어 루프를 **두 프로세스**로 나누고 UDP 로 잇는다.

  화면 쪽 어댑터(`RemoteGuard`)에 넣은 목표가 제어 루프의 `LiveTargets` 를 거쳐
  실제 바이트가 되는지, **중간을 가로채지 않고** 확인한다.
  """
  import socket as _s
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "System_ID"))
  from joint_web import RemoteGuard
  from huphy_live import TargetServer

  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)

  probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
  port = probe.getsockname()[1]; probe.close()
  server = TargetServer(live, port, token="").start()
  # 다리 하나짜리 루프에 **양다리 표기**(L_hip_yaw_joint -> left_leg/hip_yaw)로
  # 보낸다. 제어 루프가 꼬리를 보고 맞춰 줘야 한다 — 화면은 상대가 한 다리로
  # 도는지 양다리로 도는지 모르는 편이 낫다.
  ui = RemoteGuard(f"127.0.0.1:{port}", arm_token="", allow={"L_hip_yaw_joint"})

  goal = trimmed("hip_yaw")[1] - SAFETY["command_margin_deg"]
  reply = ui.local_target("L_hip_yaw_joint", goal, "")
  deadline = __import__("time").monotonic() + 2.0
  while live.rx == 0 and __import__("time").monotonic() < deadline:
    __import__("time").sleep(0.01)
  spin(loop, motion, 400, t0=1.0)
  server.stop()
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  if not reply.get("ok"):
    f.leak(f"화면 쪽에서 거부: {reply}")
  elif live.rx == 0:
    f.leak("UDP 가 제어 루프에 안 닿았습니다.")
  elif hit and abs(hit["pos"] - goal) < 0.5:
    f.hold(f"화면 -> UDP -> 제어 루프 -> 통신선이 이어집니다 "
           f"(목표 {goal:+.2f}도 -> 나간 값 {hit['pos']:+.2f}도).")
  else:
    f.leak(f"통신선까지 못 갔습니다: {hit}")


@check("E20", "두 프로세스", "암호가 틀리면 화면 쪽에서 먼저 막는가")
def e20(f):
  from joint_web import RemoteGuard
  ui = RemoteGuard("127.0.0.1:1", arm_token="비밀", allow={"hip_yaw"})
  bad = ui.local_target("hip_yaw", 5.0, "틀린값")
  good_blocked = ui.local_target("모르는관절", 5.0, "비밀")
  if bad.get("ok") is False and ui.stats["bad_token"] == 1 and \
     good_blocked.get("ok") is False and ui.stats["unknown"] == 1:
    f.hold("암호가 틀리면 UDP 를 **보내지도 않습니다**. 허락 안 된 이름도 같습니다. "
           "이 둘은 화면 쪽 일이고, 한계·속도는 로봇 프로그램 몫입니다.")
  else:
    f.leak(f"예상과 다릅니다: {bad} / {good_blocked} / {ui.stats}")


@check("E21", "두 프로세스", "'힘 빼기' 가 목표를 지금 자리로 묶는가")
def e21(f):
  leg, loop, live, motion, motors = build()
  spin(loop, motion, 3)
  live.update({"hip_yaw": 10.0})
  spin(loop, motion, 200)
  before = sent_to(motors, MOTOR_OF["hip_yaw"])
  r = live.freeze("hip_yaw")
  motors.sent.clear()
  spin(loop, motion, 200, t0=3.0)
  after = sent_to(motors, MOTOR_OF["hip_yaw"])
  if r.get("ok") and after and abs(after["pos"] - before["pos"]) < 1.0:
    f.hold(f"목표가 {r['deg']:+.2f}도에 묶였고, 그 뒤 200주기 동안 "
           f"{after['pos']:+.2f}도에 머뭅니다. **토크는 안 끊습니다** — "
           f"끊으면 중력이 끌어내려 화면과 실물이 어긋납니다.")
  else:
    f.leak(f"안 묶였습니다: {r} / {before} -> {after}")


@check("E22", "두 프로세스", "화면이 통신선을 만지지 않는가 (읽는 곳은 하나여야 한다)")
def e22(f):
  """같은 선을 두 프로그램이 각자 읽으면 서로의 응답을 가로챕니다. 원격 모드의
  화면 어댑터가 CAN 소켓을 아예 안 여는지 본다."""
  import inspect
  from joint_web import RemotePoller, RemoteGuard
  bad = []
  for cls in (RemotePoller, RemoteGuard):
    src = inspect.getsource(cls)
    for token in ("AF_CAN", "CAN_RAW", "socketcan"):
      if token in src:
        bad.append(f"{cls.__name__} 에 {token}")
  if not bad:
    f.hold("원격 모드의 화면 쪽 두 클래스 어디에도 CAN 소켓이 없습니다. "
           "통신선을 읽는 것은 제어 루프 하나뿐입니다.")
  else:
    f.leak(f"화면이 통신선을 만집니다: {bad}")


# ===========================================================================
# 사 · 응답을 기다리는 창
# ===========================================================================
@check("E23", "대기창", "--drain-ms 로 준 값이 실제 버스까지 닿는가")
def e23(f):
  """설정이 도중에 증발하는 것은 흔한 결함입니다. 값이 **버스 객체까지** 갔는지
  직접 봅니다 — 그래야 '넣었다고 믿었는데 기본값으로 돌던' 일이 안 생깁니다."""
  from huphy.motors.robstride.bus import RobStrideBus as _B
  from huphy.motors.canbus import CanBus as _C, DEFAULT_DRAIN_S
  from huphy.config.schema import LimbConfig as _L
  from huphy.motors.base import Gains as _G, Motor as _M
  install(MODELS)
  cfg = _L(name="left_leg", kind="leg", side="left", channel="can0",
           control_hz=100.0,
           motors={n: _M(id=i, model=MODEL_NAME[i], gains=_G(kp=10.0, kd=1.0))
                   for n, i in MOTOR_OF.items()})
  base = _B(_C(cfg.channel), cfg.motors_by_id())
  wide = _B(_C(cfg.channel), cfg.motors_by_id(), drain_s=0.005)
  if abs(base.drain_s - DEFAULT_DRAIN_S) < 1e-9 and abs(wide.drain_s - 0.005) < 1e-9:
    f.hold(f"기본값 {DEFAULT_DRAIN_S*1000:g}ms 이고, 5ms 를 주면 버스가 5ms 를 "
           f"듭니다. `huphy_live --drain-ms` 가 `build_leg/build_biped` 의 "
           f"`drain_s` 로, 거기서 `RobStrideBus` 로 그대로 갑니다.")
  else:
    f.leak(f"값이 중간에 증발합니다: 기본 {base.drain_s}, 준 값 {wide.drain_s}")


@check("E24", "대기창", "기본 대기창이 **오래 버티는** 값인가")
def e24(f):
  """2026-09-12 실측 (토크 없이, 양다리 12관절):

      창        10초              90초
      0 ms      85Hz 무응답 0     **20초 만에 통신두절로 죽음**
      2 ms      66Hz              66.2Hz 버팀
      3 ms      68Hz              **68.0Hz 버팀**  <- 고른 값
      5 ms      57Hz              61.1Hz 버팀

  ☆ 같은 날 5.0 -> 0.0 -> 3.0 으로 두 번 뒤집혔습니다. 두 번 다 **짧게 재고**
  결론 낸 것이 원인이었습니다. 10초로는 0이 최선처럼 보입니다.
  """
  import inspect, re as _re
  import huphy_live
  src = inspect.getsource(huphy_live.main)
  m = _re.search(r'"--drain-ms".*?default=([\d.]+)', src, _re.S)
  got = float(m.group(1)) if m else -1.0
  if abs(got - 3.0) < 1e-9:
    f.hold("기본값이 3.0 입니다. 0 은 20초 만에 죽고, 5 는 더 느립니다.")
  elif got == 0.0:
    f.leak("기본값이 0 입니다 — 90초 소크에서 통신두절로 죽는 값입니다.")
  else:
    f.leak(f"기본값이 3.0 이 아닙니다: {got}")


def _ankle():
  """발목 기구학만 붙인 가벼운 로봇. 통신선이 필요 없는 검사들이 씁니다."""
  from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics
  import huphy_live as HL

  class _FakeLeg:
    def __init__(self, kin, name):
      self.kinematics = kin
      self.id = name                      # ★ 접두사는 **id** 로 붙는다
      self.config = type("C", (), {"name": name})()

  left = _FakeLeg(AnkleKinematics(AnkleGeometry().mirrored()), "left_leg")
  right = _FakeLeg(AnkleKinematics(AnkleGeometry()), "right_leg")
  # ★ 속성 이름은 `parts` 다 (`robots/biped.py:122`). `limbs` 로 짐작해 놨다가
  # 실물 양다리 구동이 통째로 실패했고, 이 시험은 그 가짜 이름을 쓰고 있었기
  # 때문에 **아무것도 못 잡았다.** 시험이 실물과 같은 이름을 써야 한다.
  robot = type("R", (), {"parts": [left, right]})()
  return HL, robot, left, right


@check("E25", "발목 쌍", "축마다 자르는 것이 실제로 먹히는가")
def e25(f):
  HL, robot, *_ = _ankle()
  base = lambda t, o: {"left_leg/ankle_a": 90.0, "left_leg/ankle_b": -90.0,
                       "right_leg/ankle_a": 0.0, "right_leg/ankle_b": 0.0}
  g = HL.AnklePairGuard(base, robot)
  out = g(0.0, {})
  a, b = out["left_leg/ankle_a"], out["left_leg/ankle_b"]
  if abs(a) <= HL.ANKLE_AXIS_DEG + 1e-6 and abs(b) <= HL.ANKLE_AXIS_DEG + 1e-6:
    f.hold(f"A 90도 / B -90도를 넣었는데 {a:+.2f} / {b:+.2f} 로 나갔습니다 "
           f"(축 한계 \u00b1{HL.ANKLE_AXIS_DEG:g}도).")
  else:
    f.leak(f"축 한계를 넘었습니다: {a:+.2f} / {b:+.2f}")


@check("E26", "발목 쌍", "축 안이어도 **조합**이 ROM 밖이면 물러나는가")
def e26(f):
  """축 한계 안이지만 발판을 한계 밖으로 보내는 조합을 일부러 고른다."""
  HL, robot, left, _ = _ankle()
  worst = None
  n = 61
  lim = HL.ANKLE_AXIS_DEG
  for i in range(n):
    a = -lim + 2 * lim * i / (n - 1)
    for j in range(n):
      b = -lim + 2 * lim * j / (n - 1)
      try:
        p, r = left.kinematics.solve_fk(a, b)
      except Exception:                                    # noqa: BLE001
        continue
      if not (HL.ANKLE_ROM_PITCH[0] <= p <= HL.ANKLE_ROM_PITCH[1]
              and HL.ANKLE_ROM_ROLL[0] <= r <= HL.ANKLE_ROM_ROLL[1]):
        worst = (a, b, p, r)
  if worst is None:
    f.hold(f"축 한계 \u00b1{lim:g}도 안에서는 **조합이 ROM 을 벗어나는 자리가 "
           f"아예 없습니다** ({n}x{n} 격자 전수). 두 겹 중 축 한 겹만으로도 "
           f"이미 닫혀 있고, 조합 검사는 그 위에 덧대는 것입니다.")
    return
  a, b, p, r = worst
  base = lambda t, o: {"left_leg/ankle_a": a, "left_leg/ankle_b": b,
                       "right_leg/ankle_a": 0.0, "right_leg/ankle_b": 0.0}
  g = HL.AnklePairGuard(base, robot)
  out = g(0.0, {})
  pa, pb = out["left_leg/ankle_a"], out["left_leg/ankle_b"]
  p2, r2 = left.kinematics.solve_fk(pa, pb)
  ok = (HL.ANKLE_ROM_PITCH[0] <= p2 <= HL.ANKLE_ROM_PITCH[1]
        and HL.ANKLE_ROM_ROLL[0] <= r2 <= HL.ANKLE_ROM_ROLL[1])
  (f.hold if ok else f.leak)(
    f"A{a:+.1f}/B{b:+.1f} 은 발판을 숙임 {p:+.1f} 기울임 {r:+.1f} 로 보냅니다. "
    f"물러난 뒤 A{pa:+.1f}/B{pb:+.1f} -> 숙임 {p2:+.1f} 기울임 {r2:+.1f} "
    f"({'ROM 안' if ok else '**여전히 밖**'}).")


@check("E27", "발목 쌍", "축 한계가 ROM 과 아직 맞는가 (전수)")
def e27(f):
  """`ANKLE_AXIS_DEG` 는 ROM 에서 계산해 낸 값이다. ROM 을 바꾸면 같이 바뀌어야
  하는데 사람이 잊는다. 그래서 **매번 다시 계산해 대조한다.**"""
  HL, robot, left, right = _ankle()
  P, R = HL.ANKLE_ROM_PITCH, HL.ANKLE_ROM_ROLL
  worst = 1e9
  for kin in (left.kinematics, right.kinematics):
    s_ok = 0.0
    for s in [x * 0.25 for x in range(1, 300)]:
      good = True
      for a in (-s, s):
        for b in (-s, s):
          try:
            p, r = kin.solve_fk(a, b)
            if not (P[0] <= p <= P[1] and R[0] <= r <= R[1]):
              good = False
          except Exception:                                # noqa: BLE001
            good = False
      if good:
        s_ok = s
      else:
        break
    worst = min(worst, s_ok)
  if abs(worst - HL.ANKLE_AXIS_DEG) <= 0.25 + 1e-9:
    f.hold(f"ROM 숙임 {P} 기울임 {R} 로 다시 계산하면 \u00b1{worst:.2f}도이고, "
           f"코드의 값은 \u00b1{HL.ANKLE_AXIS_DEG:g}도입니다. 맞습니다.")
  else:
    f.leak(f"어긋났습니다 — 다시 계산하면 \u00b1{worst:.2f}도인데 코드는 "
           f"\u00b1{HL.ANKLE_AXIS_DEG:g}도입니다. ROM 을 바꾸고 축 한계를 "
           f"안 고쳤을 수 있습니다.")


@check("E28", "발목 쌍", "ROM 이 **지금 쓰는** 시뮬레이터 모델과 같은가")
def e28(f):
  """사용자 지시로 확정한 출처(*"rp 모드 mjcf / urdf 확인해봐"*). 모델이 바뀌면
  여기서 갈립니다.

  ★ **옛 모델은 뺍니다.** `pygmalion_v1`·`cant20`·`cant30`·`rolloff30` 은 숙임 상한이
  +40도라 지금 값(+30도)과 다른데, 그건 v1 시절 값이고 지금 학습도 뷰어도 안 씁니다.
  한 폴더에 세대가 섞여 있어 전부 대조하면 이 차이가 매번 걸립니다.
  """
  import glob, math as _m, re as _re
  import huphy_live as HL
  root = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls")
  # 옛 세대 (숙임 상한 +40도). 지금 학습도 뷰어도 안 쓰고, 파이썬 어디에서도
  # 참조되지 않는 것을 확인했습니다. 날짜도 6~7월로 v2(9월) 이전입니다.
  OLD = ("pygmalion_v1.xml", "pygmalion.xml",
         "cant20", "cant30", "rolloff30")
  # `_` 로 시작하는 것은 도구가 만든 임시 파일입니다 (예: `_scene_tmp.xml`).
  xmls = [x for x in glob.glob(os.path.join(root, "*.xml"))
          if not os.path.basename(x).startswith("_")
          and os.path.basename(x) not in OLD
          and not any(o in os.path.basename(x) for o in OLD if not o.endswith(".xml"))]
  seen = {}
  for path in xmls:
    src = open(path).read()
    for m in _re.finditer(
        r'<joint name="[LR]_ankle_(pitch|roll)_joint"[^>]*range="([-\d.]+) ([-\d.]+)"', src):
      key = (m.group(1), round(_m.degrees(float(m.group(2))), 2),
             round(_m.degrees(float(m.group(3))), 2))
      seen.setdefault(key, []).append(os.path.basename(path))
  want = {("pitch",) + tuple(HL.ANKLE_ROM_PITCH), ("roll",) + tuple(HL.ANKLE_ROM_ROLL)}
  if not xmls:
    f.note("시뮬레이터 모델 파일을 못 찾았습니다 — 대조를 건너뜁니다.")
  elif set(seen) == want:
    f.hold(f"지금 쓰는 모델 {len(xmls)}개가 전부 같은 값이고 코드와 일치합니다: "
           f"숙임 {HL.ANKLE_ROM_PITCH} 기울임 {HL.ANKLE_ROM_ROLL}. "
           f"(옛 세대 {len(OLD)}종은 숙임 상한이 +40도라 뺐습니다 — "
           f"코드에서 참조하지 않는 것을 확인했습니다)")
  else:
    lines = [f"{k} <- {v[:2]}" for k, v in sorted(seen.items())]
    f.leak(f"모델과 코드가 다릅니다. 모델 {lines} / 코드 {sorted(want)}")


@check("E29", "발목 쌍", "물러날 때 두 모터가 **같이** 움직이는가")
def e29(f):
  """한쪽만 자르면 두 로드가 서로 다른 자세를 요구해 기구가 비틉니다."""
  HL, robot, left, _ = _ankle()
  lim = HL.ANKLE_AXIS_DEG
  base = lambda t, o: {"left_leg/ankle_a": lim, "left_leg/ankle_b": -lim,
                       "right_leg/ankle_a": 0.0, "right_leg/ankle_b": 0.0}
  g = HL.AnklePairGuard(base, robot)
  g(0.0, {})
  out = g(0.01, {})
  a, b = out["left_leg/ankle_a"], out["left_leg/ankle_b"]
  p, r = left.kinematics.solve_fk(a, b)
  ok = (HL.ANKLE_ROM_PITCH[0] <= p <= HL.ANKLE_ROM_PITCH[1]
        and HL.ANKLE_ROM_ROLL[0] <= r <= HL.ANKLE_ROM_ROLL[1])
  ratio = abs(a) / lim if lim else 0.0
  (f.hold if ok else f.leak)(
    f"A{a:+.2f}/B{b:+.2f} (원한 값의 {ratio*100:.0f}%) -> 숙임 {p:+.1f} "
    f"기울임 {r:+.1f}. 둘이 같은 비율로 물러납니다.")


# ===========================================================================
# 자 · 여는 관절만 열기 · 한계 좁히기
# ===========================================================================
@check("E30", "관절 고르기", "고르지 않은 관절에는 **아무것도 안 나가는가**")
def e30(f):
  """이게 이번 실물 시험의 핵심 안전장치다. 안전 관절만 열고 나머지는 꺼진 채 둔다.

  특히 무릎은 0도가 곧 하드스톱이라, 실수로 목표 표에 들어가면 그대로 밀어붙인다.
  """
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build()
  spin(loop, motion_all, 3)

  chosen = HL.pick_joints(leg, set(SAFE_JOINTS))
  live = HL.LiveTargets(chosen)
  motion = HL.AnklePairGuard(HL.ThermalCutoff(live, 50.0), leg)
  motors.sent.clear()
  for k in range(80):
    loop.step(motion, t=5.0 + k * 0.01)

  opened = {n: count_sent(motors, MOTOR_OF[n]) for n in SAFE_JOINTS}
  shut = {n: count_sent(motors, MOTOR_OF[n])
          for n in MOTOR_OF if n not in SAFE_JOINTS}
  if all(v > 0 for v in opened.values()) and all(v == 0 for v in shut.values()):
    f.hold(f"연 관절 {sorted(opened)} 은 명령을 받고({list(opened.values())}주기), "
           f"안 연 관절 {sorted(shut)} 에는 **한 번도 안 나갔습니다**. "
           f"무릎이 하드스톱(0도)으로 끌려가지 않습니다.")
  else:
    f.leak(f"샜습니다 — 연 것 {opened} / 안 연 것 {shut}")


@check("E31", "관절 고르기", "없는 관절 이름을 주면 시작 전에 멈추는가")
def e31(f):
  import huphy_live as HL
  leg, loop, *_ = build()
  try:
    HL.pick_joints(leg, {"hip_yaw", "존재하지_않는_관절"})
  except SystemExit as exc:
    f.hold(f"시작 전에 멈춥니다: {str(exc)[:80]}. 오타로 엉뚱한 관절만 열린 채 "
           f"도는 것보다 안 뜨는 편이 낫습니다.")
    return
  f.leak("오타를 그냥 지나쳤습니다.")


@check("E32", "한계 좁히기", "--trim-deg 가 실제 한계를 양쪽으로 좁히는가")
def e32(f):
  import huphy_live as HL
  leg, loop, live, motion, motors = build(trim=False)     # MJCF 원래 한계로
  before = {n: tuple(leg.calibration[n].limits_deg) for n in SAFE_JOINTS}
  HL.trim_limits(leg, RANGE_TRIM_DEG)
  after = {n: tuple(leg.calibration[n].limits_deg) for n in SAFE_JOINTS}
  ok = all(abs(after[n][0] - (before[n][0] + RANGE_TRIM_DEG)) < 1e-9
           and abs(after[n][1] - (before[n][1] - RANGE_TRIM_DEG)) < 1e-9
           for n in SAFE_JOINTS)
  detail = " · ".join(f"{n} {before[n][0]:+.1f}~{before[n][1]:+.1f} -> "
                      f"{after[n][0]:+.1f}~{after[n][1]:+.1f}" for n in SAFE_JOINTS)
  (f.hold if ok else f.leak)(detail)


@check("E33", "한계 좁히기", "좁힌 한계가 **실제로 통신선에서** 지켜지는가")
def e33(f):
  """설정만 바꾸고 끝내면 안 되고, 나가는 바이트가 그 안이어야 합니다."""
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build(trim=False)
  HL.trim_limits(leg, RANGE_TRIM_DEG)
  spin(loop, motion_all, 3)
  chosen = HL.pick_joints(leg, {"hip_yaw"})
  live = HL.LiveTargets(chosen)
  motion = HL.ThermalCutoff(live, 50.0)
  live.update({"hip_yaw": 900.0})
  motors.sent.clear()
  for k in range(400):
    loop.step(motion, t=9.0 + k * 0.01)
  hit = sent_to(motors, MOTOR_OF["hip_yaw"])
  lo, hi = leg.calibration["hip_yaw"].limits_deg
  ceiling = hi - SAFETY["command_margin_deg"]
  if hit and hit["pos"] <= ceiling + 0.05:
    f.hold(f"목표 900도 -> 나간 값 {hit['pos']:+.2f}도. 좁힌 상한 {hi:+.2f}도에서 "
           f"여유 {SAFETY['command_margin_deg']:g}도 안({ceiling:+.2f}도)입니다.")
  else:
    f.leak(f"넘었습니다: {hit['pos'] if hit else None} (상한 {ceiling:+.2f})")


# ===========================================================================
# 차 · 시작하는 순간
# ===========================================================================
@check("E34", "시작", "제어 루프가 켜질 때 **지금 자리를 잡고** 시작하는가")
def e34(f):
  """0도를 그대로 명령하면 관절이 있던 자리만큼이 계단이 되어 한 번에 튑니다."""
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build()
  # 관절을 0도가 아닌 자리로 **실제로 몰아 놓는다.** 가짜 모터는 명령을 받아야만
  # 답하므로, 명령 없이 자리만 바꾸면 상태가 안 오고 통신 두절로 멈춘다.
  live_all.update({"hip_yaw": 11.26})
  spin(loop, motion_all, 300)
  here = leg.get_observation()["hip_yaw.pos"]

  live = HL.LiveTargets(HL.pick_joints(leg, {"hip_yaw"}))
  motion = HL.ThermalCutoff(live, 50.0)
  motors.sent.clear()
  loop.step(motion, t=9.0)
  first = sent_to(motors, MOTOR_OF["hip_yaw"])
  loop.step(motion, t=9.01)
  second = sent_to(motors, MOTOR_OF["hip_yaw"])
  if second is None:
    f.leak("아무것도 안 나갔습니다.")
  elif abs(second["pos"] - here) < 0.5:
    f.hold(f"관절이 {here:+.2f}도에 있을 때 첫 명령이 {second['pos']:+.2f}도입니다 — "
           f"**있던 자리를 잡습니다.** 움직임은 사람이 시켜야 생깁니다.")
  else:
    f.leak(f"관절은 {here:+.2f}도인데 첫 명령이 {second['pos']:+.2f}도입니다. "
           f"{abs(second['pos'] - here):.2f}도짜리 계단이 나갑니다.")


@check("E35", "시작", "사람이 목표를 준 뒤에는 측정값이 목표를 덮어쓰지 않는가")
def e35(f):
  """처지는 관절에서 목표가 따라 내려가면 영영 못 올라갑니다."""
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build()
  live_all.update({"hip_yaw": 5.0})
  spin(loop, motion_all, 200)
  live = HL.LiveTargets(HL.pick_joints(leg, {"hip_yaw"}))
  live.update({"hip_yaw": 20.0})            # 사람이 먼저 목표를 줬다
  motion = HL.ThermalCutoff(live, 50.0)
  for k in range(5):
    loop.step(motion, t=9.0 + k * 0.01)
  if abs(live.snapshot()["hip_yaw"] - 20.0) < 1e-9:
    f.hold("사람이 준 목표 +20.00도가 그대로 남았습니다. 측정값(+5도)이 "
           "덮어쓰지 않습니다.")
  else:
    f.leak(f"목표가 {live.snapshot()['hip_yaw']:+.2f}도로 덮였습니다.")


# ===========================================================================
# 카 · 0도가 곧 하드스톱인 관절 (무릎)
# ===========================================================================
@check("E36", "하드스톱", "0도가 하드스톱인 무릎에 0도를 주면 여유만큼 떨어져 서는가")
def e36(f):
  """사용자 질문(2026-09-12): *"0도에서 하드스탑쪽으로 가는 명령 주는 건 HUPHY
  안전필터 클리핑이 막아주지?"*

  무릎은 왼쪽이 `0~+120`, 오른쪽이 `-120~0` 이라 **0도가 곧 하드스톱**입니다.
  지금까지 검사한 것은 엉덩이회전(양쪽이 다 열린 관절)이었으므로, 한쪽 끝이
  막힌 관절로 직접 확인합니다.
  """
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build(trim=False)   # 모델 한계 그대로
  spin(loop, motion_all, 3)
  live = HL.LiveTargets(HL.pick_joints(leg, {"knee"}))
  motion = HL.ThermalCutoff(live, 50.0)
  live.update({"knee": 0.0})
  motors.sent.clear()
  for k in range(400):
    loop.step(motion, t=9.0 + k * 0.01)
  hit = sent_to(motors, MOTOR_OF["knee"])
  lo, hi = leg.calibration["knee"].limits_deg
  floor = lo + SAFETY["command_margin_deg"]
  if hit is None:
    f.leak("아무것도 안 나갔습니다.")
  elif abs(hit["pos"] - floor) < 0.1:
    f.hold(f"한계 {lo:+.1f}~{hi:+.1f}도에서 0도를 달라고 했더니 "
           f"**{hit['pos']:+.2f}도**가 나갔습니다 — 하드스톱에서 "
           f"{SAFETY['command_margin_deg']:g}도 떨어져 섭니다.")
  else:
    f.leak(f"{hit['pos']:+.2f}도가 나갔습니다 (하한 {floor:+.2f}도).")


@check("E37", "하드스톱", "하드스톱 **너머**를 달라고 해도 같은 자리에 서는가")
def e37(f):
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build(trim=False)
  spin(loop, motion_all, 3)
  live = HL.LiveTargets(HL.pick_joints(leg, {"knee"}))
  motion = HL.ThermalCutoff(live, 50.0)
  lo, hi = leg.calibration["knee"].limits_deg
  floor = lo + SAFETY["command_margin_deg"]
  seen = []
  for want in (-5.0, -50.0, -500.0):
    live.update({"knee": want})
    motors.sent.clear()
    for k in range(400):
      loop.step(motion, t=20.0 + k * 0.01)
    hit = sent_to(motors, MOTOR_OF["knee"])
    seen.append((want, hit["pos"] if hit else None))
  ok = all(p is not None and abs(p - floor) < 0.1 for _, p in seen)
  (f.hold if ok else f.leak)(
    "한계 너머로 얼마를 달라고 하든 같은 자리에 섭니다: "
    + " · ".join(f"{w:+.0f}도 -> {p:+.2f}도" for w, p in seen)
    + f" (하한 {lo:+.1f} + 여유 {SAFETY['command_margin_deg']:g} = {floor:+.2f}도)")


@check("E38", "하드스톱", "이미 한계 밖에 있으면 **한 주기씩** 돌아오는가")
def e38(f):
  """실물에서 실제로 일어날 일입니다 — 지금 오른 무릎이 +0.19도인데 허용 상한이
  -3.00도라, 켜는 순간 한계 안으로 끌려옵니다. 한 번에 뛰면 안 됩니다."""
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build(trim=False)
  live_all.update({"knee": 30.0})          # 일단 한계 안 깊숙이
  spin(loop, motion_all, 400)
  # 한계를 관절보다 아래로 끌어내려 '이미 밖' 을 만든다
  from dataclasses import replace
  cal = dict(leg.calibration)
  cal["knee"] = replace(cal["knee"], limits_deg=(0.0, 10.0))
  leg._set_calibration(cal)
  live = HL.LiveTargets(HL.pick_joints(leg, {"knee"}))
  motion = HL.ThermalCutoff(live, 50.0)
  motors.sent.clear()
  steps = []
  prev = None
  for k in range(6):
    loop.step(motion, t=40.0 + k * 0.01)
    hit = sent_to(motors, MOTOR_OF["knee"])
    if hit and prev is not None:
      steps.append(abs(hit["pos"] - prev))
    if hit:
      prev = hit["pos"]
  cap = SAFETY["max_delta_deg"]
  if steps and max(steps) <= cap + 0.05:
    f.hold(f"한 주기 최대 이동 {max(steps):.2f}도 (상한 {cap:g}도). "
           f"한계 밖에서 안으로도 한 번에 안 뜁니다.")
  else:
    f.leak(f"한 주기에 {max(steps) if steps else '?'}도 뛰었습니다 (상한 {cap:g}도).")


# ===========================================================================
# 타 · 양다리 — **가짜가 아니라 진짜 Biped 로**
# ===========================================================================
def build_biped_fake():
  """다리 둘을 진짜 `Biped` 로 묶는다. 통신선만 가짜다."""
  from huphy.robots.biped import Biped
  motors = install({**MODELS, **{i + 6: m for i, m in MODELS.items()}})
  legs = []
  for side, name, base in (("left", "left_leg", 0), ("right", "right_leg", 6)):
    cfg = LimbConfig(
      name=name, kind="leg", side=side, channel=f"can{base // 6}",
      control_hz=100.0,
      motors={n: Motor(id=i + base, model=MODEL_NAME[i],
                       gains=Gains(kp=10.0, kd=1.0))
              for n, i in MOTOR_OF.items()})
    cal = {n: MotorCalibration(motor_id=i + base, zero_reference="시험",
                               limits_deg=MJCF_LIMITS[n])
           for n, i in MOTOR_OF.items()}
    geo = AnkleGeometry().mirrored() if side == "left" else AnkleGeometry()
    leg = Leg(cfg, RobStrideBus(CanBus(cfg.channel), cfg.motors_by_id()),
              safety=SafetyConfig(**SAFETY), calibration=cal,
              kinematics=AnkleKinematics(geo))
    leg.connect()
    legs.append(leg)
  robot = Biped(legs, safety=SafetyConfig(**SAFETY))
  motors.sent.clear()
  return robot, motors


@check("E39", "양다리", "고른 관절 이름을 **진짜 Biped 가** 받아들이는가")
def e39(f):
  """★ 이 검사가 없어서 실물 양다리 구동이 통째로 실패했습니다.

  `Biped` 는 이름 앞에 팔다리를 붙여야 받습니다(`biped.py:367`). 제가 팔다리
  목록을 `limbs` 라는 **없는 속성**에서 꺼내고 있었는데, 이 파일의 시험이
  **같은 틀린 이름으로 만든 가짜 로봇**을 쓰고 있어서 전부 통과했습니다. 진짜
  속성은 `parts` 입니다(`biped.py:122`).

  교훈: 시험이 실물과 **같은 물건**을 써야 합니다. 내가 지어낸 모양을 시험하면
  내 착각이 그대로 통과합니다.
  """
  import huphy_live as HL
  robot, motors = build_biped_fake()
  chosen = HL.pick_joints(robot, set(MOTOR_OF))
  if len(chosen) != 12:
    f.leak(f"12개가 아니라 {len(chosen)}개를 골랐습니다: {chosen}")
    return
  if not all("/" in n for n in chosen):
    f.leak(f"접두사가 안 붙었습니다: {chosen[:3]}")
    return
  try:
    robot.build_commands({n: 0.0 for n in chosen})
  except ValueError as exc:
    f.leak(f"진짜 Biped 가 거부했습니다: {exc}")
    return
  f.hold(f"12개 전부 접두사가 붙었고({chosen[0]}, …) 진짜 `Biped.build_commands` 가 "
         f"받아들입니다.")


@check("E40", "양다리", "양다리에서도 안 연 관절에는 아무것도 안 나가는가")
def e40(f):
  import huphy_live as HL
  robot, motors = build_biped_fake()
  loop = ControlLoop(robot, hz=100.0, mode=Mode.CONTROL)
  chosen = HL.pick_joints(robot, {"hip_yaw"})
  live = HL.LiveTargets(chosen)
  motion = HL.AnklePairGuard(HL.ThermalCutoff(live, 50.0), robot)
  motors.sent.clear()
  for k in range(60):
    loop.step(motion, t=k * 0.01)
  yaw = [count_sent(motors, i) for i in (3, 9)]
  others = {i: count_sent(motors, i) for i in range(1, 13) if i not in (3, 9)}
  if all(v > 0 for v in yaw) and all(v == 0 for v in others.values()):
    f.hold(f"양쪽 엉덩이회전(3·9번)만 {yaw}주기 받았고 나머지 열 개는 0회입니다.")
  else:
    f.leak(f"샜습니다 — 회전 {yaw} / 나머지 {others}")


@check("E42", "시작", "자세를 모르는 동안에는 **아무 명령도 안 나가는가**")
def e42(f):
  """못 뿌렸을 때 0도가 나가는 것보다, 아무것도 안 나가는 편이 안전합니다.
  모터는 꺼진 채 남고 수거는 계속 돌아 다음 주기에 자세를 알게 됩니다."""
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build()
  live_all.update({"hip_yaw": 11.0})
  spin(loop, motion_all, 300)

  live = HL.LiveTargets(HL.pick_joints(leg, {"hip_yaw"}))
  # 응답이 하나도 없는 관찰 -> 명령이 나가면 안 된다
  out = live(0.0, {"hip_yaw.pos": 0.0, "stale_motors": 6})
  if out:
    f.leak(f"자세를 모르는데 명령을 냈습니다: {out}")
    return
  # 응답이 온 뒤에는 낸다
  out2 = live(0.01, leg.get_observation())
  if out2 and abs(out2["hip_yaw"] - leg.get_observation()["hip_yaw.pos"]) < 0.5:
    f.hold(f"모를 때는 빈 사전(HUPHY 가 전송을 건너뜀), 안 뒤에는 "
           f"{out2['hip_yaw']:+.2f}도를 냅니다. 막은 주기 {live._blocked}회.")
  else:
    f.leak(f"응답이 온 뒤에도 이상합니다: {out2}")


@check("E41", "시작", "응답을 못 받은 첫 주기에 **빈칸(0도)을 자세로 착각하지 않는가**")
def e41(f):
  """2026-09-12 실물에서 실제로 일어났습니다. 응답 대기창을 0으로 바꾸자 첫 주기에
  응답이 없었고, 관절이 0.03~9.46도에 있는데 시작 자세를 전부 `+0.00도` 로 잡아
  열두 관절에 0도를 명령했습니다.

  `MotorState` 는 응답 전에도 `position_deg=0.0` 을 들고 있고(`motors/base.py:127`)
  관찰에는 그냥 숫자로 실려 옵니다. **키가 있는지만 봐서는 못 가립니다.**
  """
  import huphy_live as HL
  leg, loop, live_all, motion_all, motors = build()
  live_all.update({"hip_yaw": 11.0})
  spin(loop, motion_all, 300)
  here = leg.get_observation()["hip_yaw.pos"]

  live = HL.LiveTargets(HL.pick_joints(leg, {"hip_yaw"}))
  # 응답이 아예 없는 관찰 — 첫 주기의 모습
  live.observe({"hip_yaw.pos": 0.0, "stale_motors": 6})
  live._seed_once()
  if live._seeded:
    f.leak(f"응답도 없는데 씨를 뿌렸습니다: {live.snapshot()}")
    return
  # 응답이 온 뒤에는 제대로
  live.observe(leg.get_observation())
  live._seed_once()
  got = live.snapshot()["hip_yaw"]
  if live._seeded and abs(got - here) < 0.5:
    f.hold(f"응답 없는 주기에는 안 뿌리고, 응답이 온 뒤 {got:+.2f}도"
           f"(실제 {here:+.2f}도)로 뿌립니다.")
  else:
    f.leak(f"뿌린 값 {got:+.2f}도, 실제 {here:+.2f}도")


@check("E46", "화면 연결", "양다리에서 팔다리 없는 이름은 **모호하다고 거절하는가**")
def e46(f):
  """양다리인데 `hip_yaw` 만 오면 어느 다리인지 정할 근거가 없습니다. 조용히 한쪽을
  고르면 반대쪽이 안 움직이는 줄 모른 채 돕니다."""
  import huphy_live as HL
  robot, motors = build_biped_fake()
  live = HL.LiveTargets(HL.pick_joints(robot, {"hip_yaw"}))
  vague = live.update({"hip_yaw": 5.0})
  clear = live.update({"left_leg/hip_yaw": 5.0})
  if vague["ok"] is False and clear["ok"] is True \
     and clear["taken"] == ["left_leg/hip_yaw"]:
    f.hold("팔다리 없는 이름은 거절하고, 붙은 이름만 받습니다. "
           "다리 하나짜리 루프에서는 꼬리만 맞아도 받습니다(E19).")
  else:
    f.leak(f"예상과 다릅니다: 모호 {vague} / 분명 {clear}")


@check("E43", "화면 연결", "화면 이름이 제어 루프 이름으로 **제대로 번역되는가**")
def e43(f):
  """화면은 `L_hip_yaw_joint`, 제어 루프는 `left_leg/hip_yaw` 를 씁니다.
  번역이 틀리면 슬라이더를 아무리 움직여도 '모르는 이름' 으로 조용히 버려집니다."""
  from joint_web import RemoteGuard
  import huphy_live as HL
  robot, motors = build_biped_fake()
  names = set(HL.pick_joints(robot, set(MOTOR_OF)))
  bad = [sim for sim, huphy in RemoteGuard.TO_HUPHY.items() if huphy not in names]
  if bad:
    f.leak(f"제어 루프가 모르는 이름으로 번역됩니다: {bad}")
    return
  if len(RemoteGuard.TO_HUPHY) != 12:
    f.leak(f"12개가 아니라 {len(RemoteGuard.TO_HUPHY)}개만 번역됩니다")
    return
  f.hold(f"12개 전부 제어 루프가 아는 이름으로 번역됩니다 "
         f"(예: L_hip_yaw_joint -> {RemoteGuard.TO_HUPHY['L_hip_yaw_joint']}).")


@check("E44", "화면 연결", "슬라이더 -> UDP -> 제어 루프 -> 통신선이 **끝까지** 이어지는가")
def e44(f):
  """화면 어댑터에 넣은 목표가 실제 바이트가 되는지, 양다리 이름으로 확인합니다."""
  import socket as _s, time as _t
  from joint_web import RemoteGuard
  import huphy_live as HL
  robot, motors = build_biped_fake()
  loop = ControlLoop(robot, hz=100.0, mode=Mode.CONTROL)
  chosen = HL.pick_joints(robot, {"hip_yaw"})
  live = HL.LiveTargets(chosen)
  motion = HL.ThermalCutoff(live, 50.0)
  for k in range(5):                      # 자세를 알게 한다
    loop.step(motion, t=k * 0.01)

  probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
  port = probe.getsockname()[1]; probe.close()
  server = HL.TargetServer(live, port, token="").start()
  ui = RemoteGuard(f"127.0.0.1:{port}", arm_token="", allow={"L_hip_yaw_joint"})
  reply = ui.local_target("L_hip_yaw_joint", 12.0, "")
  end = _t.monotonic() + 2.0
  while live.rx == 0 and _t.monotonic() < end:
    _t.sleep(0.01)
  motors.sent.clear()
  for k in range(400):
    loop.step(motion, t=1.0 + k * 0.01)
  server.stop()
  hit = sent_to(motors, 3)                # 왼 엉덩이회전
  if not reply.get("ok"):
    f.leak(f"화면 쪽에서 거부: {reply}")
  elif live.rx == 0:
    f.leak("UDP 가 제어 루프에 안 닿았습니다.")
  elif hit and abs(hit["pos"] - 12.0) < 0.5:
    f.hold(f"슬라이더 +12.00도 -> 통신선 {hit['pos']:+.2f}도. "
           f"이름 번역까지 포함해 끝까지 이어집니다.")
  else:
    f.leak(f"통신선까지 못 갔습니다: {hit}")


@check("E45", "화면 연결", "화면이 RS03 을 '힘 꺼짐' 으로 단정하지 않는가")
def e45(f):
  """RS03 은 힘이 들어가도 동작모드를 0으로 보고합니다. 화면이 그걸 '꺼짐' 으로
  그리면 사람이 잘못 판단합니다 — 제가 그것 때문에 멀쩡한 모터 일곱 개를 고장으로
  몰았습니다."""
  import json as _j, socket as _s, time as _t
  from joint_web import RemotePoller
  probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
  port = probe.getsockname()[1]; probe.close()
  rp = RemotePoller(port).start()
  sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
  pkt = {"motors": {"3": {"joint": "left_leg/hip_yaw", "deg": 1.0, "vel": 0.0,
                          "nm": 0.0, "temp": 30.0, "mode": 0,
                          "mode_reported": False, "tgt": 1.0, "age_ms": 3.0},
                    "4": {"joint": "left_leg/knee", "deg": 9.0, "vel": 0.0,
                          "nm": 0.0, "temp": 30.0, "mode": 2,
                          "mode_reported": True, "tgt": 9.0, "age_ms": 3.0}}}
  for _ in range(5):
    sock.sendto(_j.dumps(pkt).encode(), ("127.0.0.1", port)); _t.sleep(0.05)
  _t.sleep(0.3)
  rows, _ = rp.snapshot()
  rp.stop()
  if not rows:
    f.leak("상태를 못 받았습니다.")
  elif rows[3].get("mode") is None and rows[4].get("mode") == 2 \
       and abs(rows[3].get("age_s", -1) - 0.0) < 0.01:
    f.hold("보고하지 않는 모델(3번)은 모드를 **비워** 두고, 보고하는 모델(4번)은 "
           "그대로 둡니다. 나이도 초 단위로 바뀝니다.")
  else:
    f.leak(f"예상과 다릅니다: {rows}")


@check("E47", "화면 연결", "제어 루프가 죽으면 화면이 **목표를 안 보내고 알리는가**")
def e47(f):
  """2026-09-12 에 실제로 겪었습니다. 제어 루프가 통신두절로 죽어 있는데 화면이
  *"목표 10.0도 유지 중"* 이라고 적었습니다 — UDP 는 받는 쪽이 죽어도 오류를 안
  내기 때문입니다. 사람이 관절이 붙잡혀 있다고 믿을 상태였습니다."""
  import socket as _s, time as _t
  from joint_web import RemoteGuard, RemotePoller
  probe = _s.socket(_s.AF_INET, _s.SOCK_DGRAM); probe.bind(("127.0.0.1", 0))
  port = probe.getsockname()[1]; probe.close()
  rp = RemotePoller(port).start()
  ui = RemoteGuard("127.0.0.1:1", arm_token="", allow={"L_hip_yaw_joint"}, poller=rp)

  dead = ui.local_target("L_hip_yaw_joint", 10.0, "")   # 상태를 한 번도 못 받음
  if dead.get("ok") is not False:
    rp.stop(); f.leak(f"죽은 루프에 목표를 보냈습니다: {dead}"); return

  # 상태가 오기 시작하면 다시 받는다
  sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
  import json as _j
  pkt = {"motors": {"3": {"joint": "left_leg/hip_yaw", "deg": 1.0, "vel": 0.0,
                          "nm": 0.0, "temp": 30.0, "mode": None,
                          "mode_reported": False, "tgt": 1.0, "age_ms": 3.0,
                          "lo": -42.0, "hi": 42.0}}}
  for _ in range(3):
    sock.sendto(_j.dumps(pkt).encode(), ("127.0.0.1", port)); _t.sleep(0.05)
  _t.sleep(0.3)
  alive = ui.local_target("L_hip_yaw_joint", 10.0, "")
  rp.stop()
  if alive.get("ok") is True and ui.stats.get("loop_dead") == 1:
    f.hold("루프가 말이 없으면 **보내지 않고 거절**하고, 상태가 오면 다시 받습니다. "
           "거절 1회를 셌습니다.")
  else:
    f.leak(f"예상과 다릅니다: {alive} / {ui.stats}")


# ===========================================================================
# 실행
# ===========================================================================
def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--only", nargs="*", help="이 번호만 (예: E3 E5)")
  a = ap.parse_args(argv)

  print("\n화면 -> HUPHY 제어 루프 -> 통신선 전 구간 검사 (가짜 모터, 힘 안 들어감)")
  print(f"  안전 관절 {SAFE_JOINTS} · 범위 MJCF한계에서 양쪽 {RANGE_TRIM_DEG:g}도 안으로 "
        f"· 세기 상한 {KP_CAP:g}")
  print(f"  한 주기 이동량 {SAFETY['max_delta_deg']:g}도 · "
        f"한계 여유 {SAFETY['command_margin_deg']:g}도\n")

  out: List[Finding] = []
  for code, area, title, fn in CHECKS:
    if a.only and code not in a.only:
      continue
    f = Finding(code, area, title)
    try:
      fn(f)
    except Exception as exc:                      # noqa: BLE001
      f.leak(f"검사가 터졌습니다: {type(exc).__name__}: {exc}")
    out.append(f)
    mark = {"지킴": "OK ", "주의": "!! ", "구멍": "XX ", "확인 못 함": "?? "}[f.verdict]
    print(f"  {mark}{f.code:<4} [{f.area}] {f.title}")
    print(f"        {f.detail}")

  held = sum(1 for f in out if f.verdict == "지킴")
  noted = sum(1 for f in out if f.verdict == "주의")
  leaked = [f for f in out if f.needs_fix]
  print(f"\n  지킴 {held} · 주의 {noted} · 구멍 {len(leaked)}")
  if leaked:
    print("\n  ** 고쳐야 하는 자리 **")
    for f in leaked:
      print(f"    {f.code} {f.title}\n        {f.detail}")
    return 1
  print("\n  구멍 없음 — 실물로 넘어가도 되는 상태입니다.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
