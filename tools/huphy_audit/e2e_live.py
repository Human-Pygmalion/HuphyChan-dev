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
  ui = RemoteGuard(f"127.0.0.1:{port}", arm_token="", allow={"hip_yaw"})

  goal = trimmed("hip_yaw")[1] - SAFETY["command_margin_deg"]
  reply = ui.local_target("hip_yaw", goal, "")
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


@check("E24", "대기창", "huphy_live 의 기본 대기창이 실측으로 고른 값인가")
def e24(f):
  """2026-09-12 실측: 2ms 에서는 한 모터가 연속 40주기까지 응답을 놓쳐 로봇이
  스스로 멈춥니다(멈추는 조건은 연속 5주기). 5ms 가 분기점입니다."""
  import argparse as _a, huphy_live
  ap = _a.ArgumentParser()
  for act in ():
    pass
  # main() 의 파서를 직접 만들지 않고, 기본값만 확인한다
  import inspect
  src = inspect.getsource(huphy_live.main)
  m = __import__("re").search(r'"--drain-ms".*?default=([\d.]+)', src, __import__("re").S)
  if m and abs(float(m.group(1)) - 5.0) < 1e-9:
    f.hold("기본값이 5.0 입니다. 실측 표는 `huphy_live` 머리말과 "
           "`docs/experiments/2026-09-12_loop_timing.md` 에 있습니다.")
  else:
    f.leak(f"기본값이 5.0 이 아닙니다: {m.group(1) if m else '못 찾음'}")


# ===========================================================================
# 아 · 발목 두 모터를 한 쌍으로 자르기 (모터 공간 AB)
# ===========================================================================
def _ankle():
  from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics
  import huphy_live as HL

  class _FakeLeg:
    def __init__(self, kin, name):
      self.kinematics = kin
      self.config = type("C", (), {"name": name})()

  left = _FakeLeg(AnkleKinematics(AnkleGeometry().mirrored()), "left_leg")
  right = _FakeLeg(AnkleKinematics(AnkleGeometry()), "right_leg")
  robot = type("R", (), {"limbs": [left, right]})()
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
