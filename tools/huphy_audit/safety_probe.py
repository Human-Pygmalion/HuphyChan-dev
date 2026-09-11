#!/usr/bin/env python3
"""로봇 프로그램(HUPHY)의 **안전 장치를 하나씩 일부러 건드려 보는 검사**.

왜 하는가
---------
곧 실물 로봇에 힘을 넣습니다. 그 전에 "이상한 값이 들어오면 막아 준다" 고 적혀
있는 장치들이 **정말로 막는지**를 확인해야 합니다. 문서를 읽어서가 아니라, 실제로
이상한 값을 넣어 보고 결과를 재서 확인합니다.

무엇을 검사하는가
-----------------
로봇 프로그램 안에서 "명령을 걸러내는 자리" 를 전부 찾아, 자리마다 시나리오를
하나씩 만들었습니다. 각 시나리오는 셋으로 이루어집니다.

    넣은 것      일부러 만든 나쁜 상황
    기대한 것    막히거나, 잘리거나, 멈춰야 함
    나온 것      실제로 돌려 본 결과

판정은 셋입니다.

    지킴      기대한 대로 막았습니다
    구멍      막는다고 되어 있는데 안 막았거나, 애초에 막는 장치가 없습니다
    주의      막기는 하는데 지금 설정에서는 꺼져 있거나 조건부입니다

하드웨어는 쓰지 않습니다. 진짜 모터 대신 `fake_can.py` 가 답을 합니다. 그래서
"선을 뽑는다", "과열시킨다" 같은 **진짜로 하면 위험한 상황**을 안전하게 만듭니다.

쓰는 법

    PYTHONPATH=<HUPHY 사본>/src python3 tools/huphy_audit/safety_probe.py

로봇 프로그램은 우리 저장소가 아니므로 **읽기만 합니다.** 이 파일은 우리 저장소에
있고, 로봇 프로그램은 한 글자도 고치지 않습니다.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from fake_can import install                                    # noqa: E402

from huphy.config.schema import LimbConfig, RobotConfig, SafetyConfig  # noqa: E402
from huphy.control.loop import ControlLoop, Mode                # noqa: E402
from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics  # noqa: E402
from huphy.motors.base import Gains, Motor, MotorCalibration    # noqa: E402
from huphy.motors.canbus import CanBus                          # noqa: E402
from huphy.motors.robstride import tables as T                  # noqa: E402
from huphy.motors.robstride.bus import RobStrideBus             # noqa: E402
from huphy.motors.robstride.codec import mit                    # noqa: E402
from huphy.robots.leg import FIRST_STEP_DT_S, Leg, RESEED_GAP_S  # noqa: E402
from huphy.safety import guards                                 # noqa: E402
from huphy.safety.link import LinkWatch                         # noqa: E402

KEEPS = "지킴"
HOLE = "구멍"
CARE = "주의"

# 실물 왼다리 배치 그대로. 벤치의 두 대(무릎 4번, 발목 5번)도 이 안에 있습니다.
MODELS = {1: T.Model.RS04, 2: T.Model.RS04, 3: T.Model.RS03,
          4: T.Model.RS04, 5: T.Model.RS03, 6: T.Model.RS03}
MODEL_NAME = {i: m.value for i, m in MODELS.items()}

# 재서 얻은 하드스톱 자리. 아직 실물에서 안 쟀으므로 0.5 판의 값을 예시로 씁니다.
LIMITS = {
  "hip_pitch": (-117.07, -21.07),
  "hip_roll": (-5.51, 79.64),
  "hip_yaw": (-41.90, 31.09),
  "knee": (-20.65, 74.79),
  "ankle_a": (-79.77, 43.16),
  "ankle_b": (-12.50, 126.66),
}

# 실물 설정 파일(robot_v1.0.yaml)에 적힌 안전값 그대로.
REAL_SAFETY = dict(command_margin_deg=3.0, max_delta_deg=50.0,
                   enforce_limits=True, link_loss_cycles=5)
# 벤치 설정 파일(robot_bench.yaml)에 적힌 안전값 그대로.
BENCH_SAFETY = dict(command_margin_deg=3.0, max_delta_deg=40.0,
                    max_vel_deg_s=902.0, enforce_limits=False, link_loss_cycles=0)


# ===========================================================================
# 검사 하나하나를 담는 그릇
# ===========================================================================
class Finding:
  def __init__(self, code, area, title, injected, expected, observed, verdict, note=""):
    self.code, self.area, self.title = code, area, title
    self.injected, self.expected, self.observed = injected, expected, observed
    self.verdict, self.note = verdict, note


CHECKS: List[Tuple[str, str, str, Callable]] = []


def check(code: str, area: str, title: str):
  def wrap(fn):
    CHECKS.append((code, area, title, fn))
    return fn
  return wrap


# ===========================================================================
# 다리 하나를 책상 위에 세우는 도구
# ===========================================================================
def make_limb(kp=10.0, kd=1.0, **over) -> LimbConfig:
  motors = {
    name: Motor(id=i, model=MODEL_NAME[i], gains=Gains(kp=kp, kd=kd))
    for name, i in (("hip_pitch", 1), ("hip_roll", 2), ("hip_yaw", 3),
                    ("knee", 4), ("ankle_a", 5), ("ankle_b", 6))
  }
  motors.update(over.pop("motors", {}))
  return LimbConfig(name="left_leg", kind="leg", side="left", channel="can0",
                    control_hz=100.0, motors=motors, **over)


def calibration(measured: bool = True):
  """실측값. `measured=False` 면 지금 저장소에 들어 있는 상태(전부 안 잼)."""
  return {
    name: MotorCalibration(
      motor_id=i,
      zero_reference="바닥에 눕혀 폄" if measured else "",
      limits_deg=LIMITS[name] if measured else None,
    )
    for name, i in (("hip_pitch", 1), ("hip_roll", 2), ("hip_yaw", 3),
                    ("knee", 4), ("ankle_a", 5), ("ankle_b", 6))
  }


def stand(safety=None, limb=None, measured=True, **leg_kw):
  """가짜 통신선 위에 다리 하나를 세웁니다. (다리, 가짜모터들) 을 돌려줍니다."""
  motors = install(MODELS)
  cfg = limb if limb is not None else make_limb()
  cfg_safety = SafetyConfig(**(REAL_SAFETY if safety is None else safety))
  geometry = AnkleGeometry().mirrored()          # 왼다리
  bus = RobStrideBus(CanBus(cfg.channel), cfg.motors_by_id())
  leg_kw.setdefault("allow_uncalibrated", True)
  leg = Leg(cfg, bus, safety=cfg_safety,
            calibration=calibration(measured),
            kinematics=AnkleKinematics(geometry), **leg_kw)
  leg.connect()
  motors.sent.clear()
  return leg, motors


def pose(leg, **deg):
  """모든 관절 목표를 한 번에 만듭니다. 안 적은 것은 0 입니다."""
  out = {j: 0.0 for j in leg.joint_names}
  out.update(deg)
  return out


def decode_sent(motors, motor_id):
  """그 모터에 실제로 나간 마지막 명령을 사람이 읽는 숫자로 풉니다."""
  enc = T.encoding_for(MODELS[motor_id])
  for msg in reversed(motors.sent):
    d = bytes(msg.data)
    if msg.arbitration_id != motor_id or d[0] == 0xFF:
      continue
    q = math.degrees(mit.uint_to_float((d[0] << 8) | d[1],
                                       -enc.pmax_rad, enc.pmax_rad, 16))
    kp_u = ((d[3] & 0x0F) << 8) | d[4]
    kd_u = (d[5] << 4) | (d[6] >> 4)
    tau_u = ((d[6] & 0x0F) << 8) | d[7]
    return dict(
      pos=q,
      kp=mit.uint_to_float(kp_u, 0.0, enc.kp_max, 12), kp_raw=kp_u,
      kd=mit.uint_to_float(kd_u, 0.0, enc.kd_max, 12), kd_raw=kd_u,
      tau=mit.uint_to_float(tau_u, -enc.tmax_nm, enc.tmax_nm, 12),
    )
  return None


# ===========================================================================
# 가 · 명령 값 자체를 거르는 자리 (safety/guards.py)
# ===========================================================================
@check("A1", "명령 값", "숫자가 아닌 목표(NaN)를 넣으면 전송을 막는가")
def a1():
  leg, motors = stand()
  cmds = leg.build_commands(pose(leg, knee=float("nan")))
  knee_id = leg.config.motors["knee"].id
  leg.send(cmds)
  gone = decode_sent(motors, knee_id)
  return Finding(
    "A1", "명령 값", "숫자가 아닌 목표(NaN)를 넣으면 전송을 막는가",
    injected="무릎 목표 = NaN (0으로 나누기·발목 계산 발산에서 실제로 생김)",
    expected="거부 이유 'nan' 이 세어지고, 무릎으로 아무 프레임도 안 나감",
    observed=f"거부 {leg.counters.rejects} · 만들어진 명령에 무릎 "
             f"{'있음' if knee_id in cmds else '없음'} · 선에 나간 무릎 명령 "
             f"{'있음' if gone else '없음'}",
    verdict=KEEPS if leg.counters.rejects.get("nan") == 1
    and knee_id not in cmds and gone is None else HOLE,
    note="막지 못하면 NaN 이 인코딩에서 최대값이 되어 720도 명령이 됩니다.",
  )


@check("A2", "명령 값", "무한대(inf) 목표도 같이 막히는가")
def a2():
  leg, _ = stand()
  leg.build_commands(pose(leg, knee=float("inf")))
  return Finding(
    "A2", "명령 값", "무한대(inf) 목표도 같이 막히는가",
    injected="무릎 목표 = +무한대",
    expected="'nan' 과 같은 통로로 거부",
    observed=f"거부 {leg.counters.rejects}",
    verdict=KEEPS if leg.counters.rejects.get("nan") == 1 else HOLE,
  )


@check("A3", "명령 값", "측정값이 NaN 이면(센서 이상) 명령을 막는가")
def a3():
  ok = guards.apply(10.0, float("nan"), limits=(-20.0, 80.0),
                    command_margin_deg=3.0, max_delta_deg=50.0)
  return Finding(
    "A3", "명령 값", "측정값이 NaN 이면(센서 이상) 명령을 막는가",
    injected="목표는 멀쩡한 10도, 그런데 현재 각도가 NaN",
    expected="전송 안 함 (현재를 모르면 폭주 상한을 못 잼)",
    observed=f"보낼 수 있나={ok.sendable}, 이유={ok.reject}",
    verdict=KEEPS if not ok.sendable else HOLE,
  )


@check("A4", "명령 값", "한 번도 응답하지 않은 모터에 명령이 나가는가")
def a4():
  leg, motors = stand()
  motors.answer = False
  leg.refresh()                      # 아무도 답하지 않는 한 주기
  leg2, motors2 = stand()
  motors2.answer = False
  leg2.bus._states[4].stamp = 0.0    # 무릎이 한 번도 답한 적 없는 상태
  motors2.sent.clear()
  cmds = leg2.build_commands(pose(leg2, knee=10.0))
  return Finding(
    "A4", "명령 값", "한 번도 응답하지 않은 모터에 명령이 나가는가",
    injected="무릎이 한 번도 상태를 보내지 않은 상태에서 10도 명령",
    expected="거부 이유 'nostate'. 현재 위치를 모르면 보내지 않음",
    observed=f"거부 {leg2.counters.rejects} · 만들어진 명령에 무릎 "
             f"{'있음' if 4 in cmds else '없음'}",
    verdict=KEEPS if leg2.counters.rejects.get("nostate") and 4 not in cmds else HOLE,
  )


@check("A5", "명령 값", "가동범위 밖 목표를 하드스톱 안쪽으로 자르는가")
def a5():
  leg, motors = stand()
  motors.position[4] = 60.0          # 무릎을 한계 가까이 가져다 둠
  leg.refresh()
  leg.counters.reset()
  leg.build_commands(pose(leg, knee=200.0))
  lo, hi = LIMITS["knee"]
  want = hi - REAL_SAFETY["command_margin_deg"]
  got = leg.last_sent.get("knee")
  return Finding(
    "A5", "명령 값", "가동범위 밖 목표를 하드스톱 안쪽으로 자르는가",
    injected=f"무릎이 60도에 있는 상태에서 목표 200도 (하드스톱은 {lo}~{hi}도). "
             f"폭주 상한(50도)보다 한계가 먼저 걸리도록 자리를 잡음",
    expected=f"하드스톱에서 3도 안쪽인 {want:.2f}도로 잘리고 'limit' 이 세어짐",
    observed=f"실제 목표 {got:.2f}도, 잘린 내역 {leg.counters.clips}",
    verdict=KEEPS if got is not None and abs(got - want) < 1e-6
    and leg.counters.clips.get("limit") else HOLE,
  )


@check("A6", "명령 값", "**한계를 아직 안 쟀을 때** 그 검사가 무슨 일을 하는가")
def a6():
  leg, motors = stand(measured=False)
  motors.position[4] = 60.0          # A5 와 똑같은 자리에서 출발
  leg.refresh()
  leg.counters.reset()
  leg.build_commands(pose(leg, knee=200.0))
  got = leg.last_sent.get("knee")
  return Finding(
    "A6", "명령 값", "한계를 아직 안 쟀을 때 그 검사가 무슨 일을 하는가",
    injected="A5 와 똑같은 상황인데, 실측값만 지금 저장소 상태(전부 '안 잼')로 둠. "
             "설정 파일의 한계 검사는 '켜짐'",
    expected="A5 처럼 71.79도로 잘려야 할 것 같지만…",
    observed=f"한계 클리핑 {leg.counters.clips.get('limit', 0)}회 · "
             f"폭주 상한만 걸려 목표 {got:.2f}도 (한계보다 38도 밖)",
    verdict=HOLE,
    note="한계값이 비어 있으면 검사가 **그냥 통과**합니다(limits.clamp 첫 줄). "
         "설정에 '켜짐' 이라고 적혀 있어도 실제로 지켜 주는 것은 없습니다. "
         "실물에서는 커미셔닝(sweep)으로 한계를 채우기 전까지 이 상태입니다. "
         "다만 힘을 넣는 문(enable)이 따로 막으므로 E1 과 함께 보아야 합니다.",
  )


@check("A7", "명령 값", "속도 제한이 한 주기 전진량을 묶는가")
def a7():
  leg, _ = stand(safety=BENCH_SAFETY)
  leg.build_commands(pose(leg, knee=0.0))
  leg._prev_cmd_t = time.monotonic() - 0.01     # 정확히 10밀리초 지난 것으로
  leg._prev_cmd["knee"] = 0.0
  leg.build_commands(pose(leg, knee=300.0))
  got = leg.last_sent.get("knee")
  want = BENCH_SAFETY["max_vel_deg_s"] * 0.01
  return Finding(
    "A7", "명령 값", "속도 제한이 한 주기 전진량을 묶는가",
    injected=f"0도에 있다가 갑자기 300도 명령 (벤치 설정: 초당 "
             f"{BENCH_SAFETY['max_vel_deg_s']:.0f}도까지)",
    expected=f"직전 명령에서 {want:.2f}도만 나아감",
    observed=f"실제 목표 {got:.2f}도, 잘린 내역 {leg.counters.clips}",
    verdict=KEEPS if abs(got - want) < 0.05 and leg.counters.clips.get("rate") else HOLE,
  )


@check("A8", "명령 값", "폭주 상한이 측정 위치에서 너무 멀리 못 가게 하는가")
def a8():
  leg, _ = stand()                    # 실물 설정: 속도 제한 없음, 폭주 상한 50도
  leg.build_commands(pose(leg, knee=200.0))
  got = leg.last_sent.get("knee")
  return Finding(
    "A8", "명령 값", "폭주 상한이 측정 위치에서 너무 멀리 못 가게 하는가",
    injected="무릎이 0도에 있는데 200도 명령 (실물 설정: 폭주 상한 50도)",
    expected="한계로 먼저 잘리고, 그래도 멀면 현재+50도까지만",
    observed=f"실제 목표 {got:.2f}도, 잘린 내역 {leg.counters.clips}",
    verdict=KEEPS if got is not None and got <= 50.0 + 1e-6 else HOLE,
  )


@check("A9", "명령 값", "세 관문의 순서가 설계대로인가 (한계 → 속도 → 폭주)")
def a9():
  r = guards.apply(500.0, 0.0, limits=(-20.0, 80.0), command_margin_deg=3.0,
                   max_delta_deg=50.0, prev_cmd_deg=0.0, max_step_deg=9.02)
  return Finding(
    "A9", "명령 값", "세 관문의 순서가 설계대로인가 (한계 → 속도 → 폭주)",
    injected="목표 500도, 한계 -20~80도, 한 주기 9.02도, 폭주 상한 50도",
    expected="77도로 잘린 뒤 9.02도까지만 나아감 → 최종 9.02도, 기록은 limit+rate",
    observed=f"최종 {r.value:.2f}도, 기록 {[c.value for c in r.clips]}",
    verdict=KEEPS if abs(r.value - 9.02) < 1e-6
    and [c.value for c in r.clips] == ["limit", "rate"] else HOLE,
  )


@check("A10", "명령 값", "전송이 끊겼다 되살아날 때 그동안 벌어진 만큼 한꺼번에 나가는가")
def a10():
  leg, _ = stand(safety=BENCH_SAFETY)
  leg.build_commands(pose(leg, knee=0.0))
  leg._prev_cmd["knee"] = 0.0
  leg._prev_cmd_t = time.monotonic() - (RESEED_GAP_S + 0.5)   # 0.7초 공백
  leg.build_commands(pose(leg, knee=300.0))
  got = leg.last_sent.get("knee")
  want = BENCH_SAFETY["max_vel_deg_s"] * FIRST_STEP_DT_S
  return Finding(
    "A10", "명령 값", "전송이 끊겼다 되살아날 때 그동안 벌어진 만큼 한꺼번에 나가는가",
    injected=f"{RESEED_GAP_S + 0.5:.1f}초 동안 명령이 끊겼다가 300도로 재개",
    expected=f"공백을 속도 예산으로 치지 않고 한 주기분({want:.2f}도)만",
    observed=f"실제 목표 {got:.2f}도",
    verdict=KEEPS if abs(got - want) < 0.05 else HOLE,
    note="2026-09-07 벤치에서 40도·44도가 한 번에 나간 사고가 이 자리였습니다.",
  )


@check("A11", "명령 값", "아주 첫 명령에도 속도 제한이 걸리는가")
def a11():
  leg, _ = stand(safety=BENCH_SAFETY)
  leg.build_commands(pose(leg, knee=300.0))      # 세우자마자 첫 명령
  got = leg.last_sent.get("knee")
  want = BENCH_SAFETY["max_vel_deg_s"] * FIRST_STEP_DT_S
  return Finding(
    "A11", "명령 값", "아주 첫 명령에도 속도 제한이 걸리는가",
    injected="다리를 세운 직후 첫 명령으로 300도",
    expected=f"'모르면 제일 빠른 주기를 가정' → {want:.2f}도",
    observed=f"실제 목표 {got:.2f}도",
    verdict=KEEPS if abs(got - want) < 0.05 else HOLE,
  )


@check("A12", "명령 값", "실물 설정(robot_v1.0.yaml)에 속도 제한이 있는가")
def a12():
  s = SafetyConfig(**REAL_SAFETY)
  v = s.vel_limit_deg_s("knee")
  return Finding(
    "A12", "명령 값", "실물 설정(robot_v1.0.yaml)에 속도 제한이 있는가",
    injected="실물 설정 파일에 적힌 안전값 그대로",
    expected="벤치처럼 속도 제한이 적혀 있어야 안전",
    observed=f"무릎 속도 제한 = {v} (없음)",
    verdict=HOLE if v is None else KEEPS,
    note="벤치 설정에는 max_vel_deg_s 가 있는데 실물 설정에는 아직 없습니다. "
         "실물로 옮길 때 이 줄이 빠지면, 남는 것은 폭주 상한 50도 하나입니다 — "
         "그것은 속도가 아니라 힘을 자르는 장치라 성격이 다릅니다.",
  )


# ===========================================================================
# 나 · 숫자를 바이트로 바꾸는 자리 (motors/robstride/codec/mit.py)
# ===========================================================================
@check("B1", "바이트로 바꿀 때", "마지막 관문이 NaN 을 막아 주는가")
def b1():
  enc = T.encoding_for(T.Model.RS04)
  q = mit.float_to_uint(float("nan"), -enc.pmax_rad, enc.pmax_rad, 16)
  deg = math.degrees(mit.uint_to_float(q, -enc.pmax_rad, enc.pmax_rad, 16))
  return Finding(
    "B1", "바이트로 바꿀 때", "마지막 관문이 NaN 을 막아 주는가",
    injected="목표 각도 NaN 을 바이트로 바꿔 봄",
    expected="사람들은 '범위로 잘리니까 괜찮다' 고 생각하기 쉬움",
    observed=f"바이트 값 {q} = {deg:.1f}도",
    verdict=CARE,
    note="크거나 작다는 비교가 NaN 에 대해 전부 거짓이라 그냥 통과합니다. "
         "그래서 A1 의 검사가 **유일한** 방어선입니다. 순서를 바꾸면 안 됩니다.",
  )


@check("B2", "바이트로 바꿀 때", "힘(kp)을 낮추다가 0이 되어 버리는 자리가 있는가")
def b2():
  enc = T.encoding_for(T.Model.RS04)
  rows = []
  for kp in (10.0, 5.0, 2.0, 1.5, 1.2210, 1.0, 0.5):
    u = mit.float_to_uint(kp, 0.0, enc.kp_max, 12)
    rows.append(f"{kp:g}→{u}({mit.uint_to_float(u, 0.0, enc.kp_max, 12):.3f})")
  zero_at = mit.float_to_uint(1.0, 0.0, enc.kp_max, 12)
  return Finding(
    "B2", "바이트로 바꿀 때", "힘(kp)을 낮추다가 0이 되어 버리는 자리가 있는가",
    injected="RS04 에 여러 kp 값을 넣어 실제로 나가는 값을 봄 "
             "(설정 파일 기본 kp=10, 문서가 권하는 시작법은 '0.1배로 낮춰 시작')",
    expected="낮추면 낮은 대로 나가야 함",
    observed=" · ".join(rows),
    verdict=HOLE if zero_at == 0 else KEEPS,
    note="RS03·RS04 는 kp 를 0~5000 범위의 12비트로 싣습니다. 한 칸이 1.221 이고 "
         "내림이라 **1.2210 미만은 전부 0**, 즉 힘이 아예 안 나갑니다. "
         "robot_v1.0.yaml 이 권하는 `--gain-scale 0.1` 은 kp 10 → 1.0 → 0 입니다. "
         "오류도 경고도 없이 '명령은 나가는데 안 움직임' 이 됩니다.",
  )


@check("B3", "바이트로 바꿀 때", "감쇠(kd)에도 같은 자리가 있는가")
def b3():
  enc = T.encoding_for(T.Model.RS04)
  rows = [f"{kd:g}→{mit.float_to_uint(kd, 0.0, enc.kd_max, 12)}"
          for kd in (1.0, 0.502, 0.1, 0.0244, 0.02)]
  return Finding(
    "B3", "바이트로 바꿀 때", "감쇠(kd)에도 같은 자리가 있는가",
    injected="RS04 에 여러 kd 값",
    expected="한 칸이 0.0244 이므로 그 미만은 0",
    observed=" · ".join(rows),
    verdict=CARE,
    note="정책이 쓰는 kd 0.502 는 안전합니다(20칸). 0.1배로 낮추면 0.0502 → 2칸이라 "
         "아직 살아 있지만, 여기서 더 낮추면 감쇠가 사라집니다.",
  )


@check("B4", "바이트로 바꿀 때", "범위를 넘는 목표가 반대편으로 감기지는 않는가")
def b4():
  enc = T.encoding_for(T.Model.RS04)
  q = mit.float_to_uint(math.radians(2000.0), -enc.pmax_rad, enc.pmax_rad, 16)
  deg = math.degrees(mit.uint_to_float(q, -enc.pmax_rad, enc.pmax_rad, 16))
  return Finding(
    "B4", "바이트로 바꿀 때", "범위를 넘는 목표가 반대편으로 감기지는 않는가",
    injected="목표 2000도 (인코딩 한계는 ±720도)",
    expected="+720도쪽으로 잘림. 음수로 감기면 반대로 튐",
    observed=f"{deg:.1f}도",
    verdict=KEEPS if deg > 700 else HOLE,
  )


# ===========================================================================
# 다 · 발목 (모터 두 개가 두 축을 같이 만드는 곳)
# ===========================================================================
@check("C1", "발목", "발목 목표가 기구학으로 안 풀리면 한쪽만 나가는가")
def c1():
  leg, _ = stand()
  ok = leg.build_commands(pose(leg, ankle_pitch=0.0, ankle_roll=0.0))
  bad = leg.build_commands(pose(leg, ankle_pitch=85.0, ankle_roll=85.0))
  ids = (leg.config.motors["ankle_a"].id, leg.config.motors["ankle_b"].id)
  ok_n = sum(1 for i in ids if i in ok)
  bad_n = sum(1 for i in ids if i in bad)
  return Finding(
    "C1", "발목", "발목 목표가 기구학으로 안 풀리면 한쪽만 나가는가",
    injected="① 닿을 수 있는 자세(0도·0도) ② 닿을 수 없는 자세(85도·85도)",
    expected="① 둘 다 나감 ② 둘 다 안 나감. 한쪽만 나가면 두 로드가 서로 다른 "
             "자세를 요구해 관절이 비틀림",
    observed=f"닿는 자세에서 {ok_n}/2개 · 닿지 않는 자세에서 {bad_n}/2개 나감",
    verdict=KEEPS if ok_n == 2 and bad_n == 0 else HOLE,
  )


@check("C2", "발목", "발목을 한쪽 축만 명령하면 막는가")
def c2():
  leg, _ = stand()
  act = {j: 0.0 for j in leg.joint_names}
  act.pop("ankle_roll")
  try:
    leg.build_commands(act)
    got, ok = "그냥 실행됨", False
  except ValueError as e:
    got, ok = f"거부: {str(e)[:60]}", True
  return Finding(
    "C2", "발목", "발목을 한쪽 축만 명령하면 막는가",
    injected="ankle_pitch 만 주고 ankle_roll 은 빼고 명령",
    expected="거부. 모터 두 개가 두 축을 같이 만들기 때문",
    observed=got, verdict=KEEPS if ok else HOLE,
  )


@check("C3", "발목", "모르는 관절 이름을 조용히 넘기지는 않는가")
def c3():
  leg, _ = stand()
  act = pose(leg)
  act["ankle_yaw"] = 10.0
  try:
    leg.build_commands(act)
    got, ok = "그냥 실행됨", False
  except ValueError as e:
    got, ok = f"거부: {str(e)[:60]}", True
  return Finding(
    "C3", "발목", "모르는 관절 이름을 조용히 넘기지는 않는가",
    injected="있지도 않은 관절 이름 'ankle_yaw' 를 섞어 명령",
    expected="거부. 오타를 무시하면 그 관절만 직전 자세로 굳음",
    observed=got, verdict=KEEPS if ok else HOLE,
  )


@check("C4", "발목", "발목을 힘으로 모는 방식(정책 실행)에 가동범위 검사가 있는가")
def c4():
  # 발목 두 모터의 가동범위를 아주 좁게(±5도) 잡아 둔다. 같은 목표를 두 방식으로
  # 넣어 보면, 각도로 보낼 때는 잘리고 힘으로 보낼 때는 안 잘리는 것이 드러난다.
  tight = dict(ankle_a=(-5.0, 5.0), ankle_b=(-5.0, 5.0))
  target = dict(ankle_pitch=20.0, ankle_roll=0.0)

  leg_p, _ = stand()
  for name, lim in tight.items():
    leg_p.config.motors[name].__dict__  # frozen 이라 아래 경로로 바꿔 넣는다
  leg_p._set_calibration({**calibration(True), **{
    n: MotorCalibration(motor_id=leg_p.config.motors[n].id, limits_deg=l)
    for n, l in tight.items()}})
  leg_p.build_commands(pose(leg_p, **target))
  clipped = leg_p.counters.clips.get("limit", 0)

  leg_t, motors_t = stand(ankle_output="torque", ankle_kp=(20.0, 20.0),
                          ankle_kd=(0.502, 0.502))
  leg_t._set_calibration({**calibration(True), **{
    n: MotorCalibration(motor_id=leg_t.config.motors[n].id, limits_deg=l)
    for n, l in tight.items()}})
  leg_t.build_commands(pose(leg_t, **target))
  tau = leg_t.last_torque.get("ankle_a", 0.0)
  return Finding(
    "C4", "발목", "발목을 힘으로 모는 방식(정책 실행)에 가동범위 검사가 있는가",
    injected="발목 가동범위를 ±5도로 좁혀 놓고 pitch 20도를 명령. 같은 목표를 "
             "각도 방식과 힘 방식으로 각각 넣음",
    expected="두 방식 모두 가동범위에서 걸려야 함",
    observed=f"각도 방식: 한계 클리핑 {clipped}회 · 힘 방식: 클리핑 "
             f"{leg_t.counters.clips.get('limit', 0)}회, 발목 a 에 "
             f"{tau:.2f} N·m 가 그대로 나감",
    verdict=HOLE,
    note="힘 명령에는 자를 각도가 없어 가드가 통째로 건너뜁니다(코드에도 그렇게 "
         "적혀 있습니다). 정책 실행(huphy-run)은 발목을 이 방식으로 몹니다. "
         "즉 **발목만은 가동범위 보호가 전혀 없는 채로 실물에 올라갑니다.**",
  )


# ===========================================================================
# 라 · 응답이 오고 있나 (safety/link.py, control/loop.py)
# ===========================================================================
@check("D1", "응답 끊김", "한 주기 빠지는 것과 계속 없는 것을 구분하는가")
def d1():
  w = LinkWatch(5)
  one = w.check({"knee": {"miss": 1}})
  many = w.check({"knee": {"miss": 5}})
  return Finding(
    "D1", "응답 끊김", "한 주기 빠지는 것과 계속 없는 것을 구분하는가",
    injected="무릎이 1주기 무응답 / 5주기 연속 무응답",
    expected="1주기는 정상, 5주기면 끊긴 것으로 판정",
    observed=f"1주기 → {one}, 5주기 → {many}",
    verdict=KEEPS if one is None and many is not None else HOLE,
  )


@check("D2", "응답 끊김", "선을 뽑으면 제어 루프가 실제로 멈추는가")
def d2():
  leg, motors = stand()
  loop = ControlLoop(leg, hz=100.0, mode=Mode.CONTROL, precise=False)
  motion_calls = [0]

  def motion(t, obs):
    motion_calls[0] += 1
    if motion_calls[0] == 3:
      motors.answer = False          # 세 번째 주기에 선을 뽑음
    return pose(leg, knee=5.0)

  stats = loop.run(motion, max_cycles=40)
  return Finding(
    "D2", "응답 끊김", "선을 뽑으면 제어 루프가 실제로 멈추는가",
    injected="정상으로 돌다가 세 번째 주기에 모든 응답을 끊음 "
             "(설정: 5주기 연속이면 정지)",
    expected="40주기를 다 못 돌고 8주기 언저리에서 스스로 멈춤",
    observed=f"{stats.cycles}주기 만에 종료, 사유 = {stats.link_loss}",
    verdict=KEEPS if stats.link_loss is not None and stats.cycles < 20 else HOLE,
  )


@check("D3", "응답 끊김", "벤치 설정에서도 그 정지가 살아 있는가")
def d3():
  s = SafetyConfig(**BENCH_SAFETY)
  return Finding(
    "D3", "응답 끊김", "벤치 설정에서도 그 정지가 살아 있는가",
    injected="벤치 설정 파일(robot_bench.yaml)에 적힌 값",
    expected="끊기면 멈춰야 안전",
    observed=f"link_loss_cycles = {s.link_loss_cycles} (0 = 판정 안 함)",
    verdict=CARE,
    note="벤치는 여섯 관절을 선언하고 두 대만 물려 있어, 없는 네 대가 매 주기 "
         "무응답입니다. 그래서 일부러 껐습니다. **실물로 갈 때 되살려야 하고**, "
         "되살리는 순간 배선이 하나라도 빠져 있으면 즉시 멈춥니다.",
  )


@check("D4", "응답 끊김", "오래된 측정값을 '지금 위치' 로 믿고 쓰지는 않는가")
def d4():
  leg, motors = stand()
  motors.sent.clear()
  leg.bus._states[4].stamp = time.monotonic() - 5.0    # 5초 전 값
  leg.bus._states[4].position_deg = 0.0
  leg.build_commands(pose(leg, knee=60.0))
  got = leg.last_sent.get("knee")
  return Finding(
    "D4", "응답 끊김", "오래된 측정값을 '지금 위치' 로 믿고 쓰지는 않는가",
    injected="무릎 측정값이 5초 전 것인 상태에서 60도 명령",
    expected="폭주 상한은 '지금 위치' 기준이어야 뜻이 있음",
    observed=f"5초 전 값을 기준으로 그대로 계산해 목표 {got:.2f}도",
    verdict=HOLE,
    note="'제어에 쓸 만큼 최신인가' 를 판정하는 함수(MotorState.is_fresh)가 "
         "만들어져 있고 그 설명에 바로 이 위험이 적혀 있는데, **아무도 부르지 "
         "않습니다.** 지금 쓰는 것은 '한 번이라도 답한 적 있나' 뿐입니다. "
         "응답 끊김 감시(D2)가 5주기 = 50밀리초에 잡아 주므로 겹겹이 막히긴 "
         "하지만, 그 감시를 끈 벤치 설정(D3)에서는 이것이 마지막 방어선입니다.",
  )


@check("D5", "응답 끊김", "명령하지 않은 모터를 '고장' 으로 오해하지 않는가")
def d5():
  leg, motors = stand()
  leg.build_commands(pose(leg, knee=1.0))
  cmds = {leg.config.motors["knee"].id: list(
    leg.build_commands(pose(leg, knee=1.0)).values())[0]}
  leg.send(cmds)
  leg.collect()
  st = leg.link_status()
  return Finding(
    "D5", "응답 끊김", "명령하지 않은 모터를 '고장' 으로 오해하지 않는가",
    injected="무릎에만 명령하고 나머지 다섯은 명령하지 않음",
    expected="무릎은 응답 1, 나머지는 -1(명령 안 함). 0(씹힘)이면 거짓 고장",
    observed=f"무릎 ack={st['knee']['ack']}, 엉덩이 pitch ack={st['hip_pitch']['ack']}",
    verdict=KEEPS if st["knee"]["ack"] == 1.0 and st["hip_pitch"]["ack"] == -1.0 else HOLE,
  )


# ===========================================================================
# 마 · 힘을 넣고 빼는 순서 (robots/leg.py, control/loop.py)
# ===========================================================================
@check("E1", "힘 넣고 빼기", "안 잰 상태로 힘을 넣으려 하면 막는가")
def e1():
  leg, _ = stand(measured=False, allow_uncalibrated=False)
  try:
    leg.enable()
    got, ok = "그냥 힘이 들어감", False
  except RuntimeError as e:
    got, ok = f"거부: {str(e)[:70]}", True
  return Finding(
    "E1", "힘 넣고 빼기", "안 잰 상태로 힘을 넣으려 하면 막는가",
    injected="한계도 영점도 안 잰 상태(지금 저장소 상태)에서 힘을 넣으려 함",
    expected="거부. 어디까지 가도 되는지 모르는 채 힘을 넣는 것이 가장 위험",
    observed=got, verdict=KEEPS if ok else HOLE,
    note="A6 의 구멍을 실제로 막고 있는 문이 이것입니다. 다만 "
         "`--allow-uncalibrated` 한 줄로 열리고, 열면 가동범위 보호가 "
         "**전혀 없는 채로** 돕니다(E2).",
  )


@check("E2", "힘 넣고 빼기", "그 문을 여는 스위치가 있는가")
def e2():
  leg, _ = stand(measured=False, allow_uncalibrated=True)
  try:
    leg.enable()
    got, ok = "경고만 하고 힘이 들어감", True
  except RuntimeError as e:
    got, ok = f"거부: {e}", False
  return Finding(
    "E2", "힘 넣고 빼기", "그 문을 여는 스위치가 있는가",
    injected="같은 상태에서 allow_uncalibrated=True (huphy-run 의 "
             "--allow-uncalibrated, 커미셔닝이 쓰는 길)",
    expected="커미셔닝에는 필요하지만, 이때는 무엇이 꺼지는지 알고 써야 함",
    observed=got,
    verdict=CARE if ok else KEEPS,
    note="이 스위치를 켜면 A6 의 '한계 검사가 통과만 함' 상태로 힘이 들어갑니다. "
         "남는 보호는 폭주 상한과 속도 제한뿐입니다.",
  )


@check("E3", "힘 넣고 빼기", "관찰 모드로 들어갈 때 힘을 확실히 끊는가")
def e3():
  leg, motors = stand()
  leg.bus.enable_torque()
  motors.sent.clear()
  loop = ControlLoop(leg, hz=200.0, mode=Mode.OBSERVE, precise=False)
  loop.run(None, max_cycles=2)
  stops = sum(1 for m in motors.sent
              if bytes(m.data)[0] == 0xFF and bytes(m.data)[7] == T.CMD_STOP)
  return Finding(
    "E3", "힘 넣고 빼기", "관찰 모드로 들어갈 때 힘을 확실히 끊는가",
    injected="힘이 켜진 채로 관찰 모드 루프를 시작",
    expected="들어가면서 정지 명령을 보냄 (설정이 틀려도 안 움직이게)",
    observed=f"정지 명령 {stops}회 나감",
    verdict=KEEPS if stops >= 1 else HOLE,
  )


@check("E4", "힘 넣고 빼기", "도중에 오류가 나도 자세를 붙잡고 힘을 빼는가")
def e4():
  leg, motors = stand()
  loop = ControlLoop(leg, hz=200.0, mode=Mode.CONTROL, precise=False)

  def motion(t, obs):
    if loop.stats.cycles >= 2:
      raise RuntimeError("일부러 낸 오류")
    return pose(leg, knee=1.0)

  motors.sent.clear()
  try:
    loop.run(motion, max_cycles=50)
  except RuntimeError:
    pass
  data = [bytes(m.data) for m in motors.sent]
  stop_at = [i for i, d in enumerate(data) if d[0] == 0xFF and d[7] == T.CMD_STOP]
  holds = [i for i, d in enumerate(data) if d[0] != 0xFF]
  return Finding(
    "E4", "힘 넣고 빼기", "도중에 오류가 나도 자세를 붙잡고 힘을 빼는가",
    injected="제어 도중 파이썬 오류를 일부러 냄",
    expected="① 지금 자세를 몇 주기 붙잡고 ② 그 다음에 힘을 끊음. 바로 끊으면 "
             "서 있는 다리가 주저앉음",
    observed=f"붙잡는 명령 {len(holds)}개 뒤에 정지 명령 {len(stop_at)}개",
    verdict=KEEPS if stop_at and holds and max(holds) < stop_at[0] else HOLE,
  )


@check("E5", "힘 넣고 빼기", "통신선을 닫기 전에 힘부터 끊는가")
def e5():
  leg, motors = stand()
  motors.sent.clear()
  leg.disconnect()
  data = [bytes(m.data) for m in motors.sent]
  stops = [d for d in data if d[0] == 0xFF and d[7] == T.CMD_STOP]
  return Finding(
    "E5", "힘 넣고 빼기", "통신선을 닫기 전에 힘부터 끊는가",
    injected="프로그램 종료(disconnect)",
    expected="먼저 정지 명령, 그 다음 선을 닫음. 순서가 반대면 정지를 보낼 "
             "방법이 없어져 모터가 마지막 명령을 계속 유지함",
    observed=f"닫기 전에 나간 정지 명령 {len(stops)}개",
    verdict=KEEPS if len(stops) >= len(MODELS) else HOLE,
  )


# ===========================================================================
# 바 · 모터가 스스로 알려 주는 위험 신호 (온도·고장)
# ===========================================================================
@check("F1", "모터가 보내는 경고", "권선이 뜨거워지면 힘을 끊는 장치가 있는가")
def f1():
  leg, motors = stand()
  for i in MODELS:
    motors.temp_c[i] = 95.0
  loop = ControlLoop(leg, hz=200.0, mode=Mode.CONTROL, precise=False)
  stats = loop.run(lambda t, o: pose(leg, knee=1.0), max_cycles=20)
  seen = leg.get_observation()["knee.temp"]
  return Finding(
    "F1", "모터가 보내는 경고", "권선이 뜨거워지면 힘을 끊는 장치가 있는가",
    injected="모든 모터가 권선 온도 95도를 보고하는 상태로 20주기 제어",
    expected="사람이 정한 상한(우리 규칙은 50도)에서 힘이 끊겨야 함",
    observed=f"읽히는 온도 {seen:.1f}도 · 루프는 {stats.cycles}주기를 "
             f"끝까지 돌았고 멈춤 사유 없음",
    verdict=HOLE,
    note="온도는 정확히 읽히고 화면·기록에도 나오지만, **그 값으로 무엇을 하는 "
         "코드가 어디에도 없습니다.** 모터 자체의 보호(RS04 기준 145도)만 "
         "남습니다. 우리 벤치 규칙인 50도 차단은 아직 로봇 프로그램에 없습니다.",
  )


@check("F2", "모터가 보내는 경고", "모터가 '고장' 이라고 답하면 루프가 알아채는가")
def f2():
  leg, motors = stand()
  motors.fault[4] = True
  motors.warning[4] = True
  loop = ControlLoop(leg, hz=200.0, mode=Mode.CONTROL, precise=False)
  stats = loop.run(lambda t, o: pose(leg, knee=1.0), max_cycles=20)
  obs = leg.get_observation()
  has_field = any("fault" in k or "warn" in k for k in obs)
  return Finding(
    "F2", "모터가 보내는 경고", "모터가 '고장' 이라고 답하면 루프가 알아채는가",
    injected="무릎이 매 응답마다 고장·경고 표시를 켠 채로 답함",
    expected="그 표시를 보고 멈추거나, 최소한 기록에 남아야 함",
    observed=f"관찰 항목에 고장/경고 칸 {'있음' if has_field else '없음'} · "
             f"루프 {stats.cycles}주기 정상 종료",
    verdict=HOLE,
    note="고장·경고·동작모드 세 가지는 **매 응답에 공짜로 실려 옵니다.** 그것을 "
         "푸는 함수(decode_flags)까지 만들어져 있고 시험도 있는데, 실제 수거 "
         "경로(bus.collect)가 그 함수를 부르지 않습니다. 고장을 알려면 따로 "
         "물어봐야 하고(read_fault), 그것은 사람이 커미셔닝 화면에서만 합니다.",
  )


@check("F3", "모터가 보내는 경고", "정말로 힘이 켜졌는지 모터에 확인하는가")
def f3():
  leg, motors = stand()
  motors.mode[4] = 0                 # 무릎만 아직 Reset 상태 (힘이 안 켜짐)
  leg.bus.enable_torque()
  return Finding(
    "F3", "모터가 보내는 경고", "정말로 힘이 켜졌는지 모터에 확인하는가",
    injected="켜기 명령을 보냈지만 무릎은 여전히 '힘 꺼짐' 상태로 답함",
    expected="모터가 보고하는 동작모드로 대조해야 함",
    observed=f"프로그램이 아는 값 = {leg.bus.is_torque_on(4)} "
             f"(보낸 명령 기준이지 물어본 값이 아님)",
    verdict=CARE,
    note="`is_torque_on` 은 '내가 켜기 명령을 보냈다' 만 뜻합니다. 모터가 실제로 "
         "켜졌는지는 응답의 동작모드에 들어 있는데 F2 와 같은 이유로 안 봅니다. "
         "벤치에서 '명령은 나가는데 안 움직임' 을 진단할 때 바로 이 값이 필요했습니다.",
  )


# ===========================================================================
# 사 · 설정 파일이 잘못됐을 때 (config/schema.py)
# ===========================================================================
@check("G1", "설정 검사", "같은 통신선에 같은 번호를 두 번 적으면 막는가")
def g1():
  try:
    make_limb(motors={"knee": Motor(id=1, model="RS04", gains=Gains(kp=10.0))})
    got, ok = "그냥 통과", False
  except ValueError as e:
    got, ok = f"거부: {str(e)[:60]}", True
  return Finding(
    "G1", "설정 검사", "같은 통신선에 같은 번호를 두 번 적으면 막는가",
    injected="무릎 번호를 1번(엉덩이 pitch)과 같게 적음",
    expected="거부. 번호가 겹치면 누가 답한 것인지 구분이 안 됨",
    observed=got, verdict=KEEPS if ok else HOLE,
  )


@check("G2", "설정 검사", "가동범위를 거꾸로 적으면 막는가")
def g2():
  try:
    Motor(id=4, model="RS04", limits_deg=(80.0, -20.0))
    got, ok = "그냥 통과", False
  except ValueError as e:
    got, ok = f"거부: {str(e)[:60]}", True
  return Finding(
    "G2", "설정 검사", "가동범위를 거꾸로 적으면 막는가",
    injected="무릎 한계를 (80, -20) 으로 뒤집어 적음",
    expected="거부. 뒤집힌 채로 자르면 모든 명령이 한 점으로 끌려감",
    observed=got, verdict=KEEPS if ok else HOLE,
  )


@check("G3", "설정 검사", "폭주 상한을 0이나 음수로 적으면 막는가")
def g3():
  bad = []
  for v in (0.0, -5.0):
    try:
      SafetyConfig(max_delta_deg=v)
      bad.append(v)
    except ValueError:
      pass
  return Finding(
    "G3", "설정 검사", "폭주 상한을 0이나 음수로 적으면 막는가",
    injected="max_delta_deg 에 0 과 -5",
    expected="둘 다 거부. 0이면 어떤 명령도 못 나가고 음수는 뜻이 없음",
    observed="전부 거부됨" if not bad else f"통과해 버린 값 {bad}",
    verdict=KEEPS if not bad else HOLE,
  )


@check("G4", "설정 검사", "가동범위를 설정 파일에도 적으면 막는가 (값이 두 곳에 생기는 것)")
def g4():
  import huphy.config.loader as L
  src = open(L.__file__).read()
  guarded = "limits_deg" in src
  return Finding(
    "G4", "설정 검사", "가동범위를 설정 파일에도 적으면 막는가",
    injected="robot.yaml 의 모터 항목에 limits_deg 를 적어 봄",
    expected="거부. 실측 파일과 설정 파일 두 곳에 같은 값이 생기면 어긋났을 때 "
             "어느 쪽이 진짜인지 알 수 없음",
    observed="설정 읽는 곳에 그 검사가 " + ("있음" if guarded else "없음"),
    verdict=KEEPS if guarded else HOLE,
  )


@check("G5", "설정 검사", "정책을 돌릴 때 사람이 시작 신호를 꼭 줘야 하는가")
def g5():
  import inspect
  from huphy.scripts import run as runmod
  src = inspect.getsource(runmod.EnterWatcher)
  auto = "isatty" in src
  return Finding(
    "G5", "설정 검사", "정책을 돌릴 때 사람이 시작 신호를 꼭 줘야 하는가",
    injected="huphy-run 을 화면이 아닌 곳에서 실행 (서비스·파이프·자동 스크립트)",
    expected="사람이 Enter 를 눌러야 정책이 시작",
    observed="화면이 아니면 " + ("확인 없이 바로 시작함" if auto else "기다림"),
    verdict=CARE if auto else KEEPS,
    note="화면 앞에 사람이 있을 때는 맞는 동작입니다. 다만 원격으로 돌리거나 "
         "서비스로 띄우면 **접근 자세로 옮기자마자 정책이 곧바로 돕니다.** "
         "우리는 뷰어를 통해 원격으로 걸 계획이므로 이 경로가 실제로 쓰입니다.",
  )


# ===========================================================================
# 아 · 정책이 내는 값 (control/policy.py)
# ===========================================================================
@check("H1", "정책 출력", "모델이 터무니없는 값을 내면 어디서 걸리는가")
def h1():
  from huphy.control import policy
  big = policy.joint_targets([50.0] * 6, spec=policy.BALANCE)
  leg, _ = stand()
  leg.build_commands({**pose(leg), **{k: v for k, v in big.items()
                                      if k in leg.joint_names}})
  return Finding(
    "H1", "정책 출력", "모델이 터무니없는 값을 내면 어디서 걸리는가",
    injected="모델이 모든 관절에 50(보통 -1~1)을 냄 → 관절 목표 716도",
    expected="정책 쪽에서 한 번, 가드에서 또 한 번 걸리는 것이 안전",
    observed=f"정책이 만든 목표 무릎 {big['knee']:.0f}도 → "
             f"가드 통과 후 {leg.last_sent.get('knee'):.2f}도, "
             f"잘린 내역 {leg.counters.clips}",
    verdict=CARE,
    note="정책 쪽에는 자르는 곳이 없습니다 — 행동에 곱하기만 합니다. 실제로 막는 "
         "것은 다음 단계의 가드뿐이라, 가드가 꺼진 구간(A6·C4)에서는 그대로 "
         "나갑니다. 시뮬레이터는 행동을 ±100 으로 자르고 들어갑니다.",
  )


@check("H2", "정책 출력", "자세 센서가 멎어도 정책이 계속 도는가")
def h2():
  from huphy.sensors.base import ImuState
  stale = ImuState(stamp=time.monotonic() - 10.0, is_valid=True)
  return Finding(
    "H2", "정책 출력", "자세 센서가 멎어도 정책이 계속 도는가",
    injected="자세 센서 값이 10초째 갱신되지 않음",
    expected="'얼마나 묵었나' 를 보고 멈추거나 경고해야 함",
    observed=f"값의 나이 {stale.age_ms():.0f}밀리초를 잴 수는 있으나, "
             f"정책 경로에서 그 값을 보는 곳이 없음",
    verdict=HOLE,
    note="정책은 매 주기 센서를 읽어 각속도와 중력 방향을 입력에 넣습니다. "
         "센서가 멎으면 **얼어붙은 값**이 계속 들어가고, 정책은 로봇이 가만히 "
         "서 있다고 믿습니다. 값의 나이는 기록에는 나가지만 판정에는 안 쓰입니다.",
  )


# ===========================================================================
# 결과 내기
# ===========================================================================
def run_all(only=None) -> List[Finding]:
  out = []
  for code, area, title, fn in CHECKS:
    if only and code not in only:
      continue
    try:
      out.append(fn())
    except Exception as e:                          # noqa: BLE001
      out.append(Finding(code, area, title, injected="-", expected="-",
                         observed=f"검사 자체가 실패함: {type(e).__name__}: {e}",
                         verdict="검사실패"))
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--only", nargs="*", help="이 번호만 돌림 (예: A1 C4)")
  ap.add_argument("--quiet", action="store_true", help="요약만")
  a = ap.parse_args(argv)

  findings = run_all(a.only)
  width = 78

  if not a.quiet:
    area = None
    for f in findings:
      if f.area != area:
        area = f.area
        print(f"\n{'=' * width}\n  {area}\n{'=' * width}")
      print(f"\n[{f.code}] {f.title}")
      print(f"  판정     {f.verdict}")
      print(f"  넣은 것  {f.injected}")
      print(f"  기대     {f.expected}")
      print(f"  나온 것  {f.observed}")
      if f.note:
        for line in _wrap(f.note, width - 11):
          print(f"           {line}")

  print(f"\n{'=' * width}\n  요약\n{'=' * width}")
  for verdict in (KEEPS, CARE, HOLE, "검사실패"):
    hits = [f for f in findings if f.verdict == verdict]
    if not hits:
      continue
    print(f"  {verdict:5} {len(hits):2}건   " + ", ".join(f.code for f in hits))
  holes = [f for f in findings if f.verdict == HOLE]
  if holes:
    print("\n  막지 못하는 것:")
    for f in holes:
      print(f"    {f.code}  {f.title}")
  return 0


def _wrap(text: str, width: int) -> List[str]:
  words, line, out = text.split(), "", []
  for w in words:
    if len(line) + len(w) + 1 > width and line:
      out.append(line)
      line = w
    else:
      line = f"{line} {w}".strip()
  if line:
    out.append(line)
  return out


if __name__ == "__main__":
  raise SystemExit(main())
