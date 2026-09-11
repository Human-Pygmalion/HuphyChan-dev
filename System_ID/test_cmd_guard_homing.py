"""영자세 복귀가 실제로 슬라이더보다 먼저 오는지 시험한다.

사용자 지시 (2026-09-12): *"이제 어떤 하드웨어 명령을 보내더라도, 먼저 이 영점
위치로 복귀하고, 그 다음 관절을 움직일 수 있도록 한다."*

이 시험은 **모터 없이** 돈다 — `frame_for` 에 측정 각도를 우리가 직접 넣어 주고,
돌려주는 바이트를 되풀어 무엇을 명령했는지 본다.

    python3 -m pytest System_ID/test_cmd_guard_homing.py -q
"""

import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cmd_guard as cg


JOINT = "L_hip_yaw_joint"
MOTOR = cg.ALLOWED[JOINT]["motor"]
ENC = cg.ALLOWED[JOINT]["enc"]


def decode_pos(frame: bytes) -> float:
  """`mit_frame` 이 실은 바이트에서 목표 각도를 되풀어 낸다.

  `_f2u` 가 라디안을 16비트로 눌러 담으므로 되푸는 쪽도 라디안에서 도로 돌린다.
  한 눈금이 2*12.57/65535 라디안 = 0.022도라, 비교는 그보다 성기게 한다.
  """
  raw = (frame[0] << 8) | frame[1]
  lo, hi = -ENC["pmax"], ENC["pmax"]
  return math.degrees(lo + (hi - lo) * raw / 65535.0)


def arm(guard, deg_now, *, target=0.0):
  """켜기 프레임까지 내보낸다. 명령이 신선해야 힘이 들어가므로 목표를 먼저 준다."""
  guard.local_target(JOINT, target, "")
  return guard.frame_for(MOTOR, deg_now, 30.0, 0.02)


def make_guard(**kw):
  opts = dict(allow=[JOINT], arm_token="", dry_run=False,
              max_vel_deg_s=20.0, deadman_s=0.3, kp=10.0, kd=1.0)
  opts.update(kw)
  return cg.CommandGuard(**opts)


def drive(guard, *, deg_now, targets_deg=None, cycles=200, dt=0.02):
  """명령을 계속 주면서 주기를 돌린다. 마지막으로 명령한 각도를 돌려준다.

  `deg_now` 는 관절이 명령을 **그대로 따라간다**고 보고 갱신한다 — 이 시험이 보는
  것은 가드가 무엇을 명령하는가이지 모터의 동특성이 아니다.
  """
  last = deg_now
  for _ in range(cycles):
    if targets_deg is not None:
      guard.local_target(JOINT, targets_deg, "")
    else:
      guard.last_rx[MOTOR] = time.monotonic()      # 손을 올려둔 것과 같게
    out = guard.frame_for(MOTOR, last, 30.0, dt)
    if isinstance(out, tuple) or out is None:
      continue
    last = decode_pos(out)
  return last


class TestHomesBeforeItListens:
  def test_arming_goes_to_zero_not_to_the_slider(self):
    """30도에 있는 관절에 +40도를 달라고 해도, 먼저 0도로 간다."""
    g = make_guard(span_deg=50.0)
    g.local_target(JOINT, 40.0, "")
    arm(g, 30.0, target=40.0)                     # 켜기 프레임
    assert g.homing[MOTOR] is True

    # 한 주기만 돌려 본다 — 0 쪽으로 가야 한다.
    g.local_target(JOINT, 40.0, "")
    frame = g.frame_for(MOTOR, 30.0, 30.0, 0.02)
    assert decode_pos(frame) < 30.0

  def test_it_actually_arrives_at_zero(self):
    """+40도를 계속 조르는 동안에도 **먼저 0도를 찍고** 나서야 40도로 간다."""
    g = make_guard(span_deg=50.0)
    arm(g, 30.0)
    at, where_it_stopped_homing = 30.0, None
    for _ in range(600):
      g.local_target(JOINT, 40.0, "")
      out = g.frame_for(MOTOR, at, 30.0, 0.02)
      if isinstance(out, tuple) or out is None:
        continue
      at = decode_pos(out)
      if where_it_stopped_homing is None and not g.homing[MOTOR]:
        where_it_stopped_homing = at
    assert where_it_stopped_homing == pytest.approx(0.0, abs=0.1)
    assert at == pytest.approx(40.0, abs=0.1)     # 그 뒤에야 슬라이더로 감

  def test_the_slider_works_after_it_arrives(self):
    g = make_guard(span_deg=50.0)
    arm(g, 30.0)
    drive(g, deg_now=30.0, targets_deg=0.0, cycles=300)
    assert g.homing[MOTOR] is False
    end = drive(g, deg_now=0.0, targets_deg=12.0, cycles=300)
    assert end == pytest.approx(12.0, abs=0.1)

  def test_the_window_is_anchored_at_zero_not_where_it_was_switched_on(self):
    """이전 결함: 켠 자리를 기준으로 구간이 잡혀 켤 때마다 구간이 걸어갔다."""
    g = make_guard(span_deg=10.0)
    arm(g, 30.0)
    assert g.q0[MOTOR] == pytest.approx(0.0)      # 켠 자리(30도)가 아니라 0

  def test_turning_it_off_and_on_does_not_ratchet(self):
    """껐다 켜기를 되풀이해도 갈 수 있는 구간이 밀려나지 않는다."""
    g = make_guard(span_deg=10.0)
    reached = []
    at = 25.0
    for _ in range(3):
      arm(g, at)
      at = drive(g, deg_now=at, targets_deg=60.0, cycles=400)
      reached.append(at)
      g.last_rx[MOTOR] = 0.0                       # 손을 뗌
      g.frame_for(MOTOR, at, 30.0, 0.02)           # 끄기 프레임
    # 매번 같은 자리에서 멈춰야 한다 (0 + span).
    assert reached == pytest.approx([10.0, 10.0, 10.0], abs=0.2)


class TestHardStopMargin:
  def test_a_joint_whose_zero_is_the_hard_stop_stops_short(self):
    """무릎처럼 0도가 곧 하드스톱이면 여유만큼 안쪽을 영자세로 삼는다."""
    g = make_guard(span_deg=50.0, limit_by={JOINT: (0.0, 120.0)},
                   home_margin_deg=3.0)
    assert g._home_target(MOTOR) == pytest.approx(3.0)
    arm(g, 40.0)
    end = drive(g, deg_now=40.0, targets_deg=0.0, cycles=400)
    assert end == pytest.approx(3.0, abs=0.1)

  def test_the_slider_cannot_reach_the_hard_stop_either(self):
    """영자세에 도착한 **뒤에도** 슬라이더가 스톱 위로 갈 수 없어야 한다.

    2026-09-12 에 이 시험이 구멍을 잡아냈다 -- 그전에는 여유가 영자세 복귀에만
    걸려 있어서, 도착한 뒤 0도를 달라고 하면 그대로 하드스톱까지 갔다.
    """
    # 상대 구간을 넓게 둬서 **고정 한계가 자르는 것**만 보이게 한다.
    g = make_guard(span_deg=200.0, limit_by={JOINT: (0.0, 120.0)},
                   home_margin_deg=3.0)
    arm(g, 40.0)
    drive(g, deg_now=40.0, targets_deg=0.0, cycles=800)
    assert g.homing[MOTOR] is False
    end = drive(g, deg_now=3.0, targets_deg=-20.0, cycles=400)
    assert end == pytest.approx(3.0, abs=0.1)
    end = drive(g, deg_now=3.0, targets_deg=999.0, cycles=2000)
    assert end == pytest.approx(117.0, abs=0.1)

  def test_the_other_sign_stops_short_too(self):
    g = make_guard(span_deg=50.0, limit_by={JOINT: (-120.0, 0.0)},
                   home_margin_deg=3.0)
    assert g._home_target(MOTOR) == pytest.approx(-3.0)

  def test_a_span_too_narrow_for_a_margin_goes_to_the_middle(self):
    g = make_guard(span_deg=50.0, limit_by={JOINT: (10.0, 14.0)},
                   home_margin_deg=3.0)
    assert g._home_target(MOTOR) == pytest.approx(12.0)


class TestItStillObeysTheOtherGuards:
  def test_the_return_obeys_the_speed_limit(self):
    """영자세로 돌아가는 것도 속도 제한을 그대로 받는다."""
    g = make_guard(span_deg=90.0, max_vel_deg_s=20.0)
    arm(g, 80.0)
    g.local_target(JOINT, 0.0, "")
    frame = g.frame_for(MOTOR, 80.0, 30.0, 0.02)
    # 한 주기(0.02초) 에 20도/초 = 0.4도까지만
    assert decode_pos(frame) == pytest.approx(80.0 - 0.4, abs=0.05)

  def test_letting_go_during_the_return_cuts_torque(self):
    """영자세로 가는 도중에 손을 떼도 데드맨이 그대로 듣는다."""
    g = make_guard(span_deg=50.0)
    arm(g, 30.0)
    assert g.torque_on[MOTOR] is True
    g.last_rx[MOTOR] = 0.0                         # 0.3초보다 오래됨
    g.frame_for(MOTOR, 30.0, 30.0, 0.02)
    assert g.torque_on[MOTOR] is False
    assert MOTOR not in g.homing

  def test_too_hot_during_the_return_cuts_torque(self):
    g = make_guard(span_deg=50.0, max_temp_c=50.0)
    arm(g, 30.0)
    g.last_rx[MOTOR] = time.monotonic()
    g.frame_for(MOTOR, 30.0, 55.0, 0.02)
    assert g.torque_on[MOTOR] is False


class TestTheScreenShowsWhatTheGuardAllows:
  """슬라이더가 보여 주는 범위와 가드가 실제로 허락하는 범위가 같아야 한다.

  다르면 사람이 갈 수 없는 자리를 고르고, 왜 안 가는지 모른다.
  """

  @pytest.mark.parametrize("limit,span", [
    (None, 10.0), ((0.0, 120.0), 50.0), ((-120.0, 0.0), 50.0),
    ((0.0, 120.0), 200.0), ((-40.0, 40.0), 10.0),
  ])
  def test_the_slider_ends_are_reachable_and_nothing_beyond(self, limit, span):
    kw = dict(span_deg=span, home_margin_deg=3.0)
    if limit is not None:
      kw["limit_by"] = {JOINT: limit}
    g = make_guard(**kw)
    arm(g, g._home_target(MOTOR))
    drive(g, deg_now=g._home_target(MOTOR), targets_deg=g._home_target(MOTOR),
          cycles=200)
    lo, hi = g._slider_span(MOTOR)
    at = g._home_target(MOTOR)
    assert drive(g, deg_now=at, targets_deg=hi, cycles=4000) == pytest.approx(hi, abs=0.1)
    assert drive(g, deg_now=hi, targets_deg=lo, cycles=8000) == pytest.approx(lo, abs=0.1)
    # 끝 너머는 못 간다
    assert drive(g, deg_now=lo, targets_deg=lo - 30.0, cycles=2000) == pytest.approx(lo, abs=0.1)


class TestWhatTheScreenIsTold:
  def test_a_target_sent_while_homing_is_accepted_but_marked_waiting(self):
    g = make_guard(span_deg=50.0)
    arm(g, 30.0)
    reply = g.local_target(JOINT, 40.0, "")
    assert reply["ok"] is True and reply["homing"] is True

  def test_status_says_where_zero_is(self):
    g = make_guard(span_deg=50.0, limit_by={JOINT: (0.0, 120.0)})
    arm(g, 40.0)
    row = g.status()["motors"][str(MOTOR)]
    assert row["homing"] is True and row["home"] == pytest.approx(3.0)
