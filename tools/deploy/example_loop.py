#!/usr/bin/env python3
"""**로봇 쪽 제어 고리의 뼈대.** 모터·관성측정장치 부분만 바꿔 쓰십시오.

이 파일은 그대로 돌아갑니다 — 모터와 관성측정장치 자리에 **가짜**가 들어 있어서,
배선과 주기와 단위를 로봇 없이 확인할 수 있습니다. 진짜로 쓸 때는 `FakeRobot` 을
여러분의 것으로 바꾸면 됩니다.

    python3 example_loop.py                 # 가짜로 5 초
    python3 example_loop.py --seconds 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from policy_runner import StandingPolicy

CONTRACT = "deploy_contract.json"
ONNX = "policy.onnx"


class FakeRobot:
  """진짜 로봇 자리. **여기만 바꾸면 됩니다.**

  진짜 구현이 지켜야 할 것:
  * `gyro` 는 몸통 기준 각속도 3 개 [rad/s]
  * `gravity` 는 몸통 기준 **중력 방향** 단위벡터. 서 있으면 (0, 0, −1).
    관성측정장치가 위 방향을 주면 **부호를 뒤집어서** 돌려주십시오.
  * `joint_pos` 는 이름을 키로 하는 사전 [rad]
  * `set_targets` 는 관절 목표각 [rad] 을 받습니다 (토크가 아닙니다)
  """

  def __init__(self, joint_names, default_pose):
    self.joint_names = list(joint_names)
    self.q = dict(default_pose)

  @property
  def gyro(self):
    return (0.0, 0.0, 0.0)

  @property
  def gravity(self):
    return (0.0, 0.0, -1.0)          # 서 있는 상태. 부호에 주의 (README §3 함정 1)

  @property
  def joint_pos(self):
    return dict(self.q)

  def set_targets(self, targets: dict) -> None:
    # 진짜 로봇에서는 여기서 모터에 위치 명령을 보냅니다.
    # 가짜에서는 "목표를 그대로 따라갔다" 고 칩니다.
    self.q.update(targets)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--seconds", type=float, default=5.0)
  ap.add_argument("--contract", default=CONTRACT)
  ap.add_argument("--onnx", default=ONNX)
  n = ap.parse_args()

  pol = StandingPolicy.from_files(n.contract, n.onnx)
  pol.self_test()                                  # ★ 모터 켜기 전에 반드시

  robot = FakeRobot(pol.joint_names, pol.default_pose)

  # 정책을 켜기 전에 기준자세로. 진짜 로봇에서는 천천히 보내십시오.
  robot.set_targets(pol.default_pose)

  pol.reset()
  dt = 1.0 / pol.adapter.control_rate_hz
  t_next = time.perf_counter()
  worst_late = 0.0

  for _ in range(int(n.seconds * pol.adapter.control_rate_hz)):
    targets = pol.inference(
      gyro=robot.gyro,
      projected_gravity=robot.gravity,
      joint_pos=robot.joint_pos,
      command=(0.0, 0.0, 0.0),                     # 제자리 서기
    )
    robot.set_targets(targets)

    t_next += dt
    late = time.perf_counter() - t_next
    worst_late = max(worst_late, late)
    if late < 0:
      time.sleep(-late)
    else:                                          # 뒤처지면 따라잡지 않고 시계를 맞춘다
      t_next = time.perf_counter()

  d = np.array([targets[j] - pol.default_pose[j] for j in pol.joint_names])
  print(f"[example] {pol.steps} 주기 · 가장 늦은 지연 {worst_late * 1e3:+.2f} ms "
        f"(주기 {dt * 1e3:.0f} ms)")
  print(f"[example] 마지막 목표각, 기준자세 대비 최대 {np.degrees(np.abs(d)).max():.2f}°")
  print("  (가짜 로봇이라 물리적으로 옳은 값은 아닙니다 — 배선·단위·주기만 봅니다)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
