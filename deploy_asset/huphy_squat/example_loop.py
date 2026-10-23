#!/usr/bin/env python3
"""가짜 로봇으로 배선·단위·주기를 확인하는 50 Hz 제어 루프 예제.

  python3 example_loop.py [--seconds 20] [--checkpoint 5000]

실제 로봇에서는 `FakeRobot` 의 세 함수(read_imu, read_encoders, send_targets)만 바꾸면 됩니다.
목표 높이 h* 는 **로봇 쪽에서 정하는 숫자**입니다 (센서로 재는 값이 아님). 이 예제는
기본자세 높이에서 시작해 0.75 m 까지 천천히 내려갔다가 다시 올라옵니다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from policy_runner import DEFAULT_CHECKPOINT, SquatPolicy

H_RATE = 0.10  # h* 변화 속도 제한 [m/s] (권장값, 아래 README 참고)


class FakeRobot:
  """기본자세로 가만히 서 있는 척만 하는 로봇. 실제 드라이버로 바꿀 부분."""

  def __init__(self, default_pose: dict):
    self.q = dict(default_pose)

  def read_imu(self):
    return np.zeros(3), np.array([0.0, 0.0, -1.0])  # 자이로 [rad/s], 중력 방향 단위벡터 (IMU 좌표계)

  def read_encoders(self) -> dict:
    return dict(self.q)  # 모터 이름 → 엔코더각 [rad] (크랭크는 크랭크 모터 각)

  def send_targets(self, targets: dict) -> None:
    pass  # 여기서 각 모터 드라이버에 위치 목표를 보낸다 (PD 는 드라이버가 함)


def h_schedule(t: float, h0: float) -> float:
  """원하는 높이 (예: 조이스틱/UI 에서 오는 값). 여기서는 10 초마다 h0 ↔ 0.75 m."""
  return h0 if int(t // 10) % 2 == 0 else 0.75


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
  args = ap.parse_args()

  pol = SquatPolicy.from_package(checkpoint=args.checkpoint)
  pol.self_test()                     # 참조 입출력 대조 — 실패하면 모터를 켜지 말 것
  robot = FakeRobot(pol.default_pose)
  h0 = pol.default_pose_height        # 기본자세 높이 ≈ 0.891 m 에서 시작
  h_cmd = h0
  pol.reset()

  dt = 1.0 / 50.0
  t0 = next_t = time.monotonic()
  while (now := time.monotonic()) - t0 < args.seconds:
    h_want = h_schedule(now - t0, h0)
    h_cmd += float(np.clip(h_want - h_cmd, -H_RATE * dt, H_RATE * dt))   # 천천히 따라가게
    gyro, grav = robot.read_imu()
    targets = pol.inference(gyro=gyro, projected_gravity=grav, joint_pos=robot.read_encoders(), h_star=h_cmd)
    robot.send_targets(targets)
    if pol.steps % 50 == 0:
      print(f"t {now - t0:5.1f} s  h* {h_cmd:.3f} m  R_knee 목표 {targets['R_knee_joint']:+.3f} rad")
    next_t += dt
    time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
  main()
