#!/usr/bin/env python3
"""가짜 로봇·odometry·mapper 로 배선·단위·주기를 확인하는 50 Hz 제어 루프 예제.

  python3 example_loop.py [--seconds 20] [--map world_grid|world_fn|robot_grid|robot_fn]

실제 로봇에서는 세 클래스만 바꾸면 됩니다.

    FakeRobot      IMU (자이로, 중력 방향), 엔코더, 모터 목표 보내기
    FakeOdometry   odometry estimator: 골반 선속도 (골반 좌표계), 월드 좌표 골반 위치·자세
    FakeMapper     mapper: 지면 높이 (height_scan.py 의 네 형태 중 하나)

속도 명령은 **로봇 쪽에서 정하는 값**입니다 (조이스틱, UI 등). 이 예제는 서 있다가 0.3 m/s 로 천천히 가속합니다.
가짜 로봇은 움직이지 않으므로 배선 확인용입니다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from height_scan import RobotGrid, RobotHeightFunction, WorldGrid
from policy_runner import RoughPolicy

CMD_RATE = 0.5  # 속도 명령 변화 제한 [m/s^2] (실기 권장, README 4 절)


class FakeRobot:
  """기본자세로 평지에 가만히 서 있는 척만 하는 로봇. 실제 드라이버로 바꿀 부분."""

  def __init__(self, default_pose: dict):
    self.q = dict(default_pose)

  def read_imu(self):
    return np.zeros(3), np.array([0.0, 0.0, -1.0])        # 자이로 [rad/s], 중력 방향 단위벡터 (IMU 좌표계)

  def read_encoders(self) -> dict:
    return dict(self.q)                                    # 모터 이름 → 엔코더각 [rad] (크랭크는 크랭크 모터 각)

  def send_targets(self, targets: dict) -> None:
    pass                                                   # 여기서 각 모터 드라이버에 위치 목표를 보낸다


class FakeOdometry:
  """odometry estimator 자리. 실제로는 IMU + 다리 kinematics (+ LiDAR) 추정값."""

  def __init__(self, pelvis_height: float):
    self.pos = np.array([0.0, 0.0, pelvis_height])         # 월드 좌표 골반(base_link) 원점 [m], mapper 지도와 같은 좌표계
    self.quat = np.array([1.0, 0.0, 0.0, 0.0])             # 골반 → 월드 (w, x, y, z)
    self.v_body = np.zeros(3)                              # 골반 선속도 [m/s], 골반 좌표계


class FakeMapper:
  """mapper 자리: 평지 (지면 높이 0) 를 네 가지 형태로 낸다. 실제 mapper 는 이 중 편한 형태 하나로 주면 된다."""

  def __init__(self, kind: str):
    self.kind = kind
    if kind == "world_grid":                               # 월드 좌표 4 cm 지도 10 × 10 m, 값 = 지면 높이
      self.grid = WorldGrid(np.zeros((251, 251)), origin_xy=(-5.0, -5.0), resolution=0.04)

  def height(self, odom: FakeOdometry):
    if self.kind == "world_grid":
      return self.grid                                     # 새 지도가 오면 self.grid.update(z, origin, res)
    if self.kind == "world_fn":
      return lambda xy: np.zeros(len(xy))                  # 월드 좌표 점 (N, 2) → 지면 높이 (N,)
    if self.kind == "robot_grid":                          # 로봇 중심 33 × 21, 값 = z_pelvis − z_terrain
      return RobotGrid(np.full((33, 21), odom.pos[2]))    # 평지 (지면 z = 0) 이므로 z_pelvis − 0
    return RobotHeightFunction(lambda xy: np.full(len(xy), odom.pos[2]))   # 골반 yaw 좌표 점 → z_pelvis − z_terrain


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--map", default="world_grid", choices=["world_grid", "world_fn", "robot_grid", "robot_fn"])
  args = ap.parse_args()

  pol = RoughPolicy.from_package()
  pol.self_test()                          # 참조 입출력 대조 — 실패하면 모터를 켜지 말 것
  robot, odom, mapper = FakeRobot(pol.default_pose), FakeOdometry(pol.default_pose_height), FakeMapper(args.map)
  pol.reset()

  dt = 1.0 / 50.0
  cmd = np.zeros(3)
  t0 = next_t = time.monotonic()
  while (now := time.monotonic()) - t0 < args.seconds:
    want = np.array([0.3 if now - t0 > 5.0 else 0.0, 0.0, 0.0])          # 5 초 서 있다가 0.3 m/s
    cmd += np.clip(want - cmd, -CMD_RATE * dt, CMD_RATE * dt)
    gyro, grav = robot.read_imu()
    targets = pol.inference(gyro=gyro, projected_gravity=grav, joint_pos=robot.read_encoders(),
                            base_lin_vel=odom.v_body, base_pos=odom.pos, base_quat=odom.quat,
                            height=mapper.height(odom), command=cmd)
    robot.send_targets(targets)
    if pol.steps % 50 == 0:
      fh = pol.last_footholds
      print(f"t {now - t0:5.1f} s  명령 vx {cmd[0]:.2f}  R_knee 목표 {targets['R_knee_joint']:+.3f} rad  "
            f"착지점 R ({fh[0, 0]:+.2f}, {fh[0, 1]:+.2f}) L ({fh[1, 0]:+.2f}, {fh[1, 1]:+.2f}) m")
    next_t += dt
    time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
  main()
