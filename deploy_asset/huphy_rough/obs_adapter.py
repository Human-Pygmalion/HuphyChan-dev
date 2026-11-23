#!/usr/bin/env python3
"""**로봇에서 읽은 값 → 정책 관측 앞부분 48 칸과 FPM 입력 36 칸.** numpy 와 표준 라이브러리만 쓴다.

`huphy_squat/obs_adapter.py` 와 같은 방식입니다. 높이 부분(166 칸)은 `height_scan.py` 가 만듭니다.

## 정책 관측 214 칸 (순서 그대로)

    0-2      base_lin_vel        골반(base_link) 선속도 [m/s], 골반 좌표계 (+x 앞, +y 왼쪽, +z 위). state estimator 값
    3-5      gyro_scaled         0.2 × IMU 각속도 [rad/s], IMU(=골반) 좌표계
    6-8      projected_gravity   IMU 좌표계로 본 **중력 방향** 단위벡터. 똑바로 서면 (0, 0, -1)
    9-11     command             속도 명령 (vx [m/s], vy [m/s], wz [rad/s]), 골반 좌표계
    12-23    motor_pos_prev      직전 주기의 엔코더각 − 기본자세 [rad], 모터 12 개
    24-35    motor_pos           이번 주기의 엔코더각 − 기본자세 [rad]
    36-47    last_action         직전 주기에 정책이 낸 날 것의 값 (±10 으로 자름)
    48-213   foothold_scan       height_scan.py (R 83 칸, L 83 칸)

## FPM 입력 36 칸

    base_lin_vel 3, gyro 3 (배율 없음), projected_gravity 3, command 3, motor_pos_prev 12, motor_pos 12

FPM 은 두 발의 다음 착지점 (x, y) [m] 를 골반 yaw 좌표로 냅니다 (R, L 순).
정규화는 하지 않습니다 — 두 ONNX 가 자기 안에 품고 있습니다.

## huphy_squat 과 다른 점

* 선속도 3 칸과 속도 명령 3 칸이 들어가고, 자이로에 0.2 를 곱합니다. 마지막에 높이 166 칸이 붙습니다.
* 기본자세가 다릅니다 (hip roll 0, hip yaw 0, 크랭크 R +0.1440 / −0.1471). README 7 절.
* 관절 순서·이력 순서(오래된 것 먼저)·중력 부호·자이로 정의는 같습니다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class ContractError(ValueError):
  """명세가 이 어댑터가 다룰 수 있는 모양이 아니다."""


def gravity_from_quat(w: float, x: float, y: float, z: float) -> np.ndarray:
  """IMU 자세 쿼터니언 (w, x, y, z), "IMU → 월드" 회전 (월드 z 가 위) → IMU 좌표계의 중력 방향 단위벡터.
  IMU 가 (x, y, z, w) 순서로 준다면 순서를 바꿔 넣으십시오. 똑바로 서면 (0, 0, -1) 이 나와야 합니다."""
  n = math.sqrt(w * w + x * x + y * y + z * z)
  if n < 1e-9:
    raise ValueError("쿼터니언의 크기가 0 입니다")
  w, x, y, z = w / n, x / n, y / n, z / n
  return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def world_to_body(v_world, quat) -> np.ndarray:
  """월드 좌표 벡터 → 몸 좌표 (quat 는 (w, x, y, z), 몸 → 월드). 선속도를 월드 좌표로 받았을 때 쓴다."""
  w, x, y, z = (float(v) for v in quat)
  n = math.sqrt(w * w + x * x + y * y + z * z)
  w, x, y, z = w / n, x / n, y / n, z / n
  R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
  return R.T @ np.asarray(v_world, dtype=np.float64).reshape(3)


class ObsAdapter:
  """명세를 읽어 관측 앞부분과 FPM 입력을 만든다. 상태는 직전 엔코더 프레임과 경고 상태뿐."""

  LAYOUT = ["base_lin_vel", "gyro_scaled", "projected_gravity", "command", "motor_pos_prev", "motor_pos", "last_action",
            "foothold_scan"]
  GRAVITY_CHECK_STEPS = 5

  def __init__(self, contract: dict):
    self.c = contract
    self.obs_dim = int(contract["obs_dim"])
    self.action_dim = int(contract["action_dim"])
    self.control_rate_hz = float(contract["control_rate_hz"])
    acts = contract["actions"]
    self.action_joints = [a["joint"] for a in acts]
    self.scale = np.array([a["scale_rad_per_unit"] for a in acts], dtype=np.float64)
    self.offset = np.array([a["offset_rad"] for a in acts], dtype=np.float64)
    self.clip_lo = np.array([a["clip_rad"][0] for a in acts], dtype=np.float64)
    self.clip_hi = np.array([a["clip_rad"][1] for a in acts], dtype=np.float64)
    self.default = {a["joint"]: float(a["offset_rad"]) for a in acts}
    terms = [t["term"] for t in contract["observations"]]
    if terms != self.LAYOUT:
      raise ContractError(f"관측 배치가 이 어댑터와 다릅니다: {terms}")
    if sum(int(t["width"]) for t in contract["observations"]) != self.obs_dim:
      raise ContractError("관측 항목 폭의 합이 obs_dim 과 다릅니다")
    obs = {t["term"]: t for t in contract["observations"]}
    self.gyro_scale = float(obs["gyro_scaled"]["scale"])
    self.last_action_clip = float(obs["last_action"]["clip"])
    if list(obs["motor_pos"]["joint_names"]) != self.action_joints:
      raise ContractError("엔코더 관절 순서가 행동 순서와 다릅니다")
    cmd = contract["command"]
    self.cmd_lo = np.array(cmd["range_lo"], dtype=np.float64)
    self.cmd_hi = np.array(cmd["range_hi"], dtype=np.float64)
    self.fpm_dim = int(contract["fpm"]["input_dim"])
    self.reset()

  @classmethod
  def from_file(cls, path) -> "ObsAdapter":
    return cls(json.loads(Path(path).read_text(encoding="utf-8")))

  def reset(self) -> None:
    """이력을 비운다. **로봇을 세우거나 정책을 바꿀 때마다** 부를 것."""
    self._prev = None
    self._grav_checks_left = self.GRAVITY_CHECK_STEPS
    self._cmd_warned = False
    self.cmd_clipped_steps = 0

  # ------------------------------------------------------------------ 관측
  def build(self, *, gyro, projected_gravity, joint_pos: dict, base_lin_vel, command, last_action):
    """한 주기의 (관측 앞부분 48 칸, FPM 입력 36 칸), float32. **주기마다 정확히 한 번** 부를 것
    (부를 때마다 엔코더 이력이 한 칸 밀립니다)."""
    v = self._vec("base_lin_vel", base_lin_vel, 3)
    w = self._vec("gyro", gyro, 3)
    g = self._vec("projected_gravity", projected_gravity, 3)
    self._check_gravity_sign(g)
    c = self.clip_command(command)
    now = self._encoders(joint_pos)
    prev = now if self._prev is None else self._prev          # 리셋 직후: 이번 프레임을 두 번 (학습과 같음)
    self._prev = now
    a = np.clip(self._vec("last_action", last_action, self.action_dim), -self.last_action_clip, self.last_action_clip)
    prop = np.concatenate((v, self.gyro_scale * w, g, c, prev, now, a))
    fpm = np.concatenate((v, w, g, c, prev, now))
    return prop.astype(np.float32), fpm.astype(np.float32)

  def clip_command(self, command) -> np.ndarray:
    """속도 명령을 학습 범위로 자른다. 범위 밖이면 **한 번만** 알리고 계속 돈다."""
    c = self._vec("command", command, 3)
    k = np.clip(c, self.cmd_lo, self.cmd_hi)
    if not np.array_equal(k, c):
      self.cmd_clipped_steps += 1
      if not self._cmd_warned:
        self._cmd_warned = True
        print(f"[obs_adapter] 속도 명령 {np.round(c, 3).tolist()} 가 학습 범위 vx {self.cmd_lo[0]}~{self.cmd_hi[0]}, "
              f"vy {self.cmd_lo[1]}~{self.cmd_hi[1]}, wz {self.cmd_lo[2]}~{self.cmd_hi[2]} 밖입니다 — 잘라서 넣습니다 "
              "(이 경고는 reset() 전까지 한 번만)", flush=True)
    return k

  def _vec(self, name, x, n) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size != n:
      raise ValueError(f"{name}: {v.size} 개를 받았는데 {n} 개여야 합니다")
    if not np.all(np.isfinite(v)):
      raise ValueError(f"{name}: NaN/Inf 가 들어왔습니다 {v}")
    return v

  def _encoders(self, joint_pos: dict) -> np.ndarray:
    if not isinstance(joint_pos, Mapping):
      raise TypeError(f"joint_pos 는 모터 이름 → 엔코더각 [rad] 사전이어야 합니다 (받은 것: {type(joint_pos).__name__}). "
                      "배열이면 dict(zip(pol.joint_names, 배열)) 로 넣으십시오")
    missing = [j for j in self.action_joints if j not in joint_pos]
    if missing:
      raise ValueError(f"관절값이 없습니다: {missing}")
    q = np.array([float(joint_pos[j]) - self.default[j] for j in self.action_joints])
    if not np.all(np.isfinite(q)):
      raise ValueError(f"엔코더각에 NaN/Inf 가 있습니다 {q}")
    return q

  def _check_gravity_sign(self, v: np.ndarray) -> None:
    """리셋 직후(서 있을 때) z 가 양수면 **위 방향**을 넣은 것이다. 모터가 돌기 전에 멈춘다."""
    if self._grav_checks_left <= 0:
      return
    self._grav_checks_left -= 1
    if float(v[2]) > 0.5:
      raise ValueError(f"중력 방향의 z 성분이 {float(v[2]):+.3f} 입니다. 서 있는 로봇이면 -1 에 가까워야 합니다. "
                       "**위 방향(up vector)을 그대로 넣으신 것 같습니다** — 부호를 뒤집어 주십시오.")
    n = float(np.linalg.norm(v))
    if not 0.8 < n < 1.2:
      raise ValueError(f"중력 방향의 크기가 {n:.3f} 입니다. **단위벡터** 여야 합니다 (가속도 [m/s^2] 를 그대로 넣지 마십시오)")

  # ------------------------------------------------------------------ 행동
  def action_to_target(self, raw_action) -> dict:
    """정책의 날 것의 값 → 모터 목표각 [rad]:  목표각 = 기본자세 + 0.25 × 행동, 그 뒤 clip_rad 로 자름."""
    a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if a.size != self.action_dim:
      raise ValueError(f"행동이 {a.size} 개인데 {self.action_dim} 개여야 합니다")
    t = np.clip(self.offset + self.scale * a, self.clip_lo, self.clip_hi)
    return {name: float(t[i]) for i, name in enumerate(self.action_joints)}
