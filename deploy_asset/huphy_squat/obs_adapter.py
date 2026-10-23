#!/usr/bin/env python3
"""**로봇에서 읽은 값 → 정책이 받는 관측 벡터 43 개.** numpy 와 표준 라이브러리만 쓴다.

`pyg_rough_main_20261012/obs_adapter.py` 를 이 정책(huphy_squat)에 맞게 고친 것입니다.
달라진 점은 아래 "이전 꾸러미와 다른 점" 에 모았습니다.

## 관측 43 개 (순서 그대로)

    0-2    gyro               IMU 각속도 [rad/s], IMU(=몸통) 좌표계
    3-5    projected_gravity  IMU 좌표계로 본 **중력 방향** 단위벡터. 똑바로 서면 (0, 0, -1)
    6-29   motor_pos          모터 12 개의 엔코더각 - 기준자세각 [rad],
                              **[이전 프레임 12][지금 프레임 12]** (오래된 것이 앞)
    30-41  last_action        직전 주기에 정책이 낸 **날 것의** 값 12 개 (배율·오프셋 전)
    42     h_star             목표 골반 높이 [m] 그대로 (배율·오프셋 없음)

정규화는 하지 않습니다 — ONNX 가 자기 안에 품고 있습니다.

## 이전 꾸러미(pyg_rough_main_20261012)와 다른 점

* 마지막 칸: 속도 명령 3 개 → **목표 높이 h* 1 개** (그래서 45 → 43 차원).
* 기준자세: 엉덩이 롤이 ±0.05 rad 벌어져 있고(R +, L -), 엉덩이 요 ±0.008837,
  크랭크 A/B 값도 다릅니다. 상대각은 **이 꾸러미의 기준자세** 에서 뺍니다.
* 크랭크 목표각 자르기 범위: ±1.08 → **±0.95 rad**.
* 관절 순서·이력 순서·중력 부호·자이로 정의는 같습니다.

## 쓰는 법

    ad = ObsAdapter.from_file("deploy_contract.json")
    ad.reset()
    obs = ad.build(gyro=..., projected_gravity=..., joint_pos={"R_hip_pitch_joint": ..., ...},
                   last_action=prev_raw, h_star=0.85)
    targets = ad.action_to_target(raw)          # 이름 → 목표각 [rad]

보통은 이 파일을 직접 부르지 말고 `policy_runner.py` 의 `SquatPolicy` 를 쓰십시오.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class ContractError(ValueError):
  """명세가 이 어댑터가 다룰 수 있는 모양이 아니다."""


def gravity_from_quat(w: float, x: float, y: float, z: float) -> np.ndarray:
  """IMU 자세 쿼터니언 → IMU 좌표계의 중력 방향 단위벡터.

  쿼터니언은 **(w, x, y, z) 순서**, "IMU 좌표 → 월드 좌표" 회전(월드 z 가 위)이라고
  가정합니다. 결과는 R(q)^T · (0, 0, -1) = 월드 아래 방향을 IMU 좌표로 본 것.
  IMU 가 (x, y, z, w) 순서로 준다면 **순서를 바꿔서** 넣으십시오 — 순서를 틀리면 값이
  그럴듯하게 틀립니다. 똑바로 서 있을 때 (0, 0, -1) 이 나오는지 꼭 확인하십시오.
  """
  n = math.sqrt(w * w + x * x + y * y + z * z)
  if n < 1e-9:
    raise ValueError("쿼터니언의 크기가 0 입니다")
  w, x, y, z = w / n, x / n, y / n, z / n
  # R^T 의 셋째 열에 -1 을 곱한 것 = -(R 의 셋째 행)
  return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class ObsAdapter:
  """명세 하나를 읽어 관측 벡터를 만든다. 상태는 엔코더각 이력과 경고 상태뿐."""

  KNOWN = {"imu_gyro", "imu_gravity", "motor_encoder_rel", "last_action", "height_command"}
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
    h = contract["h_star"]
    self.h_lo, self.h_hi = float(h["range_m"][0]), float(h["range_m"][1])

    self.terms = list(contract["observations"])
    bad = [t["term"] for t in self.terms if t["func"] not in self.KNOWN]
    if bad:
      raise ContractError(f"모르는 관측 항목: {bad}. 0 으로 채우지 않고 멈춥니다")
    hist = [t for t in self.terms if t["func"] == "motor_encoder_rel"]
    if len(hist) != 1:
      raise ContractError("엔코더 항목(motor_encoder_rel)이 정확히 하나여야 합니다")
    self.hist_term = hist[0]
    self.hist_joints = list(self.hist_term["joint_names"])
    self.hist_len = int(self.hist_term["history_frames"])
    if len(self.hist_joints) * self.hist_len != int(self.hist_term["width"]):
      raise ContractError("엔코더 항목의 폭이 관절 수 x 프레임 수와 다릅니다")
    missing = [j for j in self.hist_joints if j not in self.default]
    if missing:
      raise ContractError(f"기준자세각을 모르는 관절: {missing}")
    if sum(int(t["width"]) for t in self.terms) != self.obs_dim:
      raise ContractError("관측 항목 폭의 합이 obs_dim 과 다릅니다")
    self._hist: list[np.ndarray] = []
    self._grav_checks_left = 0
    self._h_warned = False
    self.h_clipped_steps = 0

  @classmethod
  def from_file(cls, path) -> "ObsAdapter":
    return cls(json.loads(Path(path).read_text(encoding="utf-8")))

  def reset(self) -> None:
    """이력을 비운다. **로봇을 세우거나 정책을 바꿀 때마다** 부를 것."""
    self._hist = []
    self._grav_checks_left = self.GRAVITY_CHECK_STEPS
    self._h_warned = False
    self.h_clipped_steps = 0

  # ------------------------------------------------------------------ 관측
  def build(self, *, gyro, projected_gravity, joint_pos: dict, last_action, h_star) -> np.ndarray:
    """한 제어 주기의 관측 벡터(float32, 43 개). **주기마다 정확히 한 번** 부를 것
    (부를 때마다 엔코더 이력이 한 칸 밀립니다)."""
    q = self._push_history(joint_pos)
    h = self.clip_h_star(h_star)
    parts = []
    for t in self.terms:
      f = t["func"]
      if f == "imu_gyro":
        v = np.asarray(gyro, dtype=np.float64).reshape(-1)
      elif f == "imu_gravity":
        v = np.asarray(projected_gravity, dtype=np.float64).reshape(-1)
        self._check_gravity_sign(v)
      elif f == "motor_encoder_rel":
        v = q
      elif f == "last_action":
        v = np.asarray(last_action, dtype=np.float64).reshape(-1)
      else:                                        # height_command
        v = np.array([h])
      if v.size != int(t["width"]):
        raise ValueError(f"{t['term']}: {v.size} 개를 받았는데 {t['width']} 개여야 합니다")
      if not np.all(np.isfinite(v)):
        raise ValueError(f"{t['term']}: NaN/Inf 가 들어왔습니다 {v}")
      parts.append(v)
    return np.concatenate(parts).astype(np.float32)

  def clip_h_star(self, h_star) -> float:
    """h* 를 학습 범위로 자른다. 범위 밖이면 **한 번만** 알리고 계속 돈다."""
    h = float(np.asarray(h_star, dtype=np.float64).reshape(-1)[0]) if np.ndim(h_star) else float(h_star)
    if not math.isfinite(h):
      raise ValueError(f"h_star 가 {h} 입니다 — 숫자여야 합니다 [m]")
    c = min(max(h, self.h_lo), self.h_hi)
    if c != h:
      self.h_clipped_steps += 1
      if not self._h_warned:
        self._h_warned = True
        print(f"[obs_adapter] h_star {h:.3f} m 는 학습 범위 [{self.h_lo}, {self.h_hi}] m 밖입니다 — "
              f"{c:.3f} m 로 잘라서 넣습니다 (이 경고는 reset() 전까지 한 번만). "
              "단위가 m 인지 확인하십시오 (mm·cm 가 아님).", flush=True)
    return c

  def _check_gravity_sign(self, v: np.ndarray) -> None:
    """리셋 직후(서 있을 때) z 가 양수면 **위 방향**을 넣은 것이다. 모터가 돌기 전에 멈춘다."""
    if self._grav_checks_left <= 0:
      return
    self._grav_checks_left -= 1
    if float(v[2]) > 0.5:
      raise ValueError(
        f"중력 방향의 z 성분이 {float(v[2]):+.3f} 입니다. 서 있는 로봇이면 -1 에 가까워야 합니다. "
        "**위 방향(up vector)을 그대로 넣으신 것 같습니다** — 부호를 뒤집어 주십시오.")
    n = float(np.linalg.norm(v))
    if not 0.8 < n < 1.2:
      raise ValueError(f"중력 방향의 크기가 {n:.3f} 입니다. **단위벡터** 여야 합니다 "
                       "(가속도 [m/s^2] 를 그대로 넣지 마십시오)")

  def _push_history(self, joint_pos: dict) -> np.ndarray:
    """이번 프레임을 넣고 **[오래된 것 … 최신]** 순으로 이어 붙인다."""
    missing = [j for j in self.hist_joints if j not in joint_pos]
    if missing:
      raise ValueError(f"관절값이 없습니다: {missing}")
    frame = np.array([float(joint_pos[j]) - self.default[j] for j in self.hist_joints])
    self._hist.append(frame)
    if len(self._hist) > self.hist_len:
      self._hist = self._hist[-self.hist_len:]
    frames = list(self._hist)
    while len(frames) < self.hist_len:             # 리셋 직후: 첫 프레임으로 앞을 메움 (학습과 같음)
      frames.insert(0, frames[0])
    return np.concatenate(frames)

  # ------------------------------------------------------------------ 행동
  def action_to_target(self, raw_action) -> dict:
    """정책의 날 것의 값 → 모터 목표각 [rad]:  목표각 = 오프셋 + 0.25 x 행동, 그 뒤 clip_rad 로 자름."""
    a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if a.size != self.action_dim:
      raise ValueError(f"행동이 {a.size} 개인데 {self.action_dim} 개여야 합니다")
    t = np.clip(self.offset + self.scale * a, self.clip_lo, self.clip_hi)
    return {name: float(t[i]) for i, name in enumerate(self.action_joints)}
