#!/usr/bin/env python3
"""**mapper 의 높이 정보 → 정책이 받는 foothold scan 166 칸.** numpy 와 표준 라이브러리만 쓴다.

정책은 높이 정보 없이는 쓰지 않는다고 가정합니다. 로봇에는 odometry estimator (골반 위치·자세·선속도) 와
mapper (지면 높이 지도) 가 붙어 있고, 이 파일은 mapper 의 출력을 정책 입력으로 바꿉니다.

## 정책이 받는 값 (최종 입력)

골반에 붙어 골반의 **yaw 만 따라 도는 수평 격자** 를 씁니다 (roll·pitch 는 따라가지 않음).

    격자     x(앞) −0.80 … +0.80 m, y(왼쪽) −0.50 … +0.50 m, 0.05 m 간격 → 33 × 21 점
    점 (i, j) 의 위치   x = −0.80 + 0.05 i,  y = −0.50 + 0.05 j   (i = 0…32, j = 0…20)

발마다 (R 먼저, 그다음 L) FPM 이 내는 다음 착지점 (fx, fy) [m, 같은 격자 좌표] 에서 가장 가까운 점
(ax, ay) = (round((fx + 0.8) / 0.05), round((fy + 0.5) / 0.05)) 을 찾고, 그 둘레 9 × 9 점을 봅니다.

    점 k = 9 a + b  (a, b = 0…8):  i = clip(ax + a − 4, 0, 32),  j = clip(ay + b − 4, 0, 20)
    값      h_k = clip(z_pelvis − z_terrain(점 k) − 0.5, −1, 1)       [m]

한 발의 칸은 [fx, fy, h_0 … h_80] (83 칸, fx·fy 도 ±1 로 자름) 이고, R 다음 L 로 붙여 166 칸입니다.
z_pelvis 는 골반(base_link) 원점의 높이, z_terrain 은 그 점의 **지면 높이** 입니다.
그러니 정책에 실제로 필요한 것은 **각 격자점에서 "골반이 지면보다 몇 m 위에 있나"** 뿐입니다.
평지에서 기본자세로 서면 z_pelvis − z_terrain ≈ 0.91 m, 그래서 h ≈ 0.41 입니다.

## mapper 출력을 넣는 방법 (`policy_runner.RoughPolicy.inference(height=...)`)

| mapper 가 내는 것 | 클래스 | odometry 의 위치·자세 필요 |
|---|---|---|
| 월드 좌표 elevation map (2.5D 지도) | ``WorldGrid`` | 예 |
| 월드 좌표 질의 함수: 점 → 지면 높이 | ``WorldHeightFunction`` (그냥 callable 을 넣어도 이것) | 예 |
| 로봇 중심 격자 33 × 21 (위 정의), 값 = z_pelvis − z_terrain | ``RobotGrid`` (그 모양의 배열을 넣어도 이것) | 아니오 |
| 로봇 중심 질의 함수: 골반 yaw 좌표 점 → z_pelvis − z_terrain | ``RobotHeightFunction`` | 아니오 |

"예" 인 방법은 ``base_pos`` (월드 좌표 골반 원점 [m]) 와 ``base_quat`` (w, x, y, z, 골반 → 월드) 가
지도와 **같은 월드 좌표계** 여야 합니다. 높이(z)도 같은 기준이어야 합니다 (정책은 z_pelvis − z_terrain 만 봄).

## 학습 때의 높이 정보 품질

학습에서는 4 cm 격자 지도에서 가장 가까운 칸 값을 읽었고, 그 위에 다음 오차를 넣었습니다.
mapper 가 이 범위 안이면 학습 조건 안입니다.

* 칸마다 매 주기 균등 노이즈 ±0.10 m (착지점 fx, fy 에도 ±0.10 m)
* 에피소드마다 지도 전체가 xy ±0.05 m, z ±0.05 m 어긋남 (drift)

## 지도에 빈 칸 (NaN/Inf, 아직 못 본 곳) 이 있으면

그 칸은 **같은 발 패치의 유효한 칸들의 중앙값** 으로 채웁니다. 패치 전체가 비면 그 발의 **직전 주기 값** 을 씁니다.
처음 한 번만 경고하고 계속 돕니다. 직전 값도 없으면 (리셋 직후 패치 전체가 빔) 예외를 냅니다 — 지도가 준비된 뒤 켜십시오.
채운 칸 수는 ``RoughPolicy.scan_nan_count`` 로 볼 수 있습니다.
"""

from __future__ import annotations

import math

import numpy as np


def yaw_from_quat(q) -> float:
  """(w, x, y, z) 쿼터니언 (몸 → 월드) 의 yaw [rad]."""
  w, x, y, z = (float(v) for v in q)
  if not w * w + x * x + y * y + z * z > 1e-12:
    raise ValueError(f"base_quat 의 크기가 0 이거나 NaN 입니다: {q}")
  return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def quat_to_mat(q) -> np.ndarray:
  w, x, y, z = (float(v) for v in q)
  n = math.sqrt(w * w + x * x + y * y + z * z)
  if not n > 1e-6:
    raise ValueError(f"쿼터니언의 크기가 0 이거나 NaN 입니다: {q}")
  w, x, y, z = w / n, x / n, y / n, z / n
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])


# --------------------------------------------------------------------------- mapper 출력
class HeightSource:
  """격자점에서 z_pelvis − z_terrain [m] 을 돌려주는 것. 아래 클래스들이 이것을 구현한다."""

  needs_pose = False

  def relative(self, ix, iy, pts_yaw, base_pos, yaw) -> np.ndarray:
    """``ix, iy`` 격자 번호 (N,), ``pts_yaw`` 골반 yaw 좌표 점 (N, 2) → z_pelvis − z_terrain (N,)."""
    raise NotImplementedError


class WorldGrid(HeightSource):
  """월드 좌표 elevation map. ``z[i, j]`` 는 점 (origin_x + i·res, origin_y + j·res) 의 지면 높이 [m]
  (첫 번째 축이 x). 가장 가까운 칸 값을 쓰고, 지도 밖은 가장자리 칸 값을 쓴다 (학습과 같음).
  mapper 가 지도를 새로 낼 때마다 ``update`` 로 바꾸면 된다. 로봇 둘레 최소 ±0.9 m 를 덮어야 한다."""

  needs_pose = True

  def __init__(self, z, origin_xy, resolution: float):
    self.update(z, origin_xy, resolution)

  def update(self, z, origin_xy, resolution: float) -> None:
    self.z = np.asarray(z, dtype=np.float64)
    if self.z.ndim != 2:
      raise ValueError(f"지도는 2 차원 배열이어야 합니다 (받은 모양 {self.z.shape})")
    self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
    self.res = float(resolution)

  def relative(self, ix, iy, pts_yaw, base_pos, yaw):
    pts = _to_world(pts_yaw, base_pos, yaw)
    i = np.clip(np.rint((pts[:, 0] - self.ox) / self.res).astype(int), 0, self.z.shape[0] - 1)
    j = np.clip(np.rint((pts[:, 1] - self.oy) / self.res).astype(int), 0, self.z.shape[1] - 1)
    return base_pos[2] - self.z[i, j]


class WorldHeightFunction(HeightSource):
  """``fn(xy)``: 월드 좌표 점 (N, 2) [m] → 지면 높이 z (N,) [m]."""

  needs_pose = True

  def __init__(self, fn):
    self.fn = fn

  def relative(self, ix, iy, pts_yaw, base_pos, yaw):
    pts = _to_world(pts_yaw, base_pos, yaw)
    return base_pos[2] - np.asarray(self.fn(pts), dtype=np.float64).reshape(-1)


class RobotGrid(HeightSource):
  """로봇 중심 격자 33 × 21 (``grid[i, j]``, 모듈 설명의 점 정의). 값 = z_pelvis − z_terrain [m]."""

  SHAPE = (33, 21)

  def __init__(self, grid):
    self.grid = np.asarray(grid, dtype=np.float64)
    if self.grid.shape != self.SHAPE:
      raise ValueError(f"로봇 중심 격자는 {self.SHAPE[0]}×{self.SHAPE[1]} (첫 축 x: 앞 −0.8 → +0.8 m, 둘째 축 y: 오른쪽 −0.5 → "
                       f"왼쪽 +0.5 m) 이어야 합니다. 받은 모양 {self.grid.shape}" +
                       (" — x 와 y 가 바뀐 것 같습니다 (전치)" if self.grid.shape == self.SHAPE[::-1] else ""))

  def relative(self, ix, iy, pts_yaw, base_pos, yaw):
    return self.grid[ix, iy]


class RobotHeightFunction(HeightSource):
  """``fn(xy)``: 골반 yaw 좌표 점 (N, 2) [m] → z_pelvis − z_terrain (N,) [m]."""

  def __init__(self, fn):
    self.fn = fn

  def relative(self, ix, iy, pts_yaw, base_pos, yaw):
    return np.asarray(self.fn(pts_yaw), dtype=np.float64).reshape(-1)


def _to_world(pts_yaw, base_pos, yaw):
  c, s = math.cos(yaw), math.sin(yaw)
  return np.asarray(base_pos[:2]) + pts_yaw @ np.array([[c, s], [-s, c]])


# --------------------------------------------------------------------------- scan
class FootholdScan:
  """착지점 2 개 + mapper 출력 → foothold scan (166,). 격자 상수는 명세(deploy_contract.json)에서 읽는다."""

  HEIGHT_CHECK_STEPS = 5          # 리셋 직후 (서 있을 때) z_pelvis − z_terrain 이 그럴듯한지 본다
  HEIGHT_RANGE_M = (0.5, 1.3)

  def __init__(self, contract: dict):
    s = contract["foothold_scan"]
    self.res = float(s["resolution_m"])
    self.size_x, self.size_y = (float(v) for v in s["grid_size_m"])
    self.nx, self.ny = (int(v) for v in s["grid_nodes"])
    self.n = int(s["patch_n"])
    self.offset = float(s["offset_m"])
    if self.nx != int(round(self.size_x / self.res)) + 1 or self.ny != int(round(self.size_y / self.res)) + 1:
      raise ValueError("명세의 격자 크기와 점 수가 맞지 않습니다")
    k = np.arange(-(self.n // 2), self.n // 2 + 1)
    ox, oy = np.meshgrid(k, k, indexing="ij")
    self.off_x, self.off_y = ox.ravel(), oy.ravel()
    self.width = 2 * (2 + self.n * self.n)
    self.reset()

  def as_source(self, height) -> HeightSource:
    """inference(height=...) 로 들어온 mapper 출력을 높이 입력 객체로 바꾼다."""
    if height is None:
      raise ValueError("height 가 None 입니다. 이 정책은 mapper 의 높이 정보가 필요합니다 (height_scan.py)")
    if isinstance(height, HeightSource):
      return height
    if callable(height):
      return WorldHeightFunction(height)
    arr = np.asarray(height)
    if arr.shape == (self.nx, self.ny):
      return RobotGrid(arr)
    raise TypeError(f"height 로 {type(height).__name__} {getattr(arr, 'shape', '')} 를 받았습니다. "
                    f"{self.nx}×{self.ny} 로봇 중심 격자, 함수, 또는 height_scan 의 클래스여야 합니다")

  def reset(self) -> None:
    self.nan_count = 0
    self._nan_warned = False
    self._last_rel = None
    self._height_checks_left = self.HEIGHT_CHECK_STEPS

  def patch_nodes(self, footholds):
    """발마다 9 × 9 격자 번호와 골반 yaw 좌표 점: ix, iy (2, 81), pts (2, 81, 2)."""
    fh = np.asarray(footholds, dtype=np.float64).reshape(2, 2)
    ax = np.rint((fh[:, 0] + self.size_x / 2) / self.res).astype(int)
    ay = np.rint((fh[:, 1] + self.size_y / 2) / self.res).astype(int)
    ix = np.clip(ax[:, None] + self.off_x, 0, self.nx - 1)
    iy = np.clip(ay[:, None] + self.off_y, 0, self.ny - 1)
    pts = np.stack((ix * self.res - self.size_x / 2, iy * self.res - self.size_y / 2), -1)
    return ix, iy, pts

  def compute(self, footholds, source: HeightSource, base_pos=None, base_quat=None):
    """→ (scan (166,) float64, 표시용 정보 dict)."""
    fh = np.asarray(footholds, dtype=np.float64).reshape(2, 2)
    if source.needs_pose:
      if base_pos is None or base_quat is None:
        raise ValueError(f"{type(source).__name__} 은 odometry 의 base_pos 와 base_quat 가 필요합니다 (지도와 같은 월드 좌표계)")
      base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
      yaw = yaw_from_quat(base_quat)
    else:
      base_pos = None if base_pos is None else np.asarray(base_pos, dtype=np.float64).reshape(3)
      yaw = 0.0 if base_quat is None else yaw_from_quat(base_quat)
    ix, iy, pts = self.patch_nodes(fh)
    rel = []
    for f in range(2):
      r = np.asarray(source.relative(ix[f], iy[f], pts[f], base_pos, yaw), dtype=np.float64).reshape(-1)
      if r.size != len(ix[f]):
        raise ValueError(f"{type(source).__name__} 이 점 {len(ix[f])} 개에 대해 값 {r.size} 개를 돌려줬습니다 (점마다 하나여야 함)")
      rel.append(r)
    rel = self._fill_missing(np.stack(rel))
    self._check_height(rel, source)
    h = rel - self.offset
    out = np.clip(np.concatenate((fh, h), -1), -1.0, 1.0)  # (2, 83)
    return out.reshape(-1), {"footholds": fh, "points_yaw": pts, "relative": rel, "yaw": yaw}

  def _check_height(self, rel: np.ndarray, source) -> None:
    """리셋 직후 서 있는 로봇이면 z_pelvis − z_terrain 은 약 0.9 m 다. 크게 다르면 모터가 돌기 전에 멈춘다
    (지면 높이 z 자체를 넣었거나, odometry 와 지도의 높이 기준이 다를 때)."""
    if self._height_checks_left <= 0:
      return
    self._height_checks_left -= 1
    med = float(np.median(rel))
    lo, hi = self.HEIGHT_RANGE_M
    if not lo < med < hi:
      raise ValueError(f"{type(source).__name__}: 골반이 지면보다 {med:.2f} m 위에 있다고 나옵니다 (서 있으면 약 0.9 m). "
                       "로봇 중심 형태에는 지면 높이가 아니라 z_pelvis − z_terrain 을 넣어야 하고, 월드 좌표 형태는 "
                       "base_pos 의 z 와 지도의 높이 기준이 같아야 합니다.")

  def _fill_missing(self, rel: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(rel)
    if bad.any():
      rel = rel.copy()
      for f in range(2):
        if not bad[f].any():
          continue
        if (~bad[f]).any():
          rel[f, bad[f]] = np.median(rel[f, ~bad[f]])          # 같은 발 패치의 유효한 칸들의 중앙값
        elif self._last_rel is not None:
          rel[f] = self._last_rel[f]                           # 패치 전체가 빔: 직전 주기 값
        else:
          raise ValueError("리셋 직후 발 패치 전체가 NaN 입니다. mapper 가 로봇 둘레 지도를 낸 뒤에 정책을 켜십시오.")
      self.nan_count += int(bad.sum())
      if not self._nan_warned:
        self._nan_warned = True
        print(f"[height_scan] 높이 {int(bad.sum())} 칸이 NaN/Inf 입니다 — 같은 패치의 중앙값 (또는 직전 값) 으로 채웁니다 "
              "(이 경고는 reset() 전까지 한 번만)", flush=True)
    self._last_rel = rel
    return rel
