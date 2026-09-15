"""**좌표계와 벡터를 로봇 위에 그려서** 센서와 추정값을 눈으로 맞춰 본다.

2026-09-15 · 사용자 지시: *"위 센서 및 추정값까지 좌표계 시각화로 로봇위에 그려줘."*

무엇을 그리는가 — 셋입니다. 전부 **`imu_in_base` 사이트 자리**에 그립니다. 정책이
중력과 각속도를 읽는 바로 그 자리이고, 세 값을 같은 원점에서 봐야 비교가 됩니다.

| 그림 | 뜻 | 색 |
|---|---|---|
| 축 세 개(빨강·초록·파랑 = x·y·z) | `imu_in_base` 좌표계 자체 | viser 기본 |
| 굵은 화살표 | **추정** 중력 방향 (`ground_fk`, 발바닥이 평평하다는 가정) | 초록 |
| 굵은 화살표 | **실물 IMU** 가 말하는 중력 방향 | 주황 |
| 가는 화살표 | 각속도 — 추정(연초록) / 실물(연주황) | 옅은 색 |

두 굵은 화살표가 **겹치면** 관절각과 IMU 가 같은 이야기를 하는 것이고, **벌어지면**
그 각도가 곧 불일치입니다. 숫자는 조작판에도 같이 냅니다 — 그림만으로는 몇 도인지
못 읽습니다.

## 왜 화살표를 몸통 좌표계로 그리나

중력 방향(`projected gravity`)은 **몸통에서 본 값**입니다. 세계 좌표에 그리면 로봇이
기울 때 화살표가 같이 돌아 버려서 "몸통이 기울었다" 는 정보가 사라집니다. 그래서
`imu_in_base` 사이트에 붙여 그립니다 — 로봇이 기울면 화살표도 같이 기울고, 두 화살표
사이의 각만 남습니다.

## 안 그리는 것

**크기.** 중력 방향은 단위벡터라 길이에 뜻이 없고, 각속도는 단위가 달라(rad/s) 같은
자에 놓으면 거짓말이 됩니다. 각속도 화살표는 **방향만** 보이고 길이는 고정이며,
크기는 숫자로만 냅니다.
"""

from __future__ import annotations

import numpy as np

SITE = "imu_in_base"

GREEN = (60, 200, 90)          # 추정 (관절각에서 푼 것)
ORANGE = (245, 150, 40)        # 실물 IMU
GREEN_PALE = (20, 130, 60)      # 밝은 바닥에서 흰색으로 안 보이게 진하게
ORANGE_PALE = (180, 95, 10)     # 같은 이유로 진하게

GRAV_LEN = 0.28                # 중력 화살표 길이 [m] — 보기 좋으라고 정한 값
GYRO_LEN = 0.22                # 각속도 화살표의 **최대** 길이 [m]
GYRO_FULL = 2.0                # 이 크기[rad/s]에서 화살표가 최대 길이가 된다
GYRO_DEADBAND = 0.05           # 이보다 작으면 **안 그린다**

# 2026-09-15 · 사용자 지적: *"하얀색이 imu추정인가? 엄청 튀는데?"*
#
# 튀는 것이 맞았고, 원인은 제 그리기 방식이었습니다. 각속도 화살표를 **길이 고정**으로
# 그렸더니, 로봇이 가만히 있어도 방향이 마구 돌았습니다.
#
# 왜 방향이 도는가: 각속도 추정은 `-J_rot(q)·q̇` 라 관절 속도를 그대로 받습니다. 그런데
# 실물의 `vel` 은 정지 상태에서도 **표준편차 2.1~2.3 deg/s, 최대 12.8 deg/s** 의 잡음이
# 있습니다 (09-15 ROM 기록의 앞 150초, 아무도 안 만진 구간에서 잼). 사슬로 합해지면
# 몸통 각속도가 0.2 rad/s 언저리의 **방향이 무작위인** 값이 됩니다. 크기는 작은데
# 길이를 고정해 그리면 그 무작위 방향만 보입니다.
#
# 그래서 길이를 **크기에 비례**시키고 아주 작으면 아예 안 그립니다. 잡음은 짧은 토막이
# 되고, 진짜 움직임만 길어집니다. 잡음을 지운 것이 아니라 **크기를 정직하게 보이게** 한
# 것입니다 — 숫자는 조작판에 그대로 나갑니다.


def _quat_from_mat(R) -> np.ndarray:
  """3x3 회전 -> viser 의 (w, x, y, z)."""
  import mujoco

  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
  return q


class FrameOverlay:
  """viser 장면에 축과 화살표를 올리고, 틱마다 값을 갈아 끼운다.

  **장면에 물체를 새로 만들지 않고 같은 이름으로 덮어씁니다.** viser 는 같은 이름으로
  다시 부르면 갈아 끼우므로, 초당 여러 번 불러도 물체가 쌓이지 않습니다.
  """

  def __init__(self, server, core, scene=None, root: str = "/est"):
    self.server = server
    self.core = core
    self.scene = scene
    self.root = root
    self.visible = True
    import mujoco

    self._mj = mujoco
    m = core.m
    self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"robot/{SITE}")
    if self.sid < 0:
      self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    if self.sid < 0:
      raise ValueError(f"{SITE} 사이트가 이 모형에 없습니다 — 정책이 읽는 자리입니다")

  # ------------------------------------------------------------------ 내부
  def _site_pose(self):
    """사이트의 자리와 자세. **장면 오프셋을 더해서** 화면 좌표로 맞춘다.

    `ViserMujocoScene` 은 카메라 추적이 켜져 있으면 로봇 전체를 `-tracked_pos` 만큼
    옮겨 그립니다(추적 대상이 화면 원점에 오게). 그래서 세계 좌표를 그대로 쓰면
    화살표만 로봇에서 **떨어져** 붙습니다 — 실제로 처음에 그렇게 났습니다.
    자세(회전)는 오프셋과 무관하므로 그대로 씁니다.
    """
    d = self.core.d
    pos = np.array(d.site_xpos[self.sid])
    R = np.array(d.site_xmat[self.sid]).reshape(3, 3)
    off = getattr(self.scene, "_scene_offset", None)
    if off is not None:
      pos = pos + np.asarray(off, dtype=float)
    return pos, R

  def _hide(self, name) -> None:
    self.server.scene.add_arrows(f"{self.root}/{name}", points=np.zeros((1, 2, 3)),
                                 colors=np.array([(0, 0, 0)]), visible=False)

  def _arrow(self, name, vec_site, color, length, R_site, pos, radius, scale_by=None):
    """사이트 좌표계에서 준 방향을 **세계로 돌려** 화살표 하나를 놓는다.

    ``scale_by`` 를 주면 길이를 ``|v| / scale_by`` 에 비례시킵니다(최대 ``length``).
    각속도처럼 **크기에 뜻이 있는** 값에 씁니다 — 중력 방향은 단위벡터라 안 씁니다.
    """
    v = np.asarray(vec_site, dtype=float)
    n = float(np.linalg.norm(v))
    if scale_by:
      if not np.isfinite(n) or n < GYRO_DEADBAND:
        self._hide(name)                  # 잡음만 있을 때는 아무것도 안 그린다
        return
      length = length * min(1.0, n / float(scale_by))
    if not np.isfinite(n) or n < 1e-9:
      self._hide(name)
      return
    tip = pos + (R_site @ (v / n)) * length
    self.server.scene.add_arrows(
      f"{self.root}/{name}",
      points=np.array([[pos, tip]]),
      colors=np.array([color]),
      shaft_radius=radius, head_radius=radius * 2.4, head_length=length * 0.24,
      visible=self.visible,
    )

  # ------------------------------------------------------------------ 공개
  def update(self, est: dict | None, imu: dict | None) -> dict:
    """한 번 그린다. 돌려주는 것은 **숫자로 낼 비교값**.

    ``est`` 는 :class:`ground_fk.GroundFkEstimator` 가 낸 것, ``imu`` 는 실물에서 온
    ``{"proj_gravity_b":…, "gyro_b":…}``. 둘 중 없는 것은 그냥 안 그립니다 —
    없는 값을 0 으로 그려서 "겹쳤다" 처럼 보이게 하지 않습니다.
    """
    pos, R = self._site_pose()
    self.server.scene.add_frame(f"{self.root}/site", show_axes=True,
                                axes_length=0.16, axes_radius=0.006,
                                wxyz=_quat_from_mat(R), position=pos,
                                visible=self.visible)
    out: dict = {}
    if est is not None:
      self._arrow("grav_est", est["proj_gravity_b"], GREEN, GRAV_LEN, R, pos, 0.010)
      if est.get("gyro_b") is not None:
        self._arrow("gyro_est", est["gyro_b"], GREEN_PALE, GYRO_LEN, R, pos, 0.005,
                    scale_by=GYRO_FULL)
        out["gyro_est_norm"] = float(np.linalg.norm(est["gyro_b"]))
      out["residual_deg"] = est["residual_deg"]
    if imu is not None and imu.get("proj_gravity_b") is not None:
      self._arrow("grav_imu", imu["proj_gravity_b"], ORANGE, GRAV_LEN, R, pos, 0.010)
      if imu.get("gyro_b") is not None:
        self._arrow("gyro_imu", imu["gyro_b"], ORANGE_PALE, GYRO_LEN, R, pos, 0.005,
                    scale_by=GYRO_FULL)
        out["gyro_imu_norm"] = float(np.linalg.norm(imu["gyro_b"]))
    if est is not None and imu is not None and imu.get("proj_gravity_b") is not None:
      a = np.asarray(est["proj_gravity_b"], float)
      b = np.asarray(imu["proj_gravity_b"], float)
      na, nb = np.linalg.norm(a), np.linalg.norm(b)
      if na > 1e-9 and nb > 1e-9:
        out["grav_angle_deg"] = float(np.degrees(
          np.arccos(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))))
      if est.get("gyro_b") is not None and imu.get("gyro_b") is not None:
        out["gyro_err_norm"] = float(np.linalg.norm(
          np.asarray(est["gyro_b"], float) - np.asarray(imu["gyro_b"], float)))
    return out

  def set_visible(self, on: bool) -> None:
    self.visible = bool(on)
    for n in ("site", "grav_est", "grav_imu", "gyro_est", "gyro_imu"):
      h = self.server.scene._handle_from_name.get(f"{self.root}/{n}") \
        if hasattr(self.server.scene, "_handle_from_name") else None
      if h is not None:
        h.visible = self.visible
