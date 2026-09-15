"""관절각만으로 **몸통이 얼마나 기울었는지** 풀어낸다 — IMU 없이.

2026-09-15 · 사용자 지시: *"시뮬상에서 모드를 하나 추가해서, 두 발바닥이 지면에 닿고,
관절값이 정확하다는 가정하에 FK/IK 풀어서 Base_link 의 회전을 구하고, 그때의 projected
gravity와, IMU 에서 읽히는 projected gravity, 같은 원리로 gyro 값을 구하는 <추정값>."*

## 원리 — IK 를 풀 필요가 없다

풀어야 할 것처럼 보이지만, 한 줄로 끝납니다.

발바닥이 바닥에 **평평하게** 닿아 있다면, 그 발바닥의 법선은 **세계 좌표의 +z** 입니다.
그리고 발바닥 법선을 **몸통 좌표계에서** 본 값은 관절각만으로 나옵니다 (순수 FK —
몸통을 단위 자세에 두고 발까지 걸어 내려가면 됩니다). 그 둘을 이으면:

    R_wb · n_b = ẑ            (n_b = 몸통에서 본 발바닥 법선)
    => R_wbᵀ · ẑ = n_b

`projected gravity` 는 **몸통에서 본 중력 방향**, 즉 `R_wbᵀ · (0,0,-1) = -n_b` 입니다.

    ★ 추정 projected gravity = **−(몸통에서 본 발바닥 법선)**

역기구학도, 최적화도, 반복도 없습니다. 발바닥 법선 하나면 됩니다.

## 발바닥 법선이 무엇인가 — 기하에서 확인함

발 충돌 상자(`*_foot1..7_collision`)의 `quat` 이 발 몸체 기준 **단위**입니다
(2026-09-15 확인). 따라서 발바닥은 **발 좌표계 +z 에 수직**이고, 위를 보는 법선은
`(0, 0, 1)` 입니다. 상자가 `z = -0.014` 에 반높이 `0.029` 로 놓여 바닥면이
`z = -0.043` 이라는 것도 같은 확인에서 나왔습니다.

## 두 발이 서로 다른 말을 하면

관절각이 아무 값이나면 왼발과 오른발의 법선이 **서로 다릅니다.** 그러면 둘 다 평평할
수는 없고, 가정이 깨진 것입니다. 그래서 이 추정기는

  * 두 법선의 **벌어진 각**(`residual_deg`)을 같이 냅니다 — 이것이 크면 추정을
    믿으면 안 됩니다. 한 발을 들고 있거나, 영점이 어긋났거나, 관절각이 틀린 것입니다.
  * 평균을 써서 하나의 답을 냅니다 (둘 다 평평하다는 가정에 가장 가까운 자세).

**이 값을 조용히 내놓지 않습니다.** 크레인에 매달려 있으면 발이 바닥에 안 닿으므로
가정 자체가 거짓이고, 그때는 `residual_deg` 가 아니라 **사람이 상황을 알아야** 합니다.

## 각속도(gyro)도 같은 가정에서

발이 바닥에 **붙어 미끄러지지 않는다**면, 발은 세계에 대해 안 돕니다. 그러면 몸통의
각속도는 발에서 몸통까지 사슬이 만드는 각속도의 **반대**입니다.

    ω_몸통(몸통좌표) = − J_rot(q) · q̇

`J_rot` 은 몸통을 고정했을 때 발이 갖는 회전 자코비안이고, MuJoCo 가 계산해 줍니다
(`mj_jacBody`). 수치미분이 아니라 **닫힌 형태**라 잡음이 안 끼고, 실물의 `vel` 을
그대로 넣을 수 있습니다.

## 무엇과 비교하는가

정책이 실제로 먹는 값은 `base_link` 에 붙은 **`imu_in_base` 사이트**의
`framezaxis` 센서(`imu_upvector`)를 뒤집은 것이고, 각속도는 같은 사이트의
`gyro`(`imu_ang_vel`) 입니다. 사이트에 `quat` 이 없어 **몸통 축과 나란**하므로, 위에서
구한 몸통 좌표계 값과 **같은 자리**입니다.
"""

from __future__ import annotations

import numpy as np

SOLE_NORMAL_FOOT = np.array([0.0, 0.0, 1.0])
"""발 좌표계에서 본 발바닥 법선(위쪽). 충돌 상자 `quat` 이 단위인 데서 나옴."""

FEET = ("L_foot_link", "R_foot_link")
IMU_SITE = "imu_in_base"


class GroundFkEstimator:
  """모형 하나를 받아, 관절각을 주면 몸통 기울기를 돌려준다.

  **자기 전용 `MjData` 를 씁니다.** 돌고 있는 시뮬의 상태를 건드리지 않습니다 —
  추정은 관찰이지 조작이 아니고, 뷰어가 보여 주는 자세를 이것이 흔들면 안 됩니다.
  """

  def __init__(self, model, feet=FEET):
    import mujoco

    self._mj = mujoco
    self.m = model
    self.d = mujoco.MjData(model)
    # 구운 모형(`bake.py`)은 이름에 `robot/` 을 붙이고 날 XML 은 안 붙입니다. 둘 다 찾습니다.
    def body_id(name: str) -> int:
      for cand in (name, f"robot/{name}"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
        if bid >= 0:
          return bid
      return -1

    self.feet = []
    for name in feet:
      bid = body_id(name)
      if bid < 0:                       # 이름이 다른 변형이면 조용히 빼지 않고 말한다
        raise ValueError(f"{name} 이라는 몸체가 이 모형에 없습니다")
      self.feet.append((name, bid))
    self.base_bid = body_id("base_link")
    # 자유관절(몸통)의 자유도. 여기 속도는 0 으로 둬야 "몸통 고정" 자코비안이 된다.
    self.free_dofs = []
    if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
      self.free_dofs = list(range(model.jnt_dofadr[0], model.jnt_dofadr[0] + 6))

  # ------------------------------------------------------------------ 내부
  def _pose_identity(self, qpos, qvel=None) -> None:
    """관절각을 넣되 **몸통은 원점·단위자세**로 둔다. 그래야 세계 = 몸통 좌표계."""
    self.d.qpos[:] = qpos
    if self.free_dofs:
      self.d.qpos[0:3] = 0.0
      self.d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    self.d.qvel[:] = 0.0 if qvel is None else qvel
    for i in self.free_dofs:
      self.d.qvel[i] = 0.0
    self._mj.mj_forward(self.m, self.d)

  # ------------------------------------------------------------------ 공개
  def estimate(self, qpos, qvel=None) -> dict:
    """관절각(과 있으면 관절속도)에서 몸통 기울기를 푼다.

    돌려주는 것:
      ``up_b``            몸통에서 본 **위쪽**(세계 +z). 발바닥 법선의 평균.
      ``proj_gravity_b``  몸통에서 본 **중력 방향** = ``-up_b``. 정책이 먹는 값.
      ``gyro_b``          몸통 각속도(몸통 좌표). ``qvel`` 이 없으면 ``None``.
      ``residual_deg``    좌우 발바닥 법선이 벌어진 각. **작아야 믿을 수 있다.**
      ``per_foot``        발마다의 법선과, 그 발만으로 본 중력 방향.
    """
    self._pose_identity(qpos, qvel)
    normals, per_foot = [], {}
    for name, bid in self.feet:
      R = self.d.xmat[bid].reshape(3, 3)          # 발 -> 몸통 (몸통이 단위이므로)
      n = R @ SOLE_NORMAL_FOOT
      n = n / np.linalg.norm(n)
      normals.append(n)
      per_foot[name] = {"up_b": n.copy(), "proj_gravity_b": -n}

    up = np.sum(normals, axis=0)
    nrm = float(np.linalg.norm(up))
    up = up / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0)) if len(normals) > 1 else 1.0
    residual = float(np.degrees(np.arccos(dot)))

    gyro = None
    if qvel is not None:
      gyro = self._gyro(qvel)

    return {"up_b": up, "proj_gravity_b": -up, "gyro_b": gyro,
            "residual_deg": residual, "per_foot": per_foot}

  def _gyro(self, qvel) -> np.ndarray:
    """발이 세계에 붙어 있다는 가정에서 몸통 각속도. 발마다 구해 평균."""
    v = np.asarray(qvel, dtype=float).copy()
    for i in self.free_dofs:
      v[i] = 0.0
    out = []
    for _name, bid in self.feet:
      jacr = np.zeros((3, self.m.nv))
      self._mj.mj_jacBody(self.m, self.d, None, jacr, bid)
      out.append(-(jacr @ v))            # 발이 안 돌면 몸통이 반대로 도는 셈
    return np.mean(out, axis=0)

  # ------------------------------------------------------------------ 참값
  def truth_from_sim(self, data) -> dict:
    """**시뮬이 아는 진짜 값**. 자기검사와 비교용 — 실물에는 없는 값이다.

    정책이 먹는 것과 같은 자리에서 읽습니다: `imu_upvector`(뒤집으면 중력 방향)와
    `imu_ang_vel`. 센서가 없는 모형이면 몸통 자세에서 직접 계산합니다.
    """
    mj = self._mj

    def sensor_id(name: str) -> int:
      for cand in (name, f"robot/{name}"):      # 구운 모형은 접두사가 붙는다
        i = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SENSOR, cand)
        if i >= 0:
          return i
      return -1

    sid, gid = sensor_id("imu_upvector"), sensor_id("imu_ang_vel")
    out = {}
    if sid >= 0:
      adr = self.m.sensor_adr[sid]
      up = np.array(data.sensordata[adr:adr + 3])
    else:
      # 센서가 없는 모형: 몸통 자세에서 직접 구한다 (세계 +z 를 몸통 좌표로)
      up = data.xmat[self.base_bid].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    out["up_b"] = up
    out["proj_gravity_b"] = -up
    if gid >= 0:
      adr = self.m.sensor_adr[gid]
      out["gyro_b"] = np.array(data.sensordata[adr:adr + 3])
    return out
