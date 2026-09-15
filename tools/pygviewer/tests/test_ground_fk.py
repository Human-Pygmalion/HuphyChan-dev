"""접지 FK 추정기가 **시뮬이 아는 답을 되찾는지** 본다.

2026-09-15. 이 시험이 있는 이유: 추정기는 실물에서 맞는지 틀린지 알 길이 없습니다.
IMU 와 다르면 둘 중 누가 틀린지 모릅니다. 그런데 시뮬에는 **참값이 있습니다**
(`imu_upvector` / `imu_ang_vel` 센서 — 정책이 먹는 바로 그 값). 그래서 시뮬에서
먼저 맞히지 못하면 실물에서 쓸 자격이 없습니다.
"""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pygviewer.ground_fk import GroundFkEstimator, SOLE_NORMAL_FOOT  # noqa: E402

XML = (Path(__file__).resolve().parents[3] / "mujoco-sim/mjlab/src/mjlab/asset_zoo"
       / "robots/pygmalion/xmls"
       / "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_loop.xml")


@pytest.fixture(scope="module")
def model():
  if not XML.exists():
    pytest.skip(f"모형이 없습니다: {XML}")
  return mujoco.MjModel.from_xml_path(str(XML))


def _adr(m):
  out = {}
  for s in "LR":
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{j}_joint")
      out[f"{s}_{j}"] = (m.jnt_qposadr[jid], m.jnt_dofadr[jid])
  return out


def _upright(n):
  """법선 `n` 을 세계 +z 로 보내는 회전."""
  z = np.array([0.0, 0.0, 1.0])
  v, c = np.cross(n, z), float(np.dot(n, z))
  s = float(np.linalg.norm(v))
  if s < 1e-9:
    return np.eye(3)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def test_sole_normal_is_the_foot_z_axis(model):
  """발바닥 상자의 `quat` 이 단위라는 것이 이 추정기의 **유일한 기하 가정**이다."""
  fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_foot_link")
  soles = [g for g in range(model.ngeom)
           if model.geom_bodyid[g] == fid
           and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_collision")]
  assert soles, "발바닥 충돌 상자를 못 찾았습니다"
  for g in soles:
    assert np.allclose(model.geom_quat[g], [1, 0, 0, 0], atol=1e-9), (
      "발바닥 상자가 발 몸체에 대해 돌아가 있습니다 — SOLE_NORMAL_FOOT 을 다시 정해야 합니다")
  assert np.allclose(SOLE_NORMAL_FOOT, [0, 0, 1])


def test_recovers_the_tilt_the_sim_knows(model):
  """관절각에서 푼 중력 방향이 시뮬 센서의 참값과 같아야 한다."""
  d = mujoco.MjData(model)
  est = GroundFkEstimator(model)
  adr = _adr(model)
  rng = np.random.default_rng(3)
  worst = 0.0
  for _ in range(12):
    hp, hr, hy, kn = (rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2),
                      rng.uniform(-0.3, 0.3), rng.uniform(-0.6, -0.1))
    d.qpos[:] = model.qpos0
    for name, v in (("hip_pitch", hp), ("hip_roll", hr), ("hip_yaw", hy), ("knee", kn)):
      d.qpos[adr[f"L_{name}"][0]] = v          # 좌우 거울: 이 모형은 이 셋이 반대 부호
      d.qpos[adr[f"R_{name}"][0]] = -v
    e = est.estimate(d.qpos.copy())
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, _upright(e["up_b"]).flatten())
    d.qpos[3:7] = q
    mujoco.mj_forward(model, d)
    truth = est.truth_from_sim(d)
    ang = np.degrees(np.arccos(np.clip(
      float(np.dot(e["proj_gravity_b"], truth["proj_gravity_b"])), -1.0, 1.0)))
    worst = max(worst, ang)
  assert worst < 1e-3, f"중력 방향이 참값과 {worst:.4f} deg 어긋납니다"


def test_gyro_matches_the_sim_gyro_sensor(model):
  """발이 붙어 있다는 가정에서 구한 각속도가 시뮬 gyro 와 같아야 한다."""
  d = mujoco.MjData(model)
  est = GroundFkEstimator(model)
  adr = _adr(model)
  rng = np.random.default_rng(11)
  worst = 0.0
  for _ in range(12):
    d.qpos[:] = model.qpos0
    qv = np.zeros(model.nv)
    for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
      v = rng.uniform(-0.4, 0.4)
      d.qpos[adr[f"L_{name}"][0]] = v
      d.qpos[adr[f"R_{name}"][0]] = -v
      qv[adr[f"L_{name}"][1]] = rng.uniform(-1.0, 1.0)
      qv[adr[f"R_{name}"][1]] = rng.uniform(-1.0, 1.0)
    e = est.estimate(d.qpos.copy(), qv)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, _upright(e["up_b"]).flatten())
    d.qpos[3:7] = q
    d.qvel[:] = qv
    d.qvel[0:3] = 0.0
    d.qvel[3:6] = e["gyro_b"]        # MuJoCo 자유관절 각속도는 **몸통 좌표**
    mujoco.mj_forward(model, d)
    truth = est.truth_from_sim(d)
    worst = max(worst, float(np.linalg.norm(e["gyro_b"] - truth["gyro_b"])))
  assert worst < 1e-9, f"각속도가 시뮬 센서와 {worst:.2e} rad/s 어긋납니다"


def test_residual_says_when_the_assumption_is_broken(model):
  """한 발만 접으면 두 발바닥이 다른 말을 하고, 그것이 숫자로 나와야 한다."""
  d = mujoco.MjData(model)
  est = GroundFkEstimator(model)
  adr = _adr(model)
  d.qpos[:] = model.qpos0
  flat = est.estimate(d.qpos.copy())["residual_deg"]
  d.qpos[adr["L_hip_roll"][0]] = 0.35          # 왼쪽만 기울인다
  broken = est.estimate(d.qpos.copy())["residual_deg"]
  assert flat < 1.0, f"기본 자세에서 잔차가 이미 {flat:.2f} deg 입니다"
  assert broken > 15.0, f"가정이 깨졌는데 잔차가 {broken:.2f} deg 밖에 안 됩니다"
