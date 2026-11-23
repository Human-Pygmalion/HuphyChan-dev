#!/usr/bin/env python3
"""**로봇에서 huphy_rough 정책을 돌린다.** `numpy`, `onnxruntime`, 표준 라이브러리만 쓴다.

`huphy_squat/policy_runner.py` 와 같은 방식입니다 — 직전 행동과 엔코더 이력을 이 클래스가 들고 있고,
착지점 예측(FPM)과 높이 패치도 안에서 합니다. 부르는 쪽은 매 주기 네 곳의 값을 줍니다.

    IMU                 자이로, 중력 방향
    엔코더              모터 12 개 각도
    odometry estimator  골반 선속도 (골반 좌표계), 월드 좌표 골반 위치·자세
    mapper              지면 높이 (height_scan.py 의 네 가지 형태 중 하나)

선속도와 높이는 **반드시** 있어야 합니다. 선속도를 0 이나 속도 명령으로 대신하면 걷지 못합니다 (README 5 절).

## 쓰는 법

    from policy_runner import RoughPolicy

    pol = RoughPolicy.from_package()   # deploy_contract.json + policy_40000.onnx + foot_model_40000.onnx
    pol.self_test()                    # 참조 입출력 대조. 모터 켜기 전에 반드시
    pol.reset()                        # 로봇을 세울 때마다

    while True:                        # 50 Hz (20 ms)
        targets = pol.inference(
            gyro=imu.gyro_xyz,                     # rad/s, IMU(=골반) 좌표계
            projected_gravity=imu.gravity_xyz,     # 단위벡터, 서 있으면 (0, 0, -1)
            joint_pos={"R_hip_pitch_joint": ..., ...},   # 모터 엔코더각 [rad], 이름 사전
            base_lin_vel=odom.v_body,              # 골반 선속도 [m/s], 골반 좌표계
            base_pos=odom.pos, base_quat=odom.quat,  # 월드 좌표 골반 원점 [m], 자세 (w, x, y, z)
            height=mapper.height,                  # mapper 출력 (height_scan.py: WorldGrid 등)
            command=(vx, vy, wz),                  # 속도 명령, 골반 좌표계
        )
        for name, rad in targets.items():
            motor[name].set_position(rad)

## 안 하는 것

* 관측 정규화 — ONNX 안에 들어 있습니다.  * PD 이득 설정 — 모터 드라이버 쪽(motor_params.json).
* 안전 정지·시간초과·넘어짐 감지 — 부르는 쪽에서 감싸십시오.
* odometry 추정과 지도 만들기 — estimator 와 mapper 쪽 일입니다.

## 혼자서도 돌아갑니다

    python3 policy_runner.py --self-test
    python3 policy_runner.py --dry-run 50         # 가짜 입력으로 배선 확인
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

try:                                        # 같은 폴더에서 바로 쓸 때
  from height_scan import FootholdScan, RobotGrid
  from obs_adapter import ObsAdapter
except ImportError:                         # 꾸러미로 쓸 때
  from .height_scan import FootholdScan, RobotGrid  # type: ignore
  from .obs_adapter import ObsAdapter  # type: ignore

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = 40000


def _sha256(path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


class RoughPolicy:
  """명세 + ONNX 두 개(정책, FPM)를 묶어 **센서값 + 명령 + 높이 → 모터 목표각** 을 낸다."""

  PERIOD_WARMUP_STEPS = 5
  PERIOD_WINDOW = 51

  def __init__(self, contract: dict, folder, checkpoint: int = DEFAULT_CHECKPOINT, warn_period: bool = True):
    import onnxruntime as ort

    self.contract = contract
    key = str(int(checkpoint))
    if key not in contract["checkpoints"]:
      raise ValueError(f"체크포인트 {checkpoint} 는 없습니다. 있는 것: {sorted(contract['checkpoints'])}")
    self.checkpoint = int(checkpoint)
    self.ckpt_info = contract["checkpoints"][key]
    self.adapter = ObsAdapter(contract)
    self.scan = FootholdScan(contract)
    folder = Path(folder)
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    self.sessions = {}
    for net in ("policy", "fpm"):            # 짝이 틀리면 여기서 멈춘다
      path = folder / self.ckpt_info[f"{net}_onnx"]
      sha = _sha256(path)
      if sha != self.ckpt_info[f"{net}_onnx_sha256"]:
        raise ValueError(f"{path.name} (sha256 {sha[:16]}…) 가 명세와 맞지 않습니다. 명세와 ONNX 를 같은 꾸러미에서 가져오십시오.")
      self.sessions[net] = ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])
    pi_in, fpm_in = self.sessions["policy"].get_inputs()[0], self.sessions["fpm"].get_inputs()[0]
    if int(pi_in.shape[-1]) != self.adapter.obs_dim or int(fpm_in.shape[-1]) != self.adapter.fpm_dim:
      raise ValueError("ONNX 입력 크기가 명세와 다릅니다")
    self._pi_in, self._fpm_in = pi_in.name, fpm_in.name

    self.dt = 1.0 / self.adapter.control_rate_hz
    self.warn_period = bool(warn_period)
    self._last_raw = np.zeros(self.adapter.action_dim, dtype=np.float64)
    self._t_prev: float | None = None
    self._gaps: list[float] = []
    self._warned = False
    self.steps = 0
    self.last_scan_info: dict | None = None
    self.last_obs: np.ndarray | None = None

  # ---------------------------------------------------------------- 만들기
  @classmethod
  def from_package(cls, folder=HERE, checkpoint: int = DEFAULT_CHECKPOINT, **kw) -> "RoughPolicy":
    """꾸러미 폴더에서 `deploy_contract.json` 과 그 체크포인트의 ONNX 두 개를 읽는다."""
    folder = Path(folder)
    c = json.loads((folder / "deploy_contract.json").read_text(encoding="utf-8"))
    return cls(c, folder, checkpoint, **kw)

  # ---------------------------------------------------------------- 리셋
  def reset(self) -> None:
    """엔코더 이력·직전 행동·경고 상태를 비운다. **로봇을 세울 때마다 부를 것.**"""
    self.adapter.reset()
    self.scan.reset()
    self._last_raw[:] = 0.0
    self._t_prev = None
    self._gaps.clear()
    self._warned = False
    self.steps = 0
    self.last_scan_info = None
    self.last_obs = None

  # ---------------------------------------------------------------- 한 주기
  def inference(self, *, gyro, projected_gravity, joint_pos: dict, base_lin_vel, height, base_pos=None, base_quat=None,
                command=(0.0, 0.0, 0.0)) -> dict:
    """한 제어 주기. **모터 이름 → 목표각[rad]** 사전을 낸다.

    ``gyro``               IMU 각속도 3 개 [rad/s], IMU 좌표계 (+x 앞, +y 왼쪽, +z 위)
    ``projected_gravity``  IMU 좌표계의 중력 방향 단위벡터. 서 있으면 (0, 0, -1)
    ``joint_pos``          모터 12 개의 엔코더각 [rad], 이름을 키로 하는 사전 (크랭크는 크랭크 모터 각)
    ``base_lin_vel``       odometry: 골반(base_link 원점) 선속도 [m/s], 골반 좌표계. 월드 좌표면 obs_adapter.world_to_body
    ``height``             mapper 출력: WorldGrid, 월드 좌표 함수, 33×21 로봇 중심 격자, 로봇 중심 함수 — height_scan.py
    ``base_pos``, ``base_quat``  odometry: 월드 좌표 골반 원점 [m] 과 자세 (w, x, y, z), mapper 지도와 같은 좌표계.
                           월드 좌표 지도·함수에 필요 (로봇 중심 격자만 쓰면 없어도 됨)
    ``command``            (vx [m/s], vy [m/s], wz [rad/s]). 학습 범위 밖이면 잘라서 넣고 한 번 경고
    """
    self._check_period()
    prop, fpm_in = self.adapter.build(gyro=gyro, projected_gravity=projected_gravity, joint_pos=joint_pos,
                                      base_lin_vel=base_lin_vel, command=command, last_action=self._last_raw)
    fh = self.sessions["fpm"].run(None, {self._fpm_in: fpm_in[None]})[0].reshape(2, 2).astype(np.float64)
    scan, info = self.scan.compute(fh, self.scan.as_source(height), base_pos, base_quat)
    obs = np.concatenate((prop, scan)).astype(np.float32)
    raw = self.sessions["policy"].run(None, {self._pi_in: obs[None]})[0][0]
    self._last_raw = np.asarray(raw, dtype=np.float64)   # ← 다음 관측으로 되먹임
    self.last_scan_info, self.last_obs = info, obs
    self.steps += 1
    return self.adapter.action_to_target(self._last_raw)

  @property
  def last_raw_action(self) -> np.ndarray:
    """직전에 정책이 낸 날 것의 값 (기록용 — 되먹임은 이 클래스가 한다)."""
    return self._last_raw.copy()

  @property
  def last_footholds(self) -> np.ndarray | None:
    """직전 주기에 FPM 이 낸 다음 착지점 (2, 2) [m], 골반 yaw 좌표, (R, L). 기록·표시용."""
    return None if self.last_scan_info is None else self.last_scan_info["footholds"].copy()

  @property
  def command_clipped_steps(self) -> int:
    return self.adapter.cmd_clipped_steps

  @property
  def scan_nan_count(self) -> int:
    """reset() 이후 높이가 NaN/Inf 라 같은 패치의 중앙값 (또는 직전 값) 으로 채운 칸 수."""
    return self.scan.nan_count

  def _check_period(self) -> None:
    if not self.warn_period:
      return
    now = time.monotonic()
    if self.steps < self.PERIOD_WARMUP_STEPS:
      self._t_prev = now
      return
    if self._t_prev is not None and not self._warned:
      self._gaps.append(now - self._t_prev)
      if len(self._gaps) > self.PERIOD_WINDOW:
        self._gaps.pop(0)
      if len(self._gaps) == self.PERIOD_WINDOW:      # 창의 중앙값 — 한 번의 딸꾹질은 무시
        mid = sorted(self._gaps)[self.PERIOD_WINDOW // 2]
        if abs(mid - self.dt) > 0.2 * self.dt:
          self._warned = True
          print(f"[policy_runner] 제어 주기 중앙값 {mid * 1e3:.1f} ms — 명세는 {self.dt * 1e3:.1f} ms "
                f"({self.adapter.control_rate_hz:g} Hz). 정책은 이 주기를 전제로 학습했습니다.", flush=True)
    self._t_prev = now

  # ---------------------------------------------------------------- 자가진단
  def self_test(self, tol: float = 1e-4, verbose: bool = True) -> float:
    """명세에 든 **참조 입출력** 으로 이 기계가 학습 기계와 같은 답을 내는지 본다.

    1. 정책 ONNX: 관측 → 행동 쌍.  2. FPM ONNX: 입력 → 착지점 쌍.
    3. 이 runner 전체: 센서값·명령·로봇 중심 높이 격자 연속 입력 → 모터 목표각 (어댑터와 높이 패치까지 포함).

    내부 상태를 지우므로 **정책을 켜기 전에** 부르고, 그다음 reset() 을 부르십시오 (돌리는 도중에 부르지 말 것).
    """
    g = self.ckpt_info["golden"]
    worst = {}
    obs = np.asarray(g["policy"]["obs"], dtype=np.float32)
    want = np.asarray(g["policy"]["actions"], dtype=np.float64)
    got = np.stack([self.sessions["policy"].run(None, {self._pi_in: o[None]})[0][0] for o in obs])
    worst["정책"] = float(np.abs(got - want).max())
    x = np.asarray(g["fpm"]["input"], dtype=np.float32)
    want = np.asarray(g["fpm"]["footholds"], dtype=np.float64)
    got = self.sessions["fpm"].run(None, {self._fpm_in: x})[0].reshape(len(x), -1)
    worst["FPM"] = float(np.abs(got - want).max())
    seq = g["sequence"]
    warn_period = self.warn_period
    self.reset()
    self.warn_period = False
    self.adapter._prev = np.asarray(seq["start"]["motor_pos_prev"], dtype=np.float64)
    self._last_raw = np.asarray(seq["start"]["last_action"], dtype=np.float64)
    self.adapter._grav_checks_left = 0
    err = 0.0
    for s in seq["steps"]:
      tgt = self.inference(gyro=s["gyro"], projected_gravity=s["projected_gravity"],
                           joint_pos=dict(zip(self.joint_names, s["joint_pos"])), base_lin_vel=s["base_lin_vel"],
                           command=s["command"], height=RobotGrid(np.asarray(s["robot_grid"])))
      err = max(err, float(np.abs(np.array([tgt[j] for j in self.joint_names]) - np.asarray(s["targets"])).max()))
    worst["runner 전체"] = err
    self.reset()
    self.warn_period = warn_period
    w = max(worst.values())
    if verbose:
      detail = ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
      print(f"[policy_runner] 체크포인트 {self.checkpoint}: 참조 입출력 대조 ({detail}), 허용치 {tol:.0e} → "
            + ("통과" if w <= tol else "★ 불일치"))
    if w > tol:
      raise ValueError(f"참조 입출력과 최대 {w:.3e} 어긋납니다(허용치 {tol:.0e}). **모터를 켜지 마십시오.**")
    return w

  # ---------------------------------------------------------------- 편의
  @property
  def joint_names(self) -> list:
    """목표각을 내는 모터 12 개, 정책이 쓰는 순서 그대로."""
    return list(self.adapter.action_joints)

  @property
  def default_pose(self) -> dict:
    """기본자세 [rad] (모터 엔코더각). 정책을 켜기 전에 로봇을 여기로 보내 두십시오."""
    return dict(self.adapter.default)

  @property
  def default_pose_height(self) -> float:
    """기본자세로 평지에 섰을 때의 골반 높이 [m] (학습 리셋 높이)."""
    return float(self.contract["foothold_scan"]["flat_relative_height_m"])

  @property
  def command_range(self) -> tuple:
    return (tuple(self.adapter.cmd_lo), tuple(self.adapter.cmd_hi))


def main(argv=None) -> int:
  import argparse

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--folder", default=str(HERE))
  ap.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
  ap.add_argument("--tol", type=float, default=1e-4)
  ap.add_argument("--self-test", action="store_true", help="참조 입출력 대조만 하고 끝낸다")
  ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                  help="기본자세에 가만히 선 셈 치고 N 주기를 돌린다 (배선 확인용)")
  n = ap.parse_args(argv)
  pol = RoughPolicy.from_package(n.folder, n.checkpoint, warn_period=False)
  lo, hi = pol.command_range
  print(f"[policy_runner] 체크포인트 {pol.checkpoint}: 관측 {pol.adapter.obs_dim} (FPM {pol.adapter.fpm_dim}) → 행동 "
        f"{pol.adapter.action_dim}, {pol.adapter.control_rate_hz:g} Hz, 명령 범위 vx {lo[0]}~{hi[0]}, vy {lo[1]}~{hi[1]}, "
        f"wz {lo[2]}~{hi[2]}")
  pol.self_test(tol=n.tol)
  if n.dry_run > 0:
    pol.reset()
    q = pol.default_pose
    for _ in range(n.dry_run):
      tgt = pol.inference(gyro=(0.0, 0.0, 0.0), projected_gravity=(0.0, 0.0, -1.0), joint_pos=q,
                          base_lin_vel=(0.0, 0.0, 0.0), command=(0.0, 0.0, 0.0),
                          height=RobotGrid(np.full((33, 21), pol.default_pose_height)))   # 평지를 본 가짜 mapper
    d = np.array([tgt[j] - pol.default_pose[j] for j in pol.joint_names])
    print(f"[policy_runner] 빈 돌리기 {n.dry_run} 주기 — 기본자세 대비 목표각 차이 최대 {np.degrees(np.abs(d)).max():.2f}°, "
          f"착지점 R {np.round(pol.last_footholds[0], 3).tolist()} L {np.round(pol.last_footholds[1], 3).tolist()} m "
          "(가짜 입력이라 배선·단위 확인용)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
