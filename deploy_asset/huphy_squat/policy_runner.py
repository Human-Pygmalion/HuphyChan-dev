#!/usr/bin/env python3
"""**로봇에서 huphy_squat 정책을 돌린다.** `numpy`, `onnxruntime`, 표준 라이브러리만 쓴다.

`pyg_rough_main_20261012/policy_runner.py` 와 같은 방식입니다 — 직전 행동과 엔코더 이력을
이 클래스가 들고 있고, 부르는 쪽은 센서값과 **목표 높이 h*** 만 줍니다.
이전 꾸러미와 달라진 것: `command=(vx, vy, wz)` 대신 **`h_star=` [m]** (필수 인자),
클래스 이름 `StandingPolicy` → `SquatPolicy`, 체크포인트가 셋(5000 기본 / 4500 / 4000).

## 쓰는 법

    from policy_runner import SquatPolicy

    pol = SquatPolicy.from_package(".", checkpoint=5000)   # deploy_contract.json + policy_5000.onnx
    pol.self_test()            # 참조 입출력 대조. 모터 켜기 전에 반드시
    pol.reset()                # 로봇을 세울 때마다

    while True:                # 50 Hz (20 ms)
        targets = pol.inference(
            gyro=imu.gyro_xyz,                     # rad/s, IMU(=몸통) 좌표계
            projected_gravity=imu.gravity_xyz,     # 단위벡터, 서 있으면 (0, 0, -1)
            joint_pos={"R_hip_pitch_joint": ..., ...},   # 모터 엔코더각 [rad], 이름 사전
            h_star=0.85,                           # 목표 골반 높이 [m], 매 주기
        )
        for name, rad in targets.items():
            motor[name].set_position(rad)

## h* 가 범위 [0.65, 0.895] m 밖이면

**잘라서(clip) 넣고, 처음 한 번만 경고를 찍고, 계속 돕니다.** 멈추지 않습니다 — 서 있는
로봇을 예외로 세우는 것이 더 위험합니다. 잘린 횟수는 `h_clipped_steps` 로 볼 수 있습니다.
NaN/Inf 는 잘 수 없으므로 예외를 냅니다.

## 안 하는 것

* 관측 정규화 — ONNX 안에 들어 있습니다.  * PD 이득 설정 — 모터 드라이버 쪽(motor_params.json).
* 안전 정지·시간초과·넘어짐 감지 — 부르는 쪽에서 감싸십시오.
* h* 의 변화율 제한 — 필요하면 부르는 쪽에서 (README "목표 높이 주는 법").

## 혼자서도 돌아갑니다

    python3 policy_runner.py --self-test                 # 기본 5000
    python3 policy_runner.py --self-test --checkpoint 4500
    python3 policy_runner.py --self-test --all           # 셋 다
    python3 policy_runner.py --dry-run 50                # 가짜 입력으로 배선 확인
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

try:                                        # 같은 폴더에서 바로 쓸 때
  from obs_adapter import ObsAdapter
except ImportError:                         # 꾸러미로 쓸 때
  from .obs_adapter import ObsAdapter       # type: ignore

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = 5000


def _sha256(path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


class SquatPolicy:
  """명세 + ONNX 하나를 묶어 **센서값 + h* → 모터 목표각** 을 낸다."""

  PERIOD_WARMUP_STEPS = 5
  PERIOD_WINDOW = 51

  def __init__(self, contract: dict, onnx_path, warn_period: bool = True):
    import onnxruntime as ort

    self.contract = contract
    self.adapter = ObsAdapter(contract)
    self.onnx_path = Path(onnx_path)
    # 이 ONNX 가 명세의 어느 체크포인트인지 해시로 찾는다 (짝이 틀리면 여기서 멈춘다)
    sha = _sha256(self.onnx_path)
    hit = [k for k, v in contract["checkpoints"].items() if v["onnx_sha256"] == sha]
    if not hit:
      raise ValueError(f"{self.onnx_path.name} (sha256 {sha[:16]}…) 는 이 명세의 어느 체크포인트와도 "
                       "맞지 않습니다. 명세와 ONNX 를 같은 꾸러미에서 가져오십시오.")
    self.checkpoint = int(hit[0])
    self.ckpt_info = contract["checkpoints"][hit[0]]

    self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
    ins, outs = self.session.get_inputs(), self.session.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
      raise ValueError(f"입력 {len(ins)} 개, 출력 {len(outs)} 개 — 1 개씩이어야 합니다")
    self.in_name, self.out_name = ins[0].name, outs[0].name
    if int(ins[0].shape[-1]) != self.adapter.obs_dim or int(outs[0].shape[-1]) != self.adapter.action_dim:
      raise ValueError("ONNX 입출력 크기가 명세와 다릅니다")

    self.dt = 1.0 / self.adapter.control_rate_hz
    self.warn_period = bool(warn_period)
    self._last_raw = np.zeros(self.adapter.action_dim, dtype=np.float64)
    self._t_prev: float | None = None
    self._gaps: list[float] = []
    self._warned = False
    self.steps = 0

  # ---------------------------------------------------------------- 만들기
  @classmethod
  def from_files(cls, contract_path, onnx_path, **kw) -> "SquatPolicy":
    c = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return cls(c, onnx_path, **kw)

  @classmethod
  def from_package(cls, folder=HERE, checkpoint: int = DEFAULT_CHECKPOINT, **kw) -> "SquatPolicy":
    """꾸러미 폴더에서 `deploy_contract.json` 과 `policy_<checkpoint>.onnx` 를 읽는다."""
    folder = Path(folder)
    c = json.loads((folder / "deploy_contract.json").read_text(encoding="utf-8"))
    key = str(int(checkpoint))
    if key not in c["checkpoints"]:
      raise ValueError(f"체크포인트 {checkpoint} 는 없습니다. 있는 것: {sorted(c['checkpoints'])}")
    return cls(c, folder / c["checkpoints"][key]["onnx"], **kw)

  # ---------------------------------------------------------------- 리셋
  def reset(self) -> None:
    """엔코더 이력·직전 행동·경고 상태를 비운다. **로봇을 세울 때마다 부를 것.**"""
    self.adapter.reset()
    self._last_raw[:] = 0.0
    self._t_prev = None
    self._gaps.clear()
    self._warned = False
    self.steps = 0

  # ---------------------------------------------------------------- 한 주기
  def inference(self, *, gyro, projected_gravity, joint_pos: dict, h_star) -> dict:
    """한 제어 주기. **모터 이름 → 목표각[rad]** 사전을 낸다.

    ``gyro``               IMU 각속도 3 개 [rad/s], IMU 좌표계 (+x 앞, +y 왼쪽, +z 위)
    ``projected_gravity``  IMU 좌표계의 중력 방향 단위벡터. 서 있으면 (0, 0, -1)
    ``joint_pos``          모터 12 개의 엔코더각 [rad], 이름을 키로 하는 사전
                           (크랭크는 크랭크 모터 각도 — 발목 각도가 아님)
    ``h_star``             목표 골반 높이 [m]. 범위 밖이면 잘라서 넣고 한 번 경고
    """
    self._check_period()
    obs = self.adapter.build(gyro=gyro, projected_gravity=projected_gravity, joint_pos=joint_pos,
                             last_action=self._last_raw, h_star=h_star)
    raw = self.session.run([self.out_name], {self.in_name: obs[None]})[0][0]
    self._last_raw = np.asarray(raw, dtype=np.float64)   # ← 다음 관측으로 되먹임
    self.steps += 1
    return self.adapter.action_to_target(self._last_raw)

  @property
  def last_raw_action(self) -> np.ndarray:
    """직전에 정책이 낸 날 것의 값 (기록용 — 되먹임은 이 클래스가 한다)."""
    return self._last_raw.copy()

  @property
  def h_clipped_steps(self) -> int:
    """reset() 이후 h* 가 범위 밖이라 잘린 주기 수."""
    return self.adapter.h_clipped_steps

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
    """명세에 든 이 체크포인트의 **참조 입출력** 으로 이 기계가 같은 답을 내는지 본다."""
    g = self.ckpt_info["golden"]
    obs = np.asarray(g["obs"], dtype=np.float32)
    want = np.asarray(g["actions"], dtype=np.float64)
    got = np.stack([self.session.run([self.out_name], {self.in_name: o[None]})[0][0] for o in obs]).astype(np.float64)
    worst = float(np.abs(got - want).max())
    if verbose:
      print(f"[policy_runner] 체크포인트 {self.checkpoint}: 참조 입출력 {len(obs)} 쌍 대조, 최대 차이 "
            f"{worst:.3e} (허용치 {tol:.0e}) → " + ("통과" if worst <= tol else "★ 불일치"))
    if worst > tol:
      raise ValueError(f"참조 입출력과 최대 {worst:.3e} 어긋납니다(허용치 {tol:.0e}). **모터를 켜지 마십시오.**")
    return worst

  # ---------------------------------------------------------------- 편의
  @property
  def joint_names(self) -> list:
    """목표각을 내는 모터 12 개, 정책이 쓰는 순서 그대로."""
    return list(self.adapter.action_joints)

  @property
  def default_pose(self) -> dict:
    """기준자세 [rad] (모터 엔코더각). 정책을 켜기 전에 로봇을 여기로 보내 두십시오."""
    return dict(self.adapter.default)

  @property
  def default_pose_height(self) -> float:
    """기준자세로 바닥에 섰을 때의 골반 높이 [m]. 시작할 때의 h* 로 쓰십시오."""
    return float(self.contract["h_star"]["default_pose_height_m"])

  @property
  def h_star_range(self) -> tuple:
    return (self.adapter.h_lo, self.adapter.h_hi)


def main(argv=None) -> int:
  import argparse

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--contract", default=str(HERE / "deploy_contract.json"))
  ap.add_argument("--onnx", help="ONNX 파일 (주면 --checkpoint 대신 이 파일)")
  ap.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT, help="5000 (기본) / 4500 / 4000")
  ap.add_argument("--all", action="store_true", help="명세의 모든 체크포인트를 대조")
  ap.add_argument("--tol", type=float, default=1e-4)
  ap.add_argument("--self-test", action="store_true", help="참조 입출력 대조만 하고 끝낸다")
  ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                  help="기준자세에 가만히 선 셈 치고 N 주기를 돌린다 (배선 확인용)")
  n = ap.parse_args(argv)

  c = json.loads(Path(n.contract).read_text(encoding="utf-8"))
  folder = Path(n.contract).resolve().parent
  if n.onnx:
    onnxs = [Path(n.onnx)]
  elif n.all:
    onnxs = [folder / v["onnx"] for _, v in sorted(c["checkpoints"].items(), key=lambda kv: -int(kv[0]))]
  else:
    onnxs = [folder / c["checkpoints"][str(n.checkpoint)]["onnx"]]
  for p in onnxs:
    pol = SquatPolicy(c, p, warn_period=False)
    print(f"[policy_runner] {p.name}: 관측 {pol.adapter.obs_dim} → 행동 {pol.adapter.action_dim}, "
          f"{pol.adapter.control_rate_hz:g} Hz, h* 범위 {pol.h_star_range} m")
    pol.self_test(tol=n.tol)
    if n.dry_run > 0:
      pol.reset()
      q = pol.default_pose
      for _ in range(n.dry_run):
        tgt = pol.inference(gyro=(0.0, 0.0, 0.0), projected_gravity=(0.0, 0.0, -1.0),
                            joint_pos=q, h_star=pol.default_pose_height)
      d = np.array([tgt[j] - pol.default_pose[j] for j in pol.joint_names])
      print(f"[policy_runner] 빈 돌리기 {n.dry_run} 주기 — 기준자세 대비 목표각 차이 최대 "
            f"{np.degrees(np.abs(d)).max():.2f}° (가짜 입력이라 배선·단위 확인용)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
