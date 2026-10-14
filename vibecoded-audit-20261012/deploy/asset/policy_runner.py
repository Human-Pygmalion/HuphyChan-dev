#!/usr/bin/env python3
"""**로봇에서 정책을 한 줄로 돌린다.** `numpy` 와 `onnxruntime` 말고는 아무것도 안 쓴다.

2026-09-21 · `obs_adapter.py` 는 "센서값 → 관측 45 개" 까지만 했습니다. 그 뒤에 남는
일 — 신경망에 넣고, 나온 값을 관절 목표각으로 바꾸고, **그 값을 다음 관측에 다시
넣는 것** — 은 로봇을 모는 쪽이 매번 직접 짜야 했습니다. 이 파일이 그 마지막 조각입니다.

## 이 파일이 실제로 막아 주는 실수

**직전 행동(`last_action`)을 되먹이는 것.** 관측 45 칸 중 12 칸은 *정책이 직전에 낸
값* 입니다. 관절 목표각이 아니라 **배율과 오프셋을 입히기 전의 날 것** 입니다. 부르는
쪽이 이걸 직접 들고 다니면, 거의 반드시 둘 중 하나를 틀립니다.

* 목표각을 대신 넣는다 → 크기가 12 로 같아서 **아무 오류도 안 납니다.** 정책은 평생 본
  적 없는 값을 받고, 그럴듯하지만 틀린 동작을 합니다.
* 넣는 것을 잊고 0 으로 둔다 → 정책이 "나는 방금 아무것도 안 했다" 고 믿습니다.

그래서 이 클래스가 **직전 행동을 안에 들고 있습니다.** 부르는 쪽은 센서값만 주면 됩니다.

## 쓰는 법 (이게 전부입니다)

    from policy_runner import StandingPolicy

    pol = StandingPolicy.from_files("deploy_contract_v7b_robust_stand.json",
                                    "policy.onnx")
    pol.self_test()          # 참조 입출력 대조. 한 번은 반드시 부를 것
    pol.reset()              # 로봇을 세울 때마다

    while True:              # 50 Hz
        targets = pol.inference(
            gyro=imu.gyro_xyz,                     # rad/s, 몸통 기준
            projected_gravity=imu.gravity_xyz,     # 단위벡터, 서 있으면 (0,0,-1)
            joint_pos={"R_hip_pitch_joint": 0.01, ...},   # rad, **이름 사전**
            command=(0.0, 0.0, 0.0),               # 제자리 서기
        )
        for name, rad in targets.items():
            motor[name].set_position(rad)

## 제어 주기를 지켜야 합니다

명세에 `control_rate_hz` 가 있습니다(지금 50 Hz = 0.02 초). 정책은 **자기가 20 ms 마다
불린다고 가정하고** 학습했습니다. 30 ms 마다 부르면 정책이 보는 세상이 느려진 것과
같아서, 넘어지지는 않더라도 학습 때와 다르게 움직입니다. `warn_period` 가 켜져 있으면
주기가 20 % 넘게 어긋날 때 한 번 알려 줍니다(멈추지는 않습니다 — 달리는 로봇을 예외로
세우는 것이 더 위험합니다).

## 안 하는 것

* **관측 정규화를 하지 않습니다.** 내보낸 ONNX 가 정규화를 자기 안에 품고 있습니다.
  명세의 `obs_normalizer_mean/std` 는 진단용입니다.
* **PD 이득을 정하지 않습니다.** 이 파일은 관절 *목표각* 까지만 냅니다. 이득은 로봇
  쪽 설정입니다.
* **안전 정지를 넣지 않습니다.** 넣어 달라고 하신 적이 없고, 임의로 넣으면 진짜
  필요할 때 왜 멈췄는지 아무도 모릅니다. 필요하면 부르는 쪽에서 감싸십시오.

## 혼자서도 돌아갑니다

    python3 policy_runner.py --contract <명세.json> --onnx <정책.onnx> --self-test
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:                                        # 같은 폴더에서 바로 쓸 때
  from obs_adapter import ObsAdapter
except ImportError:                         # 꾸러미로 쓸 때
  from .obs_adapter import ObsAdapter       # type: ignore


class StandingPolicy:
  """명세 + ONNX 하나를 묶어 **센서값 → 관절 목표각** 을 낸다."""

  def __init__(self, contract: dict, onnx_path: str | Path,
               warn_period: bool = True):
    import onnxruntime as ort                            # 여기서만 필요하다

    self.contract = contract
    self.adapter = ObsAdapter(contract)
    self.session = ort.InferenceSession(
      str(onnx_path), providers=["CPUExecutionProvider"])

    ins, outs = self.session.get_inputs(), self.session.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
      raise ValueError(f"입력 {len(ins)} 개, 출력 {len(outs)} 개 — 1 개씩이어야 합니다")
    self.in_name, self.out_name = ins[0].name, outs[0].name
    want_in, want_out = self.adapter.obs_dim, self.adapter.action_dim
    got_in, got_out = int(ins[0].shape[-1]), int(outs[0].shape[-1])
    if got_in != want_in or got_out != want_out:
      raise ValueError(
        f"ONNX 는 관측 {got_in} → 행동 {got_out} 인데 명세는 {want_in} → {want_out} "
        "입니다. **명세와 정책이 서로 다른 학습에서 나온 것입니다** — 짝을 맞추십시오"
      )

    self.dt = 1.0 / float(self.adapter.control_rate_hz)
    self.warn_period = bool(warn_period)
    self._last_raw = np.zeros(self.adapter.action_dim, dtype=np.float64)
    self._t_prev: float | None = None
    self._gaps: list[float] = []
    self._warned = False
    self.steps = 0

  # ---------------------------------------------------------------- 만들기
  @classmethod
  def from_files(cls, contract_path, onnx_path, **kw) -> "StandingPolicy":
    c = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return cls(c, onnx_path, **kw)

  # ---------------------------------------------------------------- 리셋
  def reset(self) -> None:
    """관절각 이력과 직전 행동을 비운다. **로봇을 세울 때마다 부를 것.**

    안 부르면 직전에 서 있던 기억을 그대로 들고 시작합니다 — 정책이 있지도 않은
    움직임을 되받아 첫 몇 주기 동안 엉뚱한 목표를 냅니다.
    """
    self.adapter.reset()
    self._last_raw[:] = 0.0
    self._t_prev = None
    self._gaps.clear()
    self._warned = False
    self.steps = 0

  # ---------------------------------------------------------------- 한 주기
  def inference(self, *, gyro, projected_gravity, joint_pos: dict,
                command=(0.0, 0.0, 0.0)) -> dict:
    """한 제어 주기. **관절 이름 → 목표각[rad]** 사전을 낸다.

    ``gyro``               몸통 기준 각속도 3 개 [rad/s]
    ``projected_gravity``  몸통 기준 중력 방향 단위벡터 3 개. 서 있으면 (0, 0, −1).
                           IMU 가 **위 방향** 을 주면 부호를 뒤집어서 넣으십시오.
    ``joint_pos``          관절각 [rad], **이름을 키로 하는 사전**
    ``command``            (앞뒤, 좌우, 돌기) 속도 명령. 제자리 서기는 (0, 0, 0).
    """
    self._check_period()
    obs = self.adapter.build(
      gyro=gyro, projected_gravity=projected_gravity, joint_pos=joint_pos,
      last_action=self._last_raw, command=command)
    raw = self.session.run(
      [self.out_name], {self.in_name: obs[None].astype(np.float32)})[0][0]
    self._last_raw = np.asarray(raw, dtype=np.float64)   # ← 다음 관측으로 되먹임
    self.steps += 1
    return self.adapter.action_to_target(self._last_raw)

  @property
  def last_raw_action(self) -> np.ndarray:
    """직전에 정책이 낸 **날 것의** 값. 기록용 — 되먹임은 이 클래스가 알아서 한다."""
    return self._last_raw.copy()

  # 처음 몇 주기는 재지 않는다. ONNX 첫 호출이 자기 준비를 하느라 느려서, 여기서
  # 재면 멀쩡한 고리에도 경고가 뜹니다 (2026-09-21 에 예제 고리에서 25.7 ms 로 헛경고).
  PERIOD_WARMUP_STEPS = 5
  PERIOD_WINDOW = 51          # 중앙값을 볼 창 (홀수)

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
      # **한 번의 딸꾹질로는 말하지 않는다.** 부하가 걸린 기계에서는 150 주기에 한 번쯤
      # 4 ms 씩 밀리는 것이 정상이고, 그때마다 경고하면 아무도 안 읽게 됩니다. 창의
      # **중앙값**을 보면 한 번의 이상치는 무시되고 주기 자체가 틀린 것만 남습니다.
      if len(self._gaps) == self.PERIOD_WINDOW:
        mid = sorted(self._gaps)[self.PERIOD_WINDOW // 2]
        if abs(mid - self.dt) > 0.2 * self.dt:
          self._warned = True                            # 한 번만 말한다
          print(f"[policy_runner] 제어 주기의 중앙값이 {mid * 1e3:.1f} ms 입니다 — "
                f"명세는 {self.dt * 1e3:.1f} ms({self.adapter.control_rate_hz:g} Hz) "
                "입니다. 정책은 이 주기를 전제로 학습했습니다.", flush=True)
    self._t_prev = now

  # ---------------------------------------------------------------- 자가진단
  def self_test(self, tol: float = 1e-4, verbose: bool = True) -> float:
    """명세에 든 **참조 입출력 벡터** 로 이 실행환경이 같은 답을 내는지 본다.

    학습 기계에서 만든 정책이 로봇의 CPU·ONNX 판에서도 같은 값을 내는지는 **돌려
    봐야** 압니다. 명세에는 관측 16 개와 그 정답 행동 16 개가 들어 있습니다.
    여기서 어긋나면 모터를 켜기 전에 멈추는 것이 맞습니다.
    """
    g = self.contract.get("golden")
    if not g:
      raise ValueError("명세에 참조 입출력(golden)이 없습니다 — 대조할 것이 없습니다")
    obs = np.asarray(g["obs"], dtype=np.float32)
    want = np.asarray(g["actions"], dtype=np.float64)
    got = np.stack([
      self.session.run([self.out_name], {self.in_name: o[None]})[0][0] for o in obs
    ]).astype(np.float64)
    worst = float(np.abs(got - want).max())
    if verbose:
      ok = worst <= tol
      print(f"[policy_runner] 참조 입출력 {len(obs)} 쌍 대조: 최대 차이 "
            f"{worst:.3e} (허용치 {tol:.0e}) → " + ("통과" if ok else "★ 불일치"))
    if worst > tol:
      raise ValueError(
        f"참조 입출력과 최대 {worst:.3e} 어긋납니다(허용치 {tol:.0e}). "
        "명세와 ONNX 의 짝이 틀렸거나, 이 기계의 ONNX 판이 다릅니다. "
        "**모터를 켜지 마십시오.**")
    return worst

  # ---------------------------------------------------------------- 편의
  @property
  def joint_names(self) -> list:
    """정책이 목표각을 내는 관절, **정책이 쓰는 순서 그대로.**"""
    return list(self.adapter.action_joints)

  @property
  def default_pose(self) -> dict:
    """기준자세 [rad]. 정책을 켜기 전에 로봇을 여기로 보내 두십시오."""
    return dict(self.adapter.default)


def main(argv=None) -> int:
  import argparse

  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--contract", required=True, help="deploy_contract_*.json")
  ap.add_argument("--onnx", required=True, help="내보낸 정책 .onnx")
  ap.add_argument("--tol", type=float, default=1e-4)
  ap.add_argument("--self-test", action="store_true",
                  help="참조 입출력 대조만 하고 끝낸다")
  ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                  help="기준자세에 가만히 선 셈 치고 N 주기를 돌려 본다 "
                       "(로봇 없이 배선만 확인)")
  n = ap.parse_args(argv)

  pol = StandingPolicy.from_files(n.contract, n.onnx, warn_period=False)
  print(f"[policy_runner] 관측 {pol.adapter.obs_dim} → 행동 {pol.adapter.action_dim}, "
        f"{pol.adapter.control_rate_hz:g} Hz, 관절 {len(pol.joint_names)} 개")
  pol.self_test(tol=n.tol)

  if n.dry_run > 0:
    pol.reset()
    q = pol.default_pose                       # 기준자세에 가만히 있는 셈
    for _ in range(n.dry_run):
      tgt = pol.inference(gyro=(0.0, 0.0, 0.0),
                          projected_gravity=(0.0, 0.0, -1.0),
                          joint_pos=q, command=(0.0, 0.0, 0.0))
    d = np.array([tgt[j] - pol.default_pose[j] for j in pol.joint_names])
    print(f"[policy_runner] 빈 돌리기 {n.dry_run} 주기 — 기준자세 대비 목표각 차이 "
          f"최대 {np.abs(d).max() * 180 / np.pi:.2f}°, 평균 "
          f"{np.abs(d).mean() * 180 / np.pi:.2f}°")
    print("  (로봇이 정말로 기준자세에 가만히 있다고 거짓말한 값이라 "
          "물리적으로 옳은 값은 아닙니다 — 배선과 단위가 맞는지만 봅니다)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
