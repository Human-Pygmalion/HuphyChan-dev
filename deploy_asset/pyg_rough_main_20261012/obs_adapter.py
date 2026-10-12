#!/usr/bin/env python3
"""**로봇에서 읽은 값을 정책이 받는 관측 벡터로 바꾼다.** numpy 말고는 아무것도 안 쓴다.

2026-09-20 · 실물 배포의 마지막 빈칸. 정책(ONNX)과 관측·행동 공간 명세와 수치 동등성
검증은 준비됐는데, **로봇 쪽에서 45차원 관측을 만드는 코드가 없었습니다.**

## 왜 따로 만드나 — 시뮬레이터를 실물에 못 올린다

시뮬레이터(mjlab)의 관측 생성기는 이미 있고 검증도 돼 있습니다. 그런데 그건 **환경
객체와 MuJoCo 모형이 있어야** 돕니다. 로봇을 모는 라즈베리파이에는 둘 다 없습니다.
그래서 **명세 파일만 읽고 도는** 구현이 따로 필요합니다.

같은 것을 두 번 만드는 위험은 **대조로** 막습니다 — `verify_obs_adapter.py` 가 실제
환경을 굴리면서 두 구현이 같은 값을 내는지 확인합니다.

## 두 가지를 짐작하지 않는다

**첫째, 관절 순서.** 로봇 정의는 관측이 읽을 관절을 **정규식**으로 적습니다
(`".*_hip_.*_joint"` 처럼). 실물에서는 그걸 다시 풀 수 없으니 명세에 적힌
`joint_names` 를 씁니다. 없으면 **멈춥니다** — "행동 순서와 같겠지" 하고 짐작하면,
그럴듯하게 움직이는데 틀린 로봇이 됩니다.

그리고 이 클래스는 관절값을 **이름을 키로 하는 사전**으로 받습니다. 목록으로 받으면
부르는 쪽이 순서를 틀려도 아무도 모릅니다.

**둘째, 이력 배치.** `motor_pos_history` 24 칸은 관절 12 개 x 프레임 2 개인데, 순서가
**[오래된 프레임 12, 최신 프레임 12]** 입니다 (관절별로 묶이는 것이 아님). 값은
관절각에서 **기준자세각을 뺀 상대각**입니다. 시작 직후처럼 이력이 모자라면 **가장
오래된 프레임으로 앞을 메웁니다** — 시뮬레이터의 순환버퍼가 하는 그대로입니다.

## ★ 중력 방향의 부호 — 여기서 틀리기 쉽다

`projected_gravity` 는 **중력이 가리키는 쪽**입니다. 로봇이 똑바로 서 있으면
`(0, 0, -1)` 에 가깝습니다. 그런데 시뮬레이터의 센서 이름은 `imu_upvector` 이고 그것이
내는 값은 **위 방향** `(0, 0, +1)` 입니다 — 관측을 만들 때 **부호를 뒤집습니다**
(`mjlab/envs/mdp/observations.py`: `return -sensor.data`).

크기가 똑같고 **부호만 다르므로 길이 검사도 자릿수 검사도 통과합니다.** 2026-09-20 에
제가 대조 스크립트에서 바로 이 실수를 했고, 실제 환경과 칸 단위로 맞춰 보고서야
잡았습니다. 그래서 어댑터가 **첫 몇 번의 호출에서 직접 검사**합니다 — 로봇이 서 있는
상태에서 z 성분이 양수면 위 방향을 넣은 것이므로 **모터가 움직이기 전에 멈춥니다.**

## 정규화는 하지 않는다

명세에 관측 정규화 통계가 들어 있지만 **여기서 쓰지 않습니다.** 내보낸 정책(ONNX)이
정규화를 **자기 안에** 품고 있어서, 날 것의 관측을 그대로 넣어야 합니다. 명세의 통계는
진단용입니다. (근거: 수치 동등성 검증이 참조 관측을 **그대로** 넣어 통과합니다.)

## 쓰는 법

    ad = ObsAdapter.from_file("deploy_contract_v7b_robust_stand.json")
    ad.reset()
    obs = ad.build(gyro=..., projected_gravity=..., joint_pos={"R_hip_pitch_joint": 0.01, ...},
                   last_action=prev_raw_action, command=(0.0, 0.0, 0.0))
    raw_action = onnx_session.run(None, {"obs": obs[None]})[0][0]
    targets = ad.action_to_target(raw_action)      # 관절 목표각 [rad], 이름 사전
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ContractError(ValueError):
  """명세가 이 어댑터가 다룰 수 있는 모양이 아니다."""


class ObsAdapter:
  """명세 하나를 읽어 관측 벡터를 만든다. 상태는 관절각 이력뿐."""

  # 이 어댑터가 아는 관측 항목. 명세에 이 밖의 것이 있으면 멈춘다 —
  # 모르는 항목을 0 으로 채우면 정책은 조용히 엉뚱한 값을 받는다.
  KNOWN = {
    "builtin_sensor",                 # 각속도 (자이로)
    "projected_gravity_from_sensor",  # 중력 방향
    "joint_pos_rel",                  # 관절각 - 기준자세각 (이력 포함)
    "last_action",                    # 직전에 정책이 낸 날 것의 값
    "generated_commands",             # 속도 명령
  }

  def __init__(self, contract: dict):
    self.c = contract
    self.obs_dim = int(contract["obs_dim"])
    self.action_dim = int(contract["action_dim"])
    self.control_rate_hz = float(contract.get("control_rate_hz", 50.0))
    self.action_joints = [a["joint"] for a in contract["actions"]]
    self.scale = np.array([a["scale_rad_per_unit"] for a in contract["actions"]])
    self.offset = np.array([a["offset_rad"] for a in contract["actions"]])
    self.clip = [a.get("clip_rad") for a in contract["actions"]]
    # 기준자세: 행동의 오프셋이 곧 기준자세각이다 (use_default_offset).
    self.default = {a["joint"]: float(a["offset_rad"]) for a in contract["actions"]}

    self.terms = list(contract["observations"])
    bad = [t["term"] for t in self.terms if t["func"] not in self.KNOWN]
    if bad:
      raise ContractError(f"모르는 관측 항목: {bad}. 0 으로 채우지 않고 멈춥니다")

    self.hist_term = next((t for t in self.terms if t["func"] == "joint_pos_rel"), None)
    self.hist_len = 1
    self.hist_joints: list[str] = []
    if self.hist_term is not None:
      jn = self.hist_term.get("joint_names")
      if not jn:
        raise ContractError(
          f"{self.hist_term['term']} 에 joint_names 가 없습니다. 로봇 정의는 이 목록을 "
          "정규식으로 적으므로 실물에서는 다시 풀 수 없습니다. 행동 순서와 같다고 "
          "짐작하지 않고 멈춥니다 — deploy_contract.py 를 다시 돌려 명세를 새로 내십시오."
        )
      self.hist_joints = list(jn)
      self.hist_len = max(int(self.hist_term.get("history_frames") or 0), 1)
      want = len(self.hist_joints) * self.hist_len
      if want != int(self.hist_term["width"]):
        raise ContractError(
          f"{self.hist_term['term']}: 관절 {len(self.hist_joints)} x 프레임 "
          f"{self.hist_len} = {want} 인데 명세의 폭은 {self.hist_term['width']}"
        )
      missing = [j for j in self.hist_joints if j not in self.default]
      if missing:
        raise ContractError(f"기준자세각을 모르는 관절: {missing}")
    self._hist: list[np.ndarray] = []
    self._grav_checks_left = 0

  # ------------------------------------------------------------------ 만들기
  @classmethod
  def from_file(cls, path) -> "ObsAdapter":
    return cls(json.loads(Path(path).read_text(encoding="utf-8")))

  # 리셋 직후 몇 번은 중력 방향의 부호를 검사한다. 그때는 로봇이 서 있을 때이므로
  # z 성분이 반드시 음수다. 계속 검사하지 않는 이유는, 넘어지는 중에는 정말로 양수가
  # 될 수 있기 때문이다 — 그때 멈추면 오히려 위험하다.
  GRAVITY_CHECK_STEPS = 5

  def reset(self) -> None:
    """이력을 비운다. **로봇을 세우거나 정책을 바꿀 때마다** 부를 것."""
    self._hist = []
    self._grav_checks_left = self.GRAVITY_CHECK_STEPS

  # ------------------------------------------------------------------ 관측
  def build(self, *, gyro, projected_gravity, joint_pos: dict, last_action,
            command) -> np.ndarray:
    """한 제어 주기의 관측 벡터. 단위는 전부 라디안 · 라디안/초.

    ``joint_pos`` 는 **이름을 키로 하는 사전**입니다 — 목록이면 순서를 틀려도 아무도
    모릅니다. ``last_action`` 은 정책이 **직전에 낸 날 것의 값**이지 관절 목표가
    아닙니다 (배율과 오프셋을 입히기 전).
    """
    q = self._push_history(joint_pos)
    parts = []
    for t in self.terms:
      f = t["func"]
      if f == "builtin_sensor":
        v = np.asarray(gyro, dtype=np.float64).reshape(-1)
      elif f == "projected_gravity_from_sensor":
        v = np.asarray(projected_gravity, dtype=np.float64).reshape(-1)
        self._check_gravity_sign(v)
      elif f == "joint_pos_rel":
        v = q
      elif f == "last_action":
        v = np.asarray(last_action, dtype=np.float64).reshape(-1)
      else:                                        # generated_commands
        v = np.asarray(command, dtype=np.float64).reshape(-1)
      if v.size != int(t["width"]):
        raise ValueError(f"{t['term']}: {v.size} 개를 받았는데 {t['width']} 개여야 합니다")
      parts.append(v)
    obs = np.concatenate(parts)
    if obs.size != self.obs_dim:
      raise ValueError(f"{obs.size} 차원을 만들었는데 명세는 {self.obs_dim} 차원입니다")
    return obs.astype(np.float32)

  def _check_gravity_sign(self, v: np.ndarray) -> None:
    """서 있는데 z 가 양수면 **위 방향**을 넣은 것이다. 모터가 돌기 전에 멈춘다."""
    if self._grav_checks_left <= 0 or v.size < 3:
      return
    self._grav_checks_left -= 1
    if float(v[2]) > 0.5:
      raise ValueError(
        f"중력 방향의 z 성분이 {float(v[2]):+.3f} 입니다. 서 있는 로봇이라면 −1 에 "
        "가까워야 합니다. **위 방향(up vector)을 그대로 넣으신 것 같습니다** — "
        "부호를 뒤집어 주십시오. 시뮬레이터의 `imu_upvector` 센서는 위 방향을 내고, "
        "관측은 그것을 뒤집은 값입니다 (mjlab observations.py: `return -sensor.data`). "
        "크기가 같아 길이 검사로는 안 잡힙니다."
      )

  def _push_history(self, joint_pos: dict) -> np.ndarray:
    """이번 프레임을 넣고 **[오래된 것 … 최신]** 순으로 이어 붙인다."""
    if self.hist_term is None:
      return np.zeros(0)
    missing = [j for j in self.hist_joints if j not in joint_pos]
    if missing:
      raise ValueError(f"관절값이 없습니다: {missing}")
    frame = np.array([float(joint_pos[j]) - self.default[j] for j in self.hist_joints])
    self._hist.append(frame)
    if len(self._hist) > self.hist_len:
      self._hist = self._hist[-self.hist_len :]
    frames = list(self._hist)
    while len(frames) < self.hist_len:             # 모자라면 가장 오래된 것으로 앞을 메움
      frames.insert(0, frames[0])
    return np.concatenate(frames)

  # ------------------------------------------------------------------ 행동
  def action_to_target(self, raw_action) -> dict:
    """정책의 날 것의 값 -> 관절 목표각 [rad]. 명세의 식 그대로.

        목표각 = 오프셋 + 배율 x 행동,  그 뒤 명세의 범위로 자른다
    """
    a = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if a.size != self.action_dim:
      raise ValueError(f"행동이 {a.size} 개인데 {self.action_dim} 개여야 합니다")
    t = self.offset + self.scale * a
    out = {}
    for i, name in enumerate(self.action_joints):
      v = float(t[i])
      c = self.clip[i]
      if c is not None:
        v = min(max(v, float(c[0])), float(c[1]))
      out[name] = v
    return out
