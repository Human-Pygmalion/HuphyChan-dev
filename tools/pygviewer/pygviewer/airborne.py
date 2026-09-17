"""**로봇이 들려 있는지 알아채고, 들렸으면 다리를 붙잡아 둔다.**

2026-09-17 · 사용자 지시: *"정책 실행하다가 로봇을 공중에 들 때가 있는데, 이러면
다리가 막 흔들리네. 가만히 있도록 할수있겠나?"* — 사용자가 고른 방법: 토크로 감지해
고정(4) + 학습으로도 고침(2).

## 왜 "토크가 얼마 이하면 공중" 이 안 되나

처음엔 그렇게 하려고 했습니다. 재 보니 안 됩니다 (docs/148 §5). **우리 서기 자세는
다리가 거의 곧아서** 바닥반력이 관절축 가까이를 지나갑니다 — 모멘트 팔이 짧아서
**서 있을 때 관절 토크가 원래 작습니다** (엉덩이 롤 1.8 N·m, 무릎 1.8). 들면 0.5 와
0.3 으로 떨어지는데, 그 차이가 모터 자체 마찰 수준입니다. 게다가 자세가 바뀌면
두 값이 다 바뀌므로, 한 번 정한 숫자는 다른 자세에서 틀립니다.

## 그래서 이렇게 합니다 — **지금 자세에서 모형에게 물어본다**

숫자를 미리 박지 않습니다. 매 순간 **지금 관절각 그대로** 모형에게 묻습니다.

> "이 로봇이 지금 이 자세로 **허공에 매달려** 있다면, 모터가 내야 하는 힘은 얼마인가?"

이건 모형이 정확히 답할 수 있습니다 — 중력과 링크 무게만의 문제이고 바닥이 없으니
미지수가 없습니다. 그 답을 `tau_air(q)` 라고 합시다.

그리고 실물에서 오는 토크와 견줍니다.

* 실제로 매달려 있으면 두 값이 **거의 같습니다.**
* 발로 딛고 있으면 바닥이 몸무게를 받쳐 주는 만큼 **달라집니다.**

문턱값은 "얼마나 달라야 하나" 하나뿐이고, 그건 **모터 잡음의 크기**라 자세와 무관합니다.
자세가 바뀌면 `tau_air(q)` 가 같이 따라가므로 비교가 계속 유효합니다.

## 들렸다고 판단하면

관절 목표를 **그 순간 값으로 얼어붙입니다**(`hold`). 정책은 계속 돌지만 그 출력을
모터로 안 보냅니다. 내려놓아서 다시 땅으로 판정되면 **천천히** 정책 출력으로
돌아갑니다 — 갑자기 바꾸면 그 자체가 충격입니다.

## 함부로 켜지지 않게

* **연속 판정**(`dwell`): 한 번 넘었다고 바로 안 바꿉니다. 여러 틱 연속이어야 합니다.
* **되돌림 여유**(hysteresis): 들림으로 갈 때와 풀릴 때의 기준이 다릅니다. 경계에서
  깜빡이는 것을 막습니다.
* **명령이 0 일 때만**: 걷는 중에는 아예 안 봅니다. 걷는 동안 발이 뜨는 것은 정상이고,
  그때 얼어붙으면 넘어집니다.

이 세 가지가 없으면 **땅에 서 있는데 얼어붙어 쓰러지는** 쪽이 훨씬 위험합니다.
"""

from __future__ import annotations

import numpy as np


class HangingTorqueModel:
  """지금 자세에서 "허공에 매달렸다면 내야 할 토크" 를 낸다.

  몸통을 세계에 **붙들어 매고**(들고 있는 사람의 손) 바닥을 없앤 모형을 따로 만들어
  두고, 매번 관절각만 갈아 끼워 물어봅니다. 원래 모형을 건드리지 않으므로 화면에
  보이는 시뮬레이션과 서로 방해하지 않습니다.

  속도와 가속도는 **0 으로 둡니다.** 우리가 알고 싶은 것은 "이 자세를 유지하는 데
  드는 힘" 이고, 흔들리는 중의 관성력은 오히려 비교를 흐립니다.
  """

  def __init__(self, mj_model, act_names: list[str]):
    import mujoco

    self._mj = mujoco
    self.m = mj_model
    self.d = mujoco.MjData(mj_model)
    self.act_names = list(act_names)
    # 바닥과의 접촉을 끄는 대신, **접촉이 생길 수 없는 높이로 띄워서** 묻습니다.
    # geom 을 지우면 관성이 바뀔 수 있어서(자기충돌용 geom 이 질량을 나르지는
    # 않지만 모형을 고치는 일은 언제나 뒤탈이 있습니다) 건드리지 않습니다.
    self.lift = 5.0
    self._jadr = {}
    for n in self.act_names:
      j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
      if j < 0:
        j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
      if j < 0:
        raise ValueError(f"{n} 관절이 이 모형에 없습니다")
      self._jadr[n] = (int(mj_model.jnt_qposadr[j]), int(mj_model.jnt_dofadr[j]))

  def tau(self, q: dict[str, float]) -> np.ndarray:
    """관절각 -> 매달렸을 때 필요한 관절 토크 [N*m], `act_names` 순서."""
    mujoco = self._mj
    d = self.d
    mujoco.mj_resetData(self.m, d)
    if self.m.nq > len(self._jadr):                   # 자유 몸통이 있으면 띄운다
      d.qpos[2] += self.lift
    for n, v in q.items():
      adr = self._jadr.get(n)
      if adr is not None and v is not None and np.isfinite(v):
        d.qpos[adr[0]] = float(v)
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_forward(self.m, d)                      # 폐루프 구속을 풀어 준다
    mujoco.mj_inverse(self.m, d)
    return np.array([d.qfrc_inverse[self._jadr[n][1]] for n in self.act_names])


class AirborneDetector:
  """토크를 보고 "들렸다/딛고 있다" 를 가른다. 판단만 하고 명령은 안 만든다.

  ``enter`` 는 들림으로 넘어가는 기준, ``exit_`` 는 풀리는 기준입니다. 둘이 다른
  이유는 경계에서 깜빡이지 않게 하려는 것입니다(되돌림 여유). 단위는 관절 하나당
  평균 토크 차이 [N*m] 입니다 — 관절 수가 달라져도 뜻이 안 변합니다.
  """

  def __init__(self, hanging: HangingTorqueModel, enter: float = 1.0,
               exit_: float = 2.0, dwell: int = 10):
    if not (0.0 < enter < exit_):
      raise ValueError("enter 는 exit_ 보다 작아야 합니다 (되돌림 여유)")
    self.h = hanging
    self.enter, self.exit_, self.dwell = float(enter), float(exit_), int(dwell)
    self.airborne = False
    self._count = 0
    self.residual = float("nan")

  def update(self, q: dict[str, float], tau: dict[str, float],
             command_is_zero: bool = True) -> bool:
    """한 틱. 돌려주는 것은 **지금 들려 있다고 보는가**.

        residual = mean |tau_실측 - tau_매달림(q)|

    걷는 중(`command_is_zero=False`)에는 **아예 판단하지 않고** 무조건 땅으로 둡니다 —
    걷는 동안 발이 뜨는 것은 정상이고, 그때 얼어붙으면 넘어집니다.
        """
    if not command_is_zero:
      self.airborne, self._count, self.residual = False, 0, float("nan")
      return False
    want = self.h.tau(q)
    got = np.array([tau.get(n, np.nan) for n in self.h.act_names], dtype=float)
    ok = np.isfinite(got) & np.isfinite(want)
    if ok.sum() == 0:
      return self.airborne                           # 값이 없으면 판단을 바꾸지 않는다
    self.residual = float(np.abs(got[ok] - want[ok]).mean())
    thr = self.exit_ if self.airborne else self.enter
    hit = (self.residual < thr) if not self.airborne else (self.residual < thr)
    if hit != self.airborne:
      self._count += 1
      if self._count >= self.dwell:
        self.airborne = hit
        self._count = 0
    else:
      self._count = 0
    return self.airborne


class HoldOnAirborne:
  """들렸으면 관절 목표를 **얼려 두고**, 내려놓으면 천천히 정책으로 돌아간다.

  ``blend_s`` 동안 얼린 값에서 정책 값으로 선형으로 넘어갑니다. 갑자기 바꾸면 그
  자체가 로봇에 충격입니다 — 특히 막 땅에 닿은 순간에.
  """

  def __init__(self, detector: AirborneDetector, blend_s: float = 0.5,
               dt: float = 0.02):
    self.det = detector
    self.steps = max(1, int(round(blend_s / dt)))
    self._held: np.ndarray | None = None
    self._blend = 0

  def apply(self, policy_target: np.ndarray, q: dict[str, float],
            tau: dict[str, float], command_is_zero: bool = True) -> np.ndarray:
    air = self.det.update(q, tau, command_is_zero)
    tgt = np.asarray(policy_target, dtype=float)
    if air:
      if self._held is None:
        self._held = tgt.copy()                      # 들린 그 순간의 목표로 얼린다
      self._blend = self.steps
      return self._held.copy()
    if self._held is not None and self._blend > 0:
      a = self._blend / float(self.steps)            # 1 -> 0
      out = a * self._held + (1.0 - a) * tgt
      self._blend -= 1
      if self._blend == 0:
        self._held = None
      return out
    self._held = None
    return tgt
