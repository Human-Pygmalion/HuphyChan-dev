#!/usr/bin/env python3
"""**테더 없이 혼자 서는 자세를 탐색해서 찾는다.**

2026-09-16 · 사용자 지시: *"자립하는 각도 학습해서 찾아."*

## 왜 강화학습이 아닌가

찾을 것이 **세 숫자**(엉덩이 굽힘 · 무릎 굽힘 · 크랭크 치우침)뿐입니다. 상태에 따라
달라지는 함수가 아니라 **상수**라서, 배울 것이 없습니다. 게다가 지금 GPU 는 다른 학습이
쓰고 있어 새 학습을 올리면 그쪽이 느려집니다. 그래서 CPU 에서 도는 직접탐색으로 같은
답을 더 빨리, 남의 일을 방해하지 않고 얻습니다.

## 물리 설정을 손으로 만들지 않는 이유 — 세 번 틀렸다

2026-09-15~16 에 맨 MuJoCo 로 세우려다 **연달아 세 번** 설정 버그를 냈습니다.

1. 발목만 넣고 크랭크·로드를 안 맞춰 **폐루프가 찢긴 채** 시작 -> 구속이 튕겨 로봇을 던짐
2. 바닥 높이를 캡슐 `size[2]` 로 계산 -> 공중에 뜬 채 시작해 떨어지며 넘어짐
3. 중력 끈 정착에서 받침대를 안 붙잡음 -> 무게중심이 발목 앞 28 cm 로 흘러감.
   붙잡았더니 이번엔 구속이 **931 mm** 찢어짐

세 번 다 "로봇이 못 선다" 처럼 보였지만 전부 **제 설정** 문제였습니다. 그래서 여기서는
학습이 실제로 쓰는 `ManagerBasedRlEnv` 를 그대로 씁니다 — 키프레임·폐루프·바닥·
액추에이터·게인·T-N 한계가 이미 맞춰져 있고, 학습이 매일 그 위에서 돕니다.

## 한 번의 평가

환경을 리셋하고(일관된 기준 자세에서 시작), **행동을 상수로 고정**해 버팁니다.
행동은 `(목표 - 기준오프셋) / 배율` 로 만들어 관절 목표가 정확히 원하는 자세가 되게 합니다.
명령 속도는 0 으로 고정합니다. 넘어지면 실패.

판정은 **몸통 기울기와 높이**로 합니다. 관절이 목표를 잘 따라가는 것은 서 있는 것과
다릅니다 — 2026-09-15 에 그것만 보고 "선다" 고 잘못 보고했습니다.

    stand_search.py --grid
    stand_search.py --refine --seconds 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJLAB = REPO / "mujoco-sim/mjlab"
sys.path.insert(0, str(MJLAB / "src"))
sys.path.insert(0, str(MJLAB / "analysis"))
for k, v in (("PYG_ANKLE_MODE", "AB"), ("PYG_INIT_MID", "1"), ("PYG_MOTOR_MEAS", "1"),
             ("PYG_TN", "1"), ("PYG_V2", "1"), ("PYG_SAFE_TARGET_CLIP", "1"),
             ("PYG_STUDENT_TEACHER", "1"), ("CUDA_VISIBLE_DEVICES", ""),
             ("PYG_MODEL_TAG",
              "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix")):
  os.environ.setdefault(k, v)

TASK = "Mjlab-Velocity-Flat-Pygmalion"
# 2026-09-16: 처음엔 "기울기 25도 넘으면 넘어짐" 으로 셌는데, 계측해 보니 로봇은
# **넘어지는 게 아니라 흔들리고** 있었다 (0.8초 주기로 기울기 5.7 <-> 17.6 도를 오감,
# 8초 뒤에도 높이 0.877 m 로 살아 있음). 발목이 물러서 생기는 한계주기다.
# 그래서 판정을 바꾼다: **넘어짐은 아주 크게(45도) 잡고**, 좋은 자세는 "가라앉힌 뒤
# 평균 기울기와 흔들림 폭이 작은" 것으로 고른다. 흔들림을 넘어짐으로 세면 전부
# 탈락해서 좋고 나쁨을 구분할 수 없다.
FALL_TILT_DEG = 45.0
SETTLE_S = 4.0                 # 앞부분은 버린다 (리셋 과도상태)
OK_MEAN_TILT_DEG = 5.0         # 이 아래면 "서 있다" 로 본다
OK_SWING_DEG = 6.0             # 기울기 진폭(최대-최소)


class StandEnv:
  """학습 환경을 한 번 만들어 두고, 자세를 바꿔 가며 되풀이 평가한다."""

  def __init__(self, device="cpu"):
    import torch
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from measure_loads import load_env_cfg, load_rl_cfg
    from mjlab.asset_zoo.robots.pygmalion import pygmalion_constants as C

    self.torch, self.mj, self.C = torch, mujoco, C
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    agent = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    self.env.reset()
    u = self.env.unwrapped
    self.u, self.robot, self.sim = u, u.scene["robot"], u.sim
    term = u.action_manager.get_term("joint_pos")
    ids = np.asarray(term._target_ids).reshape(-1)
    allj = list(self.robot.joint_names)
    self.anames = [allj[i] for i in ids]

    def vec(x, n):
      a = np.asarray(x.cpu() if hasattr(x, "cpu") else x, float).reshape(-1)
      return a if a.size == n else np.full(n, float(a.reshape(-1)[0]))

    self.scale = vec(term._scale, len(self.anames))
    self.offset = vec(term._offset, len(self.anames))
    self.bent = C._bent_joint_pos()
    m = self.sim.mj_model
    self.base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if self.base_bid < 0:
      self.base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    rng = m.jnt_range
    self.sign = {}
    for n in self.anames:
      j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
      if j < 0:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
      lo, hi = rng[j]
      self.sign[n] = -1.0 if abs(lo) > abs(hi) else 1.0

  # ---------------------------------------------------------------- 자세
  def targets(self, hip_deg, knee_deg, crank_deg) -> np.ndarray:
    want = np.zeros(len(self.anames))
    for i, n in enumerate(self.anames):
      if "hip_pitch" in n:
        want[i] = np.radians(hip_deg) * self.sign[n]
      elif "knee" in n:
        want[i] = np.radians(knee_deg) * self.sign[n]
      elif "crank" in n:
        want[i] = self.bent.get(n, 0.0) + np.radians(crank_deg) * self.sign[n]
      else:                                   # hip_roll / hip_yaw = 0 (발이 앞, 곧게)
        want[i] = 0.0
    return want

  # ---------------------------------------------------------------- 평가
  def evaluate(self, hip_deg, knee_deg, crank_deg, seconds=12.0) -> dict:
    """가라앉힌 뒤의 **평균 기울기와 흔들림 폭**으로 자세를 점수 매긴다."""
    want = self.targets(hip_deg, knee_deg, crank_deg)
    act = self.torch.tensor((want - self.offset) / self.scale,
                            dtype=self.torch.float32).unsqueeze(0)
    self.env.reset()
    n, n0 = int(seconds * 50), int(SETTLE_S * 50)
    tilts, zs, fell = [], [], False
    for k in range(n):
      self.env.step(act)
      try:
        self.u.command_manager.get_term("base_velocity").vel_command_b[:] = 0.0
      except Exception:                        # noqa: BLE001
        pass
      xp = np.asarray(self.sim.data.xpos[0].clone().cpu())
      xm = np.asarray(self.sim.data.xmat[0].clone().cpu())
      up = xm[self.base_bid].reshape(3, 3)[:, 2]
      tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
      if tilt > FALL_TILT_DEG:
        fell = True
        break
      if k >= n0:
        tilts.append(tilt)
        zs.append(float(xp[self.base_bid][2]))
    if fell or not tilts:
      return dict(hip=hip_deg, knee=knee_deg, crank=crank_deg, fell=True,
                  mean_tilt=90.0, swing=90.0, z=0.0, ok=False, score=(-1e9,))
    tilts, zs = np.array(tilts), np.array(zs)
    mean_tilt = float(tilts.mean())
    swing = float(tilts.max() - tilts.min())
    ok = (mean_tilt < OK_MEAN_TILT_DEG and swing < OK_SWING_DEG)
    return dict(hip=hip_deg, knee=knee_deg, crank=crank_deg, fell=False,
                mean_tilt=mean_tilt, swing=swing, z=float(zs.mean()), ok=ok,
                score=(-mean_tilt - 0.5 * swing,))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--seconds", type=float, default=12.0)
  ap.add_argument("--refine", action="store_true")
  a = ap.parse_args(argv)

  st = StandEnv()
  print(f"행동 {len(st.anames)}칸 · {a.seconds:g} 초 중 앞 {SETTLE_S:g} 초 버림 "
        f"· 넘어짐 {FALL_TILT_DEG:g} deg\n")
  print(f"{'엉덩이':>7}{'무릎':>7}{'크랭크':>8}{'평균기울기':>11}{'흔들림폭':>10}"
        f"{'몸통높이':>10}  판정")
  best, results = None, []
  for knee in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
    for hip_r in (0.5, 0.69, 0.9):
      for crank in (-8.0, -4.0, 0.0, 4.0, 8.0):
        hip = hip_r * knee
        r = st.evaluate(hip, knee, crank, seconds=a.seconds)
        results.append(r)
        if best is None or r["score"] > best["score"]:
          best = r
        tag = "넘어짐" if r["fell"] else ("★ 자립" if r["ok"] else "")
        print(f"{hip:>7.2f}{knee:>7.2f}{crank:>8.1f}{r['mean_tilt']:>11.2f}"
              f"{r['swing']:>10.2f}{r['z']:>10.4f}  {tag}")
  stood = [r for r in results if r["ok"]]
  print(f"\n자립 {len(stood)} / {len(results)}")
  print(f"최고: 엉덩이 {best['hip']:.2f} · 무릎 {best['knee']:.2f} · 크랭크 {best['crank']:+.1f}"
        f" -> 평균기울기 {best['mean_tilt']:.2f} deg · 흔들림 {best['swing']:.2f} deg")

  if a.refine:
    print("\n국소 정밀화")
    h, k, c = best["hip"], best["knee"], best["crank"]
    for dh in (-2.0, 0.0, 2.0):
      for dk in (-3.0, 0.0, 3.0):
        for dc in (-2.0, 0.0, 2.0):
          r = st.evaluate(h + dh, k + dk, c + dc, seconds=a.seconds)
          if r["score"] > best["score"]:
            best = r
          tag = "넘어짐" if r["fell"] else ("★ 자립" if r["ok"] else "")
          print(f"{h+dh:>7.2f}{k+dk:>7.2f}{c+dc:>8.1f}{r['mean_tilt']:>11.2f}"
                f"{r['swing']:>10.2f}{r['z']:>10.4f}  {tag}")
    print(f"\n정밀화 최고: 엉덩이 {best['hip']:.2f} · 무릎 {best['knee']:.2f} "
          f"· 크랭크 {best['crank']:+.1f} -> 평균기울기 {best['mean_tilt']:.2f} deg")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
