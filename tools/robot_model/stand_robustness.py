#!/usr/bin/env python3
"""**얼마나 세게 밀어야 넘어지나, 얼마나 삐뚤게 시작해도 서나.**

2026-09-17 · 사용자 지시: *"외력 다양하게 버티도록, 초기자세 틀려도 잘 버티도록 더
학습해줘."* 그 학습(v7b)이 끝났으니 **정말 좋아졌는지** 숫자로 확인합니다.

## 왜 기존 도구로는 안 되나

`analysis/measure_loads.py --push` 는 미는 세기가 **±0.7 m/s 로 코드에 박혀** 있습니다
(옛 학습의 값). v7b 는 **±1.5** 로 학습했으므로, 그 도구로 재면 "학습한 것보다 약한
힘으로만 밀어 보고 잘 버틴다" 고 말하게 됩니다. 그건 확인이 아닙니다.

## 어떻게 재나 — 한 번에 여러 마리

로봇 **64마리를 동시에** 세워 놓고 같은 세기로 밉니다. 한 마리를 64번 돌리는 것보다
빠르고, **넘어진 비율**이라는 통계가 바로 나옵니다. 쓰러짐 판정은 학습이 쓰는 그
판정을 그대로 씁니다(`terminations`) — 제가 따로 기준을 만들면 학습과 다른 잣대가
됩니다.

세기를 0 부터 올려 가며 **처음으로 넘어지기 시작하는 지점**을 찾습니다. 그 지점이
"이 정책이 버티는 외력" 입니다.

초기자세는 같은 방식으로, 리셋할 때 관절각과 몸통 기울기를 흔드는 양을 올려 가며
봅니다 (학습이 쓰는 `reset_robot_joints` · `reset_base` 이벤트를 그대로 켭니다).

## 무엇을 같이 기록하나

크랭크(발목) 토크입니다. v7 에서 **정격의 70%** 까지 갔던 것이 v7b 에서 내려갔는지가
이 학습의 숨은 관심사였습니다.

## 왜 같은 칸을 여러 번 도나 — 씨앗

로봇 64마리를 한 번에 세우면 **한 순간의 표본은 64개**지만, 그 64마리가 어떤 자세로
시작하고 어느 방향으로 밀릴지는 **한 번의 추첨**으로 정해집니다. 추첨이 운 좋게 순하면
"안 넘어진다" 가 나옵니다.

프로젝트 규칙이 이걸 명시합니다 — *"관절값은 결정적이나 무작위화 추첨은 아니다"*.
같은 저장본이 같은 조건에서 0.00초와 4.58초로 갈린 전례가 있습니다. 그래서 칸마다
**씨앗을 바꿔 여러 번** 돌리고, 평균과 함께 **칸 사이 흔들림(표준편차)** 을 같이 냅니다.
흔들림이 차이보다 크면 그 차이는 아직 아무 말도 안 하는 것입니다.

    stand_robustness.py --checkpoint <ckpt.pt> --label v7b --mode push
    stand_robustness.py --checkpoint <ckpt.pt> --label v7b --mode init --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJLAB = REPO / "mujoco-sim/mjlab"
sys.path.insert(0, str(MJLAB / "src"))
sys.path.insert(0, str(MJLAB / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airborne_signature import apply_run_env                      # noqa: E402

TASK = "Mjlab-Velocity-Flat-Pygmalion"
OUT = REPO / "docs/data"

# 미는 세기 [m/s]. 학습값 1.5 를 가운데 두고 그 위아래를 본다.
PUSH_STEPS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
# 초기 관절각 흔들기 [rad]. 학습값 0.12 를 가운데 두고. 0.164 = 실측 최대 영점오차.
INIT_STEPS = (0.0, 0.06, 0.12, 0.18, 0.24, 0.30)
PUSH_EVERY_S = 3.0          # 3초마다 한 번 — 학습의 1~3초보다 조금 느슨하게
SETTLE_S = 2.0              # 앞부분은 버린다 (리셋 과도상태)


def build(checkpoint: str, num_envs: int):
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  # 관찰 잡음은 **켠 채로** 잽니다. 실물 센서가 잡음을 내고, 학습도 켜고 돕니다
  # (docs/148 §2e). 끄고 재면 실물보다 쉬운 시험이 됩니다.
  cfg.observations["actor"].enable_corruption = True
  _b = cfg.episode_length_s + 1e4
  cfg.commands["twist"].resampling_time_range = (_b, _b)
  # 무작위화 이벤트는 우리가 직접 켜고 끄므로 일단 다 뗀다 (미는 것/초기자세만 남긴다).
  for ev in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(ev, None)
  for ev in list(cfg.events):
    if ev.startswith("inertial_dr_"):
      cfg.events.pop(ev)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  envw = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    envw, asdict(load_rl_cfg(TASK)), device="cpu")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  return env, envw, runner.get_inference_policy(device="cpu"), torch, cfg


def set_init_noise(env, joint_rad: float, tilt_rad: float) -> None:
  """리셋할 때 관절각과 몸통 기울기를 얼마나 흔들지 정한다.

  학습이 쓰는 이벤트를 그대로 고칩니다 — 제가 리셋을 새로 짜면 학습과 다른 시작이
  됩니다. play 모드는 이 둘을 0 으로 두므로 여기서 다시 넣어 줍니다.
  """
  em = env.event_manager
  try:
    c = em.get_term_cfg("reset_robot_joints")
    c.params["position_range"] = (-joint_rad, joint_rad)
  except (KeyError, ValueError):
    pass
  try:
    c = em.get_term_cfg("reset_base")
    pr = c.params.get("pose_range")
    if isinstance(pr, dict):
      pr["roll"] = (-tilt_rad, tilt_rad)
      pr["pitch"] = (-tilt_rad, tilt_rad)
  except (KeyError, ValueError):
    pass


def set_seed(torch, seed: int) -> None:
  """이 칸의 추첨을 정한다. **환경을 다시 만들지 않고** 난수만 갈아 끼운다."""
  import random

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def merge(rows: list[dict]) -> dict:
  """같은 칸의 여러 씨앗 결과를 합친다. 평균과 **칸 사이 표준편차**를 같이 낸다."""
  out = dict(rows[0])
  out["repeats"] = len(rows)
  for k in ("falls_per_robot_minute", "tilt_mean", "tilt_p95", "tilt_max", "recover_s",
            "qd_sum_mean", "qd_sum_p95", "tau_rms", "pose_err_deg",
            "crank_tau_mean", "crank_tau_p99", "height_mean"):
    v = np.array([r[k] for r in rows], dtype=float)
    out[k] = float(v.mean())
    out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
  out["seeds"] = [r.get("seed") for r in rows]
  return out


def run_cell(env, envw, policy, torch, seconds: float, push_mps: float,
             base_bid: int, settle_s: float | None = None) -> dict:
  """한 칸을 돌린다. 넘어진 횟수와 자세·토크·떨림을 모은다.

  ``settle_s`` 는 **앞부분을 얼마나 버릴지**입니다. 미는 시험은 정상상태를 보므로
  버리지만, **초기자세 시험은 그 버리는 구간이 곧 답**입니다 — 2026-09-17 에 그걸
  버린 채로 재고 "초기자세를 흔들어도 값이 똑같다" 는 무의미한 표를 냈습니다.
  정책이 2초 안에 같은 자세로 돌아오니 당연한 일이었습니다. 0 을 주면 리셋 직후부터
  봅니다.
  """
  from mjlab.envs.mdp.events import push_by_setting_velocity

  robot = env.scene["robot"]
  obs, _ = envw.reset()
  n = int(seconds * 50)
  n0 = int((SETTLE_S if settle_s is None else settle_s) * 50)
  every = int(PUSH_EVERY_S * 50)
  # 미는 방향은 앞뒤·좌우·위아래·회전 전부. 세기 하나로 비례해서 키운다
  # (학습의 PYG_DR_STRONG 비율 그대로: xy 1.5 -> z 0.6, roll/pitch 0.8, yaw 1.2).
  k = push_mps / 1.5 if push_mps > 0 else 0.0
  prange = {"x": (-push_mps, push_mps), "y": (-push_mps, push_mps),
            "z": (-0.6 * k, 0.6 * k), "roll": (-0.8 * k, 0.8 * k),
            "pitch": (-0.8 * k, 0.8 * k), "yaw": (-1.2 * k, 1.2 * k)}
  falls, tilts, heights, cranks, pushes = 0, [], [], [], 0
  qds, taus, perrs = [], [], []
  cidx = [i for i, nm in enumerate(robot.actuator_names) if "crank" in nm]
  aterm = env.action_manager.get_term("joint_pos")
  jids = np.asarray(aterm._target_ids).reshape(-1)
  q0 = np.asarray(robot.data.default_joint_pos[0].clone().cpu())[jids]
  for k_step in range(n):
    with torch.inference_mode():
      act = policy(obs)
    obs, _, dones, _ = envw.step(act)
    try:
      env.command_manager.get_term("twist").vel_command_b[:] = 0.0
    except Exception:                                             # noqa: BLE001
      pass
    if k_step >= n0:
      falls += int(dones.sum())
      if push_mps > 0 and (k_step - n0) % every == 0:
        push_by_setting_velocity(env, None, prange)
        pushes += 1
      up = np.asarray(env.sim.data.xmat[:, base_bid].clone().cpu()).reshape(-1, 3, 3)[:, :, 2]
      tilts.append(np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0))))
      heights.append(np.asarray(env.sim.data.xpos[:, base_bid, 2].clone().cpu()))
      tau_all = np.asarray(robot.data.actuator_force.clone().cpu())
      taus.append(np.abs(tau_all))
      qd = np.asarray(robot.data.joint_vel.clone().cpu())[:, jids]
      qds.append(np.abs(qd).sum(axis=1))
      q = np.asarray(robot.data.joint_pos.clone().cpu())[:, jids]
      perrs.append(np.abs(q - q0[None, :]).mean(axis=1))
      if cidx:
        cranks.append(np.abs(tau_all[:, cidx]))
  tilts, heights = np.array(tilts), np.array(heights)
  cr = np.array(cranks) if cranks else np.zeros((1, 1, 1))
  # 넘어진 횟수를 "로봇 한 마리가 1분 서 있을 때 몇 번" 으로 바꾼다 — 환경 수와
  # 시간이 달라져도 비교되는 단위여야 한다.
  robot_minutes = (n - n0) / 50.0 / 60.0 * env.num_envs
  # 회복: 기울기가 2도 아래로 내려와 **다시 안 올라오는** 첫 시점. 잠깐 스치는 것을
  # 회복으로 세지 않으려고 뒤에서부터 찾습니다.
  t_over = np.where((tilts > 2.0).any(axis=1))[0]
  recover_s = float((t_over[-1] + 1) / 50.0) if len(t_over) else 0.0
  qd = np.array(qds) if qds else np.zeros((1, 1))
  ta = np.array(taus) if taus else np.zeros((1, 1, 1))
  pe = np.array(perrs) if perrs else np.zeros((1, 1))
  return dict(push_mps=push_mps, falls=falls, pushes=pushes,
              falls_per_robot_minute=falls / max(robot_minutes, 1e-9),
              tilt_mean=float(tilts.mean()), tilt_p95=float(np.percentile(tilts, 95)),
              height_mean=float(heights.mean()),
              crank_tau_mean=float(cr.mean()), crank_tau_p99=float(np.percentile(cr, 99)),
              # "가만히" 를 재는 세 가지. 떨림·열·자세.
              tilt_max=float(tilts.max()), recover_s=recover_s,
              qd_sum_mean=float(qd.mean()), qd_sum_p95=float(np.percentile(qd, 95)),
              tau_rms=float(np.sqrt((ta ** 2).mean())),
              pose_err_deg=float(np.degrees(pe.mean())))


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--label", required=True, help="결과 파일 이름에 쓸 이름 (예: v7b)")
  ap.add_argument("--mode", choices=("push", "init"), default="push")
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--repeats", type=int, default=3,
                  help="칸마다 씨앗을 바꿔 몇 번 돌릴지. 1 이면 옛날처럼 한 번만")
  ap.add_argument("--seed0", type=int, default=0, help="첫 씨앗")
  n = ap.parse_args(argv)

  envd = apply_run_env(n.checkpoint)
  env, envw, policy, torch, cfg = build(n.checkpoint, n.num_envs)
  import mujoco

  m = env.sim.mj_model
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
  if bid < 0:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  print(f"[stand] {n.label}  {n.checkpoint}", flush=True)
  print(f"[stand] 학습 토글 {len(envd)}개, 로봇 {n.num_envs}마리, "
        f"칸마다 {n.seconds:.0f}초 x 씨앗 {n.repeats}개", flush=True)

  rows = []
  if n.mode == "push":
    set_init_noise(env, 0.0, 0.0)          # 초기자세는 깨끗하게, 미는 것만 본다
    print(f"\n{'미는세기[m/s]':>14}{'넘어짐/로봇·분':>16}{'기울기 95%':>12}"
          f"{'떨림|qd|평균':>14}{'떨림 95%':>11}{'토크RMS':>10}"
          f"{'자세오차[도]':>13}{'크랭크 99%':>12}")
    for p in PUSH_STEPS:
      reps = []
      for i in range(max(1, n.repeats)):
        set_seed(torch, n.seed0 + i)
        c = run_cell(env, envw, policy, torch, n.seconds, p, bid)
        c["seed"] = n.seed0 + i
        reps.append(c)
      r = merge(reps)
      rows.append(r)
      print(f"{p:>14.1f}{r['falls_per_robot_minute']:>10.2f}"
            f"+-{r['falls_per_robot_minute_sd']:<5.2f}{r['tilt_p95']:>12.1f}"
            f"{r['qd_sum_mean']:>14.3f}{r['qd_sum_p95']:>11.3f}{r['tau_rms']:>10.2f}"
            f"{r['pose_err_deg']:>13.2f}{r['crank_tau_p99']:>12.2f}", flush=True)
  else:
    print(f"\n{'초기흔들기[도]':>14}{'넘어짐/로봇·분':>16}{'기울기 최대':>12}"
          f"{'회복시간[초]':>13}{'떨림|qd|평균':>14}{'토크RMS':>10}"
          f"{'자세오차[도]':>13}{'크랭크 99%':>12}")
    for j in INIT_STEPS:
      set_init_noise(env, j, j * 0.8)      # 기울기는 관절보다 조금 작게
      reps = []
      for i in range(max(1, n.repeats)):
        set_seed(torch, n.seed0 + i)
        c = run_cell(env, envw, policy, torch, n.seconds, 0.0, bid, settle_s=0.0)
        c["seed"] = n.seed0 + i
        reps.append(c)
      r = merge(reps)
      r["init_rad"], r["init_deg"] = j, float(np.degrees(j))
      rows.append(r)
      print(f"{np.degrees(j):>14.1f}{r['falls_per_robot_minute']:>10.2f}"
            f"+-{r['falls_per_robot_minute_sd']:<5.2f}"
            f"{r['tilt_max']:>12.1f}{r['recover_s']:>13.2f}{r['qd_sum_mean']:>14.3f}"
            f"{r['tau_rms']:>10.2f}{r['pose_err_deg']:>13.2f}"
            f"{r['crank_tau_p99']:>12.2f}", flush=True)

  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / f"stand_robustness_{n.label}_{n.mode}.json"
  p.write_text(json.dumps(dict(checkpoint=n.checkpoint, label=n.label, mode=n.mode,
                               num_envs=n.num_envs, seconds=n.seconds,
                               repeats=n.repeats, seed0=n.seed0, rows=rows),
                          ensure_ascii=False, indent=2, default=float), encoding="utf-8")
  print(f"기록: {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
