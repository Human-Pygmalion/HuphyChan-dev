#!/usr/bin/env python3
"""**로봇을 공중에 들었을 때와 땅에 서 있을 때의 모터 힘 차이를 잰다.**

2026-09-17 · 사용자 지시: *"정책 실행하다가 로봇을 공중에 들 때가 있는데, 이러면 다리가
막 흔들리네. 가만히 있도록 할수있겠나?"*

## 왜 재는가 — 문턱값을 짐작하지 않으려고

"공중에 들렸다" 를 알아채려면 어떤 숫자가 얼마를 넘는지 정해야 합니다. 그 숫자를
감으로 박으면 두 가지로 틀립니다. 너무 낮으면 **땅에 서 있는데 멈춰 버리고**(로봇이
쓰러집니다), 너무 높으면 **들어도 계속 흔듭니다**. 그래서 두 상황을 같은 정책으로
실제로 돌려 보고, 둘이 **겹치지 않는 구간**을 찾아서 그 가운데를 씁니다.

## 왜 모터 힘인가

우리 발바닥에는 힘 센서가 없습니다. 자세 센서(IMU)도 소용이 없습니다 — 크레인에
매달린 로봇은 똑바로 서 있는 로봇과 **중력 방향이 똑같이** 읽힙니다. 남는 것은
모터가 내는 힘뿐이고, 이건 실제로 크게 다릅니다.

* 땅에 서 있으면 다리가 **몸 전체(51.5 kg)** 를 떠받칩니다.
* 공중에 들리면 다리는 **자기 무게만** 붙잡습니다. 게다가 매달린 다리는 중력이
  곧게 펴 주는 쪽으로 당기므로, 무릎이 내야 하는 힘이 0 근처로 떨어집니다.

모터 힘(`tau`)은 실물에서 이미 **매 주기 전송**됩니다
(`tools/pygviewer/pygviewer/modes.py` 의 `FAST_MOTOR_FIELDS`). 새로 달 것이 없습니다.

## 두 상황을 어떻게 만드나

같은 정책, 같은 명령(속도 0)으로 두 번 돌립니다.

* **땅** — 평소대로. 학습 환경 그대로 리셋하고 정책이 서 있게 둡니다.
* **공중** — 매 스텝 몸통의 위치·자세를 손으로 붙잡아(`write_root_state_to_sim`)
  바닥에서 떼어 놓습니다. 사람이 로봇을 들고 있는 것과 같습니다. 다리는 자유롭게
  흔들립니다 — 정확히 사용자가 본 그 현상입니다.

무작위화(DR)는 전부 끕니다. 재는 것은 "이 로봇의 값" 이지 "무작위 추첨 한 번의 값" 이
아닙니다 (`gait_kinematics_probe.py` 의 PROBE_NODR 관례와 같습니다).

## 무엇을 내놓나

관절별 `|tau|` 평균·최대와, **후보 판정식** 몇 개를 땅/공중 각각의 분포로 냅니다.
분리도(separation)는 두 분포 사이가 얼마나 벌어졌는지입니다 — 1.0 보다 크면 겹치지
않습니다.

    airborne_signature.py --checkpoint <model.pt> --seconds 8
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # 학습이 GPU 를 쓰는 중이다


def apply_run_env(checkpoint: str) -> dict:
  """체크포인트를 낸 **그 학습의 환경변수를 그대로** 켠다.

  2026-09-17 · 처음엔 손으로 `PYG_ANKLE_MODE` 같은 것 몇 개만 적었습니다. 그랬더니
  잘 학습된 정책이 2초 만에 주저앉았고, 저는 하마터면 "정책이 못 선다" 로 잘못 볼
  뻔했습니다. 빠진 것은 `PYG_INIT_BENT` (기준 자세), `PYG_KNEE_EXT`, `PYG_MASS_DR_JSON`
  등이었습니다. **기준 자세가 다르면 행동의 오프셋이 달라져서** 정책이 내는 숫자가
  전혀 다른 관절 목표가 됩니다 — 정책은 멀쩡한데 환경이 딴 로봇입니다.

  손으로 적는 목록은 반드시 또 빠집니다. 그래서 적지 않고, 학습이 스스로 남긴
  ``repro/launch_manifest.json`` 의 ``environment`` 를 통째로 씁니다.
  (프로젝트 규칙: *"측정 전 PYG_* 를 학습과 동일 지정"*.)
  """
  run_dir = Path(checkpoint).resolve().parent
  man = run_dir / "repro" / "launch_manifest.json"
  if not man.exists():
    raise SystemExit(f"{man} 이 없습니다 — 어떤 토글로 학습했는지 알 수 없어 멈춥니다")
  envd = json.loads(man.read_text(encoding="utf-8")).get("environment") or {}
  if not envd:
    raise SystemExit(f"{man} 에 environment 가 비어 있습니다")
  for k, v in envd.items():
    os.environ[k] = str(v)
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  return envd

TASK = "Mjlab-Velocity-Flat-Pygmalion"
OUT = REPO / "docs/data"

# 무작위화는 전부 끈다 — 재는 것은 이 로봇의 값이다.
DROP_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com")


def build(checkpoint: str, obs_noise: bool = False):
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  for ev in list(cfg.events):
    if ev in DROP_EVENTS or ev.startswith("inertial_dr_"):
      cfg.events.pop(ev)
  # 에피소드가 중간에 끝나면(넘어짐 판정 등) 리셋 과도상태가 측정에 섞인다. 공중에
  # 들린 로봇은 학습 환경 기준으로는 "이상한 상태" 라 계속 끝나 버리므로, 재는 동안은
  # 종료 조건을 전부 뗀다. 우리가 재려는 것은 "그 상태에서 모터가 내는 힘" 이다.
  cfg.terminations.clear()
  _b = cfg.episode_length_s + 1e4
  for cname in ("twist", "base_velocity"):
    if cname in cfg.commands:
      cfg.commands[cname].resampling_time_range = (_b, _b)
  # 2026-09-17: play 모드는 관찰 잡음을 끕니다. 그런데 **학습은 켜고 돌립니다** —
  # 실물 센서가 잡음을 내기 때문입니다(관절속도 ±1.5 rad/s, 자세 ±0.05). 정책이 내는
  # 명령은 그 잡음을 그대로 받아 늘 조금씩 떨립니다. 땅에서는 안 보이고 공중에서는
  # 보입니다. 그래서 "실물과 같은 조건" 을 재려면 이걸 다시 켜야 합니다.
  cfg.observations["actor"].enable_corruption = bool(obs_noise)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  envw = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    envw, asdict(load_rl_cfg(TASK)), device="cpu")
  # 2026-09-20: 스펙트럼 정규화로 학습한 저장본은 가중치가
  # `mlp.N.parametrizations.weight.original` 로 감싸여 있어, 평범한 신경망에 그대로
  # 넣으면 키가 안 맞아 멈춥니다. 그래서 **싣기 전에 같은 감싸기를 재현**합니다.
  # 어떤 값으로 학습했는지는 그 런의 토글 기록에 있습니다 — 짐작하지 않습니다.
  _sn = float(os.environ.get("PYG_SN", 0.0) or 0.0)
  if _sn > 0.0:
    from mjlab.rl.smooth_ppo import SmoothPPO

    # 보통은 이미 감싸여 있습니다 — 토글이 켜져 있으면 알고리즘이 SmoothPPO 로
    # 만들어지고 그 생성자가 이미 겁니다. 그때는 0 개가 나오고 아무 일도 안 합니다.
    # 여기 있는 이유는 알고리즘이 다른 경로로 만들어졌을 때의 그물입니다.
    n = SmoothPPO._apply_spectral_norm(SmoothPPO, runner.alg.actor, _sn)
    if n:
      print(f"[air] 스펙트럼 정규화 저장본 — 층 {n} 개를 여기서 감쌌습니다 (전체 λ={_sn})",
            flush=True)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  return env, envw, runner.get_inference_policy(device="cpu"), torch


def zero_command(env) -> None:
  """명령 속도를 0 으로 못박는다 — 가만히 서 있는 상황만 잰다."""
  for cname in ("twist", "base_velocity"):
    try:
      env.command_manager.get_term(cname).vel_command_b[:] = 0.0
      return
    except Exception:                                          # noqa: BLE001
      continue


def rollout(env, envw, policy, torch, seconds: float, lift: float | None,
            settle_s: float = 2.0, hold: str = "hands") -> dict:
  """한 상황을 돌리고 관절별 `tau`·속도·추종오차를 모은다.

  ``lift`` 가 없으면 평소대로 땅 위입니다. 있으면 그 높이로 **들어 올립니다**.

  ``hold`` 는 **어떻게 잡고 있느냐**입니다. 이것이 결과를 크게 바꿉니다.

  * ``"rigid"`` — 자리도 자세도 꽉 붙잡습니다. 이러면 정책이 보는 중력방향과 각속도가
    완벽한 "똑바로 서 있음" 이라, 정책이 아무것도 안 합니다. **흔들림이 재현되지
    않습니다** (처음에 이걸로 재고 "왜 안 흔들리지" 했습니다).
  * ``"hands"`` — **자리만** 붙잡고 **자세는 놓아 줍니다.** 사람이 몸통을 들고 있는
    것과 같습니다. 로봇은 손 안에서 기울 수 있고, 정책은 그 기울기를 바로잡으려
    다리를 움직이는데, 딛을 바닥이 없으니 다리만 휘두르게 됩니다 — 그 반작용이
    몸통을 다시 기울입니다. 이것이 사용자가 본 현상입니다.
  """
  robot = env.scene["robot"]
  obs, _ = envw.reset()
  pose0 = robot.data.root_link_pose_w.clone()                   # (1, 7) 자리+자세
  vel0 = torch.zeros_like(robot.data.root_link_vel_w)           # (1, 6) 멈춘 채
  if lift is not None:
    pose0[:, 2] += float(lift)
    pose0[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])          # 똑바로 들어서 시작
  n, n0 = int(seconds * 50), int(settle_s * 50)
  aterm = env.action_manager.get_term("joint_pos")
  jids = np.asarray(aterm._target_ids).reshape(-1)
  taus, qs, qvels, qerrs, heights, ndone = [], [], [], [], [], 0
  for k in range(n):
    if lift is not None:                                        # 들기 = 매 스텝 붙잡기
      if hold == "rigid":
        robot.write_root_link_pose_to_sim(pose0)
        robot.write_root_link_velocity_to_sim(vel0)
      else:                                                     # "hands": 자리만 잡는다
        cur = robot.data.root_link_pose_w.clone()
        cur[:, :3] = pose0[:, :3]                               # 자리는 고정
        robot.write_root_link_pose_to_sim(cur)                  # 자세는 그대로 둔다
        v = robot.data.root_link_vel_w.clone()
        v[:, :3] = 0.0                                          # 병진속도만 0
        robot.write_root_link_velocity_to_sim(v)
    with torch.inference_mode():
      act = policy(obs)
    obs, _, dones, _ = envw.step(act)
    ndone += int(dones.sum())
    zero_command(env)
    if k >= n0:
      taus.append(robot.data.actuator_force[0].clone().cpu().numpy().copy())
      q = robot.data.joint_pos[0].clone().cpu().numpy()[jids]
      qd = robot.data.joint_vel[0].clone().cpu().numpy()[jids]
      # 추종오차 = 관절각 - **정책이 낸 목표각**. 실물도 `tgt`/`err` 를 그대로 보냅니다
      # (modes.FAST_MOTOR_FIELDS). 2026-09-17: 처음엔 `ctrl` 을 목표각으로 썼는데,
      # 이 액추에이터의 `ctrl` 은 **토크**라 q-tau 라는 뜻 없는 값이 나왔습니다
      # (tau_all 과 qerr_all 이 27 로 거의 같아서 들켰습니다).
      tgt = aterm._processed_actions[0].clone().cpu().numpy()
      qs.append(q.copy())
      qvels.append(qd.copy())
      qerrs.append((q - tgt).copy())
      heights.append(float(robot.data.root_link_pos_w[0, 2]))
  return dict(tau=np.array(taus), q=np.array(qs), qvel=np.array(qvels),
              qerr=np.array(qerrs), z=np.array(heights), resets=ndone,
              act_joint_names=[list(robot.joint_names)[i] for i in jids])


# ---------------------------------------------------------------- 판정 후보
def detectors(r: dict, names: list[str]) -> dict:
  """스칼라 판정 후보들. 전부 **실물에서 그대로 계산되는 값**만 씁니다.

  접촉력·몸통 높이처럼 시뮬레이터만 아는 값은 쓰지 않습니다. 실물에서 매 주기 오는
  것은 관절의 `pos`·`tgt`·`vel`·`tau` 뿐입니다 (`modes.FAST_MOTOR_FIELDS`).
  """
  tau, qd, qe = r["tau"], r["qvel"], r["qerr"]

  def pick(arr, sub):
    ii = [i for i, n in enumerate(names) if sub in n]
    return np.abs(arr[:, ii]).sum(axis=1) if ii else np.zeros(len(arr))

  return {
    "tau_all": np.abs(tau).sum(axis=1),
    "tau_ankle_crank": pick(tau, "crank"),
    "tau_hip_roll": pick(tau, "hip_roll"),
    "tau_knee": pick(tau, "knee"),
    "qd_all": np.abs(qd).sum(axis=1),
    "qerr_all": np.abs(qe).sum(axis=1),
  }


def separation(a: np.ndarray, b: np.ndarray) -> tuple[float, str, float]:
  """두 분포가 얼마나 벌어졌는지. **어느 쪽이 큰지는 미리 정하지 않습니다.**

  2026-09-17: 처음엔 "땅이 당연히 더 크다" 고 놓고 한 방향만 쟀습니다. 재 보니
  **반대**였습니다 — 우리 서기 자세는 다리가 거의 곧아서 바닥반력이 관절축을 거의
  지나갑니다(모멘트 팔이 짧다). 반면 공중에 매달린 다리는 자기 무게를 외팔보로
  들어야 해서 엉덩이 토크가 오히려 큽니다. 그래서 양쪽을 다 보고 큰 쪽을 답으로 냅니다.

  돌려주는 것: (틈/퍼짐, 어느 쪽이 큰지, 문턱값). 틈이 양수면 두 상황이 안 겹칩니다.
  """
  spread = float(a.std() + b.std()) or 1e-9
  gap_ground_high = float(a.min() - b.max())
  gap_air_high = float(b.min() - a.max())
  if gap_air_high > gap_ground_high:
    return gap_air_high / spread, "air>ground", 0.5 * (b.min() + a.max())
  return gap_ground_high / spread, "ground>air", 0.5 * (a.min() + b.max())


def sweep(env, envw, policy, torch, n) -> int:
  """명령 지연만 바꿔 가며 **공중에서 다리가 얼마나 흔들리는지** 잰다.

  지연은 행동 항목의 속성(`max_delay`)이라 **환경을 다시 만들지 않고** 바꿀 수 있습니다.
  그래서 모형·정책·자세가 완전히 같은 상태에서 지연 하나만 달라집니다 — 단일 변인입니다.
  """
  term = env.action_manager.get_term("joint_pos")
  attr = "max_delay" if n.sweep_kind == "delay" else "noise"
  if not hasattr(term, attr):
    raise SystemExit("--sweep 은 지연 항목이 켜져 있어야 합니다: "
                     "--env PYG_ACTION_DELAY=1.0 을 같이 주세요")
  # 쓸지 않는 쪽은 **0 으로 못박는다**. 2026-09-17 에 지연과 잡음을 같이 켜 놓고
  # "지연이 원인" 이라고 썼다가 쓸기에서 틀린 것이 드러났습니다 — 변인이 둘이었습니다.
  other = "noise" if attr == "max_delay" else "max_delay"
  setattr(term, other, 0.0)
  vals = [float(x) for x in n.sweep.split(",")]
  unit = "지연[스텝]" if attr == "max_delay" else "잡음[비율]"
  print(f"\n{unit:>12}{'지연[ms]':>10}"
        f"{'땅 |qd| 평균':>14}{'공중 |qd| 평균':>16}{'공중 |qd| 최대':>16}{'배수':>9}")
  rows = []
  for v in vals:
    setattr(term, attr, v)
    g = rollout(env, envw, policy, torch, n.seconds, None)
    a = rollout(env, envw, policy, torch, n.seconds, n.lift, hold=n.hold)
    gq = float(np.abs(g["qvel"]).sum(axis=1).mean())
    aq = float(np.abs(a["qvel"]).sum(axis=1).mean())
    am = float(np.abs(a["qvel"]).sum(axis=1).max())
    rows.append(dict(kind=n.sweep_kind, value=v, delay_steps=v, delay_ms=v * 20.0,
                     ground_qd=gq, air_qd=aq, air_qd_max=am))
    print(f"{v:>12.4f}{v * 20.0:>10.1f}{gq:>14.3f}{aq:>16.3f}{am:>16.3f}"
          f"{(aq / gq if gq > 1e-9 else float('inf')):>9.0f}")
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / f"{n.tag}_sweep.json").write_text(
    json.dumps(dict(checkpoint=n.checkpoint, hold=n.hold, kind=n.sweep_kind,
                    overrides=dict(kv.split("=", 1) for kv in n.env),
                    note="쓸지 않는 쪽(지연/잡음)은 0 으로 못박고 잰다", rows=rows),
               ensure_ascii=False, indent=2, default=float), encoding="utf-8")
  sweep_plot(OUT / f"{n.tag}_sweep.png", rows)
  print(f"기록: {OUT / (n.tag + '_sweep.json')}, {OUT / (n.tag + '_sweep.png')}")
  return 0


def sweep_plot(path, rows) -> None:
  """라벨은 영어로 — 한글은 이 환경에서 두부로 깨집니다."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  kind = rows[0].get("kind", "delay")
  d = [(r["delay_ms"] if kind == "delay" else r["value"]) for r in rows]
  fig, ax = plt.subplots(figsize=(7.2, 4.4))
  ax.plot(d, [r["air_qd"] for r in rows], "o-", color="#e08214",
          label="held in the air (mean)")
  ax.plot(d, [r["air_qd_max"] for r in rows], "o--", color="#e08214", alpha=0.5,
          label="held in the air (peak)")
  ax.plot(d, [r["ground_qd"] for r in rows], "s-", color="#2c7fb8",
          label="standing on the ground")
  ax.set_xlabel("commanded action delay  [ms]   (1 control step = 20 ms)"
                if kind == "delay" else "multiplicative command noise  [std, fraction]")
  ax.set_ylabel("sum |joint velocity|  [rad/s]")
  ax.set_title(f"Airborne shaking vs {kind} -- one variable at a time")
  ax.legend(fontsize=9)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def plot(path, names, ground, air, hold, delay: float = 0.0) -> None:
  """그림 한 장. **라벨은 영어로** 씁니다 — 한글은 이 환경에서 두부(tofu)로 깨집니다."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
  x = np.arange(len(names))
  g = np.abs(ground["tau"]).mean(axis=0)
  a = np.abs(air["tau"]).mean(axis=0)
  ax[0].bar(x - 0.2, g, 0.4, label="on the ground", color="#2c7fb8")
  ax[0].bar(x + 0.2, a, 0.4, label=f"lifted ({hold})", color="#e08214")
  ax[0].set_xticks(x)
  ax[0].set_xticklabels([n.replace("_joint", "") for n in names], rotation=60,
                        ha="right", fontsize=8)
  ax[0].set_ylabel("mean |tau|  [N*m]")
  ax[0].set_title("Joint torque: standing vs held in the air")
  ax[0].legend(fontsize=8)
  ax[0].grid(axis="y", alpha=0.3)

  t = np.arange(len(ground["qvel"])) / 50.0
  ax[1].plot(t, np.abs(ground["qvel"]).sum(axis=1), label="on the ground", color="#2c7fb8")
  ax[1].plot(t, np.abs(air["qvel"]).sum(axis=1), label=f"lifted ({hold})", color="#e08214")
  ax[1].set_xlabel("time [s]")
  ax[1].set_ylabel("sum |joint velocity|  [rad/s]")
  shook = np.abs(air["qvel"]).sum(axis=1).max() > 10.0 * (
    np.abs(ground["qvel"]).sum(axis=1).max() + 1e-9)
  ax[1].set_title(
    f"Leg motion  (action delay {delay * 20.0:.0f} ms)  -- "
    + ("legs DO shake when lifted" if shook else "no shaking in either case"))
  ax[1].legend(fontsize=8)
  ax[1].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--seconds", type=float, default=8.0)
  ap.add_argument("--lift", type=float, default=0.6, help="띄우는 높이 [m]")
  ap.add_argument("--hold", choices=("hands", "rigid"), default="hands",
                  help="공중에서 어떻게 잡고 있는가 (rollout 의 설명을 볼 것)")
  ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                  help="런의 토글 위에 **덧씌울** 환경변수. 학습 때는 없었던 것을 "
                       "켜 보고 차이를 재는 데 쓴다 (예: PYG_ACTION_DELAY=1.0)")
  ap.add_argument("--obs-noise", action="store_true",
                  help="관찰 잡음을 학습과 같이 켠다. play 모드는 기본으로 꺼 둔다")
  ap.add_argument("--sweep-kind", choices=("delay", "noise"), default="delay",
                  help="무엇을 쓸어볼 것인가. **한 번에 하나만** 바꾼다")
  ap.add_argument("--sweep", default="",
                  help="명령 지연을 쉼표로 나열해 쓸어본다 (예: 0,0.25,0.5,1.0). "
                       "지연이 흔들림의 원인인지 **한 변인만 바꿔** 확인하는 용도")
  ap.add_argument("--tag", default="airborne_signature")
  n = ap.parse_args(argv)

  envd = apply_run_env(n.checkpoint)
  for kv in n.env:                                   # 덧씌우기는 manifest **뒤에**
    k, _, v = kv.partition("=")
    os.environ[k] = v
    envd[k] = v
    print(f"[air] 덧씌움: {k}={v}", flush=True)
  env, envw, policy, torch = build(n.checkpoint, obs_noise=n.obs_noise)
  if n.obs_noise:
    print("[air] 관찰 잡음 ON — 학습과 같은 조건 (실물 센서 잡음에 해당)", flush=True)
  # 2026-09-17: 처음엔 **행동 항목의 관절 순서**로 이름을 붙였는데, `actuator_force` 는
  # **액추에이터(ctrl) 순서**입니다. 두 순서가 같다는 보장이 없어 이름표가 어긋납니다.
  # 그 열에 붙은 이름이 틀리면 "왼다리가 힘을 안 쓴다" 같은 엉뚱한 결론이 나옵니다.
  act_names = [n.split("/")[-1] for n in env.scene["robot"].actuator_names]

  print(f"[air] 체크포인트 {n.checkpoint}", flush=True)
  print(f"[air] 학습 토글 {len(envd)}개 적용 (PYG_INIT_BENT="
        f"{envd.get('PYG_INIT_BENT', '없음')}, PYG_REL_STANDING="
        f"{envd.get('PYG_REL_STANDING', '없음')})", flush=True)
  if n.sweep:
    return sweep(env, envw, policy, torch, n)
  ground = rollout(env, envw, policy, torch, n.seconds, None)
  print(f"[air] 땅   : 높이 {ground['z'].mean():.3f} m, 리셋 {ground['resets']}회",
        flush=True)
  air = rollout(env, envw, policy, torch, n.seconds, n.lift, hold=n.hold)
  print(f"[air] 공중 : 높이 {air['z'].mean():.3f} m, 리셋 {air['resets']}회, "
        f"잡는 법 {n.hold}, 관절속도 |qd| 합 평균 "
        f"{np.abs(air['qvel']).sum(axis=1).mean():.2f} rad/s "
        f"(땅 {np.abs(ground['qvel']).sum(axis=1).mean():.2f})", flush=True)

  print(f"\n{'관절':<22}{'땅 |tau| 평균':>14}{'공중 |tau| 평균':>16}{'배수':>8}")
  rows = []
  for i, nm in enumerate(act_names):
    g = float(np.abs(ground["tau"][:, i]).mean())
    a = float(np.abs(air["tau"][:, i]).mean())
    rows.append(dict(joint=nm, ground=g, air=a))
    print(f"{nm:<22}{g:>14.2f}{a:>16.2f}{(g / a if a > 1e-6 else float('inf')):>8.1f}")

  dg, da = detectors(ground, act_names), detectors(air, act_names)
  print(f"\n{'판정후보':<18}{'땅 최소':>10}{'땅 평균':>10}{'땅 최대':>10}"
        f"{'공중 최소':>11}{'공중 평균':>11}{'공중 최대':>11}{'방향':>12}{'분리도':>9}{'문턱':>9}")
  best, det_rows = None, []
  for k in dg:
    sep, direction, thr = separation(dg[k], da[k])
    row = dict(name=k, ground_min=float(dg[k].min()), ground_mean=float(dg[k].mean()),
               ground_max=float(dg[k].max()), air_min=float(da[k].min()),
               air_mean=float(da[k].mean()), air_max=float(da[k].max()),
               direction=direction, separation=sep, threshold=thr)
    det_rows.append(row)
    print(f"{k:<18}{dg[k].min():>10.2f}{dg[k].mean():>10.2f}{dg[k].max():>10.2f}"
          f"{da[k].min():>11.2f}{da[k].mean():>11.2f}{da[k].max():>11.2f}"
          f"{direction:>12}{sep:>9.2f}{thr:>9.2f}")
    if best is None or sep > best["separation"]:
      best = row

  if best["separation"] <= 0:
    print("\n** 겹칩니다 — 이 후보들 중에는 안전하게 가를 수 있는 것이 없습니다.")
  else:
    print(f"\n가장 잘 갈리는 것: {best['name']} ({best['direction']}), "
          f"문턱 {best['threshold']:.2f}, 분리도 {best['separation']:.2f}")
  thr = best["threshold"]
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / f"{n.tag}.json"
  p.write_text(json.dumps(dict(checkpoint=n.checkpoint, seconds=n.seconds, lift=n.lift,
                               hold=n.hold, overrides=dict(kv.split('=', 1) for kv in n.env),
                               joints=rows, detectors=det_rows,
                               best=best["name"], direction=best["direction"],
                               separation=best["separation"], threshold=thr),
                          ensure_ascii=False, indent=2, default=float), encoding="utf-8")
  np.savez(OUT / f"{n.tag}.npz", act_names=np.array(act_names),
           joint_names=np.array(ground["act_joint_names"]),
           ground_tau=ground["tau"], air_tau=air["tau"],
           ground_q=ground["q"], air_q=air["q"],
           ground_qvel=ground["qvel"], air_qvel=air["qvel"])
  plot(OUT / f"{n.tag}.png", act_names, ground, air, n.hold,
       float(os.environ.get("PYG_ACTION_DELAY", 0.0) or 0.0))
  print(f"기록: {p}, {OUT / (n.tag + '.npz')}, {OUT / (n.tag + '.png')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
