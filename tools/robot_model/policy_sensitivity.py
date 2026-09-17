#!/usr/bin/env python3
"""**정책이 관측 잡음에 얼마나 민감한지**를 저장본 하나에서 바로 잰다. 굴려 볼 필요 없다.

2026-09-18 · 사용자 질문: *"간단하게 짧은 학습으로 검증하는 방법은 부족한가?"*

## 왜 이게 빠른가

우리가 고치려는 양은 딱 하나입니다 — **"관측이 조금 움직일 때 행동이 얼마나 움직이나."**
그건 **신경망의 성질**이지 로봇의 성질이 아닙니다. 그래서 로봇을 세워 두고 20 초씩
64 마리를 굴릴 필요가 없습니다. 관측 몇 천 개를 신경망에 넣고, 살짝 흔든 것을 한 번 더
넣어서, 답이 얼마나 달라지는지 보면 끝납니다. **몇 초** 걸립니다.

    민감도 = ‖ π(s + σ·잡음) − π(s) ‖ / σ        (정규화된 관측 공간)

이 숫자가 곧 **경험적 립시츠 상수**입니다. 작을수록 잡음을 덜 키웁니다.

## 관측은 어디서 가져오나

**진짜 굴려서** 가져옵니다. 아무 숫자나 넣으면 정책이 평생 본 적 없는 자리에서 재게
되고, 그건 우리가 궁금한 것이 아닙니다. 짧게(기본 10 초) 굴려서 실제로 방문한 관측을
모으고, 그 위에서 잽니다.

## 같이 내는 것 — 행동 떨림

LCP 논문이 쓰는 지표와 같은 뜻으로, 명령의 **2 차 차분**을 냅니다. 민감도가 "잡음에
얼마나 반응하나" 라면 이쪽은 "실제로 낸 명령이 얼마나 덜덜거렸나" 입니다. 둘은 다릅니다
— 잡음이 없으면 민감한 정책도 안 떨거든요.

## 무엇을 못 하나 (중요)

**강건성은 못 잽니다.** 밀어도 안 넘어지는지는 굴려 봐야 압니다. 이 도구는
*"떨림이 줄었나"* 의 절반만 답합니다. 나머지 절반은 `stand_robustness.py` 몫입니다.

    policy_sensitivity.py --checkpoint a.pt b.pt c.pt --label v7b caps lcp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airborne_signature import apply_run_env, build                # noqa: E402

OUT = REPO / "docs/data"
SIGMAS = (0.05, 0.1, 0.2)      # 정규화 공간에서 흔드는 크기


def collect(env, envw, policy, torch, seconds: float):
  """짧게 굴려서 **실제로 방문한 관측**과 낸 명령을 모은다."""
  obs, _ = envw.reset()
  lat, acts = [], []
  actor = None
  for _ in range(int(seconds * 50)):
    with torch.inference_mode():
      a = policy(obs)
    obs, _, _, _ = envw.step(a)
    try:
      env.command_manager.get_term("twist").vel_command_b[:] = 0.0
    except Exception:                                              # noqa: BLE001
      pass
    acts.append(np.asarray(a[0].cpu()))
  return acts


def measure(checkpoint: str, seconds: float, n_obs: int = 2048) -> dict:
  import torch

  apply_run_env(checkpoint)
  env, envw, policy, _torch, _cfg = None, None, None, None, None
  env, envw, policy, torch = build(checkpoint, obs_noise=True)[:4]
  runner_actor = None
  # 정책 함수 안의 신경망을 꺼낸다 (민감도는 신경망의 성질이다)
  import mjlab  # noqa: F401

  # build() 는 추론용 함수만 주므로, 러너에서 actor 를 다시 얻는다
  from dataclasses import asdict
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  runner = (load_runner_cls("Mjlab-Velocity-Flat-Pygmalion") or MjlabOnPolicyRunner)(
    envw, asdict(load_rl_cfg("Mjlab-Velocity-Flat-Pygmalion")), device="cpu")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
  runner_actor = runner.alg.actor

  obs, _ = envw.reset()
  lats, acts = [], []
  for _ in range(int(seconds * 50)):
    with torch.inference_mode():
      lats.append(runner_actor.get_latent(obs).clone())
      a = policy(obs)
    acts.append(np.asarray(a[0].cpu()).copy())
    obs, _, _, _ = envw.step(a)
    try:
      env.command_manager.get_term("twist").vel_command_b[:] = 0.0
    except Exception:                                              # noqa: BLE001
      pass

  z = torch.cat(lats, dim=0)
  if z.shape[0] > n_obs:
    z = z[torch.randperm(z.shape[0])[:n_obs]]

  def head(x):
    out = runner_actor.mlp(x)
    d = getattr(runner_actor, "distribution", None)
    return d.deterministic_output(out) if d is not None else out

  out: dict = {"checkpoint": checkpoint}
  with torch.inference_mode():
    a0 = head(z)
    for s in SIGMAS:
      d = (head(z + s * torch.randn_like(z)) - a0).norm(dim=-1)
      out[f"sens_sigma{s}"] = float((d / s).mean())
      out[f"sens_sigma{s}_p95"] = float(np.percentile(np.asarray(d.cpu()) / s, 95))
  A = np.array(acts)
  out["action_jitter"] = float(np.abs(A[2:] - 2 * A[1:-1] + A[:-2]).sum(axis=1).mean())
  out["action_rate"] = float(np.abs(A[1:] - A[:-1]).sum(axis=1).mean())
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", nargs="+", required=True)
  ap.add_argument("--label", nargs="+", default=None)
  ap.add_argument("--seconds", type=float, default=10.0)
  ap.add_argument("--tag", default="policy_sensitivity")
  n = ap.parse_args(argv)
  labels = n.label or [Path(c).parent.name[-20:] for c in n.checkpoint]

  rows = []
  print(f"\n{'정책':<22}{'민감도 σ=0.05':>14}{'σ=0.1':>10}{'σ=0.2':>10}"
        f"{'명령 떨림':>12}{'명령 변화':>12}")
  for lab, ck in zip(labels, n.checkpoint):
    r = measure(ck, n.seconds)
    r["label"] = lab
    rows.append(r)
    print(f"{lab:<22}{r['sens_sigma0.05']:>14.3f}{r['sens_sigma0.1']:>10.3f}"
          f"{r['sens_sigma0.2']:>10.3f}{r['action_jitter']:>12.4f}"
          f"{r['action_rate']:>12.4f}", flush=True)

  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / f"{n.tag}.json"
  p.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float),
               encoding="utf-8")
  print(f"\n민감도 = ‖π(s+σ·잡음) − π(s)‖ / σ. 작을수록 잡음을 덜 키운다.")
  print(f"기록: {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
