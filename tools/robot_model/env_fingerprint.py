#!/usr/bin/env python3
"""**두 컴퓨터가 정말 같은 로봇·같은 규칙으로 학습하는지** 지문 하나로 대조한다.

2026-09-17 · 사용자 지시: *"10.8.0.14 에도 GPU 있으니 여기 환경도 활용해서 학습해."*

## 왜 필요한가 — 가장 무서운 실패

새 컴퓨터에 학습 환경을 옮길 때 제일 위험한 것은 **설치가 실패하는 것이 아닙니다.**
실패하면 바로 압니다. 무서운 것은 **잘 도는데 조금 다른 로봇을 학습하는 것**입니다.
모형 파일 한 개가 예전 판이거나, 환경변수 하나가 빠지거나, 라이브러리 판이 달라
기본 자세가 1도 다르면 — 화면에는 아무 표시도 안 나고, 며칠 뒤 결과를 비교할 때
"왜 안 맞지" 로만 나타납니다.

우리는 이미 같은 종류로 두 번 데었습니다. `PYG_MODEL_TAG` 를 빼먹어 관절 범위가 조용히
옛 값으로 돌아간 일(docs/146), `PYG_INIT_BENT` 를 빼먹어 멀쩡한 정책이 2초 만에
주저앉은 일(docs/148 §1a). 둘 다 **아무 오류 메시지도 없었습니다.**

## 무엇을 지문으로 삼나

학습 결과를 바꿀 수 있는 것만 넣습니다. 바뀌면 안 되는 것이 바뀌면 지문이 달라집니다.

* **모형** — 관절 수·자유도·액추에이터 수, 관절 범위, 링크 질량, 기어비, 힘 한계
* **기준 자세** — `default_q` (행동 오프셋의 바탕. 여기가 다르면 같은 숫자가 다른 각도)
* **관측** — 항목 이름·차원·순서, 잡음 크기
* **행동** — 배율·오프셋·자르기
* **보상** — 항목 이름과 가중치
* **토글** — `PYG_*` 환경변수 전부

일부러 **안** 넣는 것: 무작위 씨앗, 환경 개수, 장치(cpu/cuda), 경로. 이것들은 달라도
같은 로봇을 같은 규칙으로 학습하는 것입니다.

    env_fingerprint.py                 # 지문 한 줄
    env_fingerprint.py --full          # 무엇이 들어갔는지 전부
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MJLAB = REPO / "mujoco-sim/mjlab"
sys.path.insert(0, str(MJLAB / "src"))
sys.path.insert(0, str(MJLAB / "analysis"))

TASK = "Mjlab-Velocity-Flat-Pygmalion"


def _r(x, nd=9):
  """부동소수점을 **자릿수를 잘라서** 넣는다. 기계마다 마지막 자리가 다를 수 있다."""
  a = np.asarray(x, dtype=float)
  return np.round(a, nd).tolist()


def fingerprint(play: bool = False) -> dict:
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  m = env.sim.mj_model
  robot = env.scene["robot"]
  aterm = env.action_manager.get_term("joint_pos")

  import mujoco

  jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
  out = {
    "model": {
      "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "nbody": int(m.nbody),
      "njnt": int(m.njnt), "neq": int(m.neq),
      "joint_names": jnames,
      "jnt_range": _r(m.jnt_range, 6),
      "body_mass": _r(m.body_mass, 6),
      "actuator_gear": _r(m.actuator_gear, 6),
      "actuator_forcerange": _r(m.actuator_forcerange, 4),
      "actuator_gainprm": _r(m.actuator_gainprm, 4),
      "actuator_biasprm": _r(m.actuator_biasprm, 4),
      "dof_damping": _r(m.dof_damping, 6),
      "dof_frictionloss": _r(m.dof_frictionloss, 6),
    },
    "default_q": _r(robot.data.default_joint_pos[0].cpu().numpy(), 9),
    "actuator_names": [str(x) for x in robot.actuator_names],
    "action": {
      "dim": int(aterm.action_dim),
      "target_names": [str(x) for x in aterm.target_names],
      "scale": _r(getattr(aterm, "_scale", 0.0), 9),
      "offset": _r(getattr(aterm, "_offset", 0.0), 9),
      "type": type(aterm).__name__,
    },
    "obs": {},
    "rewards": {k: float(v.weight) for k, v in sorted(cfg.rewards.items())},
    "terminations": sorted(cfg.terminations.keys()),
    "events": sorted(cfg.events.keys()),
    "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("PYG_")},
  }
  for gname, group in cfg.observations.items():
    terms = getattr(group, "terms", None) or {}
    out["obs"][gname] = [
      {"name": n, "func": getattr(t.func, "__name__", str(t.func)),
       "noise": (None if t.noise is None else str(t.noise)),
       "history": int(getattr(t, "history_length", 0) or 0)}
      for n, t in terms.items()
    ]
  blob = json.dumps(out, sort_keys=True, ensure_ascii=False)
  out["_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--full", action="store_true", help="지문에 들어간 내용을 전부 낸다")
  ap.add_argument("--play", action="store_true")
  ap.add_argument("--out", default="")
  n = ap.parse_args(argv)
  fp = fingerprint(play=n.play)
  if n.full or n.out:
    txt = json.dumps(fp, ensure_ascii=False, indent=2, sort_keys=True)
    if n.out:
      Path(n.out).write_text(txt, encoding="utf-8")
    else:
      print(txt)
  print(f"FINGERPRINT {fp['_sha256']}  "
        f"(nq={fp['model']['nq']} nu={fp['model']['nu']} "
        f"action={fp['action']['dim']} rewards={len(fp['rewards'])} "
        f"PYG토글={len(fp['env'])})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
