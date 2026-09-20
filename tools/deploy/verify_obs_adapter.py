#!/usr/bin/env python3
"""**실물용 어댑터가 시뮬레이터와 같은 관측을 내는지** 실제로 굴려서 대조한다.

2026-09-20 · 같은 것을 두 번 구현했으니(시뮬레이터 쪽 생성기 + 실물용 독립 어댑터)
**두 구현이 같은 값을 내는지**가 유일한 안전장치입니다. 라즈베리파이에는 mjlab 을
못 올리므로 독립 구현은 피할 수 없고, 남는 방법은 대조뿐입니다.

## 어떻게 대조하나

학습 환경을 굴리면서 매 제어 주기에 두 가지를 같이 적습니다.

* 환경이 정책에게 실제로 준 관측 45 개 (**정답**)
* 그 순간 로봇에서 읽을 수 있는 값들 — 자이로 3, 중력방향 3, 관절각 12,
  직전 행동 12, 명령 3 — 을 어댑터에 넣어 만든 관측 45 개

둘이 **모든 스텝에서 같아야** 합니다. 다르면 어느 칸이 얼마나 다른지 찍습니다.

관측 잡음은 **끕니다**. 어댑터는 잡음을 만들지 않으므로, 켜 두면 잡음 때문에 다른 것인지
배치가 틀려서 다른 것인지 구분이 안 됩니다.

    verify_obs_adapter.py --contract <deploy_contract.json> --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/robot_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airborne_signature import apply_run_env, build                # noqa: E402
from obs_adapter import ObsAdapter                                  # noqa: E402


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--contract", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--tol", type=float, default=1e-5)
  n = ap.parse_args(argv)

  ad = ObsAdapter.from_file(n.contract)
  apply_run_env(n.checkpoint)
  env, envw, policy, torch = build(n.checkpoint, obs_noise=False)[:4]
  import mujoco

  m = env.sim.mj_model
  robot = env.scene["robot"]
  alljoints = list(robot.joint_names)

  # 센서 주소 — 명세에 적힌 센서 이름으로 찾는다
  sens = {}
  for t in ad.terms:
    sn = t.get("sensor_name")
    if sn:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn)
      if sid < 0:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn.split("/")[-1])
      if sid < 0:
        raise SystemExit(f"센서 {sn} 를 모형에서 못 찾았습니다")
      sens[t["term"]] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))

  jidx = [alljoints.index(j) for j in ad.hist_joints]
  obs, _ = envw.reset()
  ad.reset()
  last_raw = np.zeros(ad.action_dim)
  worst, worst_step, n_bad = 0.0, -1, 0
  for k in range(n.steps):
    truth = np.asarray(obs["actor"][0].cpu() if hasattr(obs, "__getitem__")
                       and "actor" in obs else obs[0].cpu(), dtype=np.float64)
    sd = np.asarray(env.sim.data.sensordata[0].clone().cpu())
    q = np.asarray(robot.data.joint_pos[0].clone().cpu())
    cmd = np.asarray(env.command_manager.get_command("twist")[0].cpu())
    mine = ad.build(
      gyro=sd[slice(*(lambda a, d: (a, a + d))(*sens["base_ang_vel"]))],
      # 센서는 **위 방향**을 낸다. 관측은 그것을 뒤집은 **중력 방향**이다
      # (mjlab observations.py: `return -sensor.data`). 2026-09-20 에 이 부호를
      # 빠뜨렸다가 이 대조에서 잡혔다 — 크기가 같아 다른 검사로는 안 걸린다.
      projected_gravity=-sd[slice(*(lambda a, d: (a, a + d))(*sens["projected_gravity"]))],
      joint_pos={j: float(q[i]) for j, i in zip(ad.hist_joints, jidx)},
      last_action=last_raw,
      command=cmd,
    ).astype(np.float64)
    d = np.abs(mine - truth)
    if d.max() > worst:
      worst, worst_step = float(d.max()), k
    if d.max() > n.tol:
      n_bad += 1
      if n_bad <= 3:
        bad = np.where(d > n.tol)[0]
        print(f"  스텝 {k}: 다른 칸 {len(bad)} 개, 최대 {d.max():.3e}")
        for i in bad[:6]:
          term = next(t["term"] for t in ad.terms if t["slice"][0] <= i < t["slice"][1])
          print(f"    칸 {i:2d} ({term}): 어댑터 {mine[i]:+.6f}  환경 {truth[i]:+.6f}")
    with torch.inference_mode():
      a = policy(obs)
    last_raw = np.asarray(a[0].cpu(), dtype=np.float64)
    obs, _, _, _ = envw.step(a)

  ok = n_bad == 0
  print(f"\n{n.steps} 스텝 대조: 최대 차이 {worst:.3e} (스텝 {worst_step}), "
        f"허용치 {n.tol:.0e} 초과 {n_bad} 스텝")
  print("판정: " + ("통과 — 어댑터가 시뮬레이터와 같은 관측을 냅니다"
                   if ok else "★ 불일치 — 위 칸을 보고 고칠 것"))
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
