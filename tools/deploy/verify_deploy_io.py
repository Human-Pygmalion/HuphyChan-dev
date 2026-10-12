#!/usr/bin/env python3
"""Deploy code vs mjlab: are the inputs AND the outputs the same, step by step?

2026-10-12 (user: "입/출력 mjlab 이랑 Deploy 코드간 동일한지 검증해서 제대로 짰는지 확인해").
`verify_obs_adapter.py` checks only the 45 observations. This checks the whole chain the robot
runs, against the chain mjlab ran in training, in one closed loop:

  1. observation  : deploy `ObsAdapter.build(...)`            vs  mjlab `obs["actor"]`
  2. network      : deploy ONNX raw action (`StandingPolicy`)  vs  mjlab torch policy raw action
  3. joint target : deploy `inference()` target [rad]          vs  mjlab `JointPositionAction`
                                                                   processed target (scale+offset+clip)

The sensor values handed to the deploy side are only what a robot can read: gyro, IMU up-vector
(sign flipped to gravity), the 12 motor angles by NAME, the command. mjlab drives the physics with
its own torch action, so both sides see the same state every step.

Several commands are held in turn (stand, walk, sideways, turn, backward+turn) so the command
slots and the clip are exercised. Observation noise, DR events and terminations are off (the deploy
side makes no noise). Encoder bias is off, so mjlab's target equals its processed action.

  cd mujoco-sim/mjlab && .venv/bin/python3 ../../tools/deploy/verify_deploy_io.py \\
      --contract <deploy_contract.json> --onnx <policy.onnx> --checkpoint <run>/model_N.pt
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/robot_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from airborne_signature import apply_run_env, build  # noqa: E402
from policy_runner import StandingPolicy  # noqa: E402

COMMANDS = [(0.0, 0.0, 0.0), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.8), (-0.8, 0.0, -0.5), (2.0, 0.0, 0.0)]


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--contract", required=True)
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--steps-per-cmd", type=int, default=150)
  ap.add_argument("--tol-obs", type=float, default=1e-4)
  ap.add_argument("--tol-act", type=float, default=1e-4)
  n = ap.parse_args(argv)

  pol = StandingPolicy.from_files(n.contract, n.onnx, warn_period=False)
  ad = pol.adapter
  apply_run_env(n.checkpoint)
  env, envw, policy, torch = build(n.checkpoint, obs_noise=False)[:4]
  import mujoco

  m = env.sim.mj_model
  robot = env.scene["robot"]
  alljoints = list(robot.joint_names)
  aterm = env.action_manager.get_term("joint_pos")
  mj_targets = list(aterm.target_names)
  if sorted(mj_targets) != sorted(pol.joint_names):
    raise SystemExit(f"joint sets differ: mjlab {mj_targets} vs deploy {pol.joint_names}")
  if mj_targets != list(pol.joint_names):
    print(f"note: action order mjlab {mj_targets} / deploy {pol.joint_names} (compared by name)")
  sens = {}
  for t in ad.terms:
    sn = t.get("sensor_name")
    if sn:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn)
      if sid < 0:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sn.split("/")[-1])
      if sid < 0:
        raise SystemExit(f"sensor {sn} not in the model")
      sens[t["term"]] = (int(m.sensor_adr[sid]), int(m.sensor_dim[sid]))
  jidx = [alljoints.index(j) for j in ad.hist_joints]
  tw = env.command_manager.get_term("twist")

  obs, _ = envw.reset()
  pol.reset()
  for attr in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
    if hasattr(tw, attr):
      getattr(tw, attr)[:] = False
  worst = {"obs": 0.0, "raw": 0.0, "target": 0.0}
  where = {"obs": None, "raw": None, "target": None}
  bad = {"obs": 0, "raw": 0, "target": 0}
  clipped = 0
  rows = []
  k = 0
  for cmd in COMMANDS:
    seg = {"obs": 0.0, "raw": 0.0, "target": 0.0}
    for _ in range(n.steps_per_cmd):
      truth = np.asarray(obs["actor"][0].cpu(), dtype=np.float64)
      sd = np.asarray(env.sim.data.sensordata[0].clone().cpu())
      q = np.asarray(robot.data.joint_pos[0].clone().cpu())
      a0, d0 = sens["base_ang_vel"]
      a1, d1 = sens["projected_gravity"]
      reading = dict(
        gyro=sd[a0:a0 + d0],
        projected_gravity=-sd[a1:a1 + d1],  # the IMU site gives "up"; the policy wants gravity
        joint_pos={j: float(q[i]) for j, i in zip(ad.hist_joints, jidx)},
        command=np.asarray(tw.vel_command_b[0].cpu()),
      )
      # the adapter keeps the joint-angle history: build the comparison observation on a COPY,
      # so the real call below pushes the history exactly once (a second build would shift it)
      mine = copy.deepcopy(pol.adapter).build(last_action=pol.last_raw_action,
                                              **reading).astype(np.float64)
      targets = pol.inference(**reading)  # deploy: obs -> ONNX -> target
      raw_dep = pol.last_raw_action
      with torch.inference_mode():
        act = policy(obs)
      # pin the command AFTER reading what the current observation saw; the next observation
      # (built at the end of this step) then carries `cmd` on both sides
      tw.vel_command_b[:] = torch.tensor(cmd, dtype=tw.vel_command_b.dtype)
      raw_mj = np.asarray(act[0].cpu(), dtype=np.float64)
      obs, _, _, _ = envw.step(act)
      proc = np.asarray(aterm._processed_actions[0].cpu(), dtype=np.float64)
      clip_lo = np.asarray(aterm._clip[0, :, 0].cpu()) if aterm._clip is not None else None
      if clip_lo is not None:
        clip_hi = np.asarray(aterm._clip[0, :, 1].cpu())
        clipped += int(((proc <= clip_lo + 1e-9) | (proc >= clip_hi - 1e-9)).sum())
      tgt_dep = np.array([targets[name] for name in mj_targets])
      d = {"obs": float(np.abs(mine - truth).max()), "raw": float(np.abs(raw_dep - raw_mj).max()),
           "target": float(np.abs(tgt_dep - proc).max())}
      for key, v in d.items():
        seg[key] = max(seg[key], v)
        if v > worst[key]:
          worst[key], where[key] = v, (k, cmd)
        tol = n.tol_obs if key == "obs" else n.tol_act
        if v > tol:
          bad[key] += 1
          if bad[key] <= 3:
            print(f"  step {k} cmd {cmd}: {key} differs by {v:.3e}")
            if key == "obs":
              for i in np.where(np.abs(mine - truth) > tol)[0][:6]:
                term = next(t["term"] for t in ad.terms if t["slice"][0] <= i < t["slice"][1])
                print(f"    slot {i:2d} ({term}): deploy {mine[i]:+.6f}  mjlab {truth[i]:+.6f}")
      k += 1
    rows.append((cmd, seg))

  print(f"\n{'command (vx, vy, wz)':<24} {'obs max diff':>13} {'raw max diff':>13} {'target max diff':>16}")
  for cmd, seg in rows:
    print(f"{str(cmd):<24} {seg['obs']:13.3e} {seg['raw']:13.3e} {seg['target']:16.3e}")
  print(f"\n{k} steps. worst: obs {worst['obs']:.3e} {where['obs']}, raw {worst['raw']:.3e} "
        f"{where['raw']}, target {worst['target']:.3e} {where['target']}")
  print(f"target values at a clip bound (clip exercised): {clipped}")
  ok = not any(bad.values())
  print("VERDICT: " + ("PASS - deploy observation, network output and joint targets equal mjlab"
                       if ok else f"FAIL - steps over tolerance {bad}"))
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
