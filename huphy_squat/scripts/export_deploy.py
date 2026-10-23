#!/usr/bin/env python3
"""Build the hardware deployment package deploy_asset/huphy_squat/ from huphy_squat checkpoints.

  cd huphy_squat && /home/roy/huphy/mjlab/.venv/bin/python scripts/export_deploy.py \
      --run-dir logs/rsl_rl/huphy_squat/<run> [--iters 5000 4500 4000]

Writes, per checkpoint N: policy_N.onnx (actor + empirical obs normalizer, deterministic mean,
input "obs" [1, 43] -> output "actions" [1, 12], opset 18 / IR 8) and policy_N.pt (copy).
Writes once: deploy_contract.json (obs / action spec read from the live env, per-checkpoint
sha256 + normalizer stats + golden I/O pairs), motor_params.json, train_env_manifest.json.

Golden pairs: actor observations sampled from the TRAINING env (full DR, pushes, observation
noise, obs / command lags, h* resampling) while the checkpoint itself drives it, plus the
torch actor's output for each (GPU, float32). The hand-written runtime files (policy_runner.py,
obs_adapter.py, example_loop.py, README.md, requirements.txt) are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "deploy_asset" / "huphy_squat"
TASK = "Huphy-Squat-Flat-v0"
GOLDEN_RESET, GOLDEN_LATER = 4, 20  # samples right after a reset / during the rollout


def sha256(p: Path) -> str:
  h = hashlib.sha256()
  h.update(p.read_bytes())
  return h.hexdigest()


def r(x, n=6):
  return [round(float(v), n) for v in x]


def build_env(num_envs: int, device: str):
  import huphy_squat  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=False)  # the training env as is
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def export_onnx(runner, path: Path, meta: dict) -> None:
  import onnx

  runner.export_policy_to_onnx(str(path.parent), path.name)
  m = onnx.load(str(path))
  m.ir_version = 8  # opset 18 needs IR >= 8; keeps onnxruntime >= 1.14 able to load it
  del m.metadata_props[:]
  for k, v in meta.items():
    e = onnx.StringStringEntryProto()
    e.key, e.value = k, v if isinstance(v, str) else json.dumps(v)
    m.metadata_props.append(e)
  onnx.checker.check_model(m)
  onnx.save(m, str(path))


def golden(env, w, pol, seed: int) -> dict:
  g = torch.Generator(device="cpu").manual_seed(seed)
  obs, _ = w.reset()
  steps = sorted(torch.randint(25, 400, (GOLDEN_LATER,), generator=g).tolist())
  obs_l, act_l = [], []

  def take(o, k):
    ids = torch.randint(0, env.num_envs, (k,), generator=g).tolist()
    with torch.no_grad():
      a = pol(o)
    x = o["actor"]
    for i in ids:
      obs_l.append(x[i].cpu().numpy().astype(np.float64))
      act_l.append(a[i].cpu().numpy().astype(np.float64))

  take(obs, GOLDEN_RESET)
  for t in range(max(steps) + 1):
    while steps and steps[0] == t:
      take(obs, 1)
      steps.pop(0)
    with torch.no_grad():
      obs, _, _, _ = w.step(pol(obs))
  O, A = np.stack(obs_l), np.stack(act_l)  # noqa: E741
  return {
    "how": "actor obs from the training env (full DR, pushes, obs noise, obs/command lags, h* "
           f"resampling) driven by this checkpoint; {GOLDEN_RESET} right after reset, {GOLDEN_LATER} "
           "at random steps 25-400; actions = torch actor output (deterministic mean, raw, before "
           "scale/offset/clip). Feed obs to the ONNX as is (float32).",
    "obs": [r(o, 7) for o in O],
    "actions": [r(a, 7) for a in A],
  }


def obs_spec(env) -> list[dict]:
  from huphy_squat import env_cfg as E
  from huphy_squat import robot

  om = env.observation_manager
  names = om.active_terms["actor"]
  dims = [int(np.prod(d)) for d in om.group_obs_term_dim["actor"]]
  assert names == ["base_ang_vel", "projected_gravity", "motor_pos", "actions", "height_command"], names
  assert dims == [3, 3, 24, 12, 1], dims
  mcfg = om._group_obs_term_cfgs["actor"][2]  # noqa: SLF001
  joints = list(mcfg.params["asset_cfg"].joint_names)
  assert mcfg.history_length == 2 and joints == list(robot.MOTOR_JOINT_NAMES)
  info = {
    "base_ang_vel": dict(func="imu_gyro", unit="rad/s", train_noise_uniform=E.GYRO_NOISE,
                         definition="IMU angular velocity in the IMU frame (+x forward, +y left, +z up). "
                                    "The IMU site imu_in_base has the base_link orientation (no rotation; "
                                    "pos -0.000001, 0.007078, -0.0695 m in base_link), so this is the pelvis "
                                    "angular velocity in the pelvis frame. Raw, no scale.",
                         sim_sensor="robot/imu_ang_vel (MuJoCo gyro at site imu_in_base)"),
    "projected_gravity": dict(func="imu_gravity", unit="unit vector", train_noise_uniform=E.GRAVITY_NOISE,
                              definition="Direction of gravity expressed in the IMU frame; (0, 0, -1) when "
                                         "upright. = R_world_imu^T (0, 0, -1). The sim sensor gives the world "
                                         "UP axis in the IMU frame and the observation negates it. Not the "
                                         "accelerometer reading.",
                              sim_sensor="-robot/imu_upvector (MuJoCo framezaxis of world in site imu_in_base)"),
    "motor_pos": dict(func="motor_encoder_rel", unit="rad", train_noise_uniform=E.ENCODER_NOISE,
                      history_frames=2, joint_names=joints,
                      definition="Motor encoder angle minus default pose angle (offset_rad of the same "
                                 "joint), 12 motors in joint_names order, 2 frames laid out [previous "
                                 "frame 12][current frame 12] (oldest first). Previous = the reading of the "
                                 "previous 20 ms control step. After a reset both frames are the first "
                                 "reading. Cranks: the crank MOTOR angle, not the ankle angle."),
    "actions": dict(func="last_action", unit="none",
                    definition="Raw network output of the previous control step (before scale / offset / "
                               "clip), same order as `actions`. Zeros after a reset."),
    "height_command": dict(func="height_command", unit="m",
                           definition="Target pelvis height h* in metres, unscaled, no offset. Clipped to "
                                      "h_star.range_m by the deploy code."),
  }
  out, s = [], 0
  for n, d in zip(names, dims, strict=True):
    e = {"term": n, "slice": [s, s + d], "width": d}
    e.update(info[n])
    e.setdefault("history_frames", 0)
    e.setdefault("train_noise_uniform", 0.0)
    out.append(e)
    s += d
  return out


def action_spec(env) -> tuple[list[dict], dict]:
  from huphy_squat import robot

  a = env.action_manager.get_term("joint_pos")
  names = list(a.target_names)
  assert names == list(robot.MOTOR_JOINT_NAMES), names
  off = a._offset[0].cpu().numpy()  # noqa: SLF001
  clip = a._clip[0].cpu().numpy()  # noqa: SLF001
  acts = []
  for i, n in enumerate(names):
    # use the exact values from robot.py (the env holds the float32 copies)
    assert abs(off[i] - robot.DEFAULT_MOTOR_POS[n]) < 1e-6 and abs(clip[i, 0] - robot.ACTION_CLIP[n][0]) < 1e-6
    acts.append({"index": i, "joint": n, "scale_rad_per_unit": robot.ACTION_SCALE,
                 "offset_rad": robot.DEFAULT_MOTOR_POS[n], "clip_rad": list(robot.ACTION_CLIP[n])})
  return acts, dict(robot.DEFAULT_MOTOR_POS)


def motor_params(env) -> dict:
  from huphy_squat import robot
  from huphy_squat.mdp.events import get_motor
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  def per(d):
    return dict(zip(*resolve_matching_names_values(d, list(robot.MOTOR_JOINT_NAMES), preserve_order=False)[1:], strict=True))

  fam, kp, kd, rated = per(robot.FAMILY), per(robot.STIFFNESS), per(robot.DAMPING), per(robot.RATED_TORQUE)
  m = get_motor(env)
  assert list(m.target_names) == list(robot.MOTOR_JOINT_NAMES)
  kp_live, kd_live = m.default_stiffness[0].tolist(), m.default_damping[0].tolist()
  rob = env.scene["robot"]
  lim = rob.data.joint_pos_limits[0].cpu().numpy()
  soft = rob.data.soft_joint_pos_limits[0].cpu().numpy()
  mj = env.sim.mj_model
  import mujoco
  motors = []
  for i, n in enumerate(robot.MOTOR_JOINT_NAMES):
    assert abs(kp_live[i] - kp[n]) < 1e-4 and abs(kd_live[i] - kd[n]) < 1e-4
    j = rob.joint_names.index(n)
    jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
    f = fam[n]
    tn = robot.tn_curve(f)
    motors.append({
      "joint": n, "family": f, "kp_Nm_per_rad": kp[n], "kd_Nms_per_rad": kd[n],
      "peak_torque_Nm": tn[0][1], "rated_continuous_torque_Nm": rated[n],
      **{k: v for k, v in robot.MOTOR_DYNAMICS[f].items()},
      "joint_axis_in_parent_link": r(mj.jnt_axis[jid], 4),
      "joint_range_rad": r(lim[j], 6), "soft_range_rad_(0.9)": r(soft[j], 6),
      "target_clip_rad": list(robot.ACTION_CLIP[n]), "default_rad": robot.DEFAULT_MOTOR_POS[n],
    })
  return {
    "control": "Joint-space PD on each motor driver (MIT mode): tau = Kp (q_target - q) - Kd qdot, "
               "q_target from the policy at 50 Hz, held for 20 ms (no interpolation in training). "
               "Velocity target 0, feed-forward torque 0. In training the PD ran every 5 ms physics step.",
    "torque_limit": "tau clamped to the measured 48 V torque-speed curve in the motoring quadrant "
                    "(tau * qdot > 0): |tau| <= T_N(|qdot|); when braking up to the peak torque. "
                    "T_N is flat (= peak) below the first curve point and 0 at the last (no-load) point.",
    "gains_note": "These Kp / Kd are the real-robot driver gains the policy was trained with (randomized "
                  "x0.8-1.2 per episode in training). Set exactly these on the drivers.",
    "command_delay_trained_s": [0.0, 0.020],
    "motors": motors,
    "tn_curves_48V": {f: {"speed_rad_s": r([p[0] for p in robot.tn_curve(f)], 5),
                          "torque_Nm": r([p[1] for p in robot.tn_curve(f)], 3),
                          "source": f"huphy_squat/huphy_squat/assets/motor_spec/{f}_TN_curve_48V.csv "
                                    "(+ a 200 rpm no-load point)"} for f in ("RS03", "RS04")},
    "motor_dynamics_note": "armature [kg m^2], frictionloss [N m], viscous_damping [N m s/rad] are "
                           "bench-measured on the output shaft and written into the sim joints (randomized "
                           "x0.8-1.2 in training). They describe the motor, nothing to set on the driver.",
  }


def manifest(git: str, iters) -> dict:
  from huphy_squat import env_cfg as E
  from huphy_squat import robot

  cfg = E.huphy_squat_env_cfg()
  ev = cfg.events
  return {
    "task": TASK,
    "training_code": f"huphy_squat/huphy_squat (git {git})",
    "checkpoints": list(iters),
    "training": "one PPO run (rsl_rl, 4096 envs, 5001 iterations, seed 43), actor 43-512-256-128-12 ELU "
                "with empirical obs normalization, CAPS smoothness loss 0.3; checkpoints 4000 / 4500 / "
                "5000 of the same run",
    "sim": {"physics_dt_s": cfg.sim.mujoco.timestep, "decimation": cfg.decimation,
            "policy_dt_s": cfg.sim.mujoco.timestep * cfg.decimation, "episode_s": cfg.episode_length_s,
            "terrain": "flat plane", "robot_mass_kg": 23.63},
    "command": {"h_star_range_m": list(E.HEIGHT_RANGE), "resampling_s": [3.0, 6.0], "keep_prob": 0.2,
                "profile": "step change to U(range) every 3-6 s (20 % keep the old value); first value "
                           "of an episode drawn fresh"},
    "reset": {"pose": "default pose + U(-0.05, 0.05) rad per motor; with prob 0.5 each hip roll gets "
                      "U(-0.10, +0.06) rad extra outward; feet on the floor, yaw random",
              "params": {k: v for k, v in ev["reset_pose"].params.items()}},
    "observation_noise_uniform": {"gyro_rad_s": E.GYRO_NOISE, "gravity": E.GRAVITY_NOISE,
                                  "encoder_rad": E.ENCODER_NOISE, "last_action": 0.0, "h_star": 0.0},
    "observation_lag": {"policy_steps": [0, E.OBS_MAX_LAG], "s": [0.0, 0.02 * E.OBS_MAX_LAG],
                        "note": "one lag per episode, same for IMU and encoders (one sensor packet); h* "
                                "and last action not delayed"},
    "command_delay": {"physics_steps": [robot.ACTUATOR.delay_min_lag, robot.ACTUATOR.delay_max_lag],
                      "s": [0.0, 0.005 * robot.ACTUATOR.delay_max_lag],
                      "note": "one lag per episode for all 12 motor targets"},
    "encoder_bias_rad": list(ev["encoder_bias"].params["bias_range"]),
    "imu_mount_rpy_deg": [-2.0, 2.0],
    "pd_gain_scale": {"kp": list(ev["pd_gains"].params["kp_range"]), "kd": list(ev["pd_gains"].params["kd_range"])},
    "motor_friction_damping_armature_scale": [0.8, 1.2],
    "passive_joint_friction_Nm": list(ev["passive_friction"].params["ranges"]),
    "passive_joint_damping": list(ev["passive_damping"].params["ranges"]),
    "foot_friction": list(E.FOOT_FRICTION),
    "payload_kg_at_pelvis_com": list(E.PAYLOAD_KG),
    "link_inertial_dr": {k: {"bodies": list(b), **{kk: list(vv) for kk, vv in p.items()}} for k, (b, p) in E.INERTIAL_DR.items()},
    "pushes": {"force_N": list(E.PUSH["force_range"]), "duration_s": list(E.PUSH["duration_s"]),
               "interval_s": list(E.PUSH["interval_s"]), "where": "pelvis, horizontal, random direction"},
    "terminations": ["timeout 20 s", "tilt > 70 deg", "non-foot ground contact", "NaN / physics blow-up"],
    "dr_schedule": "all DR at full strength from iteration 0, redrawn at every reset",
    "not_in_training": ["robot lifted / feet off the ground", "slopes, steps, soft ground",
                        "IMU gyro bias / drift", "loss of a sensor packet or control-rate jitter",
                        "motor gains other than motor_params.json (beyond x0.8-1.2)"],
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--iters", type=int, nargs="+", default=[5000, 4500, 4000])
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=7)
  n = ap.parse_args()

  from huphy_squat import robot
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  PKG.mkdir(parents=True, exist_ok=True)
  run = Path(n.run_dir).resolve()
  git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
  torch.manual_seed(n.seed)
  env = build_env(n.num_envs, n.device)
  rl = load_rl_cfg(TASK)
  w = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
  assert rl.clip_actions is None
  runner = MjlabOnPolicyRunner(w, asdict(rl), device=n.device)

  acts, default = action_spec(env)
  obs = obs_spec(env)
  _, base_z = robot.load_default_pose()
  default_h = base_z - robot.RESET_CLEARANCE
  ckpts = {}
  for it in n.iters:
    src = run / f"model_{it}.pt"
    runner.load(str(src), load_cfg={"actor": True}, strict=True, map_location=n.device)
    pol = runner.get_inference_policy(device=n.device)
    actor = runner.alg.get_policy()
    norm = actor.obs_normalizer
    onnx_path, pt_path = PKG / f"policy_{it}.onnx", PKG / f"policy_{it}.pt"
    export_onnx(runner, onnx_path, {
      "task": TASK, "package": "huphy_squat", "checkpoint_iteration": str(it),
      "obs_dim": "43", "action_dim": "12", "joint_names": list(robot.MOTOR_JOINT_NAMES),
      "default_motor_pos": [robot.DEFAULT_MOTOR_POS[j] for j in robot.MOTOR_JOINT_NAMES],
      "action_scale": str(robot.ACTION_SCALE), "h_star_range_m": [0.65, 0.895],
      "note": "obs normalizer is inside this graph; feed raw obs (see deploy_contract.json)"})
    shutil.copyfile(src, pt_path)
    g = golden(env, w, pol, seed=n.seed + it)
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = np.stack([s.run(None, {"obs": np.asarray(o, np.float32)[None]})[0][0] for o in g["obs"]])
    err = float(np.abs(got - np.asarray(g["actions"])).max())
    print(f"[export] {it}: ONNX vs torch on {len(g['obs'])} golden pairs: max diff {err:.3e}")
    assert err < 1e-4
    ckpts[str(it)] = {
      "iteration": it, "onnx": onnx_path.name, "onnx_sha256": sha256(onnx_path),
      "pt": pt_path.name, "pt_sha256": sha256(pt_path),
      "obs_normalizer_mean": r(norm.mean.cpu().numpy(), 7), "obs_normalizer_std": r(norm.std.cpu().numpy(), 7),
      "obs_normalizer_eps": float(norm.eps),
      "golden": g, "golden_onnx_max_diff_on_export_machine": err,
    }

  contract = {
    "package": "huphy_squat",
    "task": TASK,
    "what": "HUPHY LegOnly biped (12 motors) stands on flat ground and holds a commanded pelvis height h*.",
    "control_rate_hz": 50.0,
    "obs_dim": 43,
    "action_dim": 12,
    "default_checkpoint": 5000,
    "target_pos_rad": "target = offset_rad + scale_rad_per_unit * action, then clipped to clip_rad "
                      "(the action itself is NOT clipped; last_action is the unclipped raw value)",
    "actions": acts,
    "default_pose_rad": default,
    "observations": obs,
    "h_star": {
      "unit": "m", "range_m": [0.65, 0.895], "recommended_range_m": [0.70, 0.891],
      "definition": "pelvis (base_link frame origin) height above the flat ground the feet stand on: "
                    "z_base_link_origin - z_ground. Not the CoM, not the IMU height (the IMU site is "
                    "0.0695 m below the base_link origin).",
      "default_pose_height_m": round(default_h, 4),
      "default_pose_height_note": "base_link origin height with the default pose and the soles flat "
                                  "on the ground (geometry, assets/linkage_table.npz). Start with this h*.",
      "out_of_range": "clipped to range_m, one warning printed until the next reset()",
      "training_profile": "step changes every 3-6 s to a uniform value in range_m",
    },
    "frames": {
      "body": "+x forward, +y left, +z up (pelvis / base_link = IMU frame)",
      "joint_sign": "positive joint angle = right-hand rotation about the joint axis in "
                    "motor_params.json (joint_axis_in_parent_link); L joints are the mirror of R: "
                    "hip pitch / roll / yaw, knee, crank A / B all have q_L = -q_R in the mirrored pose",
      "zero": "joint angle 0 = the zero of the MJCF model the policy was trained on; the real encoders "
              "must be calibrated to the same zero and sign. The default pose is NOT zero (default_pose_rad)",
    },
    "not_in_deploy": ["observation noise", "observation / command lags", "encoder bias", "any DR"],
    "notes": [
      "actions, the encoder history and last_action all use the same 12-motor order (model order).",
      "the observation carries encoder angles RELATIVE to offset_rad, two frames, oldest first.",
      "the ONNX graph contains the empirical observation normalizer; obs_normalizer_* are diagnostic.",
    ],
    "checkpoints": ckpts,
  }
  (PKG / "deploy_contract.json").write_text(json.dumps(contract, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
  (PKG / "motor_params.json").write_text(json.dumps(motor_params(env), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
  (PKG / "train_env_manifest.json").write_text(json.dumps(manifest(git, n.iters), indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
  print(f"[export] wrote {PKG}")
  env.close()


if __name__ == "__main__":
  main()
