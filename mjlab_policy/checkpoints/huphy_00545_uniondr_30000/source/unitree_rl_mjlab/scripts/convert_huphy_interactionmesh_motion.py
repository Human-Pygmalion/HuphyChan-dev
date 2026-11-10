#!/usr/bin/env python3
"""Convert HoloSoma HUPHY qpos output to MJLab's 50 Hz mimic format."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_INPUT = Path(
  "/home/jooyoungbae/holosoma/outputs/00399_huphy_retarget/"
  "00399_huphy_interactionmesh_lower9.npz"
)
DEFAULT_XML = Path(
  "/home/jooyoungbae/HuphyChan-dev/pygmalion_locomotion/assets/"
  "pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_"
  "proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / (
  "src/assets/motions/huphy/00399_interactionmesh_50hz.npz"
)

MOTOR_JOINTS = {
  f"{side}_{name}_joint"
  for side in ("L", "R")
  for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")
}


def args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--fps", type=float, default=50.0)
  return parser.parse_args()


def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
  source_t = np.arange(len(qpos), dtype=np.float64) / source_fps
  target_t = np.arange(round(source_t[-1] * target_fps) + 1, dtype=np.float64) / target_fps
  target_t[-1] = min(target_t[-1], source_t[-1])
  result = np.empty((len(target_t), qpos.shape[1]), dtype=np.float64)
  for column in list(range(3)) + list(range(7, qpos.shape[1])):
    result[:, column] = np.interp(target_t, source_t, qpos[:, column])
  # scipy uses xyzw while MuJoCo stores wxyz.
  xyzw = np.concatenate((qpos[:, 4:7], qpos[:, 3:4]), axis=1)
  interp_xyzw = Slerp(source_t, Rotation.from_quat(xyzw))(target_t).as_quat()
  result[:, 3:7] = np.concatenate((interp_xyzw[:, 3:4], interp_xyzw[:, :3]), axis=1)
  return result


def equality_residual(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> np.ndarray:
  data.qpos[:] = q
  mujoco.mj_forward(model, data)
  rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
  return np.asarray(data.efc_pos[rows], dtype=np.float64).copy()


def close_passive_linkage(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, float]:
  data = mujoco.MjData(model)
  passive_qadr = []
  for joint_id in range(1, model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name not in MOTOR_JOINTS:
      passive_qadr.append(int(model.jnt_qposadr[joint_id]))
  passive_qadr = np.asarray(passive_qadr, dtype=np.int64)

  closed = qpos.copy()
  max_residual = 0.0
  for frame, q in enumerate(closed):
    def fun(x: np.ndarray) -> np.ndarray:
      q[passive_qadr] = x
      return equality_residual(model, data, q)

    initial = q[passive_qadr].copy()
    if np.max(np.abs(fun(initial)), initial=0.0) > 1e-8:
      solution = least_squares(
        fun, initial, method="lm", xtol=1e-11, ftol=1e-11, gtol=1e-11,
        max_nfev=40,
      )
      q[passive_qadr] = solution.x
    residual = np.max(np.abs(fun(q[passive_qadr])), initial=0.0)
    max_residual = max(max_residual, float(residual))
    if frame % 250 == 0:
      print(f"Closed-loop projection {frame + 1}/{len(closed)}", flush=True)
  return closed, max_residual


def qvel_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
  qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
  dt = 1.0 / fps
  for i in range(len(qpos)):
    if i == 0:
      mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i], qpos[i + 1])
    elif i == len(qpos) - 1:
      mujoco.mj_differentiatePos(model, qvel[i], dt, qpos[i - 1], qpos[i])
    else:
      mujoco.mj_differentiatePos(model, qvel[i], 2.0 * dt, qpos[i - 1], qpos[i + 1])
  return qvel


def body_kinematics(
  model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  nframes, nbodies = len(qpos), model.nbody - 1
  pos = np.empty((nframes, nbodies, 3), dtype=np.float32)
  quat = np.empty((nframes, nbodies, 4), dtype=np.float32)
  lin_vel = np.empty((nframes, nbodies, 3), dtype=np.float32)
  ang_vel = np.empty((nframes, nbodies, 3), dtype=np.float32)
  velocity = np.empty(6, dtype=np.float64)
  for frame, (q, v) in enumerate(zip(qpos, qvel)):
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    pos[frame] = data.xpos[1:]
    quat[frame] = data.xquat[1:]
    for body_id in range(1, model.nbody):
      mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
      )
      ang_vel[frame, body_id - 1] = velocity[:3]
      lin_vel[frame, body_id - 1] = velocity[3:]
  return pos, quat, lin_vel, ang_vel


def main() -> None:
  cfg = args()
  source = np.load(cfg.input, allow_pickle=True)
  source_qpos = np.asarray(source["qpos"], dtype=np.float64)
  source_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
  model = mujoco.MjModel.from_xml_path(str(cfg.xml))
  if source_qpos.shape[1] != model.nq:
    raise ValueError(f"qpos has {source_qpos.shape[1]} columns, model nq={model.nq}")

  qpos = resample_qpos(source_qpos, source_fps, cfg.fps)
  qpos, max_loop_residual = close_passive_linkage(model, qpos)
  qvel = qvel_from_qpos(model, qpos, cfg.fps)
  body_pos, body_quat, body_lin_vel, body_ang_vel = body_kinematics(model, qpos, qvel)
  joint_names = np.asarray([
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(1, model.njnt)
  ])
  body_names = np.asarray([
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(1, model.nbody)
  ])

  cfg.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    cfg.output,
    joint_pos=qpos[:, 7:].astype(np.float32),
    joint_vel=qvel[:, 6:].astype(np.float32),
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
    fps=np.asarray([cfg.fps], dtype=np.float64),
    qpos=qpos.astype(np.float32),
    qvel=qvel.astype(np.float32),
    joint_names=joint_names,
    body_names=body_names,
    source_file=np.asarray(str(cfg.input.resolve())),
    max_loop_residual_m=np.asarray(max_loop_residual),
  )
  print(
    f"Wrote {cfg.output}: frames={len(qpos)}, fps={cfg.fps:g}, "
    f"joints={model.njnt - 1}, bodies={model.nbody - 1}, "
    f"max_loop_residual={max_loop_residual:.3e} m"
  )


if __name__ == "__main__":
  main()
