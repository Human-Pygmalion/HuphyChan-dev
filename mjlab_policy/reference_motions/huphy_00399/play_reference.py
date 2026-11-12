#!/usr/bin/env python3
"""Play or record the exact HUPHY 00399 training reference in MuJoCo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

# Headless recording needs EGL, while the interactive native viewer needs GLFW.
if "--output" in sys.argv:
  os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_MOTION = PACKAGE_DIR / "00399_interactionmesh_50hz.npz"
DEFAULT_XML = (
  REPO_ROOT
  / "pygmalion_locomotion/assets/pygmalion_v2/"
  "LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_"
  "footv5_d455_reinf_alparts2_capsule7plus5_loop.xml"
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
  parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
  parser.add_argument(
    "--output",
    type=Path,
    help="Record an MP4 headlessly. Without this option, open the native viewer.",
  )
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--end-frame", type=int)
  parser.add_argument("--width", type=int, default=960)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--azimuth", type=float, default=90.0)
  parser.add_argument("--elevation", type=float, default=-8.0)
  parser.add_argument("--distance", type=float, default=1.7)
  parser.add_argument("--loop", action="store_true")
  return parser.parse_args()


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray | None, float]:
  with np.load(path, allow_pickle=False) as archive:
    if "qpos" not in archive:
      raise ValueError(f"{path} does not contain qpos")
    qpos = np.asarray(archive["qpos"], dtype=np.float64)
    qvel = (
      np.asarray(archive["qvel"], dtype=np.float64)
      if "qvel" in archive
      else None
    )
    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
  return qpos, qvel, fps


def frame_bounds(args: argparse.Namespace, num_frames: int) -> tuple[int, int]:
  start = args.start_frame
  end = num_frames if args.end_frame is None else args.end_frame
  if not 0 <= start < num_frames:
    raise ValueError(f"--start-frame must be in [0, {num_frames - 1}]")
  if not start < end <= num_frames:
    raise ValueError(f"--end-frame must be in [{start + 1}, {num_frames}]")
  return start, end


def set_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  qvel: np.ndarray | None,
  frame: int,
) -> None:
  data.qpos[:] = qpos[frame]
  if qvel is not None:
    data.qvel[:] = qvel[frame]
  else:
    data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def camera(args: argparse.Namespace, root_position: np.ndarray) -> mujoco.MjvCamera:
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = root_position
  cam.azimuth = args.azimuth
  cam.elevation = args.elevation
  cam.distance = args.distance
  return cam


def play_native(
  args: argparse.Namespace,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  qvel: np.ndarray | None,
  fps: float,
  start: int,
  end: int,
) -> None:
  # Import after model validation so headless recording does not require GLFW.
  import mujoco.viewer

  frame = start
  period = 1.0 / fps
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = qpos[start, :3]
    viewer.cam.azimuth = args.azimuth
    viewer.cam.elevation = args.elevation
    viewer.cam.distance = args.distance
    deadline = time.monotonic()
    while viewer.is_running():
      set_frame(model, data, qpos, qvel, frame)
      viewer.cam.lookat[:] = qpos[frame, :3]
      viewer.sync()
      frame += 1
      if frame >= end:
        if not args.loop:
          break
        frame = start
      deadline += period
      time.sleep(max(0.0, deadline - time.monotonic()))


def record_mp4(
  args: argparse.Namespace,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  qvel: np.ndarray | None,
  fps: float,
  start: int,
  end: int,
) -> None:
  assert args.output is not None
  args.output.parent.mkdir(parents=True, exist_ok=True)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
  renderer = mujoco.Renderer(model, width=args.width, height=args.height)
  cam = camera(args, qpos[start, :3])
  ffmpeg = subprocess.Popen(
    [
      "ffmpeg", "-y", "-loglevel", "error",
      "-f", "rawvideo", "-pix_fmt", "rgb24",
      "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "-",
      "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ],
    stdin=subprocess.PIPE,
  )
  try:
    for frame in range(start, end):
      set_frame(model, data, qpos, qvel, frame)
      cam.lookat[:] = qpos[frame, :3]
      renderer.update_scene(data, camera=cam)
      image = renderer.render()
      assert ffmpeg.stdin is not None
      ffmpeg.stdin.write(np.ascontiguousarray(image).tobytes())
      if (frame - start) % 250 == 0 or frame + 1 == end:
        print(f"Recorded {frame - start + 1}/{end - start}", flush=True)
  finally:
    if ffmpeg.stdin is not None:
      ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    renderer.close()
  if return_code:
    raise RuntimeError(f"ffmpeg exited with status {return_code}")
  print(f"Saved {args.output} ({end - start} frames at {fps:g} FPS)")


def main() -> None:
  args = parse_args()
  qpos, qvel, fps = load_reference(args.motion)
  model = mujoco.MjModel.from_xml_path(str(args.xml))
  data = mujoco.MjData(model)
  if qpos.ndim != 2 or qpos.shape[1] != model.nq:
    raise ValueError(
      f"qpos shape is {qpos.shape}, but {args.xml} expects nq={model.nq}"
    )
  if qvel is not None and (qvel.ndim != 2 or qvel.shape != (len(qpos), model.nv)):
    raise ValueError(
      f"qvel shape is {qvel.shape}, expected {(len(qpos), model.nv)}"
    )
  start, end = frame_bounds(args, len(qpos))
  print(
    f"Motion: {args.motion}\nRobot:  {args.xml}\n"
    f"Frames: {start}:{end} / {len(qpos)}, {fps:g} FPS",
    flush=True,
  )
  if args.output is None:
    play_native(args, model, data, qpos, qvel, fps, start, end)
  else:
    record_mp4(args, model, data, qpos, qvel, fps, start, end)


if __name__ == "__main__":
  main()
