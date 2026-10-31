#!/usr/bin/env python3
"""Real-time comparison video of several foot_variant_eval.py runs: one tile per run, all tiles on the same clock.

Each run saved the compiled model (<tag>.mjb: robot + terrain exactly as simulated) and the full qpos of its first envs.
The tiles replay that qpos; nothing is re-simulated. Frame rate = policy rate (50 Hz) / --stride, so the video plays in real time.
Views: side (tracking the base from the robot's left), front (tracking from ahead), foot (close-up of the right foot from the side).
The sole is drawn from the foot collision pieces (orange; soft-arch wedge purple): the robot visual mesh has no sole.
A caption per tile shows the run label, the command and the right-foot vertical force.
  MUJOCO_GL=egl python3 tools/foot_sole/render_eval_video.py --eval-dir <dir> --tags base v5fullm ... --labels "box" "v5" ...
      --view side --out <file.mp4> [--env 0] [--start-s 0 --dur-s 135]
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--labels", nargs="+")
ap.add_argument("--view", default="side", choices=["side", "front", "foot"])
ap.add_argument("--env", type=int, default=0)
ap.add_argument("--start-s", type=float, default=0.0)
ap.add_argument("--dur-s", type=float, default=0.0, help="0 = to the end")
ap.add_argument("--stride", type=int, default=1, help="policy steps per frame (1 = 50 fps)")
ap.add_argument("--size", default="480x360")
ap.add_argument("--cols", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
W, Hh = map(int, args.size.split("x"))
DT = 0.02
labels = args.labels or args.tags
assert len(labels) == len(args.tags)
d = Path(args.eval_dir)
runs = []
for tag, lab in zip(args.tags, labels):
  m = mujoco.MjModel.from_binary_path(str(d / f"{tag}.mjb"))
  m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(Hh, 480)
  # the sole exists only as collision geometry: show the foot collision pieces (orange; soft-arch pieces purple) with the visuals
  for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "_foot" in nm and nm.endswith("_collision") and m.geom_group[g] == 3 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
      m.geom_group[g] = 2
      m.geom_rgba[g] = (0.55, 0.36, 0.84, 1.0) if m.geom_solref[g][0] > 0.03 else (0.85, 0.47, 0.12, 1.0)
  Z = np.load(d / f"{tag}.npz")
  meta = json.loads((d / f"{tag}.json").read_text())
  runs.append(dict(m=m, data=mujoco.MjData(m), r=mujoco.Renderer(m, Hh, W), qpos=Z["qpos"][:, args.env], sole=Z["sole"][:, args.env],
                   term=Z["terminated"][:, args.env], meta=meta, label=lab, bw=meta["robot_mass_kg"] * 9.81,
                   foot=m.body("robot/R_foot_link").id, base=m.body("robot/base_link").id))
T = min(len(r["qpos"]) for r in runs)
n_per = int(round(runs[0]["meta"]["seconds_per_cmd"] / DT))
prog = runs[0]["meta"]["program"]
i0 = int(args.start_s / DT)
i1 = T if args.dur_s <= 0 else min(T, i0 + int(args.dur_s / DT))
cols = args.cols or int(np.ceil(np.sqrt(len(runs))))
rows = int(np.ceil(len(runs) / cols))
try:
  font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except OSError:
  font = ImageFont.load_default()
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
opt = mujoco.MjvOption()
opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # visual meshes only (group 2) + terrain; collision group 3 hidden
writer = imageio.get_writer(args.out, fps=round(1 / (DT * args.stride)), codec="libx264", quality=7, macro_block_size=8)
for i in range(i0, i1, args.stride):
  canvas = Image.new("RGB", (cols * W, rows * Hh), (20, 20, 20))
  for k, r in enumerate(runs):
    q = r["qpos"][i]
    if np.isfinite(q).all():
      r["data"].qpos[:] = q
      mujoco.mj_forward(r["m"], r["data"])
    bp = r["data"].xpos[r["base"]]
    if args.view == "foot":
      cam.lookat[:] = r["data"].xpos[r["foot"]] + np.array([0.0, 0.0, -0.03])
      cam.distance, cam.elevation = 0.75, -8.0
      cam.azimuth = np.degrees(np.arctan2(*r["data"].xmat[r["base"]].reshape(3, 3)[[1, 0], 0])) + 90.0
    else:
      cam.lookat[:] = bp + np.array([0.0, 0.0, -0.45])
      cam.distance, cam.elevation = 2.0, -8.0
      yaw = np.degrees(np.arctan2(*r["data"].xmat[r["base"]].reshape(3, 3)[[1, 0], 0]))
      cam.azimuth = yaw + (90.0 if args.view == "side" else 180.0)
    r["r"].update_scene(r["data"], camera=cam, scene_option=opt)
    img = Image.fromarray(r["r"].render())
    dr = ImageDraw.Draw(img)
    name = prog[min(i // n_per, len(prog) - 1)][0]
    fz = r["sole"][i, 0, 0] / r["bw"]
    dead = r["term"][: i + 1].any()
    dr.rectangle([0, 0, W, 44], fill=(0, 0, 0))
    dr.text((8, 4), r["label"], fill=(255, 220, 120), font=font)
    dr.text((8, 24), f"{name}  t={i * DT:5.1f}s  R foot {fz:4.2f} BW" + ("  FELL" if dead else ""), fill=(230, 230, 230), font=font)
    canvas.paste(img, ((k % cols) * W, (k // cols) * Hh))
  writer.append_data(np.asarray(canvas))
writer.close()
print("wrote", args.out, f"{(i1 - i0) // args.stride} frames", flush=True)
