#!/usr/bin/env python3
"""Render the three analysis videos of the real-robot walking sim-to-real study (docs/177 section 10).

Nothing is simulated here. The script PLAYS BACK recorded states:
  * simulator recordings  mujoco-sim/mjlab/analysis/out/s2r_walk/<run>.npz (made by sim_walk_record.py)
  * real-robot policy logs Expr_Data/*.csv  (kinematic replay: joint angles only, NO force / torque data)

Videos (all real time: fps = data rate / downsample; ffprobe-checked: the report json holds frames / fps / duration):
  v1  "veer"  : sim 20 ms (R3, as trained) | sim real policy-interval sequence (R8); chase camera + top-down map.
                R3 is a 50 Hz recording (downsample 2 -> 25 fps), R8 a 200 Hz one (downsample 8 -> 25 fps).
  v2  "gait"  : sim real-interval (W5_real, wz 0.5) | real robot replay of 1021vz0.5ob.csv; hip_pitch and knee
                target-vs-measured strips (moving 5 s window). 20 fps (200 Hz / 10; real log resampled to 20 fps).
  v3  "5hz"   : real replay of vx0gain1walkob(진동확인용)1021.csv, log time 30-80 s, with hip_pitch target vs measured,
                gyro x/y and a running spectrogram (2 s window) of the hip_pitch target, 5.1 Hz marker. 20 fps.

How the real robot is posed (no base pose is logged):
  * joint angles: the logged `pos` (deg, same sign convention as the simulator joints) of the 12 motors; the passive
    ankle pitch/roll hinges and the rod joints are SOLVED from the crank angles by closing the four loop
    constraints (Gauss-Newton). On simulator frames this solve reproduces the recorded ankle pitch/roll within 0.1 deg.
  * roll / pitch: from the projected gravity vector (g_b = R^T [0,0,-1]).
  * yaw: integral of the world-vertical gyro rate (gyro . up_b), trapezoid on the raw rows (drifts, see docs).
  * position: none. The base stays above the origin ("in place") and its height is set every frame so that the lower
    foot touches the floor.
  * the ~52 ms, non-uniform log rows are linearly interpolated to a uniform 20 fps timeline (fps 20 = real time).
The simulator panes of v2 are also drawn in place (translation removed) so that both panes are compared like for like.

Load visibility (simulator panes only): joint spheres  grey < rated < yellow < orange (70 % peak) < red (peak);
ground-reaction-force arrows: cyan = left foot, magenta = right foot, 0.4 m = 1 body weight, drawn 0.22 m to the
side of the leg. Real-robot panes carry the label "kinematic replay, no load data".

Model: pygmalion_legonly_ab.xml of the package `pygmalion_mjlab` (branch origin/vibecoded-audit-20261012). Default
search: share/pygmalion_mjlab_20261012/pygmalion_mjlab/assets/ in this repository (untracked), else --xml.

Run (CPU, one render at a time, niced):
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl nice -n 10 mujoco-sim/mjlab/.venv/bin/python3 \
      tools/sim2real/walk/render_s2r_videos.py --video v1 --out docs/video
  (--video all renders v1, v2, v3 one after the other; --smoke N [--first K] renders only a few frames, for tests.
   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 is needed here: the conda libstdc++ on LD path breaks mesa llvmpipe.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SIM_DIR = REPO / "mujoco-sim/mjlab/analysis/out/s2r_walk"
EXPR = REPO / "Expr_Data"
XML_CANDIDATES = [
  REPO / "share/pygmalion_mjlab_20261012/pygmalion_mjlab/assets/pygmalion_legonly_ab.xml",
  REPO / ".claude/worktrees/vibecoded-audit-20261012/vibecoded-audit-20261012/pygmalion_mjlab/assets/pygmalion_legonly_ab.xml",
]
MOTORS = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")]
PASSIVE = [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]
# (peak, rated) N*m per motor, in MOTORS order: RS04 hip_pitch/hip_roll/knee 120/40, RS03 hip_yaw/cranks 60/20
PEAK_RATED = {"hip_pitch": (120.0, 40.0), "hip_roll": (120.0, 40.0), "hip_yaw": (60.0, 20.0),
              "knee": (120.0, 40.0), "crank_A": (60.0, 20.0), "crank_B": (60.0, 20.0)}
LEVEL_RGBA = {0: (0.55, 0.59, 0.64, 1.0), 1: (0.97, 0.86, 0.0, 1.0), 2: (1.0, 0.50, 0.0, 1.0), 3: (0.95, 0.05, 0.05, 1.0)}
LEVEL_RGB = {k: tuple(int(255 * c) for c in v[:3]) for k, v in LEVEL_RGBA.items()}
GRF_RGBA = {"L": (0.10, 0.85, 0.95, 0.95), "R": (0.95, 0.25, 0.90, 0.95)}
GRF_RGB = {"L": (26, 217, 242), "R": (242, 64, 230)}
BG = (24, 26, 30)
FG = (235, 235, 235)
DIM = (170, 175, 180)
RIGHT_COL = (242, 64, 230)
LEFT_COL = (26, 217, 242)


# ------------------------------------------------------------------------------------------------ small drawing utils
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
  p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
  return ImageFont.truetype(p, size)


F11, F13, F15, F17, F20 = font(11), font(13), font(15), font(17), font(20)
F13B, F15B, F17B, F20B = font(13, True), font(15, True), font(17, True), font(20, True)


def text(dr, xy, s, f=F13, fill=FG, anchor="la", shadow=True):
  if shadow:
    dr.text((xy[0] + 1, xy[1] + 1), s, font=f, fill=(0, 0, 0), anchor=anchor)
  dr.text(xy, s, font=f, fill=fill, anchor=anchor)


def dashed_line(dr, p0, p1, fill, width=1, dash=6, gap=4):
  x0, y0 = p0
  x1, y1 = p1
  L = float(np.hypot(x1 - x0, y1 - y0))
  if L < 1e-6:
    return
  ux, uy = (x1 - x0) / L, (y1 - y0) / L
  s = 0.0
  while s < L:
    e = min(s + dash, L)
    dr.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
    s += dash + gap


def dashed_poly(dr, pts, fill, width=1, dash=5, gap=3):
  for a, b in zip(pts[:-1], pts[1:]):
    dashed_line(dr, a, b, fill, width, dash, gap)


class Writer:
  """Stream raw RGB frames to ffmpeg (H.264, yuv420p, constant fps). Nothing is kept in RAM."""

  def __init__(self, path: Path, w: int, h: int, fps: int, crf: int = 23):
    self.path, self.w, self.h, self.fps, self.n = path, w, h, fps, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(path)]
    self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

  def add(self, img: Image.Image):
    assert img.size == (self.w, self.h), (img.size, (self.w, self.h))
    self.p.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    self.n += 1

  def close(self):
    self.p.stdin.close()
    rc = self.p.wait()
    assert rc == 0, f"ffmpeg failed ({rc})"


# ------------------------------------------------------------------------------------------------------------ plots
def nice_ticks(lo, hi, n=4):
  span = hi - lo
  raw = span / max(1, n)
  mag = 10 ** np.floor(np.log10(raw))
  step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=raw)
  start = np.ceil(lo / step) * step
  return [start + i * step for i in range(int((hi - start) / step + 1e-9) + 1)]


def time_plot(img, box, title, series, t_now, window, ylim, ylabel, markers=(), t_origin=0.0, xlabel=True):
  """Moving-window line plot drawn with PIL. series: list of dict(t, y, color, dash(bool), label).
  The window is [t_now-window, t_now]; the right edge is "now"."""
  dr = ImageDraw.Draw(img)
  x0, y0, w, h = box
  ml, mr, mt, mb = 46, 8, 20, 18 if xlabel else 6
  px0, py0, pw, ph = x0 + ml, y0 + mt, w - ml - mr, h - mt - mb
  dr.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=(14, 16, 20), outline=(60, 64, 70))
  text(dr, (x0 + 6, y0 + 3), title, F13B, shadow=False)
  tl = t_now - window
  lo, hi = ylim

  def X(t):
    return px0 + (t - tl) / window * pw

  def Y(v):
    return py0 + (hi - v) / (hi - lo) * ph

  for tv in nice_ticks(lo, hi, 4):
    yy = Y(tv)
    dr.line([(px0, yy), (px0 + pw, yy)], fill=(48, 52, 58))
    text(dr, (px0 - 4, yy), f"{tv:g}", F11, DIM, anchor="rm", shadow=False)
  step = 1 if window <= 12 else 2 if window <= 24 else 5
  tt = np.ceil((tl) / step) * step
  while tt <= t_now + 1e-9:
    xx = X(tt)
    dr.line([(xx, py0), (xx, py0 + ph)], fill=(40, 44, 50))
    if xlabel:
      text(dr, (xx, py0 + ph + 2), f"{tt + t_origin:g}", F11, DIM, anchor="ma", shadow=False)
    tt += step
  if ylabel:
    text(dr, (x0 + w - 6, y0 + 3), ylabel, F11, DIM, anchor="ra", shadow=False)
  for s in series:
    t, y = s["t"], s["y"]
    m = (t >= tl - 0.2) & (t <= t_now + 1e-9)
    if m.sum() < 2:
      continue
    pts = [(X(a), Y(float(np.clip(b, lo, hi)))) for a, b in zip(t[m], y[m])]
    if s.get("dash"):
      dashed_poly(dr, pts, s["color"], 2, 5, 3)
    else:
      dr.line(pts, fill=s["color"], width=2)
  for mi, mk in enumerate(markers):
    mt_, col, lab = mk
    if tl <= mt_ <= t_now:
      xx = X(mt_)
      dashed_line(dr, (xx, py0), (xx, py0 + ph), col, 2, 6, 3)
      text(dr, (min(xx + 3, px0 + pw - 2), py0 + 1 + 13 * mi), lab, F11, col, anchor="la" if xx + 3 < px0 + pw - 120 else "ra", shadow=False)
  dr.line([(px0 + pw, py0), (px0 + pw, py0 + ph)], fill=(250, 250, 250), width=1)  # "now"
  dr.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(70, 74, 80))
  # legend
  lx = px0 + 6
  for s in series:
    if not s.get("label"):
      continue
    if s.get("dash"):
      dashed_line(dr, (lx, py0 + ph - 8), (lx + 18, py0 + ph - 8), s["color"], 2, 5, 3)
    else:
      dr.line([(lx, py0 + ph - 8), (lx + 18, py0 + ph - 8)], fill=s["color"], width=2)
    text(dr, (lx + 22, py0 + ph - 8), s["label"], F11, FG, anchor="lm", shadow=False)
    lx += 30 + int(dr.textlength(s["label"], font=F11))


def lomb_spectrogram(t, y, t_cols, win, freqs):
  """Running amplitude spectrum (deg) of the irregularly sampled y: Hann-weighted Lomb-Scargle on [t_col-win, t_col]."""
  from scipy.signal import lombscargle

  out = np.full((len(freqs), len(t_cols)), np.nan)
  om = 2 * np.pi * freqs
  for j, tc in enumerate(t_cols):
    m = (t > tc - win) & (t <= tc)
    n = int(m.sum())
    if n < 12:
      continue
    tt, yy = t[m], y[m]
    w = np.hanning(n + 2)[1:-1]
    yy = (yy - np.average(yy, weights=w)) * w
    P = lombscargle(tt - tt[0], yy, om, precenter=False, normalize=False)
    out[:, j] = np.sqrt(4.0 * np.maximum(P, 0) / n) / np.mean(w)
  return out


# ------------------------------------------------------------------------------------------------------ robot model
def find_xml(arg: str | None) -> Path:
  if arg:
    return Path(arg)
  for c in XML_CANDIDATES:
    if c.exists():
      return c
  raise SystemExit("MJCF not found; pass --xml <...>/pygmalion_legonly_ab.xml")


class Robot:
  def __init__(self, xml: Path, width: int, height: int):
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.add_texture(name="chk", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.56, 0.60, 0.66], rgb2=[0.44, 0.48, 0.54], width=512, height=512,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.78, 0.80, 0.82])
    mat = spec.add_material(name="floor_mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chk"
    mat.texrepeat = [1, 1]
    mat.texuniform = True
    mat.reflectance = 0.0
    spec.worldbody.add_geom(name="floor_vis", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[12, 12, 0.05], material="floor_mat",
                            contype=0, conaffinity=0)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.34, 0.46, 0.62], rgb2=[0.06, 0.07, 0.09], width=256, height=1536)
    spec.visual.global_.offwidth = max(width, 640)
    spec.visual.global_.offheight = max(height, 480)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
    spec.visual.quality.shadowsize = 2048
    lt = spec.worldbody.add_light(pos=[1.5, -1.0, 4.0], dir=[-0.3, 0.2, -1.0], diffuse=[0.55, 0.55, 0.55], castshadow=True)
    self.m = spec.compile()
    m = self.m
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_LIMIT |
                           mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    self.d = mujoco.MjData(m)
    jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    names = [jn(j) for j in range(m.njnt)]
    self.jid = {n: names.index(n) for n in MOTORS + PASSIVE}
    self.adr12 = np.array([m.jnt_qposadr[self.jid[n]] for n in MOTORS])
    self.adr_pas = np.array([m.jnt_qposadr[self.jid[n]] for n in PASSIVE])
    self.rod_dofs = np.array([m.jnt_dofadr[j] for j, n in enumerate(names) if "rod" in n])
    self.free_dofs = np.array(list(self.rod_dofs) + [m.jnt_dofadr[self.jid[n]] for n in PASSIVE])
    assert all(m.jnt_qposadr[j] == m.jnt_dofadr[j] + 1 for j in range(1, m.njnt))
    self.floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor_vis")
    self.feet = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("collision")
                 and "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    self.foot_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot_link") for s in "RL"}
    self.bw = float(m.body_subtreemass[0]) * 9.81
    self.d.qpos[:] = m.qpos0
    self.renderer = mujoco.Renderer(m, height=height, width=width)
    self.size = (width, height)

  def set_floor(self, z: float):
    self.m.geom_pos[self.floor_gid][2] = z

  def pose(self, pos, quat, q12):
    """Set base pose + 12 motor angles (rad); solve passive ankle + rod joints from the loop constraints."""
    d = self.d
    d.qpos[0:3] = pos
    d.qpos[3:7] = quat
    d.qpos[self.adr12] = q12
    F = self.free_dofs
    for _ in range(40):
      mujoco.mj_forward(self.m, d)
      r = d.efc_pos.copy()
      if np.abs(r).max() < 1e-9:
        break
      J = d.efc_J.reshape(d.nefc, self.m.nv)[:, F]
      d.qpos[F + 1] += -np.linalg.solve(J.T @ J + 1e-9 * np.eye(len(F)), J.T @ r)
    mujoco.mj_forward(self.m, d)
    return float(np.abs(d.efc_pos).max())

  def lowest_foot_z(self):
    d, m = self.d, self.m
    z = []
    for g in self.feet:
      R = d.geom_xmat[g].reshape(3, 3)
      z.append(d.geom_xpos[g][2] - np.abs(R[2]) @ m.geom_size[g][:3])
    return min(z)

  # ---- scene decoration
  def _geom(self, scn, typ, rgba, size=None, pos=None):
    if scn.ngeom >= scn.maxgeom:
      return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, int(typ), np.zeros(3) if size is None else np.asarray(size, float), np.zeros(3) if pos is None else np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g

  def sphere(self, scn, pos, rgba, r=0.035):
    self._geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, rgba, [r, 0, 0], pos)

  def connector(self, scn, typ, p0, p1, rgba, width):
    g = self._geom(scn, typ, rgba)
    if g is not None:
      mujoco.mjv_connector(g, int(typ), width, np.asarray(p0, float), np.asarray(p1, float))

  def arrow(self, scn, p0, p1, rgba, width=0.02):
    self.connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, p0, p1, rgba, width)

  def segment(self, scn, p0, p1, rgba, width=0.012):
    self.connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, p0, p1, rgba, width)

  def render(self, lookat, distance, azimuth, elevation, decorate=None):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # collision capsules off
    opt.geomgroup[4] = 0  # hull off
    self.renderer.update_scene(self.d, cam, opt)
    if decorate is not None:
      decorate(self.renderer.scene)
    return Image.fromarray(self.renderer.render())


def quat_from_euler_zyx(yaw, pitch, roll):
  cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
  cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
  cr, sr = np.cos(roll / 2), np.sin(roll / 2)
  return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                   cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def yaw_of_quat(q):
  w, x, y, z = q.T
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------------------------------------------------- data
class SimRun:
  """One simulator recording, one env, resampled by integer downsampling to a video time base."""

  def __init__(self, name: str, env: int | None = None):
    self.name = name
    D = np.load(SIM_DIR / f"{name}.npz")
    self.meta = json.load(open(SIM_DIR / f"{name}.json"))
    self.D = D
    self.hi = "base_pos200" in D.files  # 200 Hz base / force data present (timing runs)
    s = "200" if self.hi else ""
    self.rate = 200 if self.hi else 50
    self.suf = s
    self.bp, self.bq = D["base_pos" + s], D["base_quat" + s]
    self.q = D["q200"] if self.hi else D["q"]  # 16 joints (motors then passive), rad
    self.tau = D["tau200"] if self.hi else D["tau_mean"]
    self.force = D["foot_force_w" + s]  # (T, env, [R, L], 3)
    self.target = D["target200"] if self.hi else D["target"]  # rad, 12 motors
    self.T = self.bp.shape[0]
    self.yaw = np.stack([np.unwrap(yaw_of_quat(self.bq[:, e])) for e in range(self.bp.shape[1])], 1)
    self.env = env

  def heading_change_deg(self):
    return np.degrees(self.yaw[-1] - self.yaw[0])

  def intervals(self, env):
    """(call times, intervals [ms]) of the policy for env, or None (fixed 20 ms)."""
    if "t_call" not in self.D.files:
      return None
    n = int(self.D["n_calls"][env])
    tc = self.D["t_call"][:n, env].astype(float)
    return tc, np.diff(tc, prepend=tc[0] - 0.02) * 1000.0


def pick_env(*runs: SimRun) -> tuple[int, dict]:
  """Env whose heading change is nearest to the run median, summed over runs (spread-normalised)."""
  score = np.zeros(runs[0].yaw.shape[1])
  info = {}
  for r in runs:
    h = r.heading_change_deg()
    med, spr = np.median(h), max(np.std(h), 1e-6)
    score += np.abs(h - med) / spr
    info[r.name] = {"heading_change_deg": [round(float(x), 1) for x in h], "median": round(float(med), 1)}
  return int(np.argmin(score)), info


class RealLog:
  COLS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b"]

  def __init__(self, path: Path):
    import pandas as pd

    df = pd.read_csv(path, skiprows=1)
    self.path = path
    self.t = df["t"].values.astype(float)
    leg = lambda pre: np.stack([df[f"{pre}/{s}_leg/{c}"].values for s in ("right", "left") for c in self.COLS], 1).astype(float)
    self.pos, self.tgt = leg("pos"), leg("tgt")  # deg, 12 columns in MOTORS order
    self.gyro = df[["obs/gyro/x", "obs/gyro/y", "obs/gyro/z"]].values.astype(float)
    self.grav = df[["obs/grav/x", "obs/grav/y", "obs/grav/z"]].values.astype(float)
    self.cmd = df[["obs/cmd/vx", "obs/cmd/wz"]].values.astype(float)
    g = self.grav / np.linalg.norm(self.grav, axis=1, keepdims=True)
    up = -g
    self.pitch = np.arcsin(np.clip(-up[:, 0], -1, 1))
    self.roll = np.arctan2(up[:, 1], up[:, 2])
    yawrate = np.einsum("ij,ij->i", self.gyro, up)
    self.yawrate = yawrate
    self.yaw = np.concatenate([[0.0], np.cumsum(0.5 * (yawrate[1:] + yawrate[:-1]) * np.diff(self.t))])
    self.dt_med = float(np.median(np.diff(self.t)))

  def at(self, arr, tq):
    arr = np.asarray(arr)
    if arr.ndim == 1:
      return np.interp(tq, self.t, arr)
    return np.stack([np.interp(tq, self.t, arr[:, i]) for i in range(arr.shape[1])], 1)


# ------------------------------------------------------------------------------------------------ shared painting
def sphere_levels(tau12):
  out = []
  for i, n in enumerate(MOTORS):
    pk, rt = PEAK_RATED[n.split("_", 1)[1].replace("_joint", "")]
    a = abs(float(tau12[i]))
    out.append(3 if a >= pk else 2 if a >= 0.7 * pk else 1 if a >= rt else 0)
  return out


def hip_offset(name, side, Rb):
  """Hip pitch/roll/yaw share one anchor: separate their spheres (body frame): pitch outward, roll front, yaw up."""
  lat = -1.0 if side == "R" else 1.0
  off = {"hip_pitch": (0.0, 0.060 * lat, 0.0), "hip_roll": (0.065, 0.0, 0.0), "hip_yaw": (0.0, 0.0, 0.062)}.get(name)
  return None if off is None else Rb @ np.array(off)


def decorate_sim(rb: Robot, scn, tau12, force2, base_quat, show_load=True):
  """Joint load spheres + GRF arrows for a simulator frame."""
  m, d = rb.m, rb.d
  Rb = d.xmat[1].reshape(3, 3)
  if show_load:
    lv = sphere_levels(tau12)
    for i, n in enumerate(MOTORS):
      side, short = n[0], n.split("_", 1)[1].replace("_joint", "")
      p = d.xanchor[rb.jid[n]].copy()
      off = hip_offset(short, side, Rb)
      if off is not None:
        p = p + off
      rb.sphere(scn, p, LEVEL_RGBA[lv[i]], 0.05)
    yaw = float(yaw_of_quat(np.asarray(base_quat)[None])[0])
    lat = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    for k, s in enumerate("RL"):
      F = -np.asarray(force2[k], float)  # recorded force is the one the foot exerts on the ground (sum Fz = -1 BW): flip
      if float(np.linalg.norm(F)) > 20.0:
        p0 = d.xpos[rb.foot_body[s]] + (-0.22 if s == "R" else 0.22) * lat
        p0 = p0.copy()
        p0[2] = max(p0[2], rb.m.geom_pos[rb.floor_gid][2] + 0.02)
        rb.arrow(scn, p0, p0 + F / rb.bw * 0.4, GRF_RGBA[s], 0.02)


def legend_sim(dr, x, y):
  rows = [("< rated", 0), ("> rated", 1), ("> 70% peak", 2), ("> peak", 3)]
  text(dr, (x, y), "Joint load (spheres)", F11, DIM)
  for i, (lab, lv) in enumerate(rows):
    dr.ellipse([x + 2, y + 16 + i * 14, x + 12, y + 26 + i * 14], fill=LEVEL_RGB[lv])
    text(dr, (x + 18, y + 15 + i * 14), lab, F11, FG)
  text(dr, (x, y + 16 + 4 * 14 + 3), "GRF arrows: cyan L, magenta R", F11, DIM)
  text(dr, (x, y + 16 + 5 * 14 + 3), "0.4 m = 1 body weight", F11, DIM)


def banner(img, lines, color=(255, 214, 90), h=36):
  dr = ImageDraw.Draw(img)
  dr.rectangle([0, 0, img.width, h], fill=(8, 10, 14))
  if isinstance(lines, str):
    lines = [lines]
  text(dr, (img.width // 2, h // 2), "    |    ".join(lines), F17B, color, anchor="mm", shadow=False)


# ------------------------------------------------------------------------------------------------------------ V1
def map_panel(img, box, trail_fl, now_fl, heading, color, tag):
  """Top-down map: forward up, right of the robot to the right. trail_fl: (n,2) [forward, left] in the start frame."""
  dr = ImageDraw.Draw(img)
  x0, y0, w, h = box
  dr.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], fill=(14, 16, 20), outline=(60, 64, 70))
  text(dr, (x0 + w // 2, y0 + 5), "TOP VIEW (forward = up)", F11, DIM, anchor="ma", shadow=False)
  top, bot = y0 + 26, y0 + h - 18
  scale = (bot - top) / 8.4  # px per m
  ox = x0 + 0.30 * w  # lateral origin (reference line)
  Y = lambda fwd: bot - (fwd + 0.2) * scale
  Xm = lambda left: ox - left * scale  # left negative -> to the right of the picture
  for f in range(0, 9):
    dr.line([(x0 + 4, Y(f)), (x0 + w - 4, Y(f))], fill=(44, 48, 54))
    if f % 2 == 0:
      text(dr, (x0 + w - 6, Y(f) - 1), f"{f} m", F11, DIM, anchor="rb", shadow=False)
  for lat in range(-int((ox - x0) / scale), int((x0 + w - ox) / scale) + 1):
    xx = Xm(-lat)
    if x0 + 4 < xx < x0 + w - 4:
      dr.line([(xx, top), (xx, bot)], fill=(34, 38, 44))
  dashed_line(dr, (ox, Y(0)), (ox, Y(8.0)), (230, 230, 230), 2, 8, 5)
  text(dr, (ox + 4, Y(8.0) + 2), "straight", F11, (230, 230, 230), shadow=False)
  if len(trail_fl) > 1:
    pts = [(Xm(l), Y(f)) for f, l in trail_fl]
    dr.line(pts, fill=color, width=3)
  f, l = now_fl
  cx, cy = Xm(l), Y(f)
  dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=color, outline=(255, 255, 255))
  # heading relative to start: +yaw = counter-clockwise = to the left; direction vector (forward = cos, left = sin)
  df_, dl_ = np.cos(heading), np.sin(heading)
  dr.line([(cx, cy), (cx - dl_ * 0.7 * scale, cy - df_ * 0.7 * scale)], fill=(255, 255, 255), width=2)
  text(dr, (x0 + 6, y0 + h - 4), "arrow = heading", F11, DIM, anchor="lb", shadow=False)


def render_v1(args, rb: Robot, out: Path):
  A, B = SimRun("R3"), SimRun("R8")
  env, info = pick_env(A, B)
  print("[v1] env", env, info)
  fps = 25
  nA, nB = A.T // (A.rate // fps), B.T // (B.rate // fps)
  assert nA == nB == 500, (nA, nB)
  dsA, dsB = A.rate // fps, B.rate // fps  # 2 and 8
  runs = [(A, dsA, "SIM  policy interval 20 ms (as trained)", (90, 170, 255), None),
          (B, dsB, "SIM  policy interval = real-robot sequence (median 53 ms)", (255, 160, 60), B.intervals(env))]
  # floor level = median lowest-foot z (stance foot) on a sample of frames, per run
  floors = []
  for r, ds, *_ in runs:
    zs = []
    for k in range(0, r.T, max(1, r.T // 40)):
      rb.pose(r.bp[k, env], r.bq[k, env], r.q[k, env, :12])
      zs.append(rb.lowest_foot_z())
    floors.append(float(np.median(zs)))
  print("[v1] floor z per run", floors)
  PW, H3, MW = 520, 440, 180
  W, Hh = 2 * (PW + MW), 36 + H3 + 26
  wr = Writer(out, W, Hh, fps)
  nfr = args.smoke or nA
  base0 = [(r.bp[0, env].copy(), float(r.yaw[0, env])) for r, *_ in runs]
  fl_all = []  # (forward, left) of every video frame, start frame of the run (initial position and heading)
  for (r, ds, *_), (p0, y0) in zip(runs, base0):
    dxy = r.bp[::ds, env, :2][:nA] - p0[:2]
    fl_all.append(np.stack([np.cos(y0) * dxy[:, 0] + np.sin(y0) * dxy[:, 1], -np.sin(y0) * dxy[:, 0] + np.cos(y0) * dxy[:, 1]], 1))
  t0 = time.time()
  for i in range(args.first, nfr):
    img = Image.new("RGB", (W, Hh), BG)
    banner(img, ["CMD  vx 0.5 m/s,  wz 0", "same command, flat ground, env #%d of 8 (median drift)" % env,
                 "real robot also veers RIGHT (user report)"])
    dr = ImageDraw.Draw(img)
    for pi, (r, ds, tag, col, ivl) in enumerate(runs):
      k = i * ds
      pos, quat = r.bp[k, env], r.bq[k, env]
      rb.pose(pos, quat, r.q[k, env, :12])
      rb.set_floor(floors[pi])
      yaw_now = float(r.yaw[k, env])
      p0, y0 = base0[pi]
      fwd, left = fl_all[pi][i]
      trail = fl_all[pi][: i + 1]
      pts_w = [p0[:2] + f * np.array([np.cos(y0), np.sin(y0)]) + l * np.array([-np.sin(y0), np.cos(y0)]) for f, l in trail[::5]]
      zf = floors[pi] + 0.012

      def deco(scn, r=r, k=k, pts_w=pts_w, zf=zf, col=col, p0=p0, y0=y0):
        # reference line (dashed, world, along the initial heading) and path trail on the floor
        for s in np.arange(0.0, 9.0, 0.5):
          a = p0[:2] + s * np.array([np.cos(y0), np.sin(y0)])
          b = p0[:2] + (s + 0.28) * np.array([np.cos(y0), np.sin(y0)])
          rb.segment(scn, [a[0], a[1], zf], [b[0], b[1], zf], (1, 1, 1, 1), 0.012)
        for a, b in zip(pts_w[:-1], pts_w[1:]):
          rb.segment(scn, [a[0], a[1], zf + 0.004], [b[0], b[1], zf + 0.004], tuple(c / 255 for c in col) + (1,), 0.014)
        decorate_sim(rb, scn, r.tau[k, env], r.force[k, env], r.bq[k, env])

      look = np.array([pos[0], pos[1], floors[pi] + 0.55])
      im3 = rb.render(look, 2.7, np.degrees(yaw_now) + args.az_off_v1, -12.0, deco)
      img.paste(im3, (pi * (PW + MW), 36))
      x0 = pi * (PW + MW)
      # overlay
      text(dr, (x0 + 8, 36 + 6), tag, F15B, col)
      t_now = k / r.rate
      iv = "20 ms (fixed)"
      if ivl is not None:
        tc, ms = ivl
        j = max(0, int(np.searchsorted(tc, t_now + 1e-9) - 1))
        iv = f"{ms[j]:.0f} ms now (median {np.median(ms[1:]):.0f})"
      text(dr, (x0 + 8, 36 + 26), f"t = {t_now:5.2f} s     policy interval: {iv}", F13)
      hd = np.degrees(yaw_now - y0)
      text(dr, (x0 + 8, 36 + 44), f"heading change {hd:+6.1f} deg  ({'RIGHT' if hd < -0.5 else 'LEFT' if hd > 0.5 else 'straight'})", F15B, (255, 235, 120))
      text(dr, (x0 + 8, 36 + 64), f"lateral offset {left:+5.2f} m  ({'RIGHT' if left < -0.02 else 'LEFT' if left > 0.02 else 'on line'})", F15B, (255, 235, 120))
      text(dr, (x0 + 8, 36 + 84), f"forward distance {fwd:5.2f} m", F13)
      legend_sim(dr, x0 + 8, 36 + H3 - 112)
      map_panel(img, (x0 + PW, 36, MW, H3), trail, (fwd, left), yaw_now - y0, col, tag)
      dr.line([(x0 + PW + MW, 36), (x0 + PW + MW, 36 + H3)], fill=(90, 94, 100), width=2)
    text(dr, (W // 2, Hh - 13), f"Real-time playback, {fps} fps = {A.rate} Hz / {dsA} (left), {B.rate} Hz / {dsB} (right). "
         "Nominal simulation: no noise, no randomisation, no push. Recording: R3 and R8 (s2r_walk).", F11, DIM, anchor="mm", shadow=False)
    wr.add(img)
    if i % 50 == 0:
      print(f"[v1] frame {i}/{nfr}  {time.time() - t0:.0f}s", flush=True)
  wr.close()
  return dict(env=env, fps=fps, frames=wr.n, info=info)


# ------------------------------------------------------------------------------------------------------------ V2
def render_v2(args, rb: Robot, out: Path):
  S = SimRun("W5_real")
  R = RealLog(EXPR / "1021vz0.5ob.csv")
  env, info = pick_env(S)
  print("[v2] env", env, info)
  fps = 20
  ds = S.rate // fps  # 10
  nfr = int(round(min(S.T / S.rate, R.t[-1]) * fps))
  assert nfr == 400, nfr
  # floor of the simulator pane (median lowest foot)
  zs = []
  for k in range(0, S.T, S.T // 40):
    rb.pose(S.bp[k, env], S.bq[k, env], S.q[k, env, :12])
    zs.append(rb.lowest_foot_z())
  sim_floor = float(np.median(zs))
  tq = np.arange(nfr) / fps
  rq = dict(pos=np.radians(R.at(R.pos, tq)), roll=R.at(R.roll, tq), pitch=R.at(R.pitch, tq), yaw=R.at(R.yaw, tq))
  PW, H3, HS = 700, 350, 236
  W, Hh = 2 * PW, 36 + H3 + HS + 22
  wr = Writer(out, W, Hh, fps)
  # strips: raw series (deg)
  tS = S.D["t200"] if S.hi else np.arange(S.T) / S.rate
  simser = dict(t=tS, tgt=np.degrees(S.target[:, env, :]), meas=np.degrees(S.q[:, env, :12]))
  realser = dict(t=R.t, tgt=R.tgt, meas=R.pos)
  idx = dict(hip_pitch_R=0, knee_R=3, hip_pitch_L=6, knee_L=9)
  ylim = {"hip_pitch": (-45, 25), "knee": (-5, 90)}
  # left knee is negative in the model (mirror): plot |.| of the left side so that both legs overlay
  def strip(img, x, y, ser, t_now, label_extra):
    for ci, (jn, ttl) in enumerate((("hip_pitch", "hip_pitch [deg]"), ("knee", "knee [deg]"))):
      series = []
      for side, col in (("R", RIGHT_COL), ("L", LEFT_COL)):
        i = idx[f"{jn}_{side}"]
        sg = 1.0 if side == "R" else -1.0  # left joints are mirrored: flip the sign so both legs share an axis
        series.append(dict(t=ser["t"], y=sg * ser["tgt"][:, i], color=col, dash=True, label=f"{side} tgt"))
        series.append(dict(t=ser["t"], y=sg * ser["meas"][:, i], color=col, dash=False, label=f"{side} meas"))
      time_plot(img, (x + ci * PW // 2, y, PW // 2, HS), f"{ttl}  (L sign flipped)", series, t_now, 5.0, ylim[jn], "", xlabel=True)
  t0 = time.time()
  for i in range(args.first, nfr):
    t_now = i / fps
    img = Image.new("RGB", (W, Hh), BG)
    banner(img, ["CMD  vx 0, wz 0.5 rad/s (= 28.6 deg/s turn left)", "left: simulator, real policy intervals",
                 "right: real robot, kinematic replay"])
    dr = ImageDraw.Draw(img)
    # --- sim pane (in place)
    k = i * ds
    pos = S.bp[k, env].copy()
    pos[:2] = 0.0
    rb.pose(pos, S.bq[k, env], S.q[k, env, :12])
    rb.set_floor(sim_floor)
    yaw_s = float(S.yaw[k, env])

    def deco_s(scn):
      decorate_sim(rb, scn, S.tau[k, env], S.force[k, env], S.bq[k, env])

    im = rb.render(np.array([0, 0, sim_floor + 0.5]), 2.4, np.degrees(yaw_s) + args.az_off_v2, -12.0, deco_s)
    img.paste(im, (0, 36))
    text(dr, (8, 36 + 6), "SIM  policy interval = real-robot sequence (W5_real, env #%d)" % env, F15B, (255, 160, 60))
    text(dr, (8, 36 + 26), f"t = {t_now:5.2f} s     heading {np.degrees(yaw_s - S.yaw[0, env]):+6.0f} deg  (mean {np.degrees(yaw_s - S.yaw[0, env]) / max(t_now, 1e-6):+5.1f} deg/s)", F13)
    text(dr, (8, 36 + 44), "base held in place (translation removed); loads shown", F11, DIM)
    legend_sim(dr, 8, 36 + H3 - 112)
    # --- real pane (in place)
    yaw_r, pit_r, rol_r = float(rq["yaw"][i]), float(rq["pitch"][i]), float(rq["roll"][i])
    quat = quat_from_euler_zyx(yaw_r, pit_r, rol_r)
    rb.pose(np.array([0, 0, 1.0]), quat, rq["pos"][i])
    z = 1.0 - rb.lowest_foot_z()
    rb.pose(np.array([0, 0, z]), quat, rq["pos"][i])
    rb.set_floor(0.0)
    im = rb.render(np.array([0, 0, 0.5]), 2.4, np.degrees(yaw_r) + args.az_off_v2, -12.0, None)
    img.paste(im, (PW, 36))
    text(dr, (PW + 8, 36 + 6), "REAL ROBOT  1021vz0.5ob.csv  (policy interval median %.0f ms)" % (R.dt_med * 1000), F15B, (120, 230, 140))
    text(dr, (PW + 8, 36 + 26), f"t = {t_now:5.2f} s     yaw {np.degrees(yaw_r):+6.0f} deg  (mean {np.degrees(yaw_r) / max(t_now, 1e-6):+5.1f} deg/s, gyro integral)", F13)
    text(dr, (PW + 8, 36 + 44), "KINEMATIC REPLAY, NO LOAD DATA (no force / torque logged)", F13B, (255, 120, 120))
    text(dr, (PW + 8, 36 + 62), "base fixed in place; roll/pitch from gravity; height: lower foot on floor", F11, DIM)
    dr.line([(PW, 36), (PW, Hh)], fill=(90, 94, 100), width=2)
    # --- strips
    strip(img, 0, 36 + H3, simser, t_now, "")
    strip(img, PW, 36 + H3, realser, t_now, "")
    text(dr, (W // 2, Hh - 11), f"Real-time playback, {fps} fps (sim {S.rate} Hz / {ds}; real log ~{1 / R.dt_med:.0f} Hz rows interpolated to {fps} fps; strips use the raw rows). "
         "Window = last 5 s, right edge = now.", F11, DIM, anchor="mm", shadow=False)
    wr.add(img)
    if i % 50 == 0:
      print(f"[v2] frame {i}/{nfr}  {time.time() - t0:.0f}s", flush=True)
    if args.smoke and i + 1 >= args.smoke:
      break
  wr.close()
  return dict(env=env, fps=fps, frames=wr.n, info=info)


# ------------------------------------------------------------------------------------------------------------ V3
def render_v3(args, rb: Robot, out: Path):
  R = RealLog(EXPR / "vx0gain1walkob(진동확인용)1021.csv")
  fps = 20
  T0, T1 = 30.0, 80.0
  nfr = int(round((T1 - T0) * fps))
  tq = T0 + np.arange(nfr) / fps
  rq = dict(pos=np.radians(R.at(R.pos, tq)), roll=R.at(R.roll, tq), pitch=R.at(R.pitch, tq), yaw=R.at(R.yaw, tq))
  # event times from the data (documented in the audit: lean from ~38 s, first burst ~39.7 s)
  base = (R.t > 33) & (R.t < 37.5)
  gx0 = float(np.median(R.grav[base, 0]))
  i_lean = int(np.argmax((R.t > 36) & (np.abs(R.grav[:, 0] - gx0) > 0.01)))
  t_lean = float(R.t[i_lean])
  gnorm = np.degrees(np.linalg.norm(R.gyro, axis=1))
  i_burst = int(np.argmax((R.t > t_lean) & (gnorm > 30)))
  t_burst = float(R.t[i_burst])
  peak_lean = float(np.degrees(np.arcsin(R.grav[(R.t > t_lean) & (R.t < t_burst), 0].max())))
  print(f"[v3] lean onset {t_lean:.2f} s, burst onset {t_burst:.2f} s, peak lean {peak_lean:.1f} deg, gyro peak {gnorm[(R.t > t_lean) & (R.t < t_lean + 3)].max():.0f} deg/s")
  # spectrogram of the right hip_pitch target
  freqs = np.arange(0.5, 10.01, 0.1)
  tcols = np.arange(T0 - 20.0, T1 + 1e-9, 0.1)
  spec = lomb_spectrogram(R.t, R.tgt[:, 0], tcols, 2.0, freqs)
  amp = np.where(np.isnan(spec), np.nan, spec)
  import matplotlib

  cmap = matplotlib.colormaps["magma"]
  vmin, vmax = 0.03, 3.0  # deg, log scale

  def to_rgb(a):
    x = (np.log10(np.clip(a, vmin, vmax)) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
    rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    rgb[np.isnan(a)] = (14, 16, 20)
    return rgb

  spec_rgb = to_rgb(amp)[::-1]  # low frequency at the bottom
  print("[v3] spectrogram amplitude 5 Hz band max (deg): %.2f" % np.nanmax(spec[(freqs > 4.7) & (freqs < 5.5)]))
  W3, H3, PWR = 640, 400, 640
  WD, HP = 1280, 230
  Hh = 36 + H3 + HP + 22
  wr = Writer(out, WD, Hh, fps)
  yl_hp = (min(R.tgt[(R.t > 28) & (R.t < 82), 0].min(), R.pos[(R.t > 28) & (R.t < 82), 0].min()) - 3,
           max(R.tgt[(R.t > 28) & (R.t < 82), 0].max(), R.pos[(R.t > 28) & (R.t < 82), 0].max()) + 3)
  mk = [(t_lean, (255, 120, 120), "DISTURBANCE (lean starts)"), (t_burst, (255, 170, 60), "burst onset")]
  gyd = np.degrees(R.gyro)
  t0 = time.time()
  for i in range(args.first, nfr):
    t_now = tq[i]
    img = Image.new("RGB", (WD, Hh), BG)
    banner(img, ["CMD  vx 0, wz 0 (standing)", "real robot, kinematic replay, NO LOAD DATA",
                 f"log time {T0:g}-{T1:g} s"])
    dr = ImageDraw.Draw(img)
    yaw_r, pit_r, rol_r = float(rq["yaw"][i]), float(rq["pitch"][i]), float(rq["roll"][i])
    quat = quat_from_euler_zyx(yaw_r, pit_r, rol_r)
    rb.pose(np.array([0, 0, 1.0]), quat, rq["pos"][i])
    z = 1.0 - rb.lowest_foot_z()
    rb.pose(np.array([0, 0, z]), quat, rq["pos"][i])
    rb.set_floor(0.0)
    im = rb.render(np.array([0, 0, 0.5]), 2.4, np.degrees(yaw_r) + args.az_off_v3, -8.0, None)
    img.paste(im, (0, 36))
    text(dr, (8, 36 + 6), "REAL ROBOT  vx0gain1walkob(5Hz check)1021.csv", F15B, (120, 230, 140))
    text(dr, (8, 36 + 26), f"log time t = {t_now:6.2f} s   (video {t_now - T0:5.2f} s)   policy interval median {R.dt_med * 1000:.0f} ms", F13)
    lean = np.degrees(pit_r)
    state, scol = (("standing, quiet", (200, 200, 200)) if t_now < t_lean else
                   ("DISTURBANCE: body leans forward", (255, 110, 110)) if t_now < t_burst else
                   ("5.1 Hz OSCILLATION BURST", (255, 170, 60)))
    text(dr, (8, 36 + 46), state, F17B, scol)
    text(dr, (8, 36 + 70), f"forward lean (from gravity): {lean:+5.1f} deg", F13)
    text(dr, (8, 36 + 88), "KINEMATIC REPLAY, NO LOAD DATA (no force / torque logged)", F13B, (255, 120, 120))
    text(dr, (8, 36 + H3 - 36), "side view, base fixed in place; roll/pitch from gravity;", F11, DIM)
    text(dr, (8, 36 + H3 - 22), "yaw from gyro integral; height: lower foot on floor", F11, DIM)
    # (a) hip_pitch target vs measured ; (b) gyro x/y
    time_plot(img, (W3, 36, PWR, H3 // 2), "(a) right hip_pitch  target (dashed) vs measured (solid)",
              [dict(t=R.t, y=R.tgt[:, 0], color=(255, 190, 70), dash=True, label="target"),
               dict(t=R.t, y=R.pos[:, 0], color=(120, 220, 255), dash=False, label="measured")],
              t_now, 10.0, yl_hp, "deg", mk, xlabel=False)
    time_plot(img, (W3, 36 + H3 // 2, PWR, H3 // 2), "(b) gyro x (roll rate) and y (pitch rate)",
              [dict(t=R.t, y=gyd[:, 0], color=(255, 120, 200), dash=False, label="gyro x"),
               dict(t=R.t, y=gyd[:, 1], color=(130, 230, 150), dash=False, label="gyro y")],
              t_now, 10.0, (-150, 150), "deg/s", mk, xlabel=True)
    # (c) running spectrogram
    bx, by = 0, 36 + H3
    dr.rectangle([bx, by, WD - 1, by + HP - 1], fill=(14, 16, 20), outline=(60, 64, 70))
    text(dr, (bx + 6, by + 3), "(c) running spectrogram of right hip_pitch target (2 s window, amplitude in deg)", F13B, shadow=False)
    sx0, sy0, sw, sh = bx + 52, by + 22, WD - 52 - 90, HP - 22 - 22
    win = 20.0
    cols_per_s = 10
    j1 = int(round((t_now - tcols[0]) * cols_per_s)) + 1
    j0 = j1 - int(win * cols_per_s)
    sub = np.zeros((spec_rgb.shape[0], j1 - max(j0, 0), 3), np.uint8) + np.array((14, 16, 20), np.uint8)
    sub = spec_rgb[:, max(j0, 0):j1]
    full = np.zeros((spec_rgb.shape[0], int(win * cols_per_s), 3), np.uint8)
    full[:] = (14, 16, 20)
    full[:, full.shape[1] - sub.shape[1]:] = sub
    img.paste(Image.fromarray(full).resize((sw, sh), Image.NEAREST), (sx0, sy0))
    tl = t_now - win
    for tt in range(int(np.ceil(tl / 2.0)) * 2, int(t_now) + 1, 2):
      xx = sx0 + (tt - tl) / win * sw
      dr.line([(xx, sy0 + sh), (xx, sy0 + sh + 4)], fill=DIM)
      text(dr, (xx, sy0 + sh + 5), f"{tt}", F11, DIM, anchor="ma", shadow=False)
    for fv in (0, 2, 4, 6, 8, 10):
      yy = sy0 + sh - (fv - freqs[0]) / (freqs[-1] - freqs[0]) * sh
      if sy0 <= yy <= sy0 + sh:
        dr.line([(sx0 - 4, yy), (sx0, yy)], fill=DIM)
        text(dr, (sx0 - 6, yy), f"{fv} Hz", F11, DIM, anchor="rm", shadow=False)
    y51 = sy0 + sh - (5.1 - freqs[0]) / (freqs[-1] - freqs[0]) * sh
    dashed_line(dr, (sx0, y51), (sx0 + sw, y51), (110, 255, 255), 1, 8, 6)
    text(dr, (sx0 + sw + 4, y51), "5.1 Hz", F13B, (110, 255, 255), anchor="lm", shadow=False)
    for mi, (tm, col, lab) in enumerate(mk):
      if tl <= tm <= t_now:
        xx = sx0 + (tm - tl) / win * sw
        dashed_line(dr, (xx, sy0), (xx, sy0 + sh), col, 2, 6, 3)
        text(dr, (xx + 3, sy0 + 1 + 13 * mi), lab, F11, col, shadow=False)
    dr.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], outline=(70, 74, 80))
    # colour bar
    cb = np.linspace(1, 0, sh)[:, None]
    cbrgb = (cmap(cb)[..., :3] * 255).astype(np.uint8)
    cbrgb = np.repeat(cbrgb, 12, axis=1)
    img.paste(Image.fromarray(cbrgb), (sx0 + sw + 52, sy0))
    for av in (0.03, 0.1, 0.3, 1.0, 3.0):
      yy = sy0 + sh - (np.log10(av) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin)) * sh
      text(dr, (sx0 + sw + 66, yy), f"{av:g}", F11, DIM, anchor="lm", shadow=False)
    text(dr, (sx0 + sw + 52, sy0 - 4), "deg", F11, DIM, anchor="lb", shadow=False)
    text(dr, (WD // 2, Hh - 11), f"Real-time playback, {fps} fps (log rows ~{1 / R.dt_med:.0f} Hz, interpolated to {fps} fps for the pose; plots use the raw rows). "
         f"Spectrum: Hann-weighted Lomb-Scargle on the raw rows. Disturbance = first lean > 0.01 in gravity-x ({t_lean:.1f} s).", F11, DIM, anchor="mm", shadow=False)
    wr.add(img)
    if i % 100 == 0:
      print(f"[v3] frame {i}/{nfr}  {time.time() - t0:.0f}s", flush=True)
    if args.smoke and i + 1 >= args.smoke:
      break
  wr.close()
  return dict(fps=fps, frames=wr.n, t_lean=t_lean, t_burst=t_burst, peak_lean_deg=peak_lean)


# ------------------------------------------------------------------------------------------------------------ main
def probe(path: Path):
  o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                      "stream=codec_name,r_frame_rate,nb_read_frames,width,height,duration:format=duration",
                      "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
  j = json.loads(o)
  s = j["streams"][0]
  num, den = s["r_frame_rate"].split("/")
  fps = float(num) / float(den)
  n = int(s["nb_read_frames"])
  return dict(codec=s["codec_name"], size=f'{s["width"]}x{s["height"]}', fps=fps, frames=n, frames_over_fps=n / fps,
              duration_stream=float(s.get("duration", "nan")), duration_format=float(j["format"]["duration"]),
              size_mb=round(path.stat().st_size / 1e6, 2))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--video", default="all", choices=["v1", "v2", "v3", "all"])
  ap.add_argument("--out", default=str(REPO / "docs/video"))
  ap.add_argument("--xml", default=None)
  ap.add_argument("--smoke", type=int, default=0, help="render only N frames (test)")
  ap.add_argument("--first", type=int, default=0, help="first frame (test only)")
  ap.add_argument("--stem", default="2026-10-22_s2r_walk")
  ap.add_argument("--az-off-v1", type=float, default=50.0)
  ap.add_argument("--az-off-v2", type=float, default=60.0)
  ap.add_argument("--az-off-v3", type=float, default=70.0)
  args = ap.parse_args()
  xml = find_xml(args.xml)
  out = Path(args.out)
  todo = ["v1", "v2", "v3"] if args.video == "all" else [args.video]
  names = {"v1": "veer", "v2": "gait", "v3": "5hz"}
  report = {}
  for v in todo:
    size = {"v1": (520, 440), "v2": (700, 350), "v3": (640, 400)}[v]
    rb = Robot(xml, *size)
    path = out / f"{args.stem}_{names[v]}.mp4"
    t0 = time.time()
    res = {"v1": render_v1, "v2": render_v2, "v3": render_v3}[v](args, rb, path)
    res["render_s"] = round(time.time() - t0, 1)
    res["probe"] = probe(path)
    report[v] = res
    print(json.dumps({v: res}, indent=1, default=str), flush=True)
    del rb
  (out / f"{args.stem}_render_report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
  main()
