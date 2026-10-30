#!/usr/bin/env python3
"""Render the two stance-width analysis videos of policy terrain10_union_dr_49999 (docs/sim2real/2026-10-30_union49999_stance_width.md).

Same method and rules as render_s2r_videos.py (imported, not copied): nothing is simulated here, recorded states are PLAYED BACK.
  * real robot  : kinematic replay of Expr_Data/huphy_main_90bbd1f/*.csv. The 12 logged motor angles pose the model, the passive
                  ankle hinges and rods are solved from the loop constraints, roll / pitch come from the projected gravity vector,
                  yaw is NOT used (front camera, base held in place), the height is set so that the lower foot touches the floor.
                  No force / torque is logged: no load spheres, no arrows. Label: "kinematic replay, no load data".
  * simulator   : sim_stance_record.py recordings (npz) of the same command trace: joint load spheres + ground-reaction-force arrows
                  as in render_s2r_videos.py; base held in place (translation and yaw removed, roll / pitch kept).
  * GHOST       : in every pane the same forward-kinematics solve is repeated with the POLICY TARGET angles (`tgt`, simulator: the
                  processed action) and drawn as a translucent gold copy of the legs. Where the ghost and the solid legs differ, the
                  policy asked for a stance that the leg did not reach. Two lines on the floor in front of the feet: white = actual
                  foot-centre distance, gold = target foot-centre distance.
  * stance width: lateral distance of the two foot centres in the pelvis frame, tools/sim2real/walk/stance_fk.py (default pose 24.7 cm).

Videos (real time: fps = data rate / downsample = 50 Hz / 2 = 25 fps; the report json holds frames / fps / duration from ffprobe):
  v1 "stance_real_vs_sim"  : log 2 (1028_teleop_extra2_ob.csv, whole 65 s). Left real replay, right simulator nominal replay (nom_L2.npz).
                             Strips: width real vs sim vs default, hip_roll target vs measured (real, simulator), real hip_roll motor current iq.
  v2 "stance_variants"     : log 1 (1028_teleop_ob.csv) segment 9-29 s. Real replay + simulator base / hip_roll Kp x0.7 / pelvis +6 kg /
                             hip_roll zero offset 2 deg (true geometry) + caption and width strip.

Run (CPU, ONE render at a time, niced):
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 nice -n 10 \
    mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/render_stance_videos.py --video v1 --sim-dir <dir with nom_L2.npz ...> [--smoke 8]
Simulator recordings are made as in doc section 8 (sim_stance_record.py; nominal replay of log 2 = nom_L2; variants of log 1 --seg 9:29 =
ab_base, abx_kpR_0.7, abx_payload6, ab_encM2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import render_s2r_videos as R  # noqa: E402
from render_s2r_videos import (BG, DIM, F11, F13, F13B, F15B, F17B, FG, LEFT_COL, RIGHT_COL, Robot, Writer, decorate_sim,  # noqa: E402
                               legend_sim, probe, quat_from_euler_zyx, text, time_plot, dashed_line)
from stance_fk import FK  # noqa: E402


def series(fk: FK, deg12: np.ndarray):
  """stance_fk.series with a branch guard. The ankle loop solve (Gauss-Newton from the previous row) has several solutions: after one row
  that cannot be solved (commanded crank angles no ankle pose satisfies) the shared FK object stayed on a wrong branch and every later
  series came out up to 1.9 cm wrong in width (found while making these videos, see the doc). Here every call starts from the model's
  default pose, a row whose residual is > 1e-6 or whose ankle hinge leaves +-60 deg is re-solved from the default pose, and if that fails
  too the row is marked (resid = inf) so that the caller can hold the last solvable pose."""
  m, d = fk.m, fk.d
  free, q0 = fk.free_dofs, m.qpos0
  d.qpos[:] = q0
  keys = ("yR", "yL", "xR", "xL", "zR", "zL", "inR", "inL", "outR", "outL", "resid")
  o = {k: np.full(len(deg12), np.nan) for k in keys}
  rad = np.radians(deg12)

  def ok():
    return fk.resid <= 1e-6 and np.abs(d.qpos[fk.adr_pas]).max() < np.radians(60.0)

  for i in range(len(rad)):
    fk.pose(rad[i])
    if not ok():
      d.qpos[free + 1] = q0[free + 1]
      fk.pose(rad[i])
    if not ok():
      o["resid"][i] = np.inf
      d.qpos[free + 1] = q0[free + 1]
      continue
    f = fk.feet()
    o["yR"][i], o["yL"][i] = f["R"]["c"][1], f["L"]["c"][1]
    o["xR"][i], o["xL"][i] = f["R"]["c"][0], f["L"]["c"][0]
    o["zR"][i], o["zL"][i] = f["R"]["zmin"], f["L"]["zmin"]
    o["inR"][i], o["inL"][i] = f["R"]["y_in"], f["L"]["y_in"]
    o["outR"][i], o["outL"][i] = f["R"]["y_out"], f["L"]["y_out"]
    o["resid"][i] = fk.resid
  o["width"] = o["yL"] - o["yR"]
  return o

REPO = R.REPO
LOG1 = REPO / "Expr_Data/huphy_main_90bbd1f/1028_teleop_ob.csv"
LOG2 = REPO / "Expr_Data/huphy_main_90bbd1f/1028_teleop_extra2_ob.csv"
DEFAULT_W = 24.7  # cm, default pose
GHOST_RGBA = (1.0, 0.78, 0.10, 0.38)
REAL_COL, SIM_COL = (120, 230, 140), (255, 160, 60)
GOLD = (255, 200, 40)
REGIME_COL = {"stand": (90, 140, 255), "walk": (100, 220, 120), "turn": (220, 130, 255)}
VAR_COL = {"real": REAL_COL, "base": (255, 160, 60), "kp": (90, 190, 255), "pay": (255, 110, 110), "enc": (230, 140, 255)}
FPS, DS = 25, 2  # 50 Hz data / 2


# ----------------------------------------------------------------------------------------------------------------- data
def regime(vx, wz):
  """0 = other, 1 = stand (vx<0.05 and |wz|<0.05), 2 = walk (vx>0.1), 3 = turn (|wz|>0.1 and vx<=0.1) - same classes as the doc."""
  r = np.zeros(len(vx), int)
  r[(vx < 0.05) & (np.abs(wz) < 0.05)] = 1
  r[vx > 0.1] = 2
  r[(np.abs(wz) > 0.1) & (vx <= 0.1)] = 3
  return r


def roll_pitch_from_quat(q):
  """(N,4) wxyz base quaternions -> roll, pitch [rad] from the projected gravity vector, same convention as RealLog."""
  w, x, y, z = q.T
  # body-frame up vector = R^T e_z = (R[2,0], R[2,1], R[2,2])
  R20 = 2 * (x * z - w * y)
  R21 = 2 * (y * z + w * x)
  R22 = 1 - 2 * (x * x + y * y)
  up = np.stack([R20, R21, R22], 1)
  return np.arctan2(up[:, 1], up[:, 2]), np.arcsin(np.clip(-up[:, 0], -1, 1))


def yaw_of(q):
  w, x, y, z = q.T
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Track:
  """One pane's data on its own raw time base: actual / target motor angles [deg], widths [cm], roll / pitch, optional loads."""

  def __init__(self, name, t, pos12, tgt12, roll, pitch, fk, tau=None, force=None, yaw=None, iq=None):
    self.name, self.t = name, np.asarray(t, float)
    self.pos, self.tgt = np.asarray(pos12, float), np.asarray(tgt12, float)
    self.roll, self.pitch = roll, pitch
    self.tau, self.force, self.yaw, self.iq = tau, force, yaw, iq
    sa, st = series(fk, self.pos), series(fk, self.tgt)
    # rows where the loop constraints do not close (commanded crank angles that no ankle pose satisfies) keep the last solvable pose
    self.n_unsolved = {}
    for nm, arr, ss in (("pos", "pos", sa), ("tgt", "tgt", st)):
      bad = ss["resid"] > 1e-6
      self.n_unsolved[nm] = int(bad.sum())
      if bad.any():
        idx = np.maximum.accumulate(np.where(~bad, np.arange(len(bad)), 0))
        setattr(self, arr, getattr(self, arr)[idx])
        for k in ss:
          ss[k] = ss[k][idx]
    self.w_act, self.w_tgt = sa["width"] * 100, st["width"] * 100
    self.y_act = np.stack([sa["yR"], sa["yL"]], 1)
    self.y_tgt = np.stack([st["yR"], st["yL"]], 1)
    self.fk_resid = float(max(np.nanmax(sa["resid"]), np.nanmax(st["resid"])))
    # outward-positive hip_roll angle: right +roll, left -roll (doc section 1)
    self.ab_act = np.stack([self.pos[:, 1], -self.pos[:, 7]], 1)
    self.ab_tgt = np.stack([self.tgt[:, 1], -self.tgt[:, 7]], 1)

  def at(self, arr, tq):
    arr = np.asarray(arr)
    if arr.ndim == 1:
      return np.interp(tq, self.t, arr)
    return np.stack([np.interp(tq, self.t, arr[:, i]) for i in range(arr.shape[1])], -1)


def real_track(path, fk, t0, t1, with_iq=False):
  """Real log rows with t0 <= t < t1; time re-zeroed to t0."""
  log = R.RealLog(path)
  m = (log.t >= t0) & (log.t < t1 + 1e-9)
  t = log.t[m] - t0
  iq = None
  if with_iq:
    df = pd.read_csv(path, skiprows=1)
    iq = np.stack([df["extra/right_leg/hip_roll/iq"].values, -df["extra/left_leg/hip_roll/iq"].values], 1).astype(float)[m]  # outward +
  tr = Track("real", t, log.pos[m], log.tgt[m], log.roll[m], log.pitch[m], fk, iq=iq)
  tr.cmd = log.cmd[m]
  tr.reg = regime(tr.cmd[:, 0], tr.cmd[:, 1])
  return tr


def sim_track(sim_dir, name, env, fk, bias_deg=None):
  """Simulator recording -> Track. TRUE joint angles (q_true16) = real geometry; target = processed action. bias_deg: hip_roll encoder
  bias of the variant in degrees (reported = true + bias, right +, left -); the commanded TRUE target is processed - bias."""
  D = np.load(Path(sim_dir) / f"{name}.npz")
  q = np.degrees(D["q_true16"][:, env, :12])
  tg = np.degrees(D["target"][:, env])
  if bias_deg:
    tg = tg.copy()
    tg[:, 1] -= +bias_deg
    tg[:, 7] -= -bias_deg
  qb = D["base_quat"][:, env]
  roll, pitch = roll_pitch_from_quat(qb)
  tr = Track(name, D["t"], q, tg, roll, pitch, fk, tau=D["tau"][:, env], force=D["foot_force_w"][:, env], yaw=yaw_of(qb))
  tr.base_z = D["base_pos"][:, env, 2]
  tr.cmd = D["command"][:, env][:, [0, 2]]
  tr.reg = regime(tr.cmd[:, 0], tr.cmd[:, 1])
  return tr


def pick_env(sim_dir, name, fk):
  """Env whose mean actual width is the median over the recorded envs (deterministic, representative)."""
  D = np.load(Path(sim_dir) / f"{name}.npz")
  q = np.degrees(D["q_true16"][:, :, :12])
  m = [float(np.nanmean(series(fk, q[:, e])["width"])) * 100 for e in range(q.shape[1])]
  e = int(np.argsort(m)[len(m) // 2])
  return e, m


# ------------------------------------------------------------------------------------------------------------- rendering
class Ghost:
  """Second MjData posed with the target angles; its geoms are added to the scene as translucent copies (pelvis hidden)."""

  def __init__(self, rb: Robot):
    self.rb, self.d = rb, mujoco.MjData(rb.m)
    self.d.qpos[:] = rb.m.qpos0
    self.opt = mujoco.MjvOption()
    self.opt.geomgroup[3] = 0
    self.opt.geomgroup[4] = 0
    self.pert = mujoco.MjvPerturb()
    self.base_body = mujoco.mj_name2id(rb.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    self.resid = 0.0

  def solve(self, pos, quat, q12):
    rb, d = self.rb, self.d
    d.qpos[0:3], d.qpos[3:7], d.qpos[rb.adr12] = pos, quat, q12
    F = rb.free_dofs
    for _ in range(40):
      mujoco.mj_forward(rb.m, d)
      r = d.efc_pos.copy()
      if np.abs(r).max() < 1e-9:
        break
      J = d.efc_J.reshape(d.nefc, rb.m.nv)[:, F]
      d.qpos[F + 1] += -np.linalg.solve(J.T @ J + 1e-9 * np.eye(len(F)), J.T @ r)
    mujoco.mj_forward(rb.m, d)
    return float(np.abs(d.efc_pos).max())

  def pose(self, pos, quat, q12):
    """Warm-started solve; a row that fails (residual > 1e-6) or leaves the ankle hinge range is re-solved from the default ankle pose."""
    rb, d = self.rb, self.d
    r = self.solve(pos, quat, q12)
    if r > 1e-6 or np.abs(d.qpos[rb.adr_pas]).max() > np.radians(60.0):
      d.qpos[rb.free_dofs + 1] = rb.m.qpos0[rb.free_dofs + 1]
      r = self.solve(pos, quat, q12)
    self.resid = max(self.resid, r)

  def add(self, scn):
    m = self.rb.m
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(m, self.d, self.opt, self.pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for k in range(n0, scn.ngeom):
      g = scn.geoms[k]
      body = m.geom_bodyid[g.objid] if g.objtype == mujoco.mjtObj.mjOBJ_GEOM else -1
      g.matid = -1
      g.reflectance = 0.0
      g.rgba[:] = (0.0, 0.0, 0.0, 0.0) if body == self.base_body else GHOST_RGBA


def width_lines(rb: Robot, scn, y_act, y_tgt, floor_z):
  """Two segments on the floor in front of the feet (camera side +x): white = actual, gold = target foot-centre distance."""
  zf = floor_z + 0.012
  for xo, (yr, yl), col in ((0.20, y_act, (1, 1, 1, 1)), (0.26, y_tgt, GOLD_RGBA)):
    rb.segment(scn, [xo, yr, zf], [xo, yl, zf], col, 0.010)
    for yy in (yr, yl):
      rb.segment(scn, [xo - 0.03, yy, zf], [xo + 0.03, yy, zf], col, 0.010)


GOLD_RGBA = (1.0, 0.78, 0.10, 1.0)


def regime_bar(img, box, t_now, window, t, reg):
  """Thin bar at the top of the plot area of a time_plot: blue = standing, green = walking, purple = turning (window = last `window` s)."""
  dr = ImageDraw.Draw(img)
  x0, y0, w, h = box
  px0, py0, pw = x0 + 46, y0 + 20, w - 46 - 8
  tl = t_now - window
  m = (t >= tl - 0.2) & (t <= t_now)
  ti, ri = t[m], reg[m]
  for a, b, c in zip(ti[:-1], ti[1:], ri[:-1]):
    if c == 0:
      continue
    xa = px0 + max(0.0, (a - tl) / window) * pw
    xb = px0 + min(1.0, (b - tl) / window) * pw
    col = REGIME_COL[{1: "stand", 2: "walk", 3: "turn"}[int(c)]]
    dr.rectangle([xa, py0 + 1, max(xb, xa + 1), py0 + 6], fill=col)


def pose_guarded(rb, pos, quat, q12):
  """Robot.pose with the same branch guard as Ghost.pose."""
  r = rb.pose(pos, quat, q12)
  if r > 1e-6 or np.abs(rb.d.qpos[rb.adr_pas]).max() > np.radians(60.0):
    rb.d.qpos[rb.free_dofs + 1] = rb.m.qpos0[rb.free_dofs + 1]
    r = rb.pose(pos, quat, q12)
  return r


def draw_pane(rb, gh, tr, idx_time, floor_z, tq, pos_z, with_load, cam, size):
  """Pose actual + ghost for one frame and render. Returns (image, dict(width actual, width target, ...))."""
  roll, pitch = float(tr.at(tr.roll, tq)), float(tr.at(tr.pitch, tq))
  quat = quat_from_euler_zyx(0.0, pitch, roll)
  q_act = np.radians(tr.at(tr.pos, tq))
  q_tgt = np.radians(tr.at(tr.tgt, tq))
  if pos_z is None:  # real: lower foot touches the floor
    pose_guarded(rb, np.array([0, 0, 1.0]), quat, q_act)
    z = 1.0 - rb.lowest_foot_z()
  else:
    z = pos_z
  p = np.array([0.0, 0.0, z])
  pose_guarded(rb, p, quat, q_act)
  gh.pose(p, quat, q_tgt)
  rb.set_floor(floor_z)
  y_act, y_tgt = tr.at(tr.y_act, tq), tr.at(tr.y_tgt, tq)
  k = idx_time

  def deco(scn):
    gh.add(scn)
    width_lines(rb, scn, y_act, y_tgt, floor_z)
    if with_load:
      yaw = float(tr.yaw[k])
      c, s = np.cos(-yaw), np.sin(-yaw)
      Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
      force = np.stack([Rz @ tr.force[k, 0], Rz @ tr.force[k, 1]])
      decorate_sim(rb, scn, tr.tau[k], force, quat)

  im = rb.render(np.array([0.0, 0.0, floor_z + cam["look_h"]]), cam["dist"], cam["az"], cam["el"], deco)
  info = dict(w_act=float(tr.at(tr.w_act, tq)), w_tgt=float(tr.at(tr.w_tgt, tq)))
  return im, info


def pane_overlay(img, tr_name, title, col, info, note, note_col, x0=8, y0=6, compact=False):
  dr = ImageDraw.Draw(img)
  text(dr, (x0, y0), title, F15B, col)
  if compact:  # narrow pane (variants video): two short lines, note goes to the pane bottom (caller)
    text(dr, (x0, y0 + 20), f"actual {info['w_act']:5.1f} cm ({info['w_act'] - DEFAULT_W:+.1f})", F15B, (255, 255, 255))
    text(dr, (x0, y0 + 40), f"target {info['w_tgt']:5.1f} cm", F15B, GOLD)
    return
  text(dr, (x0, y0 + 20), f"width actual {info['w_act']:5.1f} cm", F17B, (255, 255, 255))
  text(dr, (x0, y0 + 42), f"width target {info['w_tgt']:5.1f} cm", F17B, GOLD)
  text(dr, (x0, y0 + 64), f"default {DEFAULT_W:.1f} cm   actual - default {info['w_act'] - DEFAULT_W:+5.1f} cm", F11, DIM)
  if note:
    text(dr, (x0, y0 + 80), note, F13B, note_col)


def ghost_legend(dr, x, y):
  dr.rectangle([x, y + 2, x + 14, y + 12], fill=(255, 200, 40), outline=(255, 255, 255))
  text(dr, (x + 20, y + 1), "ghost legs = policy target pose", F11, FG)
  dr.line([(x, y + 22), (x + 14, y + 22)], fill=(255, 255, 255), width=3)
  text(dr, (x + 20, y + 16), "white floor line = actual width", F11, FG)
  dr.line([(x, y + 38), (x + 14, y + 38)], fill=GOLD, width=3)
  text(dr, (x + 20, y + 32), "gold floor line = target width", F11, FG)


def header(img, t_now, t_total, cmd, reg, t_extra=""):
  dr = ImageDraw.Draw(img)
  W = img.width
  dr.rectangle([0, 0, W, 38], fill=(8, 10, 14))
  nm = {0: "other", 1: "STANDING", 2: "WALKING", 3: "TURNING"}[int(reg)]
  col = (200, 200, 200) if reg == 0 else REGIME_COL[{1: "stand", 2: "walk", 3: "turn"}[int(reg)]]
  text(dr, (12, 19), f"t = {t_now:6.2f} / {t_total:.1f} s{t_extra}", F17B, (255, 214, 90), anchor="lm", shadow=False)
  text(dr, (W // 2, 19), f"command  vx {cmd[0]:4.2f} m/s   wz {cmd[1]:+5.2f} rad/s  ({'left' if cmd[1] > 0.02 else 'right' if cmd[1] < -0.02 else 'straight'})",
       F17B, (255, 214, 90), anchor="mm", shadow=False)
  text(dr, (W - 12, 19), nm, F20B_, col, anchor="rm", shadow=False)


F20B_ = R.F20B

CAM = dict(dist=1.75, az=180.0, el=-14.0, look_h=0.50)


def sim_floor_level(rb, tr, ds_step=25):
  zs = []
  for k in range(0, len(tr.t), ds_step):
    quat = quat_from_euler_zyx(0.0, float(tr.pitch[k]), float(tr.roll[k]))
    pose_guarded(rb, np.array([0.0, 0.0, tr.base_z[k]]), quat, np.radians(tr.pos[k]))
    zs.append(rb.lowest_foot_z())
  return float(np.median(zs))


# ------------------------------------------------------------------------------------------------------------------ V1
def render_v1(args, xml, out):
  fk = FK(xml)
  real = real_track(LOG2, fk, 0.0, 1e9, with_iq=True)
  env, wmeans = pick_env(args.sim_dir, "nom_L2", fk)
  sim = sim_track(args.sim_dir, "nom_L2", env, fk)
  print(f"[v1] sim env {env} of {len(wmeans)}, mean widths {np.round(wmeans, 1)}; FK residual real/sim {real.fk_resid:.1e} {sim.fk_resid:.1e}")
  nfr = int(min(len(sim.t) // DS + (len(sim.t) % DS > 0), int(real.t[-1] * FPS) + 1))
  assert nfr * DS <= len(sim.t) + 1
  T_tot = nfr / FPS
  PW, PH, SH = 640, 440, 150
  W, H = 2 * PW, 40 + PH + 2 * SH + 26
  rb = Robot(xml, PW, PH)
  gh = Ghost(rb)
  wr = Writer(out, W, H, FPS, crf=28)
  floor_sim = sim_floor_level(rb, sim)
  print(f"[v1] sim floor z {floor_sim:.4f}")
  lo = (8, 46)
  win = 10.0
  stats = {}
  t0 = time.time()
  first = args.first
  last = nfr if not args.smoke else min(nfr, first + args.smoke)
  for i in range(first, last):
    tq = i / FPS
    k = min(i * DS, len(sim.t) - 1)
    img = Image.new("RGB", (W, H), BG)
    cmd = real.at(real.cmd, tq)
    reg = int(real.reg[int(np.argmin(np.abs(real.t - tq)))])
    header(img, tq, T_tot, cmd, reg)
    # real pane
    im, ir = draw_pane(rb, gh, real, None, 0.0, tq, None, False, CAM, (PW, PH))
    img.paste(im, (0, 40))
    pane_overlay(img, "real", "REAL ROBOT  1028_teleop_extra2_ob.csv", REAL_COL, ir,
                 "kinematic replay, no load data", (255, 120, 120), y0=46)
    # sim pane
    ts = sim.t[k]
    im, is_ = draw_pane(rb, gh, sim, k, floor_sim, ts, float(sim.base_z[k]), True, CAM, (PW, PH))
    img.paste(im, (PW, 40))
    pane_overlay(img, "sim", f"SIMULATOR  nominal replay of the same command trace (env #{env})", SIM_COL, is_,
                 "nominal: no noise, no random.", (200, 200, 200), x0=PW + 8, y0=46)
    dr = ImageDraw.Draw(img)
    legend_sim(dr, PW + 8, 40 + PH - 112)
    ghost_legend(dr, PW + 8 + 215, 40 + PH - 56)
    ghost_legend(dr, 8, 40 + PH - 56)
    dr.line([(PW, 40), (PW, 40 + PH)], fill=(90, 94, 100), width=2)
    # strips
    y0 = 40 + PH
    time_plot(img, (0, y0, PW, SH), "stance width [cm]  solid = actual, dashed = policy target",
              [dict(t=real.t, y=np.full_like(real.t, DEFAULT_W), color=(150, 150, 150), dash=True, label="default 24.7"),
               dict(t=real.t, y=real.w_tgt, color=tuple(int(c * 0.65) for c in REAL_COL), dash=True, label=""),
               dict(t=sim.t, y=sim.w_tgt, color=tuple(int(c * 0.65) for c in SIM_COL), dash=True, label=""),
               dict(t=real.t, y=real.w_act, color=REAL_COL, dash=False, label="real"),
               dict(t=sim.t, y=sim.w_act, color=SIM_COL, dash=False, label="simulator")],
              tq, win, lo, "cm", xlabel=True)
    regime_bar(img, (0, y0, PW, SH), tq, win, real.t, real.reg)
    def roll_series(tr):
      s = []
      for j, (c, lab) in enumerate(((RIGHT_COL, "R"), (LEFT_COL, "L"))):
        s.append(dict(t=tr.t, y=tr.ab_tgt[:, j], color=c, dash=True, label=f"{lab} target"))
        s.append(dict(t=tr.t, y=tr.ab_act[:, j], color=c, dash=False, label=f"{lab} measured"))
      return s
    time_plot(img, (PW, y0, PW, SH), "REAL hip_roll outward angle [deg]  dashed = target", roll_series(real), tq, win, (-8, 14), "deg", xlabel=True)
    time_plot(img, (0, y0 + SH, PW, SH), "SIMULATOR hip_roll outward angle [deg]  dashed = target", roll_series(sim), tq, win, (-8, 14), "deg", xlabel=True)
    time_plot(img, (PW, y0 + SH, PW, SH), "REAL hip_roll motor current iq, outward + [A] (no sim equivalent)",
              [dict(t=real.t, y=real.iq[:, 0], color=RIGHT_COL, dash=False, label="R iq"),
               dict(t=real.t, y=real.iq[:, 1], color=LEFT_COL, dash=False, label="L iq")],
              tq, win, (-6, 28), "A", xlabel=True)
    dr = ImageDraw.Draw(img)
    text(dr, (W // 2, H - 12),
         f"Real-time playback: {FPS} fps = 50 Hz data / {DS}. Window = last {win:g} s, right edge = now. Bar on the width plot: blue standing, green walking, purple turning. "
         "Hip_roll outward: right +, left sign flipped.", F11, DIM, anchor="mm", shadow=False)
    wr.add(img)
    if i % 50 == 0:
      print(f"[v1] frame {i}/{nfr}  {time.time() - t0:.0f}s", flush=True)
  wr.close()
  seg = real.reg
  stats.update(env=env, sim_env_mean_widths_cm=[round(x, 2) for x in wmeans], fk_resid=max(real.fk_resid, sim.fk_resid), ghost_resid=gh.resid, unsolved_rows=dict(real=real.n_unsolved, sim=sim.n_unsolved),
               real_width_mean_cm=round(float(np.mean(real.w_act)), 2), sim_width_mean_cm=round(float(np.mean(sim.w_act)), 2),
               real_target_mean_cm=round(float(np.mean(real.w_tgt)), 2), sim_target_mean_cm=round(float(np.mean(sim.w_tgt)), 2),
               real_stand_mean_cm=round(float(np.mean(real.w_act[seg == 1])), 2),
               sim_stand_mean_cm=round(float(np.mean(sim.w_act[sim.reg == 1])), 2), data_duration_s=float(real.t[-1]))
  return dict(fps=FPS, frames=wr.n, **stats)


# ------------------------------------------------------------------------------------------------------------------ V2
def render_v2(args, xml, out):
  fk = FK(xml)
  t0_seg, t1_seg = 9.0, 29.0
  real = real_track(LOG1, fk, t0_seg, t1_seg)
  specs = [("base", "ab_base", "SIM base (no change)", None),
           ("kp", "abx_kpR_0.7", "SIM hip_roll Kp x 0.7", None),
           ("pay", "abx_payload6", "SIM pelvis +6 kg", None),
           ("enc", "ab_encM2", "SIM hip_roll zero offset 2 deg (true geometry)", 2.0)]
  tracks, envs, means = {}, {}, {}
  for key, name, title, bias in specs:
    e, wm = pick_env(args.sim_dir, name, fk)
    tr = sim_track(args.sim_dir, name, e, fk, bias_deg=bias)
    tracks[key], envs[key] = tr, e
    # mean over ALL envs (comparable to the doc table)
    D = np.load(Path(args.sim_dir) / f"{name}.npz")
    qa = np.degrees(D["q_true16"][:, :, :12])
    means[key] = float(np.mean([np.nanmean(series(fk, qa[:, ee])["width"]) for ee in range(qa.shape[1])])) * 100
    print(f"[v2] {key}: env {e}, env-mean width {means[key]:.1f} cm, FK residual {tr.fk_resid:.1e}")
  means["real"] = float(np.mean(real.w_act))
  print("[v2] real mean width %.1f cm" % means["real"])
  nsim = min(len(t.t) for t in tracks.values())
  nfr = min(nsim // DS + (nsim % DS > 0), int(real.t[-1] * FPS) + 1)
  assert nfr == 500, nfr
  T_tot = nfr / FPS
  PW, PH = 480, 392
  CAP = 200
  W, H = 3 * PW, 40 + 2 * PH + 26
  rb = Robot(xml, PW, PH)
  gh = Ghost(rb)
  wr = Writer(out, W, H, FPS, crf=28)
  floors = {k: sim_floor_level(rb, t) for k, t in tracks.items()}
  cam = dict(CAM, dist=1.8, look_h=0.52)
  layout = [("real", 0, 0), ("base", 1, 0), ("kp", 2, 0), ("pay", 0, 1), ("enc", 1, 1)]
  titles = {"real": "REAL ROBOT  (log 1, 9-29 s)"}
  titles.update({k: t for k, _, t, _ in specs})
  win = 20.0
  t0 = time.time()
  first = args.first
  last = nfr if not args.smoke else min(nfr, first + args.smoke)
  for i in range(first, last):
    tq = i / FPS
    img = Image.new("RGB", (W, H), BG)
    cmd = real.at(real.cmd, tq)
    reg = int(real.reg[int(np.argmin(np.abs(real.t - tq)))])
    header(img, tq, T_tot, cmd, reg, t_extra=f"   (log {t0_seg + tq:5.2f} s)")
    for key, cx, cy in layout:
      x0, y0 = cx * PW, 40 + cy * PH
      if key == "real":
        im, inf = draw_pane(rb, gh, real, None, 0.0, tq, None, False, cam, (PW, PH))
        note, ncol, col = "kinematic replay, no load data", (255, 120, 120), REAL_COL
      else:
        tr = tracks[key]
        k = min(i * DS, len(tr.t) - 1)
        im, inf = draw_pane(rb, gh, tr, k, floors[key], tr.t[k], float(tr.base_z[k]), True, cam, (PW, PH))
        note, ncol, col = "", (200, 200, 200), VAR_COL[key]
      img.paste(im, (x0, y0))
      pane_overlay(img, key, titles[key], col, inf, note, ncol, x0=x0 + 8, y0=y0 + 6, compact=True)
      dr = ImageDraw.Draw(img)
      if note:
        text(dr, (x0 + 8, y0 + PH - 20), note, F13B, ncol)
      dr.rectangle([x0, y0, x0 + PW - 1, y0 + PH - 1], outline=(70, 74, 80))
    # caption + strip in slot (2, 1)
    cx0, cy0 = 2 * PW, 40 + PH
    dr = ImageDraw.Draw(img)
    dr.rectangle([cx0, cy0, cx0 + PW - 1, cy0 + PH - 1], fill=(14, 16, 20), outline=(70, 74, 80))
    lines = [("What to look at", F15B, FG),
             ("Gold ghost = commanded pose.", F13, GOLD),
             ("Solid legs = pose the leg reaches.", F13, FG),
             (f"Real robot mean width {means['real']:.1f} cm.", F13B, REAL_COL),
             (f"Kp x0.7: {means['kp']:.1f} cm,  +6 kg: {means['pay']:.1f} cm", F13B, (255, 255, 255)),
             ("reproduce the real width.", F13B, (255, 255, 255)),
             (f"Base: {means['base']:.1f} cm (too wide).", F13, VAR_COL["base"]),
             (f"Zero offset 2 deg: {means['enc']:.1f} cm true width,", F13, VAR_COL["enc"]),
             ("but the logged angles do not show it.", F13, VAR_COL["enc"])]
    for j, (s, f, c) in enumerate(lines):
      text(dr, (cx0 + 10, cy0 + 8 + j * 20), s, f, c, shadow=False)
    legend_sim(dr, cx0 + 10 + 270, cy0 + 8)
    ghost_legend(dr, cx0 + 10 + 270, cy0 + 8 + 112)
    cut = lambda tr: tr.t <= tq  # noqa: E731  # show only what has happened so far; axis = the whole segment (log time)
    mr = cut(real)
    sp = [dict(t=real.t[mr], y=np.full(mr.sum(), DEFAULT_W), color=(150, 150, 150), dash=True, label="default"),
          dict(t=real.t[mr], y=real.w_act[mr], color=VAR_COL["real"], dash=False, label="real")]
    for key, lab in (("base", "base"), ("kp", "Kp x0.7"), ("pay", "+6 kg"), ("enc", "offset 2deg")):
      mk = cut(tracks[key])
      sp.append(dict(t=tracks[key].t[mk], y=tracks[key].w_act[mk], color=VAR_COL[key], dash=False, label=lab))
    time_plot(img, (cx0, cy0 + 188, PW, PH - 188), "stance width, actual [cm]  (x axis: log time [s])", sp, t1_seg - t0_seg, win, (8, 36), "cm",
              markers=[(tq, (255, 255, 255), "")], t_origin=t0_seg, xlabel=True)
    text(dr, (W // 2, H - 12),
         f"Real-time playback: {FPS} fps = 50 Hz / {DS}. Simulator: nominal replay of the same command trace, starts from the default pose at 9 s, "
         "4 envs per variant (median-width env shown). Width = lateral distance of the two foot centres.", F11, DIM, anchor="mm", shadow=False)
    wr.add(img)
    if i % 50 == 0:
      print(f"[v2] frame {i}/{nfr}  {time.time() - t0:.0f}s", flush=True)
  wr.close()
  return dict(fps=FPS, frames=wr.n, envs=envs, env_mean_width_cm={k: round(v, 2) for k, v in means.items()},
              ghost_resid=gh.resid, unsolved_rows={k: t.n_unsolved for k, t in tracks.items()}, data_duration_s=float(t1_seg - t0_seg))


# ------------------------------------------------------------------------------------------------------------------ main
def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--video", required=True, choices=["v1", "v2"])
  ap.add_argument("--sim-dir", required=True, help="dir with nom_L2.npz (v1) and ab_base / abx_kpR_0.7 / abx_payload6 / ab_encM2 .npz (v2)")
  ap.add_argument("--out", default=str(REPO / "docs/video"))
  ap.add_argument("--xml", default=None)
  ap.add_argument("--smoke", type=int, default=0, help="render only N frames (test)")
  ap.add_argument("--first", type=int, default=0)
  ap.add_argument("--stem", default="2026-10-30_u49999")
  args = ap.parse_args()
  xml = R.find_xml(args.xml)
  out = Path(args.out)
  names = {"v1": "stance_real_vs_sim", "v2": "stance_variants"}
  path = out / f"{args.stem}_{names[args.video]}.mp4"
  t0 = time.time()
  res = {"v1": render_v1, "v2": render_v2}[args.video](args, xml, path)
  res["render_s"] = round(time.time() - t0, 1)
  res["file"] = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
  res["probe"] = probe(path)
  res["fps_ok"] = abs(res["probe"]["frames_over_fps"] - res["data_duration_s"]) < 0.1
  print(json.dumps({args.video: res}, indent=1, default=str), flush=True)
  if not args.smoke:
    rep = out / f"{args.stem}_render_report.json"
    old = json.loads(rep.read_text()) if rep.exists() else {}
    old[args.video] = res
    rep.write_text(json.dumps(old, indent=1, default=str))


if __name__ == "__main__":
  main()
