#!/usr/bin/env python3
"""Stance-width and hip_roll statistics of real-layout walking CSVs (real robot or simulator export). Uses stance_fk.FK.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/stance_analyze.py --xml <xml> --label real1=<csv> [--label sim=<csv>,<csv>...] \
      --out stats.json [--cache dir]

Regimes (from obs/cmd): stand: vx<0.05 and |wz|<0.05; walk: vx>0.1; turn: |wz|>0.1 and vx<=0.1.
Stance/swing of a leg is INFERRED from the FK: the leg whose sole is lower along gravity-up (up = -obs/grav) is the stance leg
when it is lower by more than 8 mm than the other sole; otherwise the row is double support. Step width = lateral centre
distance at the rows where the two feet pass each other fore-aft (sign change of x_R - x_L) during walking (vx>0.1).
Sign convention (checked by FK in stance_fk.py): right hip_roll + = abduction (foot outward), left hip_roll - = abduction.
So `abd` = +roll for the right leg and -roll for the left leg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stance_fk import COLS, FK, JT, load_csv, series


def regimes(cmd_vx, cmd_wz):
  stand = (cmd_vx < 0.05) & (np.abs(cmd_wz) < 0.05)
  walk = cmd_vx > 0.1
  turn = (np.abs(cmd_wz) > 0.1) & (cmd_vx <= 0.1)
  return {"all": np.ones_like(stand), "stand": stand, "walk": walk, "turn": turn}


def fk_cached(fk, deg12, key, cache):
  if cache:
    p = Path(cache) / (hashlib.md5((key + str(deg12.shape) + str(np.nansum(deg12))).encode()).hexdigest() + ".npz")
    if p.exists():
      return dict(np.load(p))
  s = series(fk, deg12)
  if cache:
    Path(cache).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **s)
  return s


def m(a):
  a = np.asarray(a, float)
  a = a[np.isfinite(a)]
  return float(a.mean()) if len(a) else float("nan")


def analyse_one(fk, path, cache=None):
  cols, a = load_csv(path)
  c = {n: i for i, n in enumerate(cols)}
  t = a[:, 0]
  vx, wz = a[:, c["obs/cmd/vx"]], a[:, c["obs/cmd/wz"]]
  pos = a[:, [c[f"pos/{n}"] for n in COLS]]
  tgt = a[:, [c[f"tgt/{n}"] for n in COLS]]
  g = a[:, [c["obs/grav/x"], c["obs/grav/y"], c["obs/grav/z"]]]
  return t, vx, wz, pos, tgt, g, fk_cached(fk, pos, str(path) + ":pos", cache), fk_cached(fk, tgt, str(path) + ":tgt", cache)


def stance_flags(S, g):
  """Per-row stance inference. Heights along gravity-up using the pelvis-frame foot sole z and the gravity vector."""
  up = -g / np.linalg.norm(g, axis=1, keepdims=True)
  # sole height of each foot along up: use the foot centre (x,y) and sole z in the pelvis frame
  hR = S["xR"] * up[:, 0] + S["yR"] * up[:, 1] + S["zR"] * up[:, 2]
  hL = S["xL"] * up[:, 0] + S["yL"] * up[:, 1] + S["zL"] * up[:, 2]
  d = hL - hR  # >0: left sole higher -> right is stance
  stR = d > 0.008
  stL = d < -0.008
  return stR, stL, d


def step_width(S, mask_walk):
  dx = S["xR"] - S["xL"]
  ev = np.where((np.sign(dx[1:]) != np.sign(dx[:-1])) & mask_walk[1:] & mask_walk[:-1])[0] + 1
  return S["width"][ev], ev


def summarise(label, path, fk, cache=None):
  t, vx, wz, pos, tgt, g, SP, ST = analyse_one(fk, path, cache)
  reg = regimes(vx, wz)
  stR, stL, d = stance_flags(SP, g)
  out = {"label": label, "path": str(path), "rows": int(len(t)), "dur_s": float(t[-1] - t[0]), "regimes": {}}
  swR, swL = stL, stR  # right swings while left stands
  for rn, mk in reg.items():
    r = {"rows": int(mk.sum())}
    for si, s in enumerate(("right", "left")):
      j = JT.index("hip_roll") + 6 * si
      sgn = 1.0 if si == 0 else -1.0
      r[f"{s}_roll_tgt"] = m(tgt[mk, j]); r[f"{s}_roll_pos"] = m(pos[mk, j]); r[f"{s}_roll_err"] = m(tgt[mk, j] - pos[mk, j])
      stance = (stR if si == 0 else stL) & mk
      swing = (swR if si == 0 else swL) & mk
      r[f"{s}_roll_tgt_stance"] = m(tgt[stance, j]); r[f"{s}_roll_pos_stance"] = m(pos[stance, j]); r[f"{s}_roll_err_stance"] = m(tgt[stance, j] - pos[stance, j])
      r[f"{s}_roll_tgt_swing"] = m(tgt[swing, j]); r[f"{s}_roll_pos_swing"] = m(pos[swing, j]); r[f"{s}_roll_err_swing"] = m(tgt[swing, j] - pos[swing, j])
      r[f"{s}_abd_pos"] = sgn * r[f"{s}_roll_pos"]; r[f"{s}_abd_tgt"] = sgn * r[f"{s}_roll_tgt"]
    r["width_pos"] = m(SP["width"][mk]); r["width_tgt"] = m(ST["width"][mk])
    r["gap_pos"] = m(SP["gap"][mk]); r["gap_tgt"] = m(ST["gap"][mk])
    r["width_pos_p5"] = float(np.nanpercentile(SP["width"][mk], 5)) if mk.any() else float("nan")
    r["width_pos_p95"] = float(np.nanpercentile(SP["width"][mk], 95)) if mk.any() else float("nan")
    r["gap_pos_min"] = float(np.nanmin(SP["gap"][mk])) if mk.any() else float("nan")
    r["stance_frac_R"] = m(stR[mk]); r["stance_frac_L"] = m(stL[mk])
    # all other joints: tgt, pos, tgt-pos
    for jn in range(12):
      r[f"j{jn}_tgt"] = m(tgt[mk, jn]); r[f"j{jn}_pos"] = m(pos[mk, jn]); r[f"j{jn}_err"] = m(tgt[mk, jn] - pos[mk, jn])
    out["regimes"][rn] = r
  wmask = reg["walk"]
  sw_p, ev = step_width(SP, wmask)
  sw_t, _ = step_width(ST, wmask)
  out["step_width_pos"] = {"n": int(len(sw_p)), "mean": m(sw_p), "p5": float(np.nanpercentile(sw_p, 5)) if len(sw_p) else float("nan"),
                           "p95": float(np.nanpercentile(sw_p, 95)) if len(sw_p) else float("nan")}
  out["step_width_tgt"] = {"n": int(len(sw_t)), "mean": m(sw_t)}
  out["fk_resid_max"] = float(max(np.nanmax(SP["resid"]), np.nanmax(ST["resid"])))
  return out, dict(t=t, vx=vx, wz=wz, pos=pos, tgt=tgt, SP=SP, ST=ST)


def table_row(r):
  return r


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--label", action="append", required=True, help="label=csv1,csv2,... (stats are per file)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--cache", default="")
  a = ap.parse_args()
  fk = FK(a.xml)
  res = {}
  for spec in a.label:
    lab, paths = spec.split("=", 1)
    res[lab] = []
    for p in paths.split(","):
      o, _ = summarise(lab, p, fk, a.cache or None)
      res[lab].append(o)
      w = o["regimes"]["walk"]; al = o["regimes"]["all"]
      print(f"{lab} {Path(p).name}: all width pos {al['width_pos']:.3f} tgt {al['width_tgt']:.3f} | walk width pos {w['width_pos']:.3f} tgt {w['width_tgt']:.3f} "
            f"| roll R tgt/pos {al['right_roll_tgt']:+.1f}/{al['right_roll_pos']:+.1f} L {al['left_roll_tgt']:+.1f}/{al['left_roll_pos']:+.1f} | stepw {o['step_width_pos']['mean']:.3f}", flush=True)
  Path(a.out).write_text(json.dumps(res, indent=1))
