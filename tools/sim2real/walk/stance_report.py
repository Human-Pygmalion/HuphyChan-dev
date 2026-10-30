#!/usr/bin/env python3
"""Aggregate the stance-width study: real logs vs simulator replays (nominal / training DR) vs one-variable tests; writes the
numbers JSON and the figure.   See docs/sim2real/2026-10-30_union49999_stance_width.md.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/stance_report.py --xml <xml> --real-dir Expr_Data/huphy_main_90bbd1f \
      --sim-dir <dir with csv/ and *.npz> --out-json <json> --out-fig <png> [--cache dir] [--tmp dir]
Sign convention for `abd` (abduction angle, outward leg = positive): right = +hip_roll, left = -hip_roll (checked by FK).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stance_analyze import summarise
from stance_fk import FK, load_csv

REG = ("all", "stand", "walk", "turn")


def cut_csv(src, t0, t1, dst):
  lines = Path(src).read_text(encoding="utf-8").splitlines()
  keep = [ln for ln in lines if ln.startswith("#")]
  hdr = [ln for ln in lines if not ln.startswith("#")][0]
  keep.append(hdr)
  for ln in lines:
    if ln.startswith("#") or ln == hdr:
      continue
    t = float(ln.split(",", 1)[0])
    if t0 <= t < t1:
      keep.append(ln)
  Path(dst).write_text("\n".join(keep) + "\n")


def avg_envs(per_env, key_path):
  vals = []
  for o in per_env:
    x = o
    for k in key_path:
      x = x[k]
    vals.append(x)
  v = np.array(vals, float)
  return float(np.nanmean(v)), float(np.nanstd(v))


def sim_contact_stats(npz, env_k):
  """Roll tgt/pos by contact for one env (rows before the first fall)."""
  d = np.load(npz)
  term = d["terminated"][:, env_k]
  n = int(np.argmax(term)) + 1 if term.any() else len(term)
  tgt = np.degrees(d["target"][:n, env_k])
  pos = np.degrees(d["obs"][:n, env_k, 18:30] + d["default_q12"])
  con = d["foot_contact"][:n, env_k]
  out = {}
  for si, s in enumerate("RL"):
    j = 1 + 6 * si
    other = 1 - si
    single = con[:, si] & ~con[:, other]
    swing = ~con[:, si]
    both = con[:, si] & con[:, other]
    for nm, mk in (("single_stance", single), ("swing", swing), ("double", both)):
      out[f"{s}_{nm}_tgt"] = float(tgt[mk, j].mean()) if mk.any() else float("nan")
      out[f"{s}_{nm}_pos"] = float(pos[mk, j].mean()) if mk.any() else float("nan")
      out[f"{s}_{nm}_frac"] = float(mk.mean())
  return out


def run_group(fk, sim_dir, name, cache):
  files = sorted((Path(sim_dir) / "csv").glob(f"{name}_env*.csv"))
  per = [summarise(name, f, fk, cache)[0] for f in files]
  falls = None
  jp = Path(sim_dir) / f"{name}.json"
  if jp.exists():
    falls = json.loads(jp.read_text())["falls"]
  npz = Path(sim_dir) / f"{name}.npz"
  con = [sim_contact_stats(npz, k) for k in range(len(files))] if npz.exists() else []
  return per, falls, con


def flat(per):
  """Mean over envs of the main numbers, for regimes in REG."""
  o = {}
  for rg in REG:
    r = {}
    for k in ("width_pos", "width_tgt", "gap_pos", "gap_tgt", "right_roll_tgt", "right_roll_pos", "right_roll_err", "left_roll_tgt", "left_roll_pos",
              "left_roll_err", "right_abd_pos", "left_abd_pos", "right_abd_tgt", "left_abd_tgt", "rows",
              "right_roll_err_stance", "right_roll_err_swing", "left_roll_err_stance", "left_roll_err_swing"):
      r[k] = avg_envs(per, ["regimes", rg, k])[0]
      r[k + "_sd"] = avg_envs(per, ["regimes", rg, k])[1]
    for jn in range(12):
      for q in ("tgt", "pos", "err"):
        r[f"j{jn}_{q}"] = avg_envs(per, ["regimes", rg, f"j{jn}_{q}"])[0]
    o[rg] = r
  o["step_width_pos"] = avg_envs(per, ["step_width_pos", "mean"])[0]
  o["step_width_tgt"] = avg_envs(per, ["step_width_tgt", "mean"])[0]
  return o


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--real-dir", required=True)
  ap.add_argument("--sim-dir", required=True)
  ap.add_argument("--out-json", required=True)
  ap.add_argument("--cache", default="")
  ap.add_argument("--tmp", required=True)
  a = ap.parse_args()
  fk = FK(a.xml)
  tmp = Path(a.tmp)
  tmp.mkdir(parents=True, exist_ok=True)
  res = {"real": {}, "sim": {}}
  R = Path(a.real_dir)
  logs = {"L1": R / "1028_teleop_ob.csv", "L2": R / "1028_teleop_extra2_ob.csv"}
  for k, p in logs.items():
    o, _ = summarise("real_" + k, p, fk, a.cache or None)
    res["real"][k] = o
  seg = tmp / "real_seg_9_29.csv"
  cut_csv(logs["L1"], 9.0, 29.0, seg)
  res["real"]["SEG"] = summarise("real_seg", seg, fk, None)[0]
  for name in sorted({p.name.rsplit("_env", 1)[0] for p in (Path(a.sim_dir) / "csv").glob("*_env*.csv")}):
    if name.startswith(("smoke", "vt_")):
      continue
    per, falls, con = run_group(fk, a.sim_dir, name, a.cache or None)
    res["sim"][name] = {"flat": flat(per), "falls": falls, "contact": con, "n_env": len(per)}
    f = res["sim"][name]["flat"]["all"]
    print(f"{name}: width pos {f['width_pos']:.3f} tgt {f['width_tgt']:.3f}  abd pos R {f['right_abd_pos']:+.1f} L {f['left_abd_pos']:+.1f}  "
          f"roll err R {f['right_roll_err']:+.1f} L {f['left_roll_err']:+.1f}  falls {falls}")
  Path(a.out_json).write_text(json.dumps(res, indent=1))
