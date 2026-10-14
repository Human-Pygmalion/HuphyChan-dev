"""Standing-pose audit: numbers + front/side photo grids + gap-vs-iteration plot.

Input: analysis/out/stand10_<run>_m<iter>_s<seed>.npz / _model.mjb written by
tools/notes/stand10_measure_all.sh (12 s of pure standing, command 0, no warm-up walk).
All three sub-commands read those files only; nothing is trained or re-simulated here.

  # run with the mjlab venv (needs mujoco, numpy, PIL, matplotlib, pyyaml)
  mujoco-sim/mjlab/.venv/bin/python3 tools/notes/stand_pose_grid.py audit   # CSV + default pose json
  MUJOCO_GL=egl mujoco-sim/mjlab/.venv/bin/python3 tools/notes/stand_pose_grid.py grid    # needs EGL
  mujoco-sim/mjlab/.venv/bin/python3 tools/notes/stand_pose_grid.py plot
  ... stand_pose_grid.py grid --runs v7 --ckpts 100 300       # subset for a quick look

Definitions (all measured at t = 10.0 s of the rollout, frame index where time == 10.0):
  foot gap        distance between the left and right foot_link body origins, projected on the
                  pelvis' left axis (pelvis yaw only, so tilt does not leak in), in cm.
  outward value   joint angle with the sign chosen so that + means "leg/toe moves away from the
                  body midline" for BOTH legs. The sign is DERIVED from the compiled model by
                  perturbing the joint (see sign_probe), not assumed.
  fell            base height < 0.5 m at any recorded step OR a teleport (= environment reset)
                  between two consecutive steps. The training terminations (base < 0.7 m, tilt
                  > 70 deg) reset the environment, so a fall shows up as a jump, not as a low
                  height; both are flagged.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MJLAB = ROOT / "mujoco-sim/mjlab"
OUT = MJLAB / "analysis/out"
RUNS_DIR = MJLAB / "logs/rsl_rl/pygmalion_velocity"
ASSETS = ROOT / "docs/mujoco/assets"
DATA = ROOT / "docs/data"
CSV_PATH = DATA / "stand10_pose_audit.csv"
DEFAULT_JSON = DATA / "stand10_default_pose.json"

P = {
  "v7_p1": "2026-09-16_14-25-25_legonly_ab_v7_standonly_p1",
  "v7_p2": "2026-09-16_18-21-14_legonly_ab_v7_standonly_p2",
  "v7b_p1": "2026-09-16_22-47-17_legonly_ab_v7b_robust_stand_p1",
  "v7b_p1r": "2026-09-16_22-57-26_legonly_ab_v7b_robust_stand_p1r",
  "v7b_p2": "2026-09-16_23-49-54_legonly_ab_v7b_robust_stand_p2",
  "v9caps_p1": "2026-09-20_18-09-37_legonly_ab_v9_caps_p1",
  "v9caps_p2": "2026-09-20_19-17-25_legonly_ab_v9_caps_p2",
}
# run label -> display name, color (Okabe-Ito), [(phase, checkpoint iteration), ...]
RUNS = {
  "v7": ("v7 standing-only", "#0072B2",
         [("p1", 100), ("p1", 300), ("p1", 1000), ("p1", 2000), ("p1", 4100), ("p2", 6000), ("p2", 9000)]),
  "v7b": ("v7b robust stand", "#D55E00",
          [("p1", 100), ("p1r", 300), ("p1r", 1000), ("p1r", 1999),
           ("p2", 3000), ("p2", 5000), ("p2", 7000), ("p2", 8998)]),
  "v9caps": ("v9 caps", "#009E73",
             [("p1", 100), ("p1", 300), ("p1", 1000), ("p1", 1999),
              ("p2", 3000), ("p2", 5000), ("p2", 7000), ("p2", 8998)]),
}
SEEDS = range(6)
T_SHOT = 10.0
SIDE = {"L": 1.0, "R": -1.0}  # sign of the body-midline-to-leg direction in the pelvis frame (+y = left)


def run_dir(run: str, phase: str) -> Path:
  return RUNS_DIR / P[f"{run}_{phase}"]


def tag(run: str, ck: int, seed: int) -> str:
  return f"stand10_{run}_m{ck}_s{seed}"


# ---------------------------------------------------------------- model helpers
def load_model(path: Path):
  import mujoco

  return mujoco.MjModel.from_binary_path(str(path))


def _bid(m, name):
  import mujoco

  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/" + name)


def _qadr(m, jname):
  import mujoco

  return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + jname)])


def heading(R):
  """Yaw of a rotation matrix from its x axis projected on the ground plane."""
  return float(np.arctan2(R[1, 0], R[0, 0]))


def sign_probe(m) -> dict:
  """Derive from the model which joint sign means 'outward' (L foot to +y, R foot to -y).

  Perturb one joint by +0.05 rad around the straight, upright configuration and read how the
  foot (roll), the toe direction (yaw) and the ankle (knee) move in the pelvis frame.
  """
  import mujoco

  d = mujoco.MjData(m)
  out: dict = {}

  def fk(q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    R = d.xmat[_bid(m, "base_link")].reshape(3, 3)
    return R, {k: R.T @ (d.xpos[_bid(m, k)] - d.xpos[_bid(m, "base_link")]) for k in ("L_foot_link", "R_foot_link")}, d.xmat.copy()

  q0 = m.qpos0.copy()
  q0[3:7] = [1, 0, 0, 0]
  q0[2] = 0.9
  R0, p0, X0 = fk(q0)
  for s in "LR":
    for kind in ("hip_roll", "hip_yaw", "knee"):
      q = q0.copy()
      q[_qadr(m, f"{s}_{kind}_joint")] += 0.05
      R1, p1, X1 = fk(q)
      dp = p1[f"{s}_foot_link"] - p0[f"{s}_foot_link"]
      if kind == "hip_roll":
        out[(s, kind)] = float(np.sign(dp[1] * SIDE[s]))  # foot moves away from midline -> +
      elif kind == "hip_yaw":
        b = _bid(m, f"{s}_foot_link")
        dyaw = heading(X1[b].reshape(3, 3)) - heading(X0[b].reshape(3, 3))
        out[(s, kind)] = float(np.sign(dyaw * SIDE[s]))  # toe turns away from midline -> +
      else:
        out[(s, kind)] = float(np.sign(-dp[0]))  # ankle moves backwards = flexion -> +
  return out


def default_pose(run: str, phase: str, m) -> dict:
  """Reset-pose (no noise) foot gap from the saved training env.yaml init_state."""
  import yaml
  import mujoco

  class L(yaml.SafeLoader):
    pass

  L.add_multi_constructor("tag:yaml.org,2002:python/", lambda ld, suf, node: (
    ld.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
    else ld.construct_mapping(node, deep=True) if isinstance(node, yaml.MappingNode)
    else ld.construct_scalar(node)))
  cfg = yaml.load((run_dir(run, phase) / "params/env.yaml").read_text(), Loader=L)
  ist = cfg["scene"]["entities"]["robot"]["init_state"]
  jp = ist["joint_pos"]
  d = mujoco.MjData(m)
  q = m.qpos0.copy()
  q[:3] = [0, 0, float(ist["pos"][2])]
  q[3:7] = [1, 0, 0, 0]
  applied = {}
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1] for j in range(m.njnt)]
  for j, nm in enumerate(names):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    val = None
    if nm in jp:
      val = float(jp[nm])
    else:
      for pat, v in jp.items():
        if pat != nm and re.fullmatch(pat, nm):
          val = float(v)
    if val is not None:
      q[m.jnt_qposadr[j]] = val
      applied[nm] = val
  d.qpos[:] = q
  mujoco.mj_kinematics(m, d)
  pl = d.xpos[_bid(m, "L_foot_link")]
  pr = d.xpos[_bid(m, "R_foot_link")]
  return {
    "gap_cm": float((pl[1] - pr[1]) * 100),
    "joint_pos": {k: applied.get(k, 0.0) for k in (
      "L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint", "L_knee_joint",
      "R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint", "R_knee_joint")},
    "base_z": float(ist["pos"][2]),
    "env_yaml": str((run_dir(run, phase) / "params/env.yaml").relative_to(ROOT)),
  }


# ---------------------------------------------------------------- per-rollout metrics
def metrics(run: str, phase: str, ck: int, seed: int, cache: dict) -> dict:
  import mujoco

  t = tag(run, ck, seed)
  z = np.load(OUT / f"{t}.npz")
  m = cache.get("m")
  if m is None:
    m = load_model(OUT / f"{t}_model.mjb")
    cache["m"] = m
    cache["signs"] = sign_probe(m)
    cache["d"] = mujoco.MjData(m)
  sg = cache["signs"]
  d = cache["d"]
  time = z["time"]
  i = int(np.argmin(np.abs(time - T_SHOT)))
  qf = z["qpos_full"]
  d.qpos[:] = qf[i]
  mujoco.mj_kinematics(m, d)
  R = d.xmat[_bid(m, "base_link")].reshape(3, 3)
  psi = heading(R)
  fwd = np.array([np.cos(psi), np.sin(psi), 0.0])
  left = np.array([-np.sin(psi), np.cos(psi), 0.0])
  pl, pr = d.xpos[_bid(m, "L_foot_link")], d.xpos[_bid(m, "R_foot_link")]
  deg = 180.0 / np.pi
  r: dict = {
    "run": run, "phase": phase, "ckpt": ck, "seed": seed, "t_s": round(float(time[i]), 2),
    "foot_gap_cm": float((pl - pr) @ left * 100),
    "foot_foreaft_cm": float((pl - pr) @ fwd * 100),
  }
  for s in "LR":
    roll = float(z[f"qpos_{s}_hip_roll_joint"][i]) * deg
    yaw = float(z[f"qpos_{s}_hip_yaw_joint"][i]) * deg
    knee = float(z[f"qpos_{s}_knee_joint"][i]) * deg
    r[f"hip_roll_{s}_deg"] = roll
    r[f"hip_roll_out_{s}_deg"] = roll * sg[(s, "hip_roll")]
    r[f"hip_yaw_{s}_deg"] = yaw
    r[f"hip_yaw_out_{s}_deg"] = yaw * sg[(s, "hip_yaw")]
    r[f"knee_{s}_deg"] = knee
    r[f"knee_flex_{s}_deg"] = knee * sg[(s, "knee")]
    fb = _bid(m, f"{s}_foot_link")
    r[f"foot_toe_out_{s}_deg"] = (np.degrees(heading(d.xmat[fb].reshape(3, 3)) - psi + np.pi) % 360 - 180) * SIDE[s]
  zb = qf[:, 2]
  jump = np.hypot(*np.diff(qf[:, :2], axis=0).T) > 0.2
  jump |= np.abs(np.diff(zb)) > 0.15
  r["base_height_m"] = float(zb[i])
  r["min_height_m"] = float(zb.min())
  r["drift_m"] = float(np.hypot(*(qf[i, :2] - qf[0, :2])))
  r["resets_n"] = int(jump.sum())
  r["first_reset_s"] = float(time[1:][jump][0]) if jump.any() else ""
  r["fell"] = int(zb.min() < 0.5 or jump.any())
  return r


def all_rows(only_runs=None, only_ckpts=None) -> list[dict]:
  rows = []
  cache: dict = {}
  for run, (_, _, cks) in RUNS.items():
    if only_runs and run not in only_runs:
      continue
    for ph, ck in cks:
      if only_ckpts and ck not in only_ckpts:
        continue
      for s in SEEDS:
        if not (OUT / f"{tag(run, ck, s)}.npz").exists():
          print("MISSING", tag(run, ck, s))
          continue
        rows.append(metrics(run, ph, ck, s, cache))
  return rows


def cmd_audit(a):
  DATA.mkdir(parents=True, exist_ok=True)
  rows = all_rows()
  keys = list(rows[0].keys())
  with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
      w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
  print(f"wrote {CSV_PATH} ({len(rows)} rows)")
  # default (reset) pose per run + phase, from the model of any rollout of that run
  dfl = {}
  for run, (_, _, cks) in RUNS.items():
    m = load_model(OUT / f"{tag(run, cks[0][1], 0)}_model.mjb")
    for ph in sorted({p for p, _ in cks}):
      dfl[f"{run}_{ph}"] = default_pose(run, ph, m)
  sg = sign_probe(m)
  dfl["_signs_derived_from_model"] = {f"{k[0]}_{k[1]}": v for k, v in sg.items()}
  DEFAULT_JSON.write_text(json.dumps(dfl, indent=1))
  print(f"wrote {DEFAULT_JSON}")
  for k, v in dfl.items():
    print(k, v if k.startswith("_") else round(v["gap_cm"], 2))


# ---------------------------------------------------------------- rendering
CELL_W, CELL_H = 300, 400
DIST = 1.6
LOOK_Z = 0.48
FOVY_DEG = 45.0


def cmd_grid(a):
  import mujoco
  from PIL import Image, ImageDraw, ImageFont

  f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
  f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
  ASSETS.mkdir(parents=True, exist_ok=True)
  px_per_m = (CELL_H / 2) / (DIST * np.tan(np.radians(FOVY_DEG / 2)))
  for run, (name, _, cks) in RUNS.items():
    if a.runs and run not in a.runs:
      continue
    cks = [(p, c) for p, c in cks if not a.ckpts or c in a.ckpts]
    rows = all_rows([run], [c for _, c in cks])
    byk = {(r["ckpt"], r["seed"]): r for r in rows}
    m = None
    for view, suffix, az_off in (("front", "", 180.0), ("side", "_side", -90.0)):
      LM, TM = 110, 70
      W, H = LM + CELL_W * len(SEEDS), TM + CELL_H * len(cks)
      img = Image.new("RGB", (W, H), "white")
      dr = ImageDraw.Draw(img)
      dr.text((8, 8), f"{name} - {view} view at t = 10 s (camera fixed {DIST} m from pelvis; bar = 20 cm at pelvis depth)",
              fill="black", font=f_big)
      for s in SEEDS:
        dr.text((LM + s * CELL_W + CELL_W // 2 - 30, 40), f"seed {s}", fill="black", font=f_big)
      for ri, (ph, ck) in enumerate(cks):
        dr.text((6, TM + ri * CELL_H + CELL_H // 2 - 22), f"iter {ck}", fill="black", font=f_big)
        dr.text((6, TM + ri * CELL_H + CELL_H // 2 + 2), f"({ph})", fill="gray", font=f_small)
        for s in SEEDS:
          t = tag(run, ck, s)
          if (ck, s) not in byk:
            continue
          mdl = load_model(OUT / f"{t}_model.mjb")
          mdl.vis.global_.offwidth = CELL_W
          mdl.vis.global_.offheight = CELL_H
          mdl.vis.global_.fovy = FOVY_DEG
          dat = mujoco.MjData(mdl)
          z = np.load(OUT / f"{t}.npz")
          i = int(np.argmin(np.abs(z["time"] - T_SHOT)))
          dat.qpos[:] = z["qpos_full"][i]
          mujoco.mj_forward(mdl, dat)
          bid = _bid(mdl, "base_link")
          psi = heading(dat.xmat[bid].reshape(3, 3))
          cam = mujoco.MjvCamera()
          cam.type = mujoco.mjtCamera.mjCAMERA_FREE
          cam.lookat[:] = [dat.xpos[bid][0], dat.xpos[bid][1], LOOK_Z]
          cam.distance = DIST
          cam.azimuth = (np.degrees(psi) + az_off) % 360
          cam.elevation = -4.0
          rend = mujoco.Renderer(mdl, CELL_H, CELL_W)
          rend.update_scene(dat, camera=cam)
          cell = Image.fromarray(rend.render())
          rend.close()
          r = byk[(ck, s)]
          cd = ImageDraw.Draw(cell)
          txt = f"gap {r['foot_gap_cm']:.1f} cm"
          cd.rectangle([0, 0, CELL_W, 24], fill=(255, 255, 255))
          cd.text((4, 3), txt, fill="black", font=f_small)
          if r["fell"]:
            cd.text((CELL_W - 150, 3), f"RESET {r['first_reset_s']}s" if r["first_reset_s"] != "" else "LOW", fill=(200, 0, 0), font=f_small)
          bar = 0.20 * px_per_m
          cd.line([(CELL_W - 10 - bar, CELL_H - 12), (CELL_W - 10, CELL_H - 12)], fill="black", width=3)
          img.paste(cell, (LM + s * CELL_W, TM + ri * CELL_H))
          dr.rectangle([LM + s * CELL_W, TM + ri * CELL_H, LM + (s + 1) * CELL_W - 1, TM + (ri + 1) * CELL_H - 1], outline=(180, 180, 180))
      out = ASSETS / f"stand10_{run}_grid{suffix}.png"
      if a.runs or a.ckpts:
        out = ASSETS / f"stand10_{run}_grid{suffix}_subset.png"
      img.save(out)
      print("wrote", out, img.size)


# ---------------------------------------------------------------- plot
def cmd_plot(a):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import matplotlib.ticker

  rows = list(csv.DictReader(open(CSV_PATH)))
  dfl = json.loads(DEFAULT_JSON.read_text())
  fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
  off = {"v7": 0.94, "v7b": 1.0, "v9caps": 1.06}
  for run, (name, col, cks) in RUNS.items():
    xs, ys, fx, fy, med = [], [], [], [], {}
    for r in rows:
      if r["run"] != run:
        continue
      x, y = int(r["ckpt"]) * off[run], float(r["foot_gap_cm"])
      if int(r["fell"]):
        fx.append(x); fy.append(y)
      else:
        xs.append(x); ys.append(y)
      med.setdefault(int(r["ckpt"]), []).append(y)
    ax.scatter(xs, ys, s=22, color=col, alpha=0.55, edgecolor="none")
    if fx:
      ax.scatter(fx, fy, s=40, marker="x", color=col, linewidths=1.5)
    mk = sorted(med)
    ax.plot([k * off[run] for k in mk], [float(np.median(med[k])) for k in mk], color=col, lw=2, marker="o", ms=5, label=f"{name} (median of 6 seeds)")
    dg = {round(dfl[f"{run}_{ph}"]["gap_cm"], 2) for ph in {p for p, _ in cks}}
    ax.axhline(sorted(dg)[0], color=col, ls="--", lw=1, alpha=0.8)
  ax.set_xscale("log")
  allck = sorted({c for _, (_, _, cks) in RUNS.items() for _, c in cks})
  ax.set_xticks([100, 300, 1000, 2000, 4000, 9000])
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.set_xlabel("training iteration (log scale; each dot = one seed; x = rollout with a reset or base < 0.5 m)")
  ax.set_ylabel("lateral foot gap at t = 10 s (cm)")
  ax.set_title("Standing foot gap vs training iteration (dashed = gap of the reset pose, per run)")
  ax.grid(alpha=0.3)
  ax.legend(loc="best")
  out = ASSETS / "stand10_width_vs_ckpt.png"
  fig.tight_layout()
  fig.savefig(out)
  print("wrote", out, {k: v["gap_cm"] for k, v in dfl.items() if not k.startswith("_")})


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=["audit", "grid", "plot"])
  ap.add_argument("--runs", nargs="*")
  ap.add_argument("--ckpts", nargs="*", type=int)
  a = ap.parse_args()
  {"audit": cmd_audit, "grid": cmd_grid, "plot": cmd_plot}[a.cmd](a)
