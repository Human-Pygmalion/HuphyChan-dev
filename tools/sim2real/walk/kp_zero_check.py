#!/usr/bin/env python3
"""Real-robot log test of (A) the effective joint stiffness and (B) joint zero offsets, policy terrain10_union_dr_49999.
Docs: docs/sim2real/2026-10-30_union49999_kp_and_zero_check.md

  CUDA_VISIBLE_DEVICES="" nice -n 10 mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/kp_zero_check.py \
      --xml <union_branch>/mjlab_policy/pygmalion_mjlab/assets/pygmalion_legonly_ab.xml \
      --real Expr_Data/huphy_main_90bbd1f/1028_teleop_extra2_ob.csv --sim-dir <out>/sim --name nom_L2 \
      --out-json docs/sim2real/data/2026-10-30_union49999_kp_and_zero_check.json --out-png docs/sim2real/img/....png

A. The motor runs MIT mode: tau = Kp*(tgt-pos) + Kd*(0-vel) (firmware, high rate). The log has tgt (pre-guard), pos (14 bit encoder,
   0.022 deg step) at 50 Hz and the motor current iq ("iq filter value", A) which is read round-robin: a NEW iq value arrives
   every 6th row (8.3 Hz), the 5 rows between repeat it. Only the rows where iq changes are used.
   tau_iq = KT * iq with KT = 1.485 N.m/A (RS04: rated 40 N.m / 27 Apk; 2.1 N.m/Arms / sqrt 2; output shaft, gear 9:1 included).
   KT cancels in the ratio hip_roll / hip_pitch (same motor model), so the verdict does not depend on it.
   Time alignment (use pos of the NEXT row, lag in rows, first-order smoothing) is chosen on the CONTROL joints
   (hip_pitch, knee) only, then applied to hip_roll unchanged.
B. FK of the logged pos (stance_fk.FK) + IMU gravity (obs/grav = pelvis-frame gravity direction): world height of the lowest sole point
   of each foot and the tilt of each sole plane. The same method is run on the simulator export of the same command trace (soles
   are truly flat there) = method error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stance_fk import COLS, FK, JT, load_csv  # noqa: E402

KT = 1.485  # N.m per A of iq (output shaft)
GAIN = {"hip_pitch": (135.0, 6.0), "hip_roll": (135.0, 6.0), "knee": (130.0, 6.0), "hip_yaw": (75.0, 4.0)}
SEG_A = {"all": None, "stand": "stand", "walk": "walk", "turn": "turn", "stand_54_65s": "long"}


def vel(x, t, h=2):
  v = np.full_like(x, np.nan)
  v[h:-h] = (x[2 * h:] - x[:-2 * h]) / (t[2 * h:] - t[:-2 * h])
  return v


def ema(x, alpha):
  y = np.empty_like(x)
  y[0] = x[0]
  for i in range(1, len(x)):
    y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
  return y


def shift(x, L):
  return x if L == 0 else np.r_[np.full(L, np.nan), x[:len(x) - L]]


def prep(x, s, L, tf):
  """x[k+s] smoothed (EMA time constant tf rows, 0 = none) and delayed by L rows -> aligned with row k."""
  y = np.r_[x[s:], np.full(s, np.nan)]
  if tf:
    y = ema(np.nan_to_num(y), 1.0 / (tf + 1.0))
  return shift(y, L)


def ols(y, X):
  A = np.c_[X, np.ones(len(y))]
  b = np.linalg.lstsq(A, y, rcond=None)[0]
  r = y - A @ b
  return b, 1 - r.var() / y.var(), r


def load(path):
  cols, a = load_csv(path)
  return {n: i for i, n in enumerate(cols)}, a


def regimes(t, vx, wz):
  st = (vx < 0.05) & (np.abs(wz) < 0.05)
  return {"all": np.ones_like(st), "stand": st, "walk": vx > 0.1, "turn": (np.abs(wz) > 0.1) & (vx <= 0.1), "stand_54_65s": st & (t >= 54.8)}


# ----------------------------------------------------------------------------------------------------------------------- A
def part_a(c, a, rng):
  t = a[:, 0]
  reg = regimes(t, a[:, c["obs/cmd/vx"]], a[:, c["obs/cmd/wz"]])
  J = {}
  for leg in ("right_leg", "left_leg"):
    for j in ("hip_pitch", "hip_roll", "knee", "hip_yaw"):
      Kp, Kd = GAIN[j]
      pos = np.radians(a[:, c[f"pos/{leg}/{j}"]])
      tgt = np.radians(a[:, c[f"tgt/{leg}/{j}"]])
      iq = a[:, c[f"extra/{leg}/{j}/iq"]]
      J[(leg, j)] = dict(pos=pos, tgt=tgt, iq=iq, v=vel(pos, t), fresh=np.r_[True, np.diff(iq) != 0], Kp=Kp, Kd=Kd)

  def aligned(d, s, L, tf):
    n = len(d["pos"])
    err_raw = d["tgt"][:n - s] - d["pos"][s:]  # tgt of row k against the position one row later
    err = prep(np.r_[err_raw, np.full(s, np.nan)], 0, L, tf)
    v = prep(np.r_[d["v"][s:], np.full(s, np.nan)], 0, L, tf)
    return err, v

  # ---- choose alignment on the control joints only
  grid = [(s, L, tf) for s in (0, 1) for L in range(0, 4) for tf in (0, 1, 2, 4)]
  best = []
  for g in grid:
    Y, X = [], []
    for leg in ("right_leg", "left_leg"):
      for j in ("hip_pitch", "knee"):
        d = J[(leg, j)]
        err, v = aligned(d, *g)
        tau = d["Kp"] * err - d["Kd"] * v
        m = d["fresh"] & np.isfinite(tau)
        Y.append(KT * d["iq"][m])
        X.append(tau[m])
    b, R, _ = ols(np.concatenate(Y), np.concatenate(X))
    best.append((R, g, b[0]))
  best.sort(key=lambda z: -z[0])
  g0 = best[0][1]
  top = [dict(R2=float(R), s=g[0], L=g[1], tf=g[2], k=float(k)) for R, g, k in best[:6]]

  # ---- sensitivity of the factor k (all rows) to the alignment choice: the 6 best alignments on the controls
  sens = []
  for R, g, _ in best[:6]:
    row = {"s": g[0], "L": g[1], "tf": g[2]}
    for (leg, j), d in J.items():
      err, v = aligned(d, *g)
      tau = d["Kp"] * err - d["Kd"] * v
      m = d["fresh"] & np.isfinite(tau)
      row[f"{leg[0].upper()}_{j}"] = float(ols(KT * d["iq"][m], tau[m][:, None])[0][0])
    sens.append(row)

  # ---- per joint / leg / segment
  tab = {}
  tblocks = (t // 1.0).astype(int)
  for (leg, j), d in J.items():
    Kp, Kd = d["Kp"], d["Kd"]
    err, v = aligned(d, *g0)
    tau = Kp * err - Kd * v
    tq = KT * d["iq"]
    ok = d["fresh"] & np.isfinite(tau)
    d["err_f"], d["tau_f"] = err, tau
    for sname, msk in reg.items():
      m = ok & msk
      if m.sum() < 15:
        continue
      y, x = tq[m], tau[m]
      b1, R1, _ = ols(y, x[:, None])  # factor k on the nominal PD torque
      b2, R2, _ = ols(y, np.c_[err[m], -v[m]])  # Kp_eff, Kd_eff
      b3, R3, _ = ols(y, err[m][:, None])  # Kp_eff, position error only
      # reverse regression (error on torque): 1/slope is an upper bound of Kp_eff
      br, _, _ = ols(err[m], y[:, None])
      # block bootstrap (1 s blocks) of the factor k
      idx = np.where(m)[0]
      blk = np.unique(tblocks[idx])
      ks = []
      for _ in range(300):
        pick = rng.choice(blk, len(blk))
        ii = np.concatenate([idx[tblocks[idx] == q] for q in pick])
        ks.append(ols(tq[ii], tau[ii][:, None])[0][0])
      key = f"{leg}/{j}/{sname}"
      tab[key] = dict(n=int(m.sum()), k=float(b1[0]), k_ci=[float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))], R2_k=float(R1),
                      icpt_Nm=float(b1[1]), Kp_eff=float(b2[0]), Kd_eff=float(b2[1]), R2_pd=float(R2), Kp_err_only=float(b3[0]), R2_err_only=float(R3),
                      Kp_rev_upper=float(1 / br[0]) if br[0] > 0 else None, Kp_set=Kp, Kd_set=Kd,
                      mean_tau_iq=float(y.mean()), mean_tau_pd=float(x.mean()), ratio_of_means=float(y.mean() / x.mean()) if abs(x.mean()) > 0.3 else None)
  return dict(KT=KT, chosen=dict(s=g0[0], L=g0[1], tf=g0[2]), top_alignments=top, alignment_sensitivity=sens, table=tab, J=J, reg=reg)


# ----------------------------------------------------------------------------------------------------------------------- B
def sole_series(fk, deg12, grav):
  """Per row: world-frame sole heights (lowest corner along up), sole tilt (deg). up = -grav (pelvis frame)."""
  n = len(deg12)
  o = {k: np.full(n, np.nan) for k in ("hR", "hL", "rollR", "rollL", "pitR", "pitL", "width")}
  up = -grav / np.linalg.norm(grav, axis=1, keepdims=True)
  rad = np.radians(deg12)
  for i in range(n):
    fk.pose(rad[i])
    d, m = fk.d, fk.m
    u = up[i]
    cen = {}
    for s in "RL":
      pts = []
      for g in fk.foot_geoms[s]:
        R = d.geom_xmat[g].reshape(3, 3)
        cg = d.geom_xpos[g]
        h = m.geom_size[g][:3]
        for sx in (-1, 1):
          for sy in (-1, 1):
            for sz in (-1, 1):
              pts.append(cg + R @ (h * np.array([sx, sy, sz])))
      pts = np.array(pts)
      o["h" + s][i] = (pts @ u).min()
      cen[s] = 0.5 * (pts.min(0) + pts.max(0))
      Rb = d.xmat[fk.foot_body[s]].reshape(3, 3)
      xf, yf = Rb[:, 0], Rb[:, 1]
      o["pit" + s][i] = np.degrees(np.arcsin(np.clip(xf @ u, -1, 1)))  # toe up positive
      ly = np.degrees(np.arcsin(np.clip(yf @ u, -1, 1)))  # left edge up positive
      o["roll" + s][i] = -ly if s == "R" else ly  # outer edge up positive
    o["width"][i] = cen["L"][1] - cen["R"][1]
  o["dh"] = o["hL"] - o["hR"]  # left sole higher: positive
  return o


def imu_vs_gyro(c, a):
  """Walking segments: pelvis roll from the gravity vector vs the integral of the roll gyro, both high-passed at 0.4 Hz. Ratio ~1 and
  |corr| ~1 = the IMU attitude follows the true pelvis roll in the gait band (sign of the corr only reflects the axis convention)."""
  t = a[:, 0]
  dt = float(np.median(np.diff(t)))
  g = a[:, [c["obs/grav/x"], c["obs/grav/y"], c["obs/grav/z"]]]
  roll = np.arctan2(g[:, 1], -g[:, 2])
  th = np.cumsum(a[:, c["obs/gyro/x"]]) * dt
  al = 1 - np.exp(-2 * np.pi * 0.4 * dt)

  def hp(x):
    return x - ema(x, al)
  out = {}
  for name, (t0, t1) in {"walk_8.0-11.7s": (8.0, 11.7), "walk_29.2-33.4s": (29.2, 33.4), "walk_50.6-54.7s": (50.6, 54.7)}.items():
    m = (t >= t0) & (t <= t1)
    r, q = hp(roll)[m], hp(th)[m]
    out[name] = dict(imu_roll_sd_deg=float(np.degrees(r.std())), gyro_int_sd_deg=float(np.degrees(q.std())), ratio=float(r.std() / q.std()), corr=float(np.corrcoef(r, q)[0, 1]))
  return out


def stat(x):
  x = x[np.isfinite(x)]
  return dict(n=int(len(x)), mean=float(x.mean()), sd=float(x.std()), med=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True)
  ap.add_argument("--real", required=True)
  ap.add_argument("--sim-dir", required=True)
  ap.add_argument("--name", default="nom_L2")
  ap.add_argument("--out-json", required=True)
  ap.add_argument("--out-png", default="")
  a_ = ap.parse_args()
  rng = np.random.default_rng(0)
  c, a = load(a_.real)
  t = a[:, 0]
  res = {"real": str(a_.real)}
  A = part_a(c, a, rng)
  res["A"] = {k: A[k] for k in ("KT", "chosen", "top_alignments", "alignment_sensitivity", "table")}

  # ---- A3: standing hip_roll torque, real vs simulator (same command trace)
  sim = np.load(Path(a_.sim_dir) / f"{a_.name}.npz")
  J, reg = A["J"], A["reg"]
  tq_sim = sim["tau"]  # (T,E,12) N.m, order R: hip_pitch hip_roll hip_yaw knee crankA crankB, then L
  fc = sim["foot_contact"]
  cmd = sim["command"]
  st_s = (cmd[:, :, 0] < 0.05) & (np.abs(cmd[:, :, 2]) < 0.05)
  names12 = [f"{s}/{j}" for s in ("right_leg", "left_leg") for j in JT]
  # real foot-down inference by FK (needed for double / single support split)
  fk = FK(a_.xml)
  pos12 = a[:, [c[f"pos/{n}"] for n in COLS]]
  grav = a[:, [c["obs/grav/x"], c["obs/grav/y"], c["obs/grav/z"]]]
  S = sole_series(fk, pos12, grav)
  vxs, wzs = a[:, c["obs/cmd/vx"]], a[:, c["obs/cmd/wz"]]
  stand = (vxs < 0.05) & (np.abs(wzs) < 0.05)
  # joint speed of the 12 motors (deg/s), 4 row central difference
  sp = np.vstack([vel(pos12[:, i], t) for i in range(12)]).T
  spd = np.nanmax(np.abs(sp[:, [0, 1, 2, 3, 6, 7, 8, 9]]), axis=1)  # hip + knee, both legs
  quiet = stand & (spd < 5.0) & (np.hypot(a[:, c["obs/gyro/x"]], a[:, c["obs/gyro/y"]]) < 0.15)
  dh = S["dh"]
  dbl = quiet & (np.abs(dh) < 0.008)  # both soles at the same height within 8 mm (same rule as stance_analyze)
  rsing = (~stand) & (dh > 0.008)  # R stance, L swing (walking)
  lsing = (~stand) & (dh < -0.008)


  def real_tau(leg, jn, mask):
    d = J[(leg, jn)]
    m = mask & d["fresh"]
    return stat(KT * d["iq"][m])

  def sim_tau(col, mask):
    x = tq_sim[:, :, col][mask]
    return stat(x)

  cmp = {}
  for jn, col_r, col_l in (("hip_roll", 1, 7), ("hip_pitch", 0, 6), ("knee", 3, 9)):
    for nm, mreal, msim_r, msim_l in (
        ("double_support_stand", dbl, st_s & fc[:, :, 0] & fc[:, :, 1], st_s & fc[:, :, 0] & fc[:, :, 1]),
        ("right_single_support_walk", rsing, (~st_s) & fc[:, :, 0] & ~fc[:, :, 1], (~st_s) & fc[:, :, 0] & ~fc[:, :, 1]),
        ("left_single_support_walk", lsing, (~st_s) & fc[:, :, 1] & ~fc[:, :, 0], (~st_s) & fc[:, :, 1] & ~fc[:, :, 0])):
      cmp[f"{jn}/{nm}"] = dict(real_R=real_tau("right_leg", jn, mreal), real_L=real_tau("left_leg", jn, mreal), sim_R=sim_tau(col_r, msim_r), sim_L=sim_tau(col_l, msim_l))
  res["A3_torque_real_vs_sim"] = cmp

  # ---- B: sole height / tilt, real vs simulator (stand, both feet down, low motion)
  segs = {"0-7.9s": (0.0, 7.9), "18.9-23.7s": (18.9, 23.7), "45.7-50.5s": (45.7, 50.5), "54.8-65s": (54.8, 65.0)}
  Bt = {}
  for sname, (t0, t1) in list(segs.items()) + [("pooled", (0, 1e9))]:
    w = quiet & (t >= t0) & (t <= t1)
    if w.sum() < 10:
      continue
    Bt[sname] = {k: stat(S[k][w]) for k in ("rollR", "rollL", "pitR", "pitL", "dh", "width")}
    Bt[sname]["n_quiet_rows"] = int(w.sum())
  res["B_real"] = Bt
  res["B_quiet_rows"] = int(quiet.sum())
  res["B_quiet_double_rows"] = int((quiet & (np.abs(dh) < 0.008)).sum())

  # simulator: stand rows with both feet in contact, 8 envs; FK on the CSV export (pos columns + obs/grav)
  simS = {k: [] for k in ("rollR", "rollL", "pitR", "pitL", "dh", "width")}
  per_env = []
  for e in range(sim["obs"].shape[1]):
    cs, asim = load(Path(a_.sim_dir) / "csv" / f"{a_.name}_env{e}.csv")
    ps = asim[:, [cs[f"pos/{n}"] for n in COLS]]
    gs = asim[:, [cs["obs/grav/x"], cs["obs/grav/y"], cs["obs/grav/z"]]]
    ts = asim[:, 0]
    spd_s = np.nanmax(np.abs(np.vstack([vel(ps[:, i], ts) for i in (0, 1, 2, 3, 6, 7, 8, 9)])), axis=0)
    ok = st_s[:, e] & fc[:len(ts), e, 0] & fc[:len(ts), e, 1] & (spd_s < 5.0) & (ts > 3.0)
    ii = np.where(ok)[0][::3]
    if len(ii) == 0:
      continue
    Ss = sole_series(fk, ps[ii], gs[ii])
    for k in simS:
      simS[k].append(Ss[k])
    per_env.append(dict(env=e, n=int(len(ii)), rollR=float(np.mean(Ss["rollR"])), rollL=float(np.mean(Ss["rollL"])), pitR=float(np.mean(Ss["pitR"])),
                        pitL=float(np.mean(Ss["pitL"])), dh_mm=float(1e3 * np.mean(Ss["dh"]))))
  res["B_sim"] = {k: stat(np.concatenate(v)) for k, v in simS.items()}
  res["B_sim_per_env"] = per_env
  res["_simS"] = {k: np.concatenate(v) for k, v in simS.items()}


  # ---- C: does the stance sole tilt grow with the hip_roll torque? (series compliance between motor encoder and joint would show here)
  comp = {}
  walk = vxs > 0.1
  for side, leg, col, st_real, other in (("R", "right_leg", 1, walk & (dh > 0.008), 1), ("L", "left_leg", 7, walk & (dh < -0.008), 0)):
    d = J[(leg, "hip_roll")]
    m = st_real & d["fresh"]
    x = KT * d["iq"][m]
    y = S["roll" + side][m]
    b, R2, _ = ols(y, x[:, None])
    comp[f"real_{side}"] = dict(n=int(m.sum()), deg_per_Nm=float(b[0]), icpt_deg=float(b[1]), R2=float(R2), torque_med=float(np.median(x)), roll_med=float(np.median(y)))
  res["C_compliance_real"] = comp
  res["C_imu_vs_gyro"] = imu_vs_gyro(c, a)
  csim = {"R": ([], []), "L": ([], [])}
  for e in range(sim["obs"].shape[1]):
    cs, asim = load(Path(a_.sim_dir) / "csv" / f"{a_.name}_env{e}.csv")
    ps = asim[:, [cs[f"pos/{n}"] for n in COLS]]
    gs = asim[:, [cs["obs/grav/x"], cs["obs/grav/y"], cs["obs/grav/z"]]]
    n = len(ps)
    for side, col, mine, other in (("R", 1, 0, 1), ("L", 7, 1, 0)):
      ok = (~st_s[:n, e]) & fc[:n, e, mine] & ~fc[:n, e, other]
      ii = np.where(ok)[0][::2]
      if len(ii) == 0:
        continue
      Ss = sole_series(fk, ps[ii], gs[ii])
      csim[side][0].append(tq_sim[ii, e, col])
      csim[side][1].append(Ss["roll" + side])
  csm = {}
  res["_csim"] = {}
  for side in "RL":
    x, y = np.concatenate(csim[side][0]), np.concatenate(csim[side][1])
    b, R2, _ = ols(y, x[:, None])
    csm[f"sim_{side}"] = dict(n=int(len(x)), deg_per_Nm=float(b[0]), icpt_deg=float(b[1]), R2=float(R2), torque_med=float(np.median(x)), roll_med=float(np.median(y)))
    res["_csim"][side] = (x, y)
  res["C_compliance_sim"] = csm
  res["_Creal"] = {side: (KT * J[(leg, "hip_roll")]["iq"][(walk & ((dh > 0.008) if side == "R" else (dh < -0.008))) & J[(leg, "hip_roll")]["fresh"]],
                          S["roll" + side][(walk & ((dh > 0.008) if side == "R" else (dh < -0.008))) & J[(leg, "hip_roll")]["fresh"]]) for side, leg in (("R", "right_leg"), ("L", "left_leg"))}

  # ---- B3: first row vs default pose
  q0 = a[0, [c[f"pos/{n}"] for n in COLS]]
  qrel = a[0, [c[f"obs/q/{n}"] for n in COLS]]
  default = q0 - np.degrees(qrel)
  d12 = np.degrees(sim["default_q12"])
  tgt0 = a[0, [c[f"tgt/{n}"] for n in COLS]]
  res["B3_first_row"] = {n: dict(pos=float(q0[i]), default=float(d12[i]), default_from_obs=float(default[i]), diff=float(q0[i] - d12[i]), tgt0=float(tgt0[i])) for i, n in enumerate(COLS)}
  # first 0.5 s mean (settling)
  res["B3_first_row_t"] = float(t[0])

  # ---- offsets -> width scenarios (stand, pooled quiet rows)
  w = quiet
  rR = float(np.median(S["rollR"][w]) - np.median(res["_simS"]["rollR"]))
  rL = float(np.median(S["rollL"][w]) - np.median(res["_simS"]["rollL"]))
  dhr = float(np.median(S["dh"][w]) - np.median(res["_simS"]["dh"]))
  pR = float(np.median(S["pitR"][w]) - np.median(res["_simS"]["pitR"]))
  pL = float(np.median(S["pitL"][w]) - np.median(res["_simS"]["pitL"]))
  segmed = {k: [float(np.median(S[k][quiet & (t >= t0) & (t <= t1)])) for (t0, t1) in segs.values() if (quiet & (t >= t0) & (t <= t1)).sum() > 10] for k in ("rollR", "rollL", "pitR", "pitL")}
  res["B_residual"] = dict(rollR_deg=rR, rollL_deg=rL, dh_mm=1e3 * dhr, common_mode_deg=(rR + rL) / 2, antisym_deg=(rR - rL) / 2,
                           pitchR_deg=pR, pitchL_deg=pL, width_lever_antisym_mm=float(1e3 * np.median(S["width"][w]) * np.tan(np.radians((rR - rL) / 2))),
                           per_segment_median=segmed)
  # hip_roll abduction offset that would flatten each sole: true = logged - rho (abduction = +roll right, -roll left)
  base_w = float(np.median(S["width"][w]))
  idx = np.where(w)[0][::5]
  scen = {}
  for name, (dR, dL) in {"all_in_hip_roll": (rR, rL), "symmetric_part_in_hip_roll": ((rR + rL) / 2, (rR + rL) / 2)}.items():
    p2 = pos12[idx].copy()
    ir, il = COLS.index("right_leg/hip_roll"), COLS.index("left_leg/hip_roll")
    p2[:, ir] -= dR
    p2[:, il] += dL  # left abduction is negative roll, so removing +dL abduction means adding +dL to the roll value
    S2 = sole_series(fk, p2, grav[idx])
    scen[name] = dict(offR_deg=dR, offL_deg=dL, width_cm=float(100 * np.median(S2["width"])), width_logged_cm=100 * base_w,
                      rollR_after=float(np.median(S2["rollR"])), rollL_after=float(np.median(S2["rollL"])), dh_after_mm=float(1e3 * np.median(S2["dh"])))
  res["B_width_scenarios"] = scen

  # ---- figure
  if a_.out_png:
    figure(a_.out_png, A, S, quiet, res, t)

  res.pop("_simS")
  res.pop("_csim")
  res.pop("_Creal")
  Path(a_.out_json).write_text(json.dumps(res, indent=1, default=float))
  print(json.dumps(dict(chosen=res["A"]["chosen"], top=res["A"]["top_alignments"][:3]), indent=1))


def figure(path, A, S, quiet, res, t):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  J, tab = A["J"], A["table"]
  fig = plt.figure(figsize=(16, 17))
  gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1.05, 1.05], hspace=0.45, wspace=0.32)
  panels = [("right_leg", "hip_roll"), ("left_leg", "hip_roll"), ("right_leg", "hip_pitch"), ("left_leg", "hip_pitch"), ("right_leg", "knee"), ("left_leg", "knee")]
  cols = {"stand": "#2a9d8f", "walk": "#e76f51", "turn": "#8d99ae"}
  for p, (leg, j) in enumerate(panels):
    ax = fig.add_subplot(gs[p // 3, p % 3])
    d = J[(leg, j)]
    m = d["fresh"] & np.isfinite(d["tau_f"])
    tq = KT * d["iq"]
    reg = A["reg"]
    for sname in ("stand", "walk", "turn"):
      mm = m & reg[sname]
      ax.plot(d["tau_f"][mm], tq[mm], ".", ms=3.5, alpha=0.55, color=cols[sname], label=sname)
    lim = np.array([min(d["tau_f"][m].min(), tq[m].min()), max(d["tau_f"][m].max(), tq[m].max())])
    ax.plot(lim, lim, "k--", lw=1, label="slope 1 (as set)")
    r = tab[f"{leg}/{j}/all"]
    ax.plot(lim, r["k"] * lim + r["icpt_Nm"], color="#c1121f", lw=1.6, label=f"fit slope {r['k']:.2f}")
    ax.set_title(f"{leg.split('_')[0]} {j}: Kp_eff {r['Kp_eff']:.0f} (set {r['Kp_set']:.0f}), R2 {r['R2_k']:.2f}", fontsize=10)
    ax.set_xlabel("PD torque Kp*(tgt-pos) - Kd*vel [N·m]")
    ax.set_ylabel("Kt * iq [N·m]")
    ax.grid(alpha=0.3)
    if p == 0:
      ax.legend(fontsize=8, loc="upper left")
  # bar of k with CI
  ax = fig.add_subplot(gs[0:2, 3])
  labs, ks, lo, hi = [], [], [], []
  for j in ("hip_roll", "hip_pitch", "knee", "hip_yaw"):
    for leg in ("right_leg", "left_leg"):
      r = tab[f"{leg}/{j}/all"]
      labs.append(f"{leg[0].upper()} {j}")
      ks.append(r["k"])
      lo.append(r["k"] - r["k_ci"][0])
      hi.append(r["k_ci"][1] - r["k"])
  y = np.arange(len(labs))[::-1]
  ax.barh(y, ks, xerr=[lo, hi], color=["#c1121f" if "roll" in l else "#577590" for l in labs], alpha=0.85)
  ax.axvline(1.0, color="k", ls="--", lw=1)
  ax.axvline(0.7, color="#c1121f", ls=":", lw=1)
  ax.text(0.7, y.max() + 0.7, "0.7", color="#c1121f", ha="center", fontsize=9)
  ax.set_yticks(y)
  ax.set_yticklabels(labs, fontsize=9)
  ax.set_xlim(0, 1.5)
  ax.set_xlabel("effective gain / set gain (all rows, 95% CI)")
  ax.set_title("Effective stiffness factor", fontsize=10)
  # sole roll / height residual
  simS = res["_simS"]
  w = quiet
  ax = fig.add_subplot(gs[2, 0])
  bins = np.linspace(-6, 6, 49)
  ax.hist(S["rollR"][w], bins, alpha=0.6, color="#e76f51", label="real R")
  ax.hist(S["rollL"][w], bins, alpha=0.6, color="#264653", label="real L")
  ax.hist(simS["rollR"], bins, histtype="step", color="#e76f51", lw=1.8, label="sim R", density=False, weights=np.full(len(simS["rollR"]), w.sum() / len(simS["rollR"])))
  ax.hist(simS["rollL"], bins, histtype="step", color="#264653", lw=1.8, label="sim L", weights=np.full(len(simS["rollL"]), w.sum() / len(simS["rollL"])))
  ax.set_xlabel("sole roll in world frame, outer edge up [deg]")
  ax.set_ylabel("rows (sim scaled to real count)")
  ax.set_title("Standing sole roll: real vs simulator", fontsize=10)
  ax.legend(fontsize=8)
  ax = fig.add_subplot(gs[2, 1])
  b2 = np.linspace(-25, 25, 51)
  ax.hist(1e3 * S["dh"][w], b2, alpha=0.6, color="#2a9d8f", label="real")
  ax.hist(1e3 * simS["dh"], b2, histtype="step", color="k", lw=1.8, label="sim", weights=np.full(len(simS["dh"]), w.sum() / len(simS["dh"])))
  ax.set_xlabel("left sole height - right sole height [mm]")
  ax.set_ylabel("rows (sim scaled)")
  ax.set_title("Standing sole height difference", fontsize=10)
  ax.legend(fontsize=8)
  ax = fig.add_subplot(gs[2, 2])
  B = res["B_real"]
  names = [k for k in B if k != "pooled"] + ["pooled"]
  x = np.arange(len(names))
  ax.bar(x - 0.2, [B[k]["rollR"]["med"] for k in names], 0.2, color="#e76f51", label="real R roll")
  ax.bar(x, [B[k]["rollL"]["med"] for k in names], 0.2, color="#264653", label="real L roll")
  ax.bar(x + 0.2, [B[k]["pitR"]["med"] for k in names], 0.2, color="#e9c46a", label="real R pitch")
  ax.axhline(res["B_sim"]["rollR"]["med"], color="#e76f51", ls=":", lw=1)
  ax.axhline(res["B_sim"]["rollL"]["med"], color="#264653", ls=":", lw=1)
  ax.set_xticks(x)
  ax.set_xticklabels(names, fontsize=8, rotation=20)
  ax.set_ylabel("median sole tilt [deg] (dotted: sim)")
  ax.set_title("Per standing segment", fontsize=10)
  ax.legend(fontsize=7)
  ax = fig.add_subplot(gs[2, 3])
  cm = res["A3_torque_real_vs_sim"]
  lab = ["R roll\ndouble", "L roll\ndouble", "R roll\nR-single"]
  vals = [(cm["hip_roll/double_support_stand"]["real_R"]["med"], cm["hip_roll/double_support_stand"]["sim_R"]["med"]),
          (cm["hip_roll/double_support_stand"]["real_L"]["med"], cm["hip_roll/double_support_stand"]["sim_L"]["med"]),
          (cm["hip_roll/right_single_support_walk"]["real_R"]["med"], cm["hip_roll/right_single_support_walk"]["sim_R"]["med"])]
  x = np.arange(3)
  ax.bar(x - 0.18, [v[0] for v in vals], 0.36, color="#c1121f", label="real Kt*iq")
  ax.bar(x + 0.18, [v[1] for v in vals], 0.36, color="#577590", label="sim applied")
  ax.set_xticks(x)
  ax.set_xticklabels(lab, fontsize=8)
  ax.set_ylabel("hip_roll torque, median [N·m]")
  ax.set_title("hip_roll torque at the same stance", fontsize=10)
  ax.legend(fontsize=8)
  ax.axhline(0, color="k", lw=0.6)
  # row 4: stance sole roll vs hip_roll torque, real vs sim ; first row vs default
  ax = fig.add_subplot(gs[3, 0:2])
  for side, col in (("R", "#e76f51"), ("L", "#264653")):
    xs, ys = res["_csim"][side]
    ax.plot(xs, ys, ".", ms=2, alpha=0.15, color=col)
    xr, yr = res["_Creal"][side]
    ax.plot(xr, yr, "o", ms=4, alpha=0.7, color=col, label=f"real {side} stance")
    cr, cs_ = res["C_compliance_real"][f"real_{side}"], res["C_compliance_sim"][f"sim_{side}"]
    xx = np.array([min(xr.min(), xs.min()), max(xr.max(), xs.max())])
    ax.plot(xx, cr["deg_per_Nm"] * xx + cr["icpt_deg"], "-", color=col, lw=1.8)
    ax.plot(xx, cs_["deg_per_Nm"] * xx + cs_["icpt_deg"], "--", color=col, lw=1.4)
  ax.set_xlabel("stance leg hip_roll torque [N·m]  (real: Kt*iq, sim: applied)")
  ax.set_ylabel("stance sole roll from logged angles [deg]")
  ax.set_title("Stance sole roll vs hip_roll torque (dots small = sim, solid fit = real, dashed fit = sim)", fontsize=10)
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  ax = fig.add_subplot(gs[3, 2:4])
  fr = res["B3_first_row"]
  nm = list(fr.keys())
  x = np.arange(len(nm))
  ax.bar(x, [fr[k]["diff"] for k in nm], color=["#e76f51" if k.startswith("right") else "#264653" for k in nm])
  ax.set_xticks(x)
  ax.set_xticklabels([k.replace("right_leg/", "R ").replace("left_leg/", "L ") for k in nm], rotation=60, fontsize=8)
  ax.set_ylabel("first logged row - default pose [deg]")
  ax.set_title("Joint angles at the first logged row vs default pose", fontsize=10)
  ax.axhline(0, color="k", lw=0.6)
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
  main()
