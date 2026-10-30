#!/usr/bin/env python3
"""Foot slide, standing width and standing hip_roll torque of sim_stance_record.py runs (contact-model variants).
docs/sim2real/2026-10-30_union49999_sim_contact_creep.md.

  python3 contact_creep_analyze.py --sim-dir <out> --prefix cr --variants base,impratio10,... --real-json <stance_width.json> \
      --out-json <json> --out-png <png>

Per run (npz from sim_stance_record.py with foot_pos_w / foot_quat_w / foot_contact / base_pos / base_quat / tau / command):
  foot centre = foot link origin + R * (0.05, 0, -0.014) m (mean of the three sole boxes), world frame.
  pelvis frame = base_link pose; width = |y_R - y_L| of the two foot centres in the pelvis frame.
  standing segment = run of >= 3 s with |vx|,|wz| < 0.05 (sim time). The first SETTLE seconds are dropped.
  net = foot displacement over all frames; slope = least-squares speed of that displacement.
  "kept" = displacement summed over frame pairs in which both feet are in contact at both frames (own contact kept);
  the rest = frame pairs with a contact flag off at one end. lift_* tell whether those are real steps (long episodes, large lift)
  or one-frame contact chatter (short episodes, ~1 mm lift).
"""
import argparse
import json
from pathlib import Path

import numpy as np

SETTLE = 1.0
DT = 0.02
FOOT_C = np.array([0.05, 0.0, -0.014])  # mean centre of the three sole boxes (foot link frame), used for the width like stance_fk.py
SOLE_C = np.array([0.05, 0.0, -0.043])  # same x, y at the sole surface (box centre z -0.014 minus half height 0.029), used for slide


def rot(q):  # q (..., 4) wxyz -> (..., 3, 3)
  w, x, y, z = [q[..., i] for i in range(4)]
  R = np.empty(q.shape[:-1] + (3, 3), q.dtype)
  R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
  R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
  R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
  return R


def runs(mask, min_len):
  out, s = [], None
  for i, m in enumerate(mask):
    if m and s is None:
      s = i
    if (not m) and s is not None:
      if i - s >= min_len:
        out.append((s, i))
      s = None
  if s is not None and len(mask) - s >= min_len:
    out.append((s, len(mask)))
  return out


def lift_stats(c, z):
  """Contact-loss episodes of one foot (c: (n,E) bool, z: (n,E) foot-centre height): count per 10 s, mean length [s], max lift above its own median height [mm]."""
  n, E = c.shape
  cnt, ln, lift = np.zeros(E), np.full(E, np.nan), np.zeros(E)
  for e in range(E):
    r = runs(~c[:, e], 1)
    cnt[e] = len(r) * 10.0 / (n * DT)
    if r:
      ln[e] = np.mean([(j - i) * DT for i, j in r])
    lift[e] = (z[:, e].max() - np.median(z[:, e])) * 1e3
  return dict(lift_episodes_per_10s=cnt, lift_mean_len_s=ln, lift_max_mm=lift)


def analyse(npz):
  d = np.load(npz)
  T, E = d["foot_contact"].shape[:2]
  Rb = rot(d["base_quat"])  # (T,E,3,3)
  Rf = rot(d["foot_quat_w"])  # (T,E,2,3,3)
  fc = d["foot_pos_w"] + np.einsum("teijk,k->teij", Rf, FOOT_C)  # world foot centre (T,E,2,3)
  rel = np.einsum("teki,tejk->teji", Rb, fc - d["base_pos"][:, :, None, :])  # pelvis frame: R^T (p - b)
  sole = d["foot_pos_w"] + np.einsum("teijk,k->teij", Rf, SOLE_C)  # world sole point (T,E,2,3)
  yR, yL = rel[..., 0, 1], rel[..., 1, 1]
  xR, xL = rel[..., 0, 0], rel[..., 1, 0]
  width = np.abs(yR - yL)  # (T,E)
  cmd = d["command"][:, 0]
  zero = (np.abs(cmd[:, 0]) < 0.05) & (np.abs(cmd[:, 2]) < 0.05)
  walk = cmd[:, 0] > 0.1
  contact = d["foot_contact"]
  both = contact.all(-1)  # (T,E)
  tau = d["tau"]
  res = dict(T=int(T), E=int(E), falls=int(d["terminated"].any(0).sum()), seg={}, width_t=None)
  segs = runs(zero, int(3 / DT))
  # per-segment slide statistics
  for si, (a, b) in enumerate(segs):
    a2 = a + int(SETTLE / DT)
    # lateral and forward axes of the pelvis at a2 (world yaw projected)
    ex = Rb[a2, :, :, 0].copy(); ex[:, 2] = 0; ex /= np.linalg.norm(ex, axis=-1, keepdims=True)
    ey = np.stack([-ex[:, 1], ex[:, 0], np.zeros(E)], -1)  # left (+y)
    out = {}
    for fi, fn in enumerate(("R", "L")):
      sgn = -1.0 if fn == "R" else +1.0  # outward = -y for the right foot, +y for the left foot
      p = sole[a2:b, :, fi, :]  # (n,E,3)
      lat = sgn * np.einsum("nej,ej->ne", p - p[0], ey)
      fwd = np.einsum("nej,ej->ne", p - p[0], ex)
      dlat = np.diff(lat, axis=0); dfwd = np.diff(fwd, axis=0)
      cf_own = contact[a2:b, :, fi]
      kept = cf_own[:-1] & cf_own[1:]  # this foot in contact at both frames of the pair
      cl, cf = np.cumsum(np.where(kept, dlat, 0.0), 0), np.cumsum(np.where(kept, dfwd, 0.0), 0)
      n = len(cl)
      tt = np.arange(n) * DT
      kept_n = kept.sum(0)
      lost_n = (~kept).sum(0)
      lost_out = np.where(~kept, dlat, 0.0).sum(0)
      # speed = total sliding displacement / time in contact (frames where contact kept)
      spd_lat = np.where(kept_n > 0, cl[-1] / np.maximum(kept_n * DT, 1e-9), np.nan)
      spd_fwd = np.where(kept_n > 0, cf[-1] / np.maximum(kept_n * DT, 1e-9), np.nan)
      tn = np.arange(len(lat))[:, None] * DT
      sl = lambda y: ((tn - tn.mean()) * (y - y.mean(0))).sum(0) / ((tn - tn.mean()) ** 2).sum()  # noqa: E731  least-squares slope per env
      out[fn] = dict(kept_out_mm_s=spd_lat * 1e3, lost_out_mm_s=np.where(lost_n > 0, lost_out / np.maximum(lost_n * DT, 1e-9), np.nan) * 1e3,
                     lost_pair_frac=lost_n / len(kept), net_out_mm_s=sl(lat) * 1e3, net_fwd_mm_s=sl(fwd) * 1e3, net_out_mm=lat[-1] * 1e3, net_fwd_mm=fwd[-1] * 1e3, slide_out_mm=cl[-1] * 1e3, slide_fwd_mm=cf[-1] * 1e3,
                     step_out_mm=(lat[-1] - cl[-1]) * 1e3, step_fwd_mm=(fwd[-1] - cf[-1]) * 1e3,
                     slide_out_mm_s=spd_lat * 1e3, slide_fwd_mm_s=spd_fwd * 1e3,
                     contact_lost_frac=1.0 - contact[a2:b, :, fi].mean(0), **lift_stats(contact[a2:b, :, fi], sole[a2:b, :, fi, 2]))
    t0, t1 = a * DT, b * DT
    ms = lambda x: [float(np.nanmean(x)), float(np.nanstd(x))]  # noqa: E731
    w_a2, w_end = width[a2], width[b - 1]
    last2 = slice(max(a2, b - int(2 / DT)), b)
    ok = both[a2:b]
    trq = [np.where(ok, tau[a2:b, :, 1], np.nan), np.where(ok, tau[a2:b, :, 7], np.nan)]
    ok2 = both[last2]
    trq2 = [np.where(ok2, tau[last2, :, 1], np.nan), np.where(ok2, tau[last2, :, 7], np.nan)]
    res["seg"][f"{t0:.1f}-{t1:.1f}"] = dict(
      t0=t0, t1=t1, dur=t1 - t0, width_start_cm=ms(w_a2 * 100), width_end_cm=ms(w_end * 100), width_last2s_cm=ms(width[last2] * 100),
      growth_cm=ms((w_end - w_a2) * 100), both_contact_frac=float(ok.mean()),
      tau_roll_R=float(np.nanmedian(trq[0])), tau_roll_L=float(np.nanmedian(trq[1])),
      tau_roll_R_last2s=float(np.nanmedian(trq2[0])), tau_roll_L_last2s=float(np.nanmedian(trq2[1])),
      foot={fn: {k: ms(v) for k, v in out[fn].items()} for fn in out})
    res["seg"][f"{t0:.1f}-{t1:.1f}"]["_curve"] = dict(
      t=(np.arange(b - a2) * DT + a2 * DT).tolist(),
      lat_R=(-1.0 * np.einsum("nej,ej->ne", sole[a2:b, :, 0, :] - sole[a2, :, 0, :], ey)).mean(1).tolist(),
      lat_L=(+1.0 * np.einsum("nej,ej->ne", sole[a2:b, :, 1, :] - sole[a2, :, 1, :], ey)).mean(1).tolist())
  # walking
  wm = walk & ~d["terminated"].any(1)
  res["walk_width_cm"] = float(np.mean(width[wm]) * 100) if wm.any() else None
  sg = np.sign(xR - xL)
  cross = (sg[1:] * sg[:-1] < 0) & walk[1:, None]
  res["pass_width_cm"] = float(np.mean(((yR - yL)[1:][cross]) * -1) * 100) if cross.any() else None
  res["pass_n"] = int(cross.sum())
  res["width_all_cm"] = float(np.mean(width) * 100)
  res["width_t"] = dict(t=(np.arange(T) * DT).tolist(), mean=(width.mean(1) * 100).tolist(), std=(width.std(1) * 100).tolist())
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sim-dir", required=True)
  ap.add_argument("--prefix", required=True)
  ap.add_argument("--variants", required=True)
  ap.add_argument("--out-json", required=True)
  ap.add_argument("--out-png", default="")
  a = ap.parse_args()
  R = {}
  for v in a.variants.split(","):
    f = Path(a.sim_dir) / f"{a.prefix}_{v}.npz"
    if not f.exists():
      R[v] = None
      continue
    R[v] = analyse(f)
    R[v]["contact"] = json.load(open(Path(a.sim_dir) / f"{a.prefix}_{v}.json")).get("contact")
  Path(a.out_json).write_text(json.dumps(R))
  if a.out_png:
    figure(R, a.out_png)


def figure(R, png):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  vs = [v for v in R if R[v]]
  col = dict(zip(vs, ["#444444", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]))
  fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
  for v in vs:
    wt = R[v]["width_t"]
    t, m = np.array(wt["t"]), np.array(wt["mean"])
    ax[0].plot(t, m, color=col[v], lw=1.1, label=v)
  segs = list(R[vs[0]]["seg"].values())
  for s in segs:
    ax[0].axvspan(s["t0"], s["t1"], color="0.9", zorder=0)
  ax[0].axhline(22.5, color="k", ls="--", lw=1, label="real standing (22.5)")
  ax[0].set_xlabel("sim time [s] (shaded = zero command)"); ax[0].set_ylabel("stance width [cm]"); ax[0].set_title("Stance width, mean of 4 envs")
  ax[0].legend(fontsize=7)
  last = list(R[vs[0]]["seg"].keys())[-1]
  for v in vs:
    c = R[v]["seg"][last]["_curve"]
    t = np.array(c["t"]); t = t - t[0]
    ax[1].plot(t, np.array(c["lat_R"]) * 1e3, color=col[v], lw=1.1, label=v)
    ax[1].plot(t, np.array(c["lat_L"]) * 1e3, color=col[v], lw=1.1, ls=":")
  ax[1].set_xlabel(f"time since settle in standing segment {last} s [s]"); ax[1].set_ylabel("foot outward position [mm]")
  ax[1].set_title("Foot lateral position (solid R, dotted L)")
  x = np.arange(len(vs)); wd = 0.38
  ax[2].bar(x - wd / 2, [R[v]["seg"][last]["tau_roll_R_last2s"] for v in vs], wd, label="R hip_roll", color="#d62728")
  ax[2].bar(x + wd / 2, [R[v]["seg"][last]["tau_roll_L_last2s"] for v in vs], wd, label="L hip_roll", color="#1f77b4")
  ax[2].axhline(10.9, color="#d62728", ls="--", lw=1); ax[2].axhline(-10.6, color="#1f77b4", ls="--", lw=1)
  ax[2].text(len(vs) - 0.5, 11.3, "real +10.9", fontsize=7, ha="right"); ax[2].text(len(vs) - 0.5, -12.2, "real -10.6", fontsize=7, ha="right")
  ax[2].set_xticks(x); ax[2].set_xticklabels(vs, rotation=20, fontsize=7); ax[2].set_ylabel("median torque [N m]")
  ax[2].set_title("Standing hip_roll torque, last 2 s"); ax[2].legend(fontsize=7)
  fig.tight_layout(); fig.savefig(png, dpi=130)


if __name__ == "__main__":
  main()
