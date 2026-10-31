#!/usr/bin/env python3
"""Summarise foot_variant_eval.py runs: per command segment and per foot variant.

Stance = foot normal force > 0.1 body weight (BW); loaded stance (for CoP) = > 0.3 BW. First 3 s of every 15 s segment dropped;
an environment is dropped from the fall onward. CoP x is measured from the ankle origin (toe +), CoP y is turned into
"medial +" (right foot +y, left foot -y). Writes <dir>/summary.json and three figures (English labels).
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/foot_variant_analyze.py <eval_dir> [variants...]
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(sys.argv[1])
VARS = sys.argv[2:] or ["base", "v3full", "v3lat", "v3same", "v3sph"]
DT, SKIP = 0.02, float(__import__("os").environ.get("SKIP_S", "3.0"))
COL = {"base": "tab:gray", "v3full": "tab:red", "v3same": "tab:orange", "v3sph": "tab:blue", "v3lat": "tab:purple", "v5full": "tab:green"}
CRANK = [4, 5, 10, 11]
SOLE_Z = {"base": -0.043, "v3full": -0.057, "v3same": -0.043, "v3sph": -0.057, "v3lat": -0.057, "v5full": -0.057}  # sole bottom in foot_link [m]


def quat_to_mat(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def pitch_of(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))


def load(v):
  D = dict(np.load(DIR / f"{v}.npz"))
  meta = json.loads((DIR / f"{v}.json").read_text())
  T, E = D["terminated"].shape
  alive = np.ones((T, E), bool)
  for e in range(E):
    if D["terminated"][:, e].any():
      alive[np.argmax(D["terminated"][:, e]):, e] = False
  D["alive"] = alive
  return D, meta


def q(a, p):
  a = a[np.isfinite(a)]
  return float(np.percentile(a, p)) if a.size else float("nan")


def segment_stats(D, meta, sl):
  BW = meta["robot_mass_kg"] * 9.81
  al = D["alive"][sl]
  S = D["sole"][sl]  # t,e,foot,10
  fn = S[..., 0]
  st = (fn > 0.1 * BW) & al[..., None]
  ld = (fn > 0.3 * BW) & al[..., None]
  copx = S[..., 2] * 1000
  copy = S[..., 3] * 1000 * np.array([1.0, -1.0])  # medial +
  dfn = np.diff(fn, axis=0) / DT
  rise = dfn[(st[1:]) & (dfn > 0)] / BW
  # slip = horizontal speed of the foot MATERIAL point under the CoP (the ankle origin also moves when the foot rolls)
  zb = SOLE_Z[meta["variant"]]
  r = np.concatenate([np.nan_to_num(S[..., 2:4]), np.full(S.shape[:-1] + (1,), zb)], -1)
  Rm = quat_to_mat(D["foot_quat_w"][sl])
  pw = D["foot_pos_w"][sl] + np.einsum("tfij,tfj->tfi", Rm.reshape(-1, 2, 3, 3), r.reshape(-1, 2, 3)).reshape(r.shape)
  pw_next = D["foot_pos_w"][sl][1:] + np.einsum("tfij,tfj->tfi", Rm[1:].reshape(-1, 2, 3, 3), r[:-1].reshape(-1, 2, 3)).reshape(r[:-1].shape)
  vxy = np.linalg.norm((pw_next - pw[:-1])[..., :2], axis=-1) / DT * 1000
  loaded = (fn > 0.5 * BW) & al[..., None]
  slip = vxy[loaded[:-1] & loaded[1:]]
  cmd = D["command"][sl]
  v = D["base_lin_vel_b"][sl]
  w = D["base_ang_vel_b"][sl]
  tau = D["tau"][sl]
  qd = D["qd16"][sl][..., :12]
  power = np.sum(np.abs(tau * qd), -1)[al]
  # touchdown foot pitch: stance onset
  td = (~st[:-1]) & st[1:]
  fp = pitch_of(D["foot_quat_w"][sl])[1:]
  return dict(
    n_alive_frac=float(al.mean()),
    vx_err=float(np.nanmean(np.abs(v[..., 0] - cmd[..., 0])[al])), wz_err=float(np.nanmean(np.abs(w[..., 2] - cmd[..., 2])[al])),
    vx_mean=float(np.nanmean(v[..., 0][al])), wz_mean=float(np.nanmean(w[..., 2][al])),
    grf_p99_bw=q(fn[st] / BW, 99), grf_max_bw=q(fn[st] / BW, 100), load_rate_p99_bw_s=q(rise, 99),
    copx_p1=q(copx[ld], 1), copx_p5=q(copx[ld], 5), copx_p50=q(copx[ld], 50), copx_p95=q(copx[ld], 95), copx_p99=q(copx[ld], 99),
    copy_med_p99=q(copy[ld], 99), copy_lat_p1=q(copy[ld], 1),
    n_contacts=float(np.nanmean(S[..., 4][st])), patch_len_mm=float(np.nanmean((S[..., 6] - S[..., 5])[st] * 1000)),
    slip_p95_mm_s=q(slip, 95), pen_p99_mm=q(S[..., 9][st] * 1000, 99),
    crank_tau_p99=q(np.abs(tau[..., CRANK])[al], 99), crank_tau_max=q(np.abs(tau[..., CRANK])[al], 100),
    power_mean_w=float(np.mean(power)) if power.size else float("nan"),
    td_pitch_med_deg=q(fp[td], 50), td_pitch_p90_deg=q(fp[td], 90),
    base_z_mm=float(np.nanmean(D["base_pos"][sl][..., 2][al]) * 1000),
    stance_frac=float(st.mean()),
  )


summary, raw = {}, {}
for v in VARS:
  if not (DIR / f"{v}.npz").exists():
    continue
  D, meta = load(v)
  n = int(round(meta["seconds_per_cmd"] / DT))
  segs = {}
  for k, (name, vx, wz) in enumerate(meta["program"]):
    sl = slice(k * n + int(SKIP / DT), (k + 1) * n)
    segs[name] = segment_stats(D, meta, sl)
  walk = slice(n + int(SKIP / DT), len(meta["program"]) * n)
  segs["ALL_MOVING"] = segment_stats(D, meta, walk)
  summary[v] = dict(meta=dict(falls=meta["falls"], envs=meta["num_envs"], mass=meta["robot_mass_kg"], foot_geoms=meta.get("foot_geoms")),
                    segments=segs)
  BW = meta["robot_mass_kg"] * 9.81
  S = D["sole"][walk]
  ld = (S[..., 0] > 0.3 * BW) & D["alive"][walk][..., None]
  raw[v] = dict(copx=S[..., 2][ld] * 1000, copy=(S[..., 3] * 1000 * np.array([1.0, -1.0]))[ld])
(DIR / "summary.json").write_text(json.dumps(summary, indent=1))

# ---- figure 1: CoP x distribution while walking (loaded stance) + sole geometry marks
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
bins = np.arange(-90, 191, 2)
for v in raw:
  ax[0].hist(raw[v]["copx"], bins=bins, histtype="step", lw=1.6, density=True, color=COL[v], label=f"{v} (P1 {np.percentile(raw[v]['copx'],1):.0f} / P99 {np.percentile(raw[v]['copx'],99):.0f} mm)")
for x, lab in ((-80, "heel tip"), (-58, "v3 heel flat starts"), (0, "ankle"), (135, "v3 toe flat ends"), (180, "toe tip")):
  ax[0].axvline(x, color="k", lw=0.6, ls=":")
  ax[0].text(x, ax[0].get_ylim()[1] * 0.95, lab, rotation=90, fontsize=7, va="top", ha="right")
ax[0].set_xlabel("CoP x from ankle [mm] (toe +)")
ax[0].set_ylabel("density")
ax[0].set_title("CoP along the foot, all moving commands, stance > 0.3 BW")
ax[0].legend(fontsize=7)
for v in raw:
  ax[1].hist(raw[v]["copy"], bins=np.arange(-60, 61, 2), histtype="step", lw=1.6, density=True, color=COL[v], label=v)
ax[1].set_xlabel("CoP y [mm] (medial +)")
ax[1].set_title("CoP across the foot")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(DIR / "fig_cop_hist.png", dpi=110)

# ---- figure 2: per-segment metrics
names = [n for n in next(iter(summary.values()))["segments"] if n != "ALL_MOVING"]
mets = [("grf_p99_bw", "GRF P99 [BW]"), ("load_rate_p99_bw_s", "loading rate P99 [BW/s]"), ("crank_tau_p99", "crank motor |tau| P99 [N m]"),
        ("copx_p1", "rear-most CoP P1 [mm from ankle]"), ("vx_err", "|vx error| mean [m/s]"), ("slip_p95_mm_s", "stance foot slip P95 [mm/s]"),
        ("power_mean_w", "mechanical power mean [W]"), ("td_pitch_med_deg", "foot pitch at touchdown, median [deg]"), ("n_contacts", "contact points per stance foot")]
fig, axs = plt.subplots(3, 3, figsize=(17, 11))
xw = 0.8 / max(1, len(summary))
for a, (mk, lab) in zip(axs.ravel(), mets):
  for j, v in enumerate(summary):
    vals = [summary[v]["segments"][n][mk] for n in names]
    a.bar(np.arange(len(names)) + j * xw, vals, xw, color=COL[v], label=v)
  a.set_xticks(np.arange(len(names)) + 0.4 - xw / 2)
  a.set_xticklabels(names, rotation=40, fontsize=7)
  a.set_title(lab, fontsize=9)
  a.grid(axis="y", alpha=0.3)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Union-DR 49999 policy (trained on the base foot) with swapped foot collision - per command, 8 envs, 12 s each", fontsize=11)
fig.tight_layout()
fig.savefig(DIR / "fig_segments.png", dpi=100)

# ---- figure 3: CoP clouds on the foot
fig, axs = plt.subplots(1, len(raw), figsize=(4.6 * len(raw), 4.2), sharey=True)
for a, v in zip(np.atleast_1d(axs), raw):
  a.hist2d(raw[v]["copx"], raw[v]["copy"], bins=[np.arange(-90, 191, 4), np.arange(-60, 61, 4)], cmap="magma", cmin=1)
  a.add_patch(plt.Rectangle((-80, -57.5), 260, 115, fill=False, ec="c", lw=1))
  a.axvline(0, color="w", lw=0.6, ls=":")
  a.set_title(f"{v}: CoP cloud (moving)", fontsize=9)
  a.set_xlabel("x from ankle [mm]")
np.atleast_1d(axs)[0].set_ylabel("y [mm] (medial +)")
fig.tight_layout()
fig.savefig(DIR / "fig_cop_cloud.png", dpi=105)
print(json.dumps({v: {k: round(val, 3) if isinstance(val, float) else val for k, val in summary[v]["segments"]["ALL_MOVING"].items()} for v in summary}, indent=1))
