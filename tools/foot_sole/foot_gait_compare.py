#!/usr/bin/env python3
"""How does the walking style change with the foot? Two outputs from foot_variant_eval.py recordings (no re-simulation):

1. fig_gait_phase.png - every right/left stance (normal force > 0.1 BW, >= 0.12 s) normalised to 0-100 %: foot pitch (toe up +),
   CoP x in the foot frame (ankle 0, toe +), normal force; median and inter-quartile band per variant, forward (vx 0.6 / 1.0)
   and backward (vx -0.5) walking. Plus touchdown / lift-off pitch and swing clearance numbers in gait_phase.json.
2. foot_side_<seg>.mp4 - real time (50 Hz data -> 50 fps), side view zoomed on the RIGHT foot, variants side by side:
   collision shape of the foot (projected), shank (ankle -> knee by forward kinematics of the recorded joints), ground,
   CoP and normal force arrow (0.1 m per body weight).
Usage: mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/foot_gait_compare.py <eval_dir> <assets_dir> <out_dir> [seg] [env]
"""
import json
import sys
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))

matplotlib.use("Agg")
import matplotlib.animation as anim  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

EV, AS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEG = sys.argv[4] if len(sys.argv) > 4 else "vx0.6"
ENV = int(sys.argv[5]) if len(sys.argv) > 5 else 0
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.02
XML = {"base": "pygmalion_legonly_ab.xml", "v3full": "pygmalion_legonly_ab_v3full.xml", "v3same": "pygmalion_legonly_ab_v3same.xml",
       "v3sph": "pygmalion_legonly_ab_v3sph.xml"}
COL = {"base": "tab:gray", "v3full": "tab:red", "v3same": "tab:orange", "v3sph": "tab:blue"}
JOINTS16 = [f"{s}_{j}_joint" for s in "RL" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "crank_A", "crank_B")] + \
           [f"{s}_ankle_{j}_joint" for s in "RL" for j in ("pitch", "roll")]


from foot_gait_compare_lib import foot_geoms, quat_mat  # noqa: E402


def load(v):
  D = dict(np.load(EV / f"{v}.npz"))
  meta = json.loads((EV / f"{v}.json").read_text())
  return D, meta


def stances(fn, bw, min_len=6):
  on = fn > 0.1 * bw
  out, t = [], 0
  while t < len(on):
    if on[t]:
      s = t
      while t < len(on) and on[t]:
        t += 1
      if t - s >= min_len and s > 0 and t < len(on):
        out.append((s, t))
    else:
      t += 1
  return out


# ---------------- 1. gait-phase statistics ----------------
variants = [v for v in ("base", "v3full", "v3same", "v3sph") if (EV / f"{v}.npz").exists()]
groups = {"forward (vx 0.6 and 1.0)": ("vx0.6", "vx1.0"), "backward (vx -0.5)": ("vx-0.5",)}
res, curves = {}, {}
ph = np.linspace(0, 100, 51)
for v in variants:
  D, meta = load(v)
  bw = meta["robot_mass_kg"] * 9.81
  n = int(round(meta["seconds_per_cmd"] / DT))
  names = [p[0] for p in meta["program"]]
  m = mujoco.MjModel.from_xml_path(str(AS / XML[v]))
  gv = {f: np.concatenate([g for g, _ in foot_geoms(m, f"{f}_foot_link")]) for f in "RL"}
  R = quat_mat(D["foot_quat_w"])  # T,E,2,3,3
  pitch = np.degrees(np.arcsin(np.clip(R[..., 2, 0], -1, 1)))  # toe up +
  # lowest point of the foot collision shape, world z
  low = np.stack([np.min(np.einsum("tej,kj->tek", R[:, :, i, 2, :], gv[f]), -1) + D["foot_pos_w"][:, :, i, 2]
                  for i, f in enumerate("RL")], -1)
  ground = np.nanpercentile(low[D["sole"][..., 0] > 0.5 * bw], 50)
  for gname, segs in groups.items():
    P, C, F, td, lo, clear = [], [], [], [], [], []
    for sg in segs:
      k = names.index(sg)
      a, b = k * n + 150, (k + 1) * n
      for e in range(D["sole"].shape[1]):
        for fi in range(2):
          fn = D["sole"][a:b, e, fi, 0]
          st = stances(fn, bw)
          for s, t in st:
            idx = np.linspace(s, t - 1, 51)
            P.append(np.interp(idx, np.arange(len(fn)), pitch[a:b, e, fi]))
            C.append(np.interp(idx, np.arange(len(fn)), np.nan_to_num(D["sole"][a:b, e, fi, 2] * 1000, nan=np.nan)))
            F.append(np.interp(idx, np.arange(len(fn)), fn / bw))
            td.append(pitch[a + s, e, fi])
            lo.append(pitch[a + t - 1, e, fi])
          for (s0, t0), (s1, _) in zip(st[:-1], st[1:]):
            clear.append((np.max(low[a + t0:a + s1, e, fi]) - ground) * 1000)
    P, C, F = map(np.array, (P, C, F))
    curves[(v, gname)] = (P, C, F)
    res.setdefault(v, {})[gname] = dict(
      n_stance=int(len(P)), stance_ms_median=None,
      touchdown_pitch_deg=float(np.median(td)), liftoff_pitch_deg=float(np.median(lo)),
      cop_x_at_10pct=float(np.nanmedian(C[:, 5])), cop_x_at_50pct=float(np.nanmedian(C[:, 25])), cop_x_at_90pct=float(np.nanmedian(C[:, 45])),
      cop_x_range_median=float(np.nanmedian(np.nanmax(C, 1) - np.nanmin(C, 1))),
      peak_fn_bw_median=float(np.median(F.max(1))), swing_clearance_mm_median=float(np.median(clear)) if clear else None,
      ground_z_mm=float(ground * 1000))
(OUT / "gait_phase.json").write_text(json.dumps(res, indent=1))

fig, ax = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
for c, gname in enumerate(groups):
  for v in variants:
    P, C, F = curves[(v, gname)]
    for r, (Y, lab) in enumerate(((P, "foot pitch [deg] (toe up +)"), (C, "CoP x from ankle [mm] (toe +)"), (F, "normal force [BW]"))):
      med = np.nanmedian(Y, 0)
      q1, q3 = np.nanpercentile(Y, 25, 0), np.nanpercentile(Y, 75, 0)
      ax[r, c].fill_between(ph, q1, q3, color=COL[v], alpha=0.15)
      ax[r, c].plot(ph, med, color=COL[v], lw=2, label=f"{v} (n={len(Y)})")
      ax[r, c].set_ylabel(lab)
      ax[r, c].grid(alpha=0.3)
  ax[0, c].set_title(f"{gname}: one stance, 0 % = touchdown, 100 % = lift-off", fontsize=10)
  for yv, lab in ((-80, "heel tip"), (-58, "v3 heel flat starts"), (0, "ankle"), (135, "v3 toe flat ends"), (180, "toe tip")):
    ax[1, c].axhline(yv, color="k", lw=0.5, ls=":")
    ax[1, c].text(1, yv + 2, lab, fontsize=7)
  ax[2, c].set_xlabel("stance phase [%]")
  ax[0, c].legend(fontsize=8)
fig.suptitle("Walking style with each foot, same policy (Union-DR 49999), median and inter-quartile band", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_gait_phase.png", dpi=105)
print(json.dumps({v: {g: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in d.items()} for g, d in r.items()} for v, r in res.items()}, indent=1))

# ---------------- 2. side-view video, right foot ----------------
vid = [v for v in ("base", "v3full") if v in variants]
frames = {}
for v in vid:
  D, meta = load(v)
  bw = meta["robot_mass_kg"] * 9.81
  n = int(round(meta["seconds_per_cmd"] / DT))
  k = [p[0] for p in meta["program"]].index(SEG)
  a, b = k * n, (k + 1) * n
  m = mujoco.MjModel.from_xml_path(str(AS / XML[v]))
  d = mujoco.MjData(m)
  jadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS16]
  knee = []
  for t in range(a, b):
    d.qpos[:] = 0
    d.qpos[0:3] = D["base_pos"][t, ENV]
    d.qpos[3:7] = D["base_quat"][t, ENV]
    d.qpos[jadr] = D["q16"][t, ENV]
    mujoco.mj_kinematics(m, d)
    knee.append(d.xpos[m.body("R_shin_link").id].copy())
  geoms = foot_geoms(m, "R_foot_link")
  frames[v] = dict(fp=D["foot_pos_w"][a:b, ENV, 0], R=quat_mat(D["foot_quat_w"][a:b, ENV, 0]), knee=np.array(knee),
                   fn=D["sole"][a:b, ENV, 0, 0] / bw, cop=D["sole"][a:b, ENV, 0, 2:4], geoms=geoms,
                   ground=res[v][next(iter(groups))]["ground_z_mm"] / 1000)
T = min(len(f["fp"]) for f in frames.values())
fig, axs = plt.subplots(1, len(vid), figsize=(7.2 * len(vid), 4.6))
axs = np.atleast_1d(axs)
arts = []
for a_, v in zip(axs, vid):
  a_.set_aspect("equal")
  a_.set_xlabel("x along walking direction [m] (camera follows the right foot)")
  a_.set_title(f"{v}: right foot, {SEG}, env {ENV}", fontsize=10)
  ground = a_.axhline(frames[v]["ground"], color="saddlebrown", lw=2)
  polys = [a_.add_patch(Polygon(np.zeros((3, 2)), closed=True, fc=("darkorange" if kind == "sole" else "lightsteelblue"), ec="k", lw=0.4, alpha=0.9))
           for _, kind in frames[v]["geoms"]]
  shank, = a_.plot([], [], color="dimgray", lw=6, solid_capstyle="round")
  cop, = a_.plot([], [], "o", color="red", ms=6)
  arrow, = a_.plot([], [], color="red", lw=2.5)
  txt = a_.text(0.02, 0.96, "", transform=a_.transAxes, va="top", fontsize=9, family="monospace")
  arts.append((a_, v, polys, shank, cop, arrow, txt))


def draw(t):
  out = []
  for a_, v, polys, shank, cop, arrow, txt in arts:
    f = frames[v]
    p, R = f["fp"][t], f["R"][t]
    cx = p[0] + 0.05
    a_.set_xlim(cx - 0.24, cx + 0.24)
    a_.set_ylim(f["ground"] - 0.03, f["ground"] + 0.24)
    for poly, (gvtx, _) in zip(polys, f["geoms"]):
      w = gvtx @ R.T + p
      xz = w[:, [0, 2]]
      h = ConvexHull(xz)
      poly.set_xy(xz[h.vertices])
    k = f["knee"][t]
    shank.set_data([p[0], k[0]], [p[2], k[2]])
    if f["fn"][t] > 0.05 and np.isfinite(f["cop"][t, 0]):
      c = p + R @ np.array([f["cop"][t, 0], f["cop"][t, 1], 0.0])
      cop.set_data([c[0]], [f["ground"]])
      arrow.set_data([c[0], c[0]], [f["ground"], f["ground"] + 0.1 * f["fn"][t]])
    else:
      cop.set_data([], [])
      arrow.set_data([], [])
    pitch = np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    txt.set_text(f"t {t * DT:5.2f} s   foot pitch {pitch:+5.1f} deg\nnormal force {f['fn'][t]:4.2f} BW   CoP x "
                 + (f"{f['cop'][t, 0] * 1000:+5.0f} mm" if f["fn"][t] > 0.05 and np.isfinite(f["cop"][t, 0]) else "  --  "))
    out += [shank, cop, arrow, txt] + polys
  return out


fig.suptitle("Side view, real time. Orange = sole collision shape, blue-grey = plate/bracket collision, grey bar = shank, red = CoP and normal force (0.1 m per body weight)", fontsize=9)
fig.tight_layout()
ani = anim.FuncAnimation(fig, draw, frames=T, blit=False)
mp4 = OUT / f"foot_side_{SEG}_env{ENV}.mp4"
ani.save(mp4, writer=anim.FFMpegWriter(fps=int(round(1 / DT)), bitrate=3000))
print("saved", OUT / "fig_gait_phase.png", mp4, f"{T} frames at {int(round(1 / DT))} fps = {T * DT:.1f} s")
