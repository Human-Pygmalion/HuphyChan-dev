#!/usr/bin/env python3
"""Experiment conditions of HUPHY real-robot logs (archivedlogs/) - reproducible version of
docs/sim2real/2026-10-13_huphy_archivedlogs_conditions.md §0-§3.

  python3 tools/sim2real/huphy_log_conditions.py --huphy <extracted HUPHY tree> \
      [--weights NAME=path.pt ...]

--huphy is a checkout/archive of the HUPHY biped branch (src/, config/, archivedlogs/). Its own
control/rsl_rl.py and control/stand.py are imported so the policy is evaluated exactly as on the
robot. For every policy log: which weights reproduce the logged actions, internal consistency
(target formula, history order), policy period, start pose, guard clipping. For every latency log:
requested vs achieved send rate, late/lost share.
"""
import argparse, csv, glob, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--huphy", required=True)
ap.add_argument("--weights", nargs="*", default=[], help="NAME=path.pt candidates (default: config/policies/stand_bi*.pt)")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.huphy, "src"))
from huphy.control import rsl_rl, stand  # noqa: E402

cands = dict(w.split("=", 1) for w in a.weights) or {
  os.path.basename(p): p for p in sorted(glob.glob(os.path.join(a.huphy, "config/policies/stand_bi*.pt")))}
models = {k: rsl_rl.load(p, spec=stand.SPEC, action_dim=12) for k, p in cands.items()}
O = list(stand.ORDER)
dflt = np.array([stand.DEFAULT_POSE_DEG[j] for j in O])
lo = np.array([stand.LIMITS_DEG[j][0] for j in O]); hi = np.array([stand.LIMITS_DEG[j][1] for j in O])
logs = os.path.join(a.huphy, "archivedlogs")

print("== policy logs")
for f in sorted(glob.glob(os.path.join(logs, "*.csv"))):
  if not open(f).readline().startswith("t,obs"):
    continue
  d = np.genfromtxt(f, delimiter=",", skip_header=1)
  t, obs, act, tgt, pos = d[:, 0], d[:, 1:46], d[:, 46:58], d[:, 58:70], d[:, 70:82]
  match = {k: float(np.abs(np.stack([m(o) for o in obs]) - act).max()) for k, m in models.items()}
  e_tgt = np.abs(np.clip(dflt + np.degrees(0.25 * act), lo, hi) - tgt).max()
  e_q = np.abs(np.radians(pos - dflt) - obs[:, 18:30]).max()
  e_hist = max(np.abs(obs[1:, 6:18] - obs[:-1, 18:30]).max(), np.abs(obs[1:, 30:42] - act[:-1]).max())
  lim = np.clip(tgt, lo + 3, hi - 3); sent = np.clip(lim, pos - 50, pos + 50)
  dt = np.diff(t) * 1000
  print(f"{os.path.basename(f)}: {len(t)} calls {t[-1]:.1f}s | weights max|diff| " + ", ".join(f"{k} {v:.1e}" for k, v in match.items())
        + f" | tgt formula {e_tgt:.1e}deg, q {e_q:.1e}rad, history {e_hist:.1e}"
        + f" | cmd {np.unique(obs[:, 42:45].round(3), axis=0).tolist()} | dt med {np.median(dt):.1f} p99 {np.percentile(dt, 99):.1f} ms"
        + f" | start |pos-default| max {np.abs(pos[0] - dflt).max():.1f}deg | limit clip {100*(np.abs(lim-tgt)>1e-6).mean():.1f}% jump clip {100*(np.abs(sent-lim)>1e-6).mean():.1f}%")

print("== latency logs")
for f in sorted(glob.glob(os.path.join(logs, "*.csv"))):
  head = open(f).readline().strip()
  if not head.startswith("#"):
    continue
  rows = list(csv.reader(open(f)))[2:]
  span = max(float(r[0]) for r in rows if r[0]); n = len(rows)
  hz = float(head.split("hz=")[1])
  st = {s: 100 * sum(r[3] == s for r in rows) / n for s in ("ontime", "late", "lost")}
  lat = np.array([float(r[2]) for r in rows if r[3] == "ontime"])
  print(f"{os.path.basename(f)}: {head} | achieved {n/12/span:.1f} Hz (requested {hz:g}) | ontime p50 {np.median(lat):.2f} ms | late {st['late']:.1f}% lost {st['lost']:.1f}%")
