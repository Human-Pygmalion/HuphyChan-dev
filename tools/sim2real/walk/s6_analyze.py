#!/usr/bin/env python3
"""Analyse stage-6 lab runs (s6_oscillation_lab.py npz) and the real log with one code path -> JSON table + figure.

  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_analyze.py --runs name=path.npz[,..] --json out.json
Per environment: pre window [2.5, 4.9] s, post window [7, end] s (impulse 5.0-5.08 s), falls cut the record. See s6_metrics.py for definitions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from s6_metrics import (JIDX, band_power, decay_time, ls_peak, peak_fit, sliding_band_amp)  # noqa: E402

PRE = (2.5, 4.9)
POST0 = 7.0
T_IMP = 5.0


def env_record(d, e):
  n = int(d["n_calls"][e])
  t = d["t_call"][:n, e].astype(float)
  rec = dict(t=t, tgt=np.degrees(d["tgt"][:n, e]), pos=np.degrees(d["pos"][:n, e]), act=d["act"][:n, e],
             gyro=d["obs_gyro_grav"][:n, e, :3], tilt=None)
  fk = int(d["fell_k"][e])
  if fk >= 0:
    m = t < fk * 0.005
    rec = {k: (v[m] if v is not None else None) for k, v in rec.items()}
  return rec, fk


def one_env(d, e, seconds):
  rec, fk = env_record(d, e)
  t = rec["t"]
  end = t[-1] if len(t) else 0.0
  r = dict(fell=fk >= 0, fell_t=(fk * 0.005 if fk >= 0 else None), t_end=float(end))
  r["bp_pre_tgt"] = band_power(t, rec["tgt"], *PRE) if end > PRE[1] else float("nan")
  r["bp_pre_pos"] = band_power(t, rec["pos"], *PRE) if end > PRE[1] else float("nan")
  if end - POST0 < 4.0:
    return r
  r["bp_post_tgt"] = band_power(t, rec["tgt"], POST0, end)
  r["bp_post_pos"] = band_power(t, rec["pos"], POST0, end)
  late0 = max(POST0, end - 10.0)
  r["bp_late_tgt"] = band_power(t, rec["tgt"], late0, end)
  for nm, key in (("act", "act"), ("pos", "pos")):
    for j, (a, b) in JIDX.items():
      f, h = ls_peak(t, rec[key][:, [a, b]], POST0, end)
      r[f"f_{nm}_{j}"] = f
      r[f"h_{nm}_{j}"] = h
  fit = peak_fit(rec, POST0, end, "hip_pitch")
  if fit:
    r.update({f"fit_{k}": v for k, v in fit.items()})
  cs, am = sliding_band_amp(rec, "tgt", PRE[0], end)
  pre_amp = np.sqrt(2 * r["bp_pre_tgt"]) if np.isfinite(r["bp_pre_tgt"]) else 0.3
  thr = max(0.3, 2.0 * pre_amp)
  r["decay_thr"] = thr
  r["decay_s"] = decay_time(cs, am, T_IMP + 0.5, thr)
  r["amp_series"] = (cs.tolist(), am.tolist())
  return r


def summarize(rows):
  """aggregate one condition (list of per-env dicts)"""
  def med(k, sel=None):
    v = np.array([x[k] for x in rows if k in x and np.isfinite(x[k])])
    return float(np.median(v)) if len(v) else float("nan")

  out = dict(n=len(rows), falls=int(sum(x["fell"] for x in rows)))
  ok = [x for x in rows if "bp_post_tgt" in x]
  out["n_valid"] = len(ok)
  for k in ("bp_pre_tgt", "bp_post_tgt", "bp_late_tgt", "bp_pre_pos", "bp_post_pos", "fit_f", "fit_tgt_amp", "fit_pos_amp", "fit_ratio",
            "fit_lag_ms", "f_act_hip_pitch", "f_act_knee", "f_act_hip_roll", "f_pos_hip_pitch", "f_pos_knee", "f_pos_hip_roll"):
    out[k] = med(k)
  out["amp_pre_tgt"] = float(np.sqrt(2 * out["bp_pre_tgt"])) if np.isfinite(out["bp_pre_tgt"]) else float("nan")
  out["amp_post_tgt"] = float(np.sqrt(2 * out["bp_post_tgt"])) if np.isfinite(out["bp_post_tgt"]) else float("nan")
  out["amp_late_tgt"] = float(np.sqrt(2 * out["bp_late_tgt"])) if np.isfinite(out["bp_late_tgt"]) else float("nan")
  out["amp_post_pos"] = float(np.sqrt(2 * out["bp_post_pos"])) if np.isfinite(out["bp_post_pos"]) else float("nan")
  dec = np.array([x["decay_s"] for x in ok if "decay_s" in x], float)
  out["decay_median_s"] = float(np.median(dec)) if len(dec) else float("nan")
  out["n_sustained"] = int(np.isinf(dec).sum())
  out["n_decay_zero"] = int((dec == 0).sum())
  # oscillation-like per env: post band amplitude >= 3x pre and >= 1 deg, LS peak of the hip_pitch target in 4-6.5 Hz
  osc = []
  for x in ok:
    a_pre, a_post = np.sqrt(2 * x["bp_pre_tgt"]), np.sqrt(2 * x["bp_post_tgt"])
    osc.append(bool(a_post >= 1.0 and a_post >= 3 * a_pre and 4.0 <= x.get("f_act_hip_pitch", 0) <= 6.5))
  out["n_5hz_like"] = int(sum(osc))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", required=True, help="label=path.npz,label=path.npz")
  ap.add_argument("--json", required=True)
  a = ap.parse_args()
  res = {}
  for item in a.runs.split(","):
    label, path = item.split("=", 1)
    d = np.load(path)
    conds = list(dict.fromkeys(d["cond"].tolist()))
    for c in conds:
      es = np.where(d["cond"] == c)[0]
      rows = [one_env(d, int(e), None) for e in es]
      s = summarize(rows)
      s["rows"] = [{k: v for k, v in r.items() if k != "amp_series"} for r in rows]
      s["amp_series"] = [r.get("amp_series") for r in rows]
      res[f"{label}|{c}"] = s
      print(f"{label}|{c[:44]:44s} falls {s['falls']}/{s['n']}  amp pre {s['amp_pre_tgt']:.2f} post {s['amp_post_tgt']:.2f} late {s['amp_late_tgt']:.2f} deg  "
            f"f_act hp {s['f_act_hip_pitch']:.2f} knee {s['f_act_knee']:.2f} roll {s['f_act_hip_roll']:.2f}  5Hz-like {s['n_5hz_like']}/{s['n_valid']}  "
            f"decay {s['decay_median_s']:.1f}s sustained {s['n_sustained']}")
  json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
  main()
