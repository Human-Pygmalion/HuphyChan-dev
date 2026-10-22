#!/usr/bin/env python3
"""Pool the per-environment stage-6 results (s6_analyze.py JSON files) by condition name over runs/seeds and write the condition table.
  mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/s6_collect.py --in main_s0.json b2_s0.json b3_s1.json b4_s2.json --out table.json --md table.md
An environment is 'oscillating' when the 12-joint 3-8 Hz band amplitude of the targets after the impulse (7 s to the end) is >= 1.0 deg and >= 3x its own pre-impulse
value (2.5-4.9 s). 'line' = oscillating and the Lomb-Scargle peak of the hip_pitch action (R+L) lies in 4.0-6.5 Hz.
"""
import argparse, json
import numpy as np


ALIAS = {"I hold k3000 c20 interval 20 ms": "H k3000 c20 interval 20 ms"}  # same parameters, named in a later batch


def pool(files):
  P = {}
  for f in files:
    for k, v in json.load(open(f)).items():
      name = ALIAS.get(k.split("|", 1)[1], k.split("|", 1)[1])
      for r, ser in zip(v["rows"], v["amp_series"]):
        if ser is not None and len(ser[0]):
          cs, am = np.array(ser[0]), np.array(ser[1])
          m = (cs >= 8.0) & np.isfinite(am)
          r["duty"] = float((am[m] >= 1.0).mean()) if m.any() else float("nan")
        P.setdefault(name, []).append(r)
  return P


def fmt(x, nd=2):
  return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summarize(rows):
  ok = [r for r in rows if "bp_post_tgt" in r]
  o = dict(n=len(rows), falls=int(sum(r["fell"] for r in rows)), n_valid=len(ok))
  pre = np.array([np.sqrt(2 * r["bp_pre_tgt"]) for r in ok])
  post = np.array([np.sqrt(2 * r["bp_post_tgt"]) for r in ok])
  late = np.array([np.sqrt(2 * r["bp_late_tgt"]) for r in ok])
  o["amp_pre"], o["amp_post"], o["amp_late"] = (float(np.median(x)) if len(x) else float("nan") for x in (pre, post, late))
  o["power_pre"] = float(np.median([r["bp_pre_tgt"] for r in ok])) if ok else float("nan")
  o["power_post"] = float(np.median([r["bp_post_tgt"] for r in ok])) if ok else float("nan")
  osc = [r for r, a, b in zip(ok, pre, post) if b >= 1.0 and b >= 3 * a]
  o["n_osc"] = len(osc)
  fs = [r["f_act_hip_pitch"] for r in osc]
  o["f_osc_median"] = float(np.median(fs)) if fs else float("nan")
  o["f_osc_min"], o["f_osc_max"] = (float(min(fs)), float(max(fs))) if fs else (float("nan"),) * 2
  o["n_line"] = int(sum(4.0 <= f <= 6.5 for f in fs))
  o["f_all_median"] = float(np.median([r["f_act_hip_pitch"] for r in ok])) if ok else float("nan")
  fit = [r for r in osc if "fit_ratio" in r]
  o["ratio_osc"] = float(np.median([r["fit_ratio"] for r in fit])) if fit else float("nan")
  o["lag_osc_ms"] = float(np.median([r["fit_lag_ms"] for r in fit])) if fit else float("nan")
  o["tgt_hp_osc"] = float(np.median([r["fit_tgt_amp"] for r in fit])) if fit else float("nan")
  du = [r["duty"] for r in ok if "duty" in r and np.isfinite(r["duty"])]
  o["duty_median"] = float(np.median(du)) if du else float("nan")
  dec = np.array([r["decay_s"] for r in ok if "decay_s" in r], float)
  o["n_sustained"] = int(np.isinf(dec).sum())
  fin = dec[np.isfinite(dec)]
  o["decay_median_s"] = float(np.median(dec)) if len(dec) and np.isfinite(np.median(dec)) else (float("inf") if len(dec) else float("nan"))
  return o


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--in", dest="inp", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--md", required=True)
  a = ap.parse_args()
  P = pool(a.inp)
  S = {k: summarize(v) for k, v in P.items()}
  json.dump(S, open(a.out, "w"))
  lines = ["| condition | n (falls) | band amp pre -> post [deg] | band power pre -> post [deg^2] | oscillating | line 4-6.5 Hz | f of oscillating [Hz] (median, range) | sustained | duty [%] | joint/target, lag [ms] |", "|---|---|---|---|---|---|---|---|---|---|"]
  for k, o in S.items():
    f = f"{fmt(o['f_osc_median'], 1)} ({fmt(o['f_osc_min'], 1)}-{fmt(o['f_osc_max'], 1)})" if o["n_osc"] else "-"
    rl = f"{fmt(o['ratio_osc'])}, {fmt(o['lag_osc_ms'], 0)}" if o["n_osc"] else "-"
    lines.append(f"| {k} | {o['n']} ({o['falls']}) | {fmt(o['amp_pre'])} -> {fmt(o['amp_post'])} | {fmt(o['power_pre'])} -> {fmt(o['power_post'])} | {o['n_osc']}/{o['n_valid']} | {o['n_line']}/{o['n_valid']} | {f} | {o['n_sustained']}/{o['n_valid']} | {fmt(100 * o['duty_median'], 0)} | {rl} |")
  open(a.md, "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
