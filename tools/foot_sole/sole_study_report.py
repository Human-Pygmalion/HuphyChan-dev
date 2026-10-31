#!/usr/bin/env python3
"""Foot-sole study: statistics, comparison tables and figures from the eval_queue.py results.

Unit of analysis = one evaluation environment (32 per run and terrain). For every run x terrain the per-environment metrics of
sole_study_metrics.py are cached (<eval>/metrics/<tag>.json). Comparisons are differences of means with a 95 % bootstrap
confidence interval over environments (10 000 resamples). A difference counts as an effect only when (a) the interval excludes 0
and (b) it is larger than the seed noise floor: the difference between two training seeds of the same foot (base and v5fullm,
seed 43 vs 44) at the same iteration and terrain. Falls are compared as counts (Fisher exact test).
  mujoco-sim/mjlab/.venv/bin/python3 tools/foot_sole/sole_study_report.py <eval_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sole_study_metrics as sm  # noqa: E402

EVAL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
(EVAL / "metrics").mkdir(exist_ok=True)
TERRAINS = ["flat", "ridge", "slope_rough", "gravel", "stones"]
TLAB = {"flat": "flat", "ridge": "ridges 1-2 cm", "slope_rough": "6° up + ridges", "gravel": "gravel 0-2 cm (OOD)", "stones": "stones 2 cm (OOD)"}
METRICS = [  # key, label, unit, better (-1 lower is better, +1 higher is better)
  ("td_peak_bw", "touchdown peak force", "BW", -1),
  ("td_rate_bw_s", "loading rate", "BW/s", -1),
  ("grf_p99_bw", "vertical force P99", "BW", -1),
  ("cot_pos", "cost of transport (positive work)", "-", -1),
  ("tau2_per_m", "squared torque per metre", "N²m²s/m", -1),
  ("fell", "fall rate", "-", -1),
  ("tilt_rms_deg", "base tilt RMS", "deg", -1),
  ("angvel_rms", "base roll/pitch rate RMS", "rad/s", -1),
  ("vx_err", "forward speed error", "m/s", -1),
  ("wz_err", "yaw rate error", "rad/s", -1),
  ("slip_p95_mm_s", "foot slip P95", "mm/s", -1),
  ("foot_tilt_p95_deg", "stance foot tilt P95", "deg", -1),
  ("cop_margin_p5_mm", "CoP margin P5", "mm", +1),
  ("onsets_per_s", "stance onsets per foot", "1/s", -1),
]
RNG = np.random.default_rng(0)


def metrics(tag):
  cache = EVAL / "metrics" / f"{tag}.json"
  src = EVAL / f"{tag}.npz"
  if not src.exists():
    return None
  if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
    return {k: np.array(v, float) for k, v in json.loads(cache.read_text()).items()}
  D, meta = sm.load(EVAL / tag)
  m = sm.env_metrics(D, meta)
  cache.write_text(json.dumps({k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in m.items()}))
  return {k: np.asarray(v, float) for k, v in m.items()}


def boot_diff(a, b, n=10000):
  a, b = a[np.isfinite(a)], b[np.isfinite(b)]
  if len(a) < 3 or len(b) < 3:
    return np.nan, np.nan, np.nan
  ia = RNG.integers(0, len(a), (n, len(a)))
  ib = RNG.integers(0, len(b), (n, len(b)))
  d = a[ia].mean(1) - b[ib].mean(1)
  return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mean_ci(a, n=10000):
  a = a[np.isfinite(a)]
  if len(a) < 2:
    return (float(a.mean()) if len(a) else np.nan), np.nan
  bs = a[RNG.integers(0, len(a), (n, len(a)))].mean(1)
  return float(a.mean()), float((np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2)


def runs_available():
  tags = sorted(p.stem for p in EVAL.glob("*.json") if (EVAL / f"{p.stem}.npz").exists())
  out = {}
  for t in tags:
    for ter in sorted(TERRAINS, key=len, reverse=True):
      if t.endswith("_" + ter):
        out.setdefault(t[: -len(ter) - 1], {})[ter] = t
        break
  return out  # name_it -> terrain -> tag


RUNS = runs_available()
M = {r: {ter: metrics(tag) for ter, tag in d.items()} for r, d in RUNS.items()}


def get(run, ter):
  return M.get(run, {}).get(ter)


# ---------- noise floor from seed replicates
def noise_floor(it):
  nf = {}
  for ter in TERRAINS:
    for k, *_ in METRICS:
      ds = []
      for a, b in ((f"base_{it}", f"base_s44_{it}"), (f"v5fullm_{it}", f"v5fullm_s44_{it}")):
        A, B = get(a, ter), get(b, ter)
        if A is not None and B is not None:
          ds.append(abs(np.nanmean(A[k]) - np.nanmean(B[k])))
      nf[(ter, k)] = max(ds) if ds else np.nan
  return nf


NF = {it: noise_floor(it) for it in (51000, 52498)}


def compare(a, b, it):
  """rows: terrain, metric, mean a, mean b, diff, ci lo, ci hi, noise floor, verdict"""
  rows = []
  for ter in TERRAINS:
    A, B = get(a, ter), get(b, ter)
    if A is None or B is None:
      continue
    for k, lab, unit, better in METRICS:
      if k == "fell":
        fa, fb = int(np.nansum(A[k])), int(np.nansum(B[k]))
        p = fisher_exact([[fa, len(A[k]) - fa], [fb, len(B[k]) - fb]])[1]
        rows.append(dict(terrain=ter, metric=k, a=fa, b=fb, diff=fa - fb, lo=np.nan, hi=np.nan, nf=np.nan,
                         verdict=("better" if fa < fb else "worse") if p < 0.05 else "n.s.", p=p))
        continue
      d, lo, hi = boot_diff(A[k], B[k])
      nf = NF.get(it, {}).get((ter, k), np.nan)
      sig = np.isfinite(lo) and (lo > 0 or hi < 0)
      big = not np.isfinite(nf) or abs(d) > nf
      v = ("better" if d * better > 0 else "worse") if (sig and big) else ("within noise" if sig else "n.s.")
      rows.append(dict(terrain=ter, metric=k, a=float(np.nanmean(A[k])), b=float(np.nanmean(B[k])), diff=d, lo=lo, hi=hi, nf=nf, verdict=v))
  return rows


def fmt(x, nd=3):
  return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}g}"


def table_runs(names, title, keys=None):
  keys = keys or [m[0] for m in METRICS]
  lines = [f"### {title}", ""]
  for ter in TERRAINS:
    avail = [n for n in names if get(n, ter) is not None]
    if not avail:
      continue
    lines += [f"**{TLAB[ter]}** (mean ± 95 % CI over environments; falls = count of 32)", "",
              "| metric | " + " | ".join(avail) + " |", "|---|" + "---|" * len(avail)]
    for k, lab, unit, _ in METRICS:
      if k not in keys:
        continue
      cells = []
      for n in avail:
        a = get(n, ter)[k]
        if k == "fell":
          cells.append(f"{int(np.nansum(a))}")
        else:
          mu, ci = mean_ci(a)
          cells.append(f"{fmt(mu)} ± {fmt(ci, 2)}")
      lines.append(f"| {lab} [{unit}] | " + " | ".join(cells) + " |")
    lines.append("")
  return lines


def table_compare(a, b, it, title):
  rows = compare(a, b, it)
  if not rows:
    return [f"### {title}", "", "(no data yet)", ""]
  lines = [f"### {title}", "", f"difference = {a} − {b}; effect = 95 % CI excludes 0 and |difference| > seed noise floor", "",
           "| terrain | metric | " + a + " | " + b + " | difference [95 % CI] | noise floor | verdict |", "|---|---|---|---|---|---|---|"]
  for r in rows:
    if r["metric"] == "fell":
      lines.append(f"| {TLAB[r['terrain']]} | falls | {r['a']} | {r['b']} | {r['diff']:+d} (Fisher p = {r['p']:.2g}) | — | {r['verdict']} |")
    else:
      lab = next(m[1] for m in METRICS if m[0] == r["metric"])
      lines.append(f"| {TLAB[r['terrain']]} | {lab} | {fmt(r['a'])} | {fmt(r['b'])} | {fmt(r['diff'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] | {fmt(r['nf'])} | {r['verdict']} |")
  return lines + [""]


SWEEP = [("heel_flat", "heel flat end x [mm]", [("v5_hf10", 10), ("v5fullm", 20), ("v5_hf30", 30)]),
         ("heel_rise", "heel end height [mm]", [("v5_hr2", 2), ("v5fullm", 3), ("v5_hr5", 5)]),
         ("toe_flat", "toe flat end x [mm]", [("v5_tf200", 200), ("v5fullm", 215), ("v5_tf230", 230)]),
         ("toe_rise", "toe end height [mm]", [("v5_tr3", 3), ("v5fullm", 5), ("v5_tr8", 8)]),
         ("arch_lift", "arch depth [mm]", [("v5_a0", 0), ("v5_a2", 2), ("v5fullm", 4.5), ("v5_a6", 6), ("v5_a9", 9), ("v5_a12", 12)]),
         ("apex_x", "arch apex x [mm]", [("v5fullm", 108), ("v5_ax130", 130)]),
         ("soft", "arch wedge", [("v5fullm", 0), ("v5_softarch", 1)])]
FIG_METRICS = ["td_peak_bw", "td_rate_bw_s", "cot_pos", "tilt_rms_deg", "slip_p95_mm_s", "foot_tilt_p95_deg", "cop_margin_p5_mm", "fell"]


def fig_sweep():
  it = 51000
  cols = plt.cm.viridis(np.linspace(0, 0.9, len(TERRAINS)))
  for pname, plab, pts in SWEEP:
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for a, k in zip(ax.flat, FIG_METRICS):
      lab = next(m for m in METRICS if m[0] == k)
      for c, ter in zip(cols, TERRAINS):
        xs, ys, es = [], [], []
        for n, x in pts:
          v = get(f"{n}_{it}", ter)
          if v is None:
            continue
          mu, ci = (float(np.nansum(v[k])), 0.0) if k == "fell" else mean_ci(v[k])
          xs.append(x)
          ys.append(mu)
          es.append(ci)
        if xs:
          a.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=3, color=c, label=TLAB[ter])
          nf = NF[it].get((ter, k), np.nan)
          base = [y for (n, x), y in zip([p for p in pts if get(f"{p[0]}_{it}", ter) is not None], ys) if n == "v5fullm"]
          if base and np.isfinite(nf):
            a.axhspan(base[0] - nf, base[0] + nf, color=c, alpha=0.08)
      a.set_title(f"{lab[1]} [{lab[2]}]" + (" (count of 32)" if k == "fell" else ""), fontsize=9)
      a.set_xlabel(plab, fontsize=8)
      a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"Sole sweep: {plab} (others at v5; 1,002 fine-tune iterations; band = v5 ± seed noise floor)")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_sweep_{pname}.png", dpi=110)
    plt.close(fig)


def fig_pareto(names, it, fname, title):
  fig, ax = plt.subplots(1, 3, figsize=(16, 5))
  groups = [("flat", ["flat"]), ("trained rough", ["ridge", "slope_rough"]), ("untrained rough", ["gravel", "stones"])]
  for a, (gl, ters) in zip(ax, groups):
    for n in names:
      vals = [get(f"{n}_{it}", t) for t in ters]
      if any(v is None for v in vals):
        continue
      imp = np.mean([np.nanmean(v["td_peak_bw"]) for v in vals])
      cot = np.mean([np.nanmean(v["cot_pos"]) for v in vals])
      falls = sum(int(np.nansum(v["fell"])) for v in vals)
      tilt = np.mean([np.nanmean(v["tilt_rms_deg"]) for v in vals])
      a.scatter(cot, imp, s=30 + 15 * tilt ** 2, c="tab:red" if falls else "tab:blue", alpha=0.7, edgecolor="k")
      a.annotate(n + (f" ({falls} falls)" if falls else ""), (cot, imp), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("cost of transport (positive work)")
    a.set_ylabel("touchdown peak force [BW]")
    a.set_title(f"{gl}: marker size = base tilt RMS, red = falls", fontsize=9)
    a.grid(alpha=0.3)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(OUT / fname, dpi=110)
  plt.close(fig)


if __name__ == "__main__":
  lines = ["# Foot-sole study — results tables (generated by tools/foot_sole/sole_study_report.py)", "",
           f"Runs with data: {len(RUNS)}; evaluation environments per run and terrain: 32.", ""]
  lines += ["## Seed noise floor (largest |seed 43 − seed 44| of base and v5fullm, same iteration and terrain)", ""]
  for it in (51000, 52498):
    lines += [f"**iteration {it}**", "", "| metric | " + " | ".join(TLAB[t] for t in TERRAINS) + " |", "|---|" + "---|" * len(TERRAINS)]
    for k, lab, unit, _ in METRICS:
      if k == "fell":
        continue
      lines.append(f"| {lab} [{unit}] | " + " | ".join(fmt(NF[it].get((t, k))) for t in TERRAINS) + " |")
    lines.append("")
  lines += table_runs(["orig_49999", "base_52498", "v3full_52498", "v5full_52498", "v5fullm_52498"],
                      "Stage 0: foot types after 2,500 fine-tune iterations (orig = deployed 49999, box foot, no fine-tune)")
  lines += table_compare("v5fullm_52498", "base_52498", 52498, "v5 (CAD mass) vs box foot, 2,500 iterations")
  lines += table_compare("v3full_52498", "base_52498", 52498, "v3 vs box foot, 2,500 iterations")
  lines += table_compare("v5full_52498", "v3full_52498", 52498, "v5 vs v3 (both 2,500 iterations)")
  lines += table_compare("v5fullm_52498", "v5full_52498", 52498, "Foot mass only: v5 CAD mass (1.186 kg) vs 1.097 kg")
  lines += table_compare("base_52498", "orig_49999", 52498, "Control: box foot fine-tuned vs not fine-tuned (effect of the extra training alone)")
  lines += table_runs(["v5fullm_51000"] + [p[0] + "_51000" for _, _, pts in SWEEP for p in pts if p[0] != "v5fullm"],
                      "Stage 1 sweep after 1,002 iterations (v5fullm_51000 = reference at the same iteration)")
  for _, plab, pts in SWEEP:
    for n, x in pts:
      if n != "v5fullm":
        lines += table_compare(f"{n}_51000", "v5fullm_51000", 51000, f"Sweep {plab} = {x}: {n} vs v5")
  lines += table_compare("v5fullm_sr01_52498", "base_sr01_52498", 52498, "Stiffer foot contact (solref 0.01): v5 vs box foot")
  lines += table_compare("base_sr01_52498", "base_52498", 52498, "Contact stiffness only, box foot: solref 0.01 vs 0.02")
  lines += table_compare("v5fullm_sr01_52498", "v5fullm_52498", 52498, "Contact stiffness only, v5: solref 0.01 vs 0.02")
  (OUT / "results_tables.md").write_text("\n".join(lines))
  fig_sweep()
  fig_pareto(["base", "v3full", "v5full", "v5fullm"], 52498, "fig_pareto_stage0.png", "Stage 0 foot types after 2,500 iterations")
  fig_pareto(["base", "v5fullm"] + [p[0] for _, _, pts in SWEEP for p in pts if p[0] != "v5fullm"], 51000, "fig_pareto_sweep.png",
             "Stage 1 sweep after 1,002 iterations (base = box foot at the same iteration)")
  json.dump({r: {t: {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in mm.items()} for t, mm in d.items() if mm is not None}
             for r, d in M.items()}, open(OUT / "per_env_metrics.json", "w"))
  print("wrote", OUT / "results_tables.md", "runs:", sorted(RUNS))
