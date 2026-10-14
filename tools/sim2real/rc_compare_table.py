#!/usr/bin/env python3
"""Per-condition summary of real_condition_sim.py json files, two policies side by side.

  python rc_compare_table.py --a v7b=file1.json,file2.json --b rough=file3.json,... --conds-from a [--md]

Each side is a name plus a comma list of json files; a condition is read from the first file that has it.
Columns: mean survival (s), reps surviving the whole run, |action| p99 (mean over reps),
knee angle std (deg), dominant knee frequency (Hz), gyro RMS (deg/s, root of the mean of the three axes squared).
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--conds", default="", help="'|' separated condition names; default = all of a")
ap.add_argument("--json-out", default="")
a = ap.parse_args()


def load(spec):
    name, files = spec.split("=")
    by = {}
    for f in files.split(","):
        for r in json.load(open(f))["rows"]:
            by.setdefault(r["cond"], {}).setdefault(f, []).append(r)
    # first file that has the condition
    return name, {c: v[next(k for k in files.split(",") if k in v)] for c, v in by.items()}


def summ(rows):
    if not rows:
        return None
    g = np.array([r["gyro_rms_dps"] for r in rows])
    return dict(n=len(rows), surv=float(np.mean([r["survived_s"] for r in rows])),
                full=int(sum(not r["fell"] for r in rows)),
                p99=float(np.mean([r["act_p99"] for r in rows])),
                kstd=float(np.mean([r["knee_std_deg"] for r in rows])),
                khz=float(np.nanmean([r["knee_hz"] for r in rows])),
                gyro=float(np.sqrt((g ** 2).mean())))


na, A = load(a.a)
nb, B = load(a.b)
conds = a.conds.split("|") if a.conds else list(A)
out = {}
fmt = lambda s: "—" if s is None else f"{s['surv']:.1f} | {s['full']}/{s['n']} | {s['p99']:.2f} | {s['kstd']:.1f} | {s['khz']:.2f} | {s['gyro']:.0f}"  # noqa: E731
print(f"| 조건 | {na} 생존(초) | 끝까지 | |행동| p99 | 무릎 표준편차(도) | 무릎 주파수(Hz) | 자이로(도/초) | {nb} 생존(초) | 끝까지 | |행동| p99 | 무릎 표준편차(도) | 무릎 주파수(Hz) | 자이로(도/초) |")
print("|" + "---|" * 13)
for c in conds:
    sa, sb = summ(A.get(c)), summ(B.get(c))
    out[c] = {na: sa, nb: sb}
    print(f"| {c} | {fmt(sa)} | {fmt(sb)} |")
if a.json_out:
    json.dump(out, open(a.json_out, "w"), indent=1)
