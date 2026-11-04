#!/usr/bin/env python3
"""Real 11-04 walk log vs the same command trace replayed in simulation on several models (sim_stance_record.py output).
Per run: stand / walk stance width (forward kinematics of measured joint angles), hip_roll target-minus-measured, and for DR runs the percentile
of the real value inside the simulated distribution (environment = one sample).
  mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/m19_transfer/replay_compare.py <xml for FK> <real.json> <replay out dir> <out.json> name [name ...]
"""
import glob
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np

xml, real, rdir, out = sys.argv[1:5]
names = sys.argv[5:]
tool = Path(__file__).resolve().parents[2] / "sim2real" / "walk" / "real_vs_sim_stand.py"
res = {}
for n in names:
  files = sorted(glob.glob(f"{rdir}/csv/{n}_true_env*.csv"))
  if not files:
    print("no csv for", n)
    continue
  tmp = tempfile.mkdtemp()
  for f in files:
    shutil.copy(f, tmp)
  oj = Path(tmp) / "o.json"
  subprocess.run([sys.executable, str(tool), xml, real, tmp, str(oj)], check=True, stdout=subprocess.DEVNULL)
  j = json.load(open(oj))
  res["real"] = j["real"]
  sims = [v for k, v in j.items() if k.startswith("sim")]
  def col(ph, f):
    return np.array([f(s[ph]) for s in sims])
  r = dict(n_env=len(sims))
  for ph in ("stand", "walk"):
    for key, f in (("w_pos", lambda p: p["w_pos"]), ("hip_roll_R_err", lambda p: p["right"]["err"]), ("hip_roll_L_err", lambda p: p["left"]["err"])):
      v = col(ph, f)
      v = v[np.isfinite(v)]  # environments that fell before the phase have no value
      rv = f(j["real"][ph])
      r[f"{ph}_{key}"] = dict(n_valid=int(len(v)), mean=float(v.mean()), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()), real=float(rv),
                              real_percentile=float((v < rv).mean() * 100))
  res[n] = r
  print(n, {k: (round(v["mean"], 2), round(v["real"], 2), round(v["real_percentile"])) for k, v in r.items() if isinstance(v, dict)})
Path(out).write_text(json.dumps(res, indent=1))
