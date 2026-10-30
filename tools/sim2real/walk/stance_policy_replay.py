#!/usr/bin/env python3
"""Replay logged 44-input rows through policy.onnx (and model_49999.pt actor) and compare with the logged `act/` columns.
Also checks the log contract: tgt = clip(default + 0.25*act) in degrees, last_act[t+1] = act[t].
  CUDA_VISIBLE_DEVICES="" mujoco-sim/mjlab/.venv/bin/python3 tools/sim2real/walk/stance_policy_replay.py --ckpt-dir <.../terrain10_union_dr_49999> --csv a.csv [b.csv]
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np, onnxruntime as ort, torch
from stance_fk import load_csv, COLS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--json", default="")
a = ap.parse_args()
D = Path(a.ckpt_dir)
sess = ort.InferenceSession(str(D / "policy.onnx"), providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
sd = torch.load(D / "model_49999.pt", map_location="cpu", weights_only=False)
ms = sd["actor_state_dict"]
W = [(ms[f"mlp.{i}.weight"], ms[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]
def torch_actor(x):
  x = torch.tensor(x, dtype=torch.float32)
  x = (x - ms["obs_normalizer._mean"]) / (ms["obs_normalizer._std"] + 1e-2)
  for i, (w, b) in enumerate(W):
    x = x @ w.T + b
    if i < 3: x = torch.nn.functional.elu(x)
  return x.numpy()
DEFAULT = np.degrees([-0.175,0,0,0.35,0.1439548737,-0.1470630753,0.175,0,0,-0.35,-0.1438849966,0.1468702056])
res = {}
for p in a.csv:
  cols, arr = load_csv(p)
  c = {n: i for i, n in enumerate(cols)}
  obs_cols = (["obs/gyro/" + x for x in "xyz"] + ["obs/grav/" + x for x in "xyz"] + [f"obs/q_prev/{j}" for j in COLS] + [f"obs/q/{j}" for j in COLS]
              + [f"obs/last_act/{j}" for j in COLS] + ["obs/cmd/vx", "obs/cmd/wz"])
  X = arr[:, [c[n] for n in obs_cols]].astype(np.float32)
  act = arr[:, [c[f"act/{j}"] for j in COLS]]
  tgt = arr[:, [c[f"tgt/{j}"] for j in COLS]]
  onnx = np.concatenate([sess.run(None, {iname: X[i:i+1]})[0] for i in range(len(X))])
  tor = torch_actor(X)
  e1 = np.abs(onnx - act); e2 = np.abs(tor - act); e3 = np.abs(tor - onnx)
  # tgt contract
  lo = np.degrees(np.array([-1.9678634,-0.3403389,-0.7068582,0.10472,-1.08,-1.08,-0.3097954,-1.3875369,-0.7068582,-1.98968,-1.08,-1.08]))
  hi = np.degrees(np.array([0.3097954,1.3875369,0.7068582,1.98968,1.08,1.08,1.9678634,0.3403389,0.7068582,-0.10472,1.08,1.08]))
  tg = np.clip(DEFAULT + np.degrees(0.25) * act, lo, hi)
  r = dict(rows=len(X), onnx_vs_logged_act_max=float(e1.max()), onnx_vs_logged_act_mean=float(e1.mean()),
           torch_vs_logged_act_max=float(e2.max()), torch_vs_onnx_max=float(e3.max()),
           tgt_contract_max_deg=float(np.abs(tg - tgt).max()),
           lastact_chain_max=float(np.abs(X[1:, 30:42] - act[:-1]).max()),
           cmd_vx_range=[float(X[:, 42].min()), float(X[:, 42].max())], cmd_wz_range=[float(X[:, 43].min()), float(X[:, 43].max())])
  print(Path(p).name, json.dumps(r)); res[Path(p).name] = r
res["onnx_sha256"] = hashlib.sha256((D / "policy.onnx").read_bytes()).hexdigest()
if a.json: Path(a.json).write_text(json.dumps(res, indent=1))
