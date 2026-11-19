#!/usr/bin/env python3
"""Verify hashes, full checkpoints, motion, and both ONNX actors."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

def actor(state: dict[str, torch.Tensor], obs: torch.Tensor) -> np.ndarray:
  value = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 0.01)
  for layer in (0, 2, 4):
    value = torch.nn.functional.elu(torch.nn.functional.linear(
      value, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"]))
  return torch.nn.functional.linear(value, state["mlp.6.weight"], state["mlp.6.bias"]).numpy()

manifest = yaml.safe_load((ROOT / "deploy.yaml").read_text())
for relative, info in manifest["artifacts"].items():
  assert sha256(ROOT / relative) == info["sha256"], relative
policy = manifest["policy"]
parent = torch.load(ROOT / policy["parent_checkpoint"], map_location="cpu", weights_only=False)
latest = torch.load(ROOT / policy["checkpoint"], map_location="cpu", weights_only=False)
assert parent["iter"] == policy["parent_iteration"]
assert latest["iter"] == policy["iteration"]
assert latest["infos"]["env_state"]["common_step_counter"] == policy["common_step_counter"]
assert latest["actor_state_dict"]["mlp.0.weight"].shape == (512, 114)
assert latest["actor_state_dict"]["mlp.6.weight"].shape == (12, 128)
assert latest["critic_state_dict"]["mlp.0.weight"].shape == (512, 204)
for filename in ("policy.onnx", "motion_policy.onnx"):
  onnx.checker.check_model(onnx.load(ROOT / filename))
torch.manual_seed(20261119)
obs = torch.randn(16, 114)
expected = actor(latest["actor_state_dict"], obs)
for filename in ("policy.onnx", "motion_policy.onnx"):
  session = ort.InferenceSession(str(ROOT / filename), providers=["CPUExecutionProvider"])
  outputs = []
  for index in range(len(obs)):
    inputs = {"obs": obs[index:index + 1].numpy()}
    if filename == "motion_policy.onnx":
      inputs["time_step"] = np.asarray([[index]], dtype=np.float32)
    outputs.append(session.run(None, inputs)[0])
  np.testing.assert_allclose(np.concatenate(outputs), expected, rtol=1e-5, atol=1e-6)
motion = np.load(ROOT / manifest["motion"]["training_file"], allow_pickle=True)
assert motion["joint_pos"].shape == (manifest["motion"]["frames"], 24)
assert float(np.asarray(motion["fps"]).reshape(-1)[0]) == manifest["motion"]["fps"]
assert float(motion["max_loop_residual_m"]) < 1e-8
assert manifest["pd_gains"]["stiffness_nm_per_rad"] == [135.0,135.0,75.0,130.0,60.0,60.0,135.0,135.0,75.0,130.0,60.0,60.0]
print("PASS: hashes, checkpoints, motion, DeployGains contract, and both ONNX actors")
