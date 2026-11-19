#!/usr/bin/env python3
"""Verify release hashes, checkpoint structure, and ONNX/PyTorch actor parity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


PACKAGE = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
  "model_100898.pt": "7f5f3e5f50eab1fa6991f017c2f35c490bff779e96d8c52ab88e2a9ec869072c",
  "model_129400.pt": "caa1c9ea3c98be75bae6552f1715d43d0188bd61f6b4dffe0233277af1b5957b",
  "policy.onnx": "2158ec446d976c2af0639b79b964b02f7a3fb6ed194eeed52096676b1688e171",
  "env.yaml": "e8b4c846f78fe491cf50922877b3f7c52607a92a182e904d06ceb68f3d8f202f",
  "agent.yaml": "55806db977be1ca0c9432f84f4c59f3e21d9d8c3095e053ffabd5b94d56b2514",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


for filename, expected in EXPECTED_HASHES.items():
  actual = sha256(PACKAGE / filename)
  assert actual == expected, f"hash mismatch for {filename}: {actual}"

parent = torch.load(PACKAGE / "model_100898.pt", map_location="cpu", weights_only=False)
latest = torch.load(PACKAGE / "model_129400.pt", map_location="cpu", weights_only=False)
for checkpoint, iteration in ((parent, 100898), (latest, 129400)):
  assert checkpoint["iter"] == iteration
  assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"} <= checkpoint.keys()

actor = latest["actor_state_dict"]
critic = latest["critic_state_dict"]
assert actor["mlp.0.weight"].shape == (512, 45)
assert actor["mlp.6.weight"].shape == (12, 128)
assert critic["mlp.0.weight"].shape == (512, 82)
assert latest["infos"]["env_state"]["common_step_counter"] == 3105720

onnx_model = onnx.load(PACKAGE / "policy.onnx")
onnx.checker.check_model(onnx_model)
assert [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim] == [1, 45]
assert [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim] == [1, 12]

torch.manual_seed(7)
observations = torch.randn(1, 45)
value = (observations - actor["obs_normalizer._mean"]) / (
  actor["obs_normalizer._std"] + 0.01
)
for layer in (0, 2, 4):
  value = torch.nn.functional.elu(
    torch.nn.functional.linear(value, actor[f"mlp.{layer}.weight"], actor[f"mlp.{layer}.bias"])
  )
torch_actions = torch.nn.functional.linear(
  value, actor["mlp.6.weight"], actor["mlp.6.bias"]
).detach().numpy()
onnx_actions = ort.InferenceSession(
  str(PACKAGE / "policy.onnx"), providers=["CPUExecutionProvider"]
).run(None, {"obs": observations.numpy()})[0]
max_error = float(np.max(np.abs(torch_actions - onnx_actions)))
assert np.allclose(torch_actions, onnx_actions, rtol=1e-5, atol=1e-6), max_error

print(f"PASS: hashes, checkpoints, dimensions, and ONNX parity (max abs error {max_error:.3e})")
