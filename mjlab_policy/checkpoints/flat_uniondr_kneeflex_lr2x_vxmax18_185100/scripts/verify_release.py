#!/usr/bin/env python3
"""Verify release hashes, checkpoint state, and ONNX/PyTorch actor parity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


PACKAGE = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
  "model_172200.pt": "cb1669d45e26aa8e9565f082cbcc28e472450a2302ab9332c73d75bf49e1db0d",
  "model_185100.pt": "09f8aa416dfe6c58a1ebcb3d86992bf2eac544bdf877e2809069f30647932322",
  "policy.onnx": "4b318577e3d7bfa321ce6a76c29eb119b0d222dd04fc419e0e049462f65b5b79",
  "env.yaml": "08ff4b49aa22a95b0bcbf1d8566ad2521563e0e7ebef288ce2bcae5385482542",
  "agent.yaml": "39207d3f29ad69fbc52390890886d7c932866621035e21a566331d9c352fcdcb",
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

parent = torch.load(PACKAGE / "model_172200.pt", map_location="cpu", weights_only=False)
latest = torch.load(PACKAGE / "model_185100.pt", map_location="cpu", weights_only=False)
for checkpoint, iteration, counter in (
  (parent, 172200, 4132992),
  (latest, 185100, 4442592),
):
  assert checkpoint["iter"] == iteration
  assert checkpoint["infos"]["env_state"]["common_step_counter"] == counter
  assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"} <= checkpoint.keys()

actor = latest["actor_state_dict"]
critic = latest["critic_state_dict"]
assert actor["mlp.0.weight"].shape == (512, 45)
assert actor["mlp.6.weight"].shape == (12, 128)
assert critic["mlp.0.weight"].shape == (512, 82)

onnx_model = onnx.load(PACKAGE / "policy.onnx")
onnx.checker.check_model(onnx_model)
assert [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim] == [1, 45]
assert [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim] == [1, 12]

# A zero observation keeps the parity probe in the policy's normal operating
# magnitude; arbitrary unit-Gaussian raw observations are far outside the
# normalized training distribution for several channels.
observations = torch.zeros(1, 45)
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

env_snapshot = (PACKAGE / "env.yaml").read_text()
assert "- 1.8" in env_snapshot
assert "target_height: 0.1" in env_snapshot
assert "mass_range: !!python/tuple\n      - -1.0\n      - 15.0" in env_snapshot

print(
  "PASS: hashes, full resume state, dimensions, configuration sentinels, "
  f"and ONNX parity (max abs error {max_error:.3e})"
)
