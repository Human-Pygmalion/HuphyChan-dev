#!/usr/bin/env python3
"""Verify the released model_99998 checkpoint, hashes, and ONNX actor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml


DEFAULT_PACKAGE = Path("checkpoints/stairs_slopes12_union_dr_99998")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def build_actor(state: dict[str, torch.Tensor]) -> torch.nn.Module:
  actor = torch.nn.Sequential(
    torch.nn.Linear(44, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, 12),
  )
  actor.load_state_dict(
    {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}
  )
  return actor.eval()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("package", nargs="?", type=Path, default=DEFAULT_PACKAGE)
  args = parser.parse_args()
  package = args.package.resolve()
  manifest = yaml.safe_load((package / "deploy.yaml").read_text())

  for name in ("model_99998.pt", "policy.onnx", "env.yaml", "agent.yaml"):
    actual = sha256(package / name)
    expected = manifest["artifacts"][name]["sha256"]
    assert actual == expected, f"{name}: {actual} != {expected}"
  print("PASS artifact SHA-256 hashes")

  checkpoint = torch.load(package / "model_99998.pt", map_location="cpu", weights_only=False)
  required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
  assert not (required - checkpoint.keys()), required - checkpoint.keys()
  assert checkpoint["iter"] == 99_998
  assert checkpoint["infos"]["env_state"]["common_step_counter"] == 2_400_048
  actor_state = checkpoint["actor_state_dict"]
  assert actor_state["mlp.0.weight"].shape == (512, 44)
  assert actor_state["mlp.6.weight"].shape == (12, 128)
  assert checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1] == 268
  print("PASS full checkpoint metadata and model dimensions")

  actor = build_actor(actor_state)
  generator = torch.Generator().manual_seed(manifest["onnx_verification"]["seed"])
  observations = torch.randn(
    manifest["onnx_verification"]["samples"], 44, generator=generator
  )
  # mjlab/rsl_rl's EmpiricalNormalization forward path divides by (_std + 0.01).
  normalized = (
    observations - actor_state["obs_normalizer._mean"]
  ) / (actor_state["obs_normalizer._std"] + 0.01)
  with torch.no_grad():
    torch_actions = actor(normalized).numpy()

  session = ort.InferenceSession(
    str(package / "policy.onnx"), providers=["CPUExecutionProvider"]
  )
  input_meta = session.get_inputs()[0]
  output_meta = session.get_outputs()[0]
  assert input_meta.shape == [1, 44]
  assert output_meta.shape == [1, 12]
  onnx_actions = np.concatenate(
    [session.run(None, {input_meta.name: obs.numpy()[None]})[0] for obs in observations]
  )
  difference = np.abs(torch_actions - onnx_actions)
  rtol = manifest["onnx_verification"]["allclose_rtol"]
  atol = manifest["onnx_verification"]["allclose_atol"]
  assert np.allclose(torch_actions, onnx_actions, rtol=rtol, atol=atol)
  print(
    "PASS ONNX parity: "
    f"max_abs={difference.max():.9g}, mean_abs={difference.mean():.9g}, "
    f"samples={len(observations)}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
