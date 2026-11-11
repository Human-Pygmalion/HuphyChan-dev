#!/usr/bin/env python3
"""Verify artifact hashes, checkpoint/ONNX parity, motion, and robot MJCF."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_actor(state: dict[str, torch.Tensor], obs: torch.Tensor) -> np.ndarray:
  value = (obs - state["obs_normalizer._mean"]) / (
    state["obs_normalizer._std"] + 0.01
  )
  for layer in (0, 2, 4):
    value = torch.nn.functional.elu(
      torch.nn.functional.linear(
        value, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"]
      )
    )
  value = torch.nn.functional.linear(
    value, state["mlp.6.weight"], state["mlp.6.bias"]
  )
  return value.numpy()


def main() -> None:
  manifest = yaml.safe_load((ROOT / "deploy.yaml").read_text())
  for relative, info in manifest["artifacts"].items():
    actual = sha256(ROOT / relative)
    assert actual == info["sha256"], (relative, actual, info["sha256"])

  checkpoint = torch.load(ROOT / "model_30000.pt", map_location="cpu", weights_only=False)
  assert checkpoint["iter"] == 30000
  assert checkpoint["infos"]["env_state"]["common_step_counter"] == 720024
  assert checkpoint["actor_state_dict"]["mlp.0.weight"].shape == (512, 114)
  assert checkpoint["actor_state_dict"]["mlp.6.weight"].shape == (12, 128)
  assert checkpoint["critic_state_dict"]["mlp.0.weight"].shape == (512, 204)

  torch.manual_seed(20261110)
  observations = torch.randn(32, 114)
  expected = checkpoint_actor(checkpoint["actor_state_dict"], observations)
  for filename in ("policy.onnx", "motion_policy.onnx"):
    session = ort.InferenceSession(str(ROOT / filename), providers=["CPUExecutionProvider"])
    outputs = []
    for index in range(len(observations)):
      inputs = {"obs": observations[index : index + 1].numpy()}
      if filename == "motion_policy.onnx":
        inputs["time_step"] = np.asarray([[index]], dtype=np.float32)
      outputs.append(session.run(None, inputs)[0])
    actual = np.concatenate(outputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

  motion = np.load(ROOT / "motions/00545_interactionmesh_50hz.npz", allow_pickle=True)
  assert motion["joint_pos"].shape == (659, 24)
  assert motion["body_pos_w"].shape == (659, 23, 3)
  assert float(motion["fps"][0]) == 50.0
  assert float(motion["max_loop_residual_m"]) < 1e-8

  rollout_manifest = yaml.safe_load(
    (ROOT / "observation_rollouts/manifest.json").read_text()
  )
  assert len(rollout_manifest["rollouts"]) == 4
  for info in rollout_manifest["rollouts"]:
    rollout_path = ROOT / "observation_rollouts" / info["file"]
    assert sha256(rollout_path) == info["sha256"]
    rollout = np.load(rollout_path)
    obs = rollout["actor_observation"]
    action = rollout["raw_action"]
    assert obs.shape == (659, 114)
    assert action.shape == (659, 12)
    np.testing.assert_array_equal(rollout["reference_frame"], np.arange(659))
    np.testing.assert_array_equal(obs[0, 102:114], np.zeros(12, dtype=np.float32))
    np.testing.assert_array_equal(obs[1:, 102:114], action[:-1])
    assert np.isfinite(obs).all() and np.isfinite(action).all()

  robot_xml = REPO / manifest["robot"]["mjcf"]
  model = mujoco.MjModel.from_xml_path(str(robot_xml))
  assert model.njnt == 25 and model.nq == 31 and model.nv == 30
  for side in ("L", "R"):
    names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
      for index in range(model.ngeom)
    }
    assert all(f"{side}_foot{index}_collision" in names for index in range(1, 13))

  print("PASS: hashes, checkpoint, both ONNX actors, motion, and capsule7plus5 MJCF")


if __name__ == "__main__":
  main()
