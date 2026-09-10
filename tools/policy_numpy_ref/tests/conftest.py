"""Shared fixtures for the policy_numpy_ref verification suite.

All fixtures point at the baked artifacts for legonly_ab_v3_keyfix/model_27999.
If a fixture's file is missing, the depending tests are skipped (not failed) --
see docs/policy_numpy_ref/README.md for how to regenerate them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE_DIR = Path("/home/syaro/pyg_fea/pygviewer/cache")
BAKE_NAME = "LegOnly-AB__2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2__model_27999"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

WEIGHTS_NPZ = ARTIFACTS_DIR / "legonly_ab_v3_keyfix_weights.npz"
CONTRACT_JSON = CACHE_DIR / f"{BAKE_NAME}.policy_contract.json"
PARITY_NPZ = CACHE_DIR / f"{BAKE_NAME}.parity.npz"
OBS_PARITY_NPZ = CACHE_DIR / f"{BAKE_NAME}.obs_parity.npz"
ONNX_PATH = CACHE_DIR / f"{BAKE_NAME}.onnx"


def _skip_if_missing(path: Path):
  if not path.exists():
    pytest.skip(f"missing artifact: {path} (see tools/policy_numpy_ref/README.md)")


@pytest.fixture(scope="session")
def weights():
  from policy_numpy_ref.weights import load_npz

  _skip_if_missing(WEIGHTS_NPZ)
  return load_npz(WEIGHTS_NPZ)


@pytest.fixture(scope="session")
def contract():
  _skip_if_missing(CONTRACT_JSON)
  return json.loads(CONTRACT_JSON.read_text())


@pytest.fixture(scope="session")
def parity():
  _skip_if_missing(PARITY_NPZ)
  return dict(np.load(PARITY_NPZ, allow_pickle=True))


@pytest.fixture(scope="session")
def obs_parity():
  _skip_if_missing(OBS_PARITY_NPZ)
  return dict(np.load(OBS_PARITY_NPZ, allow_pickle=True))


@pytest.fixture(scope="session")
def onnx_session():
  pytest.importorskip("onnxruntime")
  import onnxruntime as ort

  _skip_if_missing(ONNX_PATH)
  return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
