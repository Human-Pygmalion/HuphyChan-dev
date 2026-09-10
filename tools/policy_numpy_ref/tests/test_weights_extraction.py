import numpy as np

from policy_numpy_ref.weights import ACTION_DIM, HIDDEN_DIMS, OBS_DIM, save_npz


def test_shapes(weights):
  assert weights.obs_mean.shape == (OBS_DIM,)
  assert weights.obs_std.shape == (OBS_DIM,)
  dims = (OBS_DIM, *HIDDEN_DIMS, ACTION_DIM)
  for i, (w, b) in enumerate(zip(weights.linear_w, weights.linear_b)):
    assert w.shape == (dims[i + 1], dims[i])
    assert b.shape == (dims[i + 1],)


def test_roundtrip(weights, tmp_path):
  from policy_numpy_ref.weights import load_npz

  out = tmp_path / "roundtrip.npz"
  save_npz(weights, out)
  reloaded = load_npz(out)
  assert np.array_equal(weights.obs_mean, reloaded.obs_mean)
  assert np.array_equal(weights.obs_std, reloaded.obs_std)
  for w0, w1 in zip(weights.linear_w, reloaded.linear_w):
    assert np.array_equal(w0, w1)


def test_checkpoint_sha_matches_bake(weights, contract):
  assert weights.meta["source_sha256"] == contract["ckpt_sha256"]
