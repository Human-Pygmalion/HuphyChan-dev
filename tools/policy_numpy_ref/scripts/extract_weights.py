#!/usr/bin/env python3
"""Extract a .pt checkpoint's actor weights into a plain numpy .npz.

Run with the mjlab venv (needs torch once, at extraction time only):

    mujoco-sim/mjlab/.venv/bin/python3 tools/policy_numpy_ref/scripts/extract_weights.py \\
        --pt mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity/2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt \\
        --out tools/policy_numpy_ref/artifacts/legonly_ab_v3_keyfix_weights.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_numpy_ref.weights import load_checkpoint_numpy, save_npz  # noqa: E402


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--pt", required=True)
  p.add_argument("--out", required=True)
  a = p.parse_args()

  weights = load_checkpoint_numpy(a.pt)
  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  save_npz(weights, a.out)
  print(f"wrote {a.out}")
  print(f"  source: {weights.meta['source_checkpoint']}")
  print(f"  sha256: {weights.meta['source_sha256']}")
  print(f"  iter:   {weights.meta['iter']}")
  print(f"  obs_mean[:3]: {weights.obs_mean[:3]}")
  print(f"  obs_std[:3]:  {weights.obs_std[:3]}")


if __name__ == "__main__":
  main()
