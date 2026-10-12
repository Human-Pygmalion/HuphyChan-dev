#!/usr/bin/env python3
"""Compare the learning curves of two runs over their common iterations (TensorBoard logs).

  python scripts/compare_curves.py RUN_DIR_A RUN_DIR_B [--label-a pkg --label-b ref] \\
      [--upto 1000] [--png curves.png]

Prints, per tag, the windowed mean of both runs at several iterations, and writes a PNG grid.
GPU training is not bit-deterministic, so the runs are expected to agree in trend and level,
not sample by sample.
"""

from __future__ import annotations

import argparse

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Curriculum/terrain_levels/mean",
  "Policy/mean_std",
  "Loss/value",
  "Episode_Termination/fell_over",
  "Episode_Termination/time_out",
  "Metrics/twist/error_vel_xy",
  "Metrics/twist/error_vel_yaw",
  "Curriculum/dr_levels/dr_factor",
)


def load(run: str, upto: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  ea = EventAccumulator(run, size_guidance={"scalars": 0})
  ea.Reload()
  out = {}
  for t in TAGS:
    if t in ea.Tags()["scalars"]:
      ev = [e for e in ea.Scalars(t) if e.step <= upto]
      out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
  return out


def wmean(steps: np.ndarray, vals: np.ndarray, at: int, half: int = 25) -> float:
  m = (steps >= at - half) & (steps <= at + half)
  return float(vals[m].mean()) if m.any() else float("nan")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("a")
  ap.add_argument("b")
  ap.add_argument("--label-a", default="A")
  ap.add_argument("--label-b", default="B")
  ap.add_argument("--upto", type=int, default=1000)
  ap.add_argument("--png", default="")
  args = ap.parse_args()
  A, B = load(args.a, args.upto), load(args.b, args.upto)
  marks = [m for m in (50, 300, 500, 750, args.upto - 25) if m <= args.upto]
  print(f"{'tag':38s} " + " ".join(f"{'it'+str(m):>17s}" for m in marks))
  for t in TAGS:
    if t not in A or t not in B:
      print(f"{t:38s} missing in {'A' if t not in A else 'B'}")
      continue
    cells = [f"{wmean(*A[t], m):8.3f}/{wmean(*B[t], m):<8.3f}" for m in marks]
    print(f"{t:38s} " + " ".join(cells))
  print(f"(cells: {args.label_a}/{args.label_b}, mean over +-25 iterations)")
  if args.png:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t in TAGS if t in A and t in B]
    cols = 3
    rows = -(-len(tags) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for ax, t in zip(axs.flat, tags):
      ax.plot(*A[t], lw=1, label=args.label_a)
      ax.plot(*B[t], lw=1, label=args.label_b, alpha=0.8)
      ax.set_title(t, fontsize=9)
      ax.set_xlabel("iteration", fontsize=8)
    for ax in list(axs.flat)[len(tags):]:
      ax.axis("off")
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.png, dpi=110)
    print("wrote", args.png)


if __name__ == "__main__":
  main()
