#!/usr/bin/env python3
"""Run a checkpoint with stock mjlab's player (`mjlab.scripts.play`) on the Pygmalion task.

  python scripts/play.py Pygmalion-Velocity-Rough-Box --checkpoint-file checkpoints/rough_main_model_11999.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygmalion_mjlab  # noqa: E402,F401  (registers the task)
from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
  main()
