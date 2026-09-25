#!/usr/bin/env python3
"""Thin the intermediate checkpoints of FINISHED training runs to free disk.

Why: the shared store fills up and the training launcher refuses to start below 30 GiB
free (2026-09-25: 26 GiB left, both v10 arms refused). A 30,000-iteration run saves a
checkpoint every 100 iterations - about 300 files, 1.4 GB - and after the run is over
almost none of them is ever loaded again: measurement, deployment and the generation
viewer all use the final checkpoint.

What it keeps, per run directory:
  * the LAST checkpoint (the final policy, or the phase end, e.g. model_15999.pt)
  * every checkpoint whose iteration is a multiple of --keep-every (default 1000)
  * every checkpoint of any run that wrote a checkpoint in the last --active-hours
    (default 6) - it may still be training
It never touches notes, measurements, videos, params or event files.

DRY RUN BY DEFAULT: it only lists what it would remove. Deleting needs --apply, and is
meant to be run by a person - it is permanent.

  python3 tools/disk/thin_checkpoints.py                 # list, change nothing
  python3 tools/disk/thin_checkpoints.py --apply         # actually delete
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time

ROOTS = (
  "mujoco-sim/mjlab/logs/rsl_rl",
  "pygmalion_locomotion/logs/rsl_rl",
)
CK = re.compile(r"model_(\d+)\.pt$")


def plan(repo: str, keep_every: int, active_hours: float):
  now = time.time()
  remove, keep_bytes, skipped_active = [], 0, []
  for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(repo, root, "*", "*", ""))):
      cks = []
      for f in glob.glob(d + "model_*.pt"):
        m = CK.search(f)
        if m:
          cks.append((int(m.group(1)), f))
      if not cks:
        continue
      if now - max(os.path.getmtime(f) for _, f in cks) < active_hours * 3600:
        skipped_active.append(d)
        continue
      last = max(i for i, _ in cks)
      for i, f in cks:
        if i == last or i % keep_every == 0:
          keep_bytes += os.path.getsize(f)
        else:
          remove.append(f)
  return remove, keep_bytes, skipped_active


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
  ap.add_argument("--keep-every", type=int, default=1000)
  ap.add_argument("--active-hours", type=float, default=6.0)
  ap.add_argument("--apply", action="store_true", help="actually delete (permanent)")
  a = ap.parse_args()

  remove, keep_bytes, active = plan(a.repo, a.keep_every, a.active_hours)
  total = sum(os.path.getsize(f) for f in remove)
  print(f"would remove {len(remove):,} intermediate checkpoints = {total / 2**30:.1f} GiB")
  print(f"keeps       {keep_bytes / 2**30:.1f} GiB (every {a.keep_every} + last per run)")
  print(f"skipped     {len(active)} run(s) with a checkpoint in the last {a.active_hours} h")
  if not a.apply:
    print("\nDRY RUN - nothing deleted. Re-run with --apply to delete.")
    return
  freed = 0
  for f in remove:
    s = os.path.getsize(f)
    os.remove(f)
    freed += s
  print(f"deleted {len(remove):,} files, freed {freed / 2**30:.1f} GiB")


if __name__ == "__main__":
  main()
