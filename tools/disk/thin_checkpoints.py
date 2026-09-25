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
  * every iteration that is NAMED anywhere in the notes or the analysis code
    (docs/, mujoco-sim/mjlab/analysis/, tools/) - e.g. model_5400.pt at a forced
    promotion, model_9000.pt measured in a note. Kept in every run, which is
    deliberately conservative: a name is cheap to keep and costly to lose.
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


REF_DIRS = ("docs", "mujoco-sim/mjlab/analysis", "tools")
REF = re.compile(r"model_(\d+)\.pt")


def referenced_iterations(repo: str) -> set[int]:
  """Iterations whose checkpoint file is named in a note or in analysis code."""
  found: set[int] = set()
  for d in REF_DIRS:
    for dp, dns, fns in os.walk(os.path.join(repo, d)):
      dns[:] = [x for x in dns if x not in ("out", "__pycache__", ".git")]
      for f in fns:
        if not f.endswith((".md", ".py", ".sh", ".json", ".yaml", ".txt")):
          continue
        try:
          with open(os.path.join(dp, f), errors="ignore") as fh:
            for m in REF.finditer(fh.read()):
              found.add(int(m.group(1)))
        except OSError:
          continue
  return found


def plan(repo: str, keep_every: int, active_hours: float):
  now = time.time()
  named = referenced_iterations(repo)
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
        if i == last or i % keep_every == 0 or i in named:
          keep_bytes += os.path.getsize(f)
        else:
          remove.append(f)
  return remove, keep_bytes, skipped_active, named


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

  remove, keep_bytes, active, named = plan(a.repo, a.keep_every, a.active_hours)
  total = sum(os.path.getsize(f) for f in remove)
  print(f"would remove {len(remove):,} intermediate checkpoints = {total / 2**30:.1f} GiB")
  print(f"keeps       {keep_bytes / 2**30:.1f} GiB (every {a.keep_every} + last per run)")
  print(f"skipped     {len(active)} run(s) with a checkpoint in the last {a.active_hours} h")
  extra = sorted(i for i in named if i % a.keep_every)
  print(f"also keeps  {len(extra)} iteration(s) named in notes/code: {extra[:40]}")
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
