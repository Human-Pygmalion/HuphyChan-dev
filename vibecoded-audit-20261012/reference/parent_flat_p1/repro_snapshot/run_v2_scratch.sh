#!/bin/bash
# V2 scratch-training launcher (docs/103 §4). Thin wrapper: cd to mjlab + use the venv
# python (the system python3 has no mjlab/mujoco), then hand every argument to the
# orchestrator. See analysis/run_v2_scratch.py for the design notes.
#
#   bash analysis/run_v2_scratch.sh --dry-run
#   bash analysis/run_v2_scratch.sh --smoke
#   nohup bash analysis/run_v2_scratch.sh --run v2s1_AB --ankle AB \
#         > analysis/out/v2s1_AB.out 2>&1 &
set -euo pipefail
cd /home/syaro/MikuchanRemote/Human-Pygmalion/mujoco-sim/mjlab
exec .venv/bin/python3 -u analysis/run_v2_scratch.py "$@"
