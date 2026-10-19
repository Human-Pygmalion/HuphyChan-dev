"""MIT-mode Command 14 probe: does the motor answer a parameter read while it runs MIT control?

User question (2026-10-19): the MIT command list ends at Command 11, so can VBUS / iqf really
be read in MIT mode?  The 2025-11-12 manuals list Commands 1-11 only.  The newer manuals
(RS04 260713, RS03 261013) add Commands 12-15, and Command 14 "Read motor parameters" needs
firmware RS04 >= 0.4.1.29 / RS03 >= 0.3.1.41.  This script tests it on the real motor.

What it does (one motor, nothing else may use the bus -- disconnect bench_server first):
  phase "off"  : motor disabled.  Each cycle: zero-gain MIT Command 3 frame, then one
                 Command 14 read (VBUS, iqf, mechPos in turn).
  phase "run"  : Command 1 (enable), the same cycles, then Command 2 (disable).
                 kp = kd = torque = 0, so the motor makes no torque.
  Every received frame is kept raw.  The report says:
    * extended (29-bit) frames seen  -> must be 0 in MIT protocol (private protocol is 29-bit)
    * state replies per MIT command, and the motor mode in them (2 = Run)
    * Command 14 replies per request, by index, with the values
    * mechPos (Command 14) minus position (state frame): two independent channels must agree

Usage (on the 3090):  python3 mit_cmd14_probe.py --channel can0 --id 4 --model RS04 \
                         [--cycles 300] [--phases off,run] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import emf_frames as fr  # noqa: E402
from bench_motor import BenchMotor  # noqa: E402
from rscodec import mit  # noqa: E402

READS = [("vbus", fr.PARAM_VBUS), ("iqf", fr.PARAM_IQF), ("mech_pos", fr.PARAM_MECH_POS)]
REPLY_WAIT_S = 0.004


def collect(m: BenchMotor, wait_s: float, raw: list) -> list:
  """Read every frame that arrives within wait_s; keep them raw."""
  out = []
  t_end = time.monotonic() + wait_s
  while True:
    f = m.recv_frame(max(0.0, t_end - time.monotonic()))
    if f is None:
      return out
    cid, data = f
    raw.append({"t": time.monotonic(), "id": cid, "ext": bool(cid & fr.CAN_EFF_FLAG), "data": data.hex(" ")})
    out.append(f)
    if time.monotonic() >= t_end:
      return out


def run_phase(m: BenchMotor, name: str, cycles: int) -> dict:
  raw: list = []
  n_cmd = n_state = 0
  modes: dict = {}
  req = {k: 0 for k, _ in READS}
  got = {k: 0 for k, _ in READS}
  vals = {k: [] for k, _ in READS}
  pos_pairs = []          # (Command 14 mechPos in deg, state position in deg)
  last_state_pos = None
  m.drain()
  if name == "run":
    m.enable()
    collect(m, 0.02, raw)
  for i in range(cycles):
    m.send_passive()                                   # MIT Command 3, all gains zero
    n_cmd += 1
    for cid, data in collect(m, REPLY_WAIT_S, raw):
      if fr.is_state_candidate(cid, data, m.motor_id) and fr.parse_param_reply(cid, data, m.motor_id) is None:
        n_state += 1
        mode = mit.decode_flags(data).mode
        modes[mode] = modes.get(mode, 0) + 1
        last_state_pos = mit.decode_state(data, enc=m.enc)[1]
    key, index = READS[i % len(READS)]
    m.send_param_read(index)
    req[key] += 1
    for cid, data in collect(m, REPLY_WAIT_S, raw):
      pr = fr.parse_param_reply(cid, data, m.motor_id)
      if pr is None:
        continue
      k = next((kk for kk, ix in READS if ix == pr.index), None)
      if k is None:
        continue
      got[k] += 1
      vals[k].append(pr.value)
      if k == "mech_pos" and last_state_pos is not None:
        pos_pairs.append((math.degrees(pr.value), last_state_pos))
    time.sleep(0.002)
  if name == "run":
    m.disable()
    collect(m, 0.02, raw)

  def stats(v):
    return None if not v else {"n": len(v), "min": min(v), "max": max(v), "mean": sum(v) / len(v)}

  diffs = [((a - b + 180.0) % 360.0) - 180.0 for a, b in pos_pairs]
  ids = {}
  for r in raw:
    key = ("ext " if r["ext"] else "std ") + f"0x{r['id'] & 0x1FFFFFFF:03X}"
    ids[key] = ids.get(key, 0) + 1
  return {
      "phase": name, "cycles": cycles,
      "mit_commands_sent": n_cmd, "state_replies": n_state,
      "state_modes": {str(k): v for k, v in sorted(modes.items())},
      "extended_frames_seen": sum(1 for r in raw if r["ext"]),
      "rx_ids": ids,
      "cmd14": {k: {"requested": req[k], "answered": got[k],
                    "no_reply_pct": round(100.0 * (1 - got[k] / req[k]), 3) if req[k] else None,
                    "values": stats(vals[k])} for k, _ in READS},
      "mechpos_minus_statepos_deg": stats(diffs),
      "raw_first": raw[:12],
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--cycles", type=int, default=300)
  ap.add_argument("--phases", default="off,run")
  ap.add_argument("--out", default="")
  a = ap.parse_args()
  m = BenchMotor(channel=a.channel, motor_id=a.id, model=a.model)
  res = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "channel": a.channel, "motor_id": a.id,
         "model": a.model, "phases": []}
  try:
    for ph in a.phases.split(","):
      res["phases"].append(run_phase(m, ph.strip(), a.cycles))
  finally:
    m.disable()
  txt = json.dumps(res, indent=1)
  if a.out:
    with open(a.out, "w") as fh:
      fh.write(txt)
  print(txt)
  return 0


if __name__ == "__main__":
  sys.exit(main())
