#!/usr/bin/env python3
"""READ-ONLY loop-rate test: how fast can the bench loop run with VBUS read every cycle?

For each rate it runs the real bench Controller loop (the same _cycle() the web page uses) with
VBUS priority on: every cycle sends one zero-gain MIT frame (state request, no torque) and one
Command 14 VBUS read.  It NEVER sends enable, and it closes the socket directly instead of via
Motor.close() (which would send five disable frames), so the motor sees nothing but zero-gain
frames and parameter reads.  The web server must not be connected to the same motor meanwhile.

    python3 rate_test.py --channel can0 --id 4 --model RS04 --rates 200,400,500,800 --secs 10
    python3 rate_test.py --fake --rates 200,400            # no hardware

Prints one line per rate: achieved loop Hz, lost state %, lost VBUS %, worst gap, late cycles.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_server import Controller  # noqa: E402


def run_one(rate: float, secs: float, a) -> dict:
  ns = argparse.Namespace(model=a.model, id=a.id, channel=a.channel, rate=rate, kp=0.0, kd=0.0, fake=a.fake,
                          fake_no_param=False, data_dir=tempfile.mkdtemp(prefix="rate_test_"), vbus_priority=True)
  ctl = Controller(ns)
  ctl.start()
  r = ctl.submit("connect", {})
  if not r.get("ok") or ctl.motor is None:
    ctl.shutdown()
    return {"rate": rate, "error": r.get("error") or "connect failed"}
  m = ctl.motor
  m.close = lambda: m._s.close()          # read-only: no disable frames on exit
  time.sleep(1.0)                          # warm-up, not counted
  n0, ls0, lv0, ov0, sf0 = ctl.cycle_n, ctl.lost_state, ctl.lost_param[0x701C], ctl.overruns, ctl.send_fail
  t0 = time.monotonic()
  worst = 0.0
  while time.monotonic() - t0 < secs:
    time.sleep(0.25)
    worst = max(worst, ctl.rates["worst_gap_ms"])
  dt = time.monotonic() - t0
  n = ctl.cycle_n - n0
  out = dict(rate=rate, cycles=n, loop_hz=n / dt, lost_state_pct=100.0 * (ctl.lost_state - ls0) / max(n, 1),
             lost_vbus_pct=100.0 * (ctl.lost_param[0x701C] - lv0) / max(n, 1), worst_gap_ms=worst,
             late=ctl.overruns - ov0, send_fail=ctl.send_fail - sf0, mode=ctl.last_state.mode if ctl.last_state else None,
             vbus=ctl.last_param_v[0x701C], enabled=ctl.enabled)
  ctl.shutdown()
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--rates", default="200,400,500,800")
  ap.add_argument("--secs", type=float, default=10.0)
  ap.add_argument("--fake", action="store_true")
  a = ap.parse_args()
  print(f"rate_test  {a.model} id={a.id} {'FAKE' if a.fake else a.channel}  VBUS every cycle, {a.secs:g} s each, read-only", flush=True)
  print(f"{'target Hz':>9} {'loop Hz':>8} {'lost state %':>13} {'lost VBUS %':>12} {'worst gap ms':>13} {'late':>6} {'sendfail':>8}  mode  VBUS", flush=True)
  for r in a.rates.split(","):
    o = run_one(float(r), a.secs, a)
    if "error" in o:
      print(f"{o['rate']:9.0f}  ERROR {o['error']}", flush=True)
      return 1
    print(f"{o['rate']:9.0f} {o['loop_hz']:8.1f} {o['lost_state_pct']:13.2f} {o['lost_vbus_pct']:12.2f} {o['worst_gap_ms']:13.1f} "
          f"{o['late']:6d} {o['send_fail']:8d}  {o['mode']}     {o['vbus'] if o['vbus'] is None else round(o['vbus'], 2)}"
          f"{'  ENABLED?!' if o['enabled'] else ''}", flush=True)
    time.sleep(0.5)
  return 0


if __name__ == "__main__":
  sys.exit(main())
