#!/usr/bin/env python3
"""burst_timing.py -- when do the replies of a back-to-back burst arrive?

Question (2026-10-19): does the adapter deliver replies in ~1 ms USB batches (Raspberry Pi
slcan measurement, 2026-09-12), or one by one at wire spacing?  And does the motor still drop
the 4th unanswered frame?

READ-ONLY: sends only zero-gain MIT frames (checked by frames.assert_passive) and Command 14
reads.  Refuses to run if the motor answers in Motor mode.

For k = 1..K frames sent back to back to ONE motor (frame 1 = zero-gain MIT, frames 2..k =
Command 14 reads of VBUS, iqf, mechPos, mechVel), repeated --reps times at --rate Hz:
  * arrival of reply j  = kernel receive time - time just before the FIRST send   (ms)
  * gap between reply j and reply j-1                                           (ms)
  * replies received per burst (k sent -> how many back)
Gaps near the wire time (about 0.11-0.14 ms per 8-byte frame at 1 Mbit/s) mean "delivered one
by one"; gaps near 0 with jumps near 1 ms mean "USB batches".

  python3 burst_timing.py --channel can0 --id 4 --model RS04 --kmax 4 --reps 500
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import canraw  # noqa: E402
import frames as fr  # noqa: E402

READS = [0x701C, 0x701A, 0x7019, 0x701B]     # VBUS, iqf, mechPos, mechVel
WAIT_S = 0.008


def q(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(p / 100.0 * len(s)))], 4)


def summ(v):
    return None if not v else {"n": len(v), "p1": q(v, 1), "median": round(statistics.median(v), 4),
                               "p90": q(v, 90), "p99": q(v, 99), "max": round(max(v), 4)}


def hist(v, step=0.1, top=3.0):
    h = {}
    for x in v:
        b = min(top, int(x / step) * step)
        key = f"{b:.1f}"
        h[key] = h.get(key, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: float(kv[0])))


def burst(bus, mid, model, k):
    sent = [fr.passive_mit_frame(mid, model)] + [fr.param_read_frame(mid, READS[i]) for i in range(k - 1)]
    t0 = None
    for cid, data in sent:
        t = bus.send(cid, data)
        t0 = t if t0 is None else t0
    arr = []
    t_end = time.time() + WAIT_S
    while True:
        r = bus.recv(max(0.0, t_end - time.time()))
        if r is None:
            break
        ts, cid, data, is_err = r
        if is_err or cid != 0xFD:
            continue
        arr.append((ts - t0) * 1e3)
        if len(arr) >= k:
            # keep listening a little for duplicates, but stop once the window ends
            pass
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--id", type=int, default=4)
    ap.add_argument("--model", default="RS04")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    bus = canraw.RawCan(a.channel)
    # hello: refuse if enabled
    cid, data = fr.passive_mit_frame(a.id, a.model)
    bus.send(cid, data)
    r = bus.recv(0.05)
    if r is None:
        sys.exit("no reply from motor")
    if fr.state_mode(r[2]) == 2:
        sys.exit("motor is in Motor mode -- zero-gain frames would release it; refusing")
    time.sleep(0.05)
    while bus.recv(0.002):
        pass
    res = {"tool": "burst_timing", "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(),
           "channel": a.channel, "motor_id": a.id, "model": a.model, "reps": a.reps, "rate_hz": a.rate, "k": {}}
    period = 1.0 / a.rate
    for k in range(1, a.kmax + 1):
        got = {}
        arrival = [[] for _ in range(k)]
        gaps = [[] for _ in range(k)]
        for _ in range(a.reps):
            tn = time.time() + period
            arr = burst(bus, a.id, a.model, k)
            got[len(arr)] = got.get(len(arr), 0) + 1
            for j, t in enumerate(arr[:k]):
                arrival[j].append(t)
                if j:
                    gaps[j].append(t - arr[j - 1])
            time.sleep(max(0.0, tn - time.time()))
        res["k"][str(k)] = {
            "replies_per_burst": {str(n): c for n, c in sorted(got.items())},
            "arrival_ms": [summ(v) for v in arrival],
            "gap_ms": [summ(v) for v in gaps[1:]],
            "gap_hist_0p1ms": [hist(v) for v in gaps[1:]],
            "first_arrival_hist_0p1ms": hist(arrival[0]),
        }
        print(f"k={k}: replies/burst {res['k'][str(k)]['replies_per_burst']}  "
              f"arrival med {[x and x['median'] for x in res['k'][str(k)]['arrival_ms']]}  "
              f"gap med {[x and x['median'] for x in res['k'][str(k)]['gap_ms']]}  "
              f"gap p90 {[x and x['p90'] for x in res['k'][str(k)]['gap_ms']]}", flush=True)
    bus.close()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
