#!/usr/bin/env python3
"""bus_probe.py -- measure a CAN bus with RobStride motors WITHOUT commanding anything.

READ-ONLY BY CONSTRUCTION.  The only frames this program can send are
  * zero-gain MIT frames (kp = kd = torque = 0; checked bit by bit before every send), and
  * Command 14 parameter reads (selector 3).
No enable, no stop, no mode/zero/id/protocol frames exist in this file (see frames.py).
Zero torque is only "nothing happens" if the motor is NOT enabled: on a motor that reports
Motor mode the zero-gain frame would release it.  The program therefore first asks each motor
once and REFUSES to continue if any answers in Motor mode (override: --allow-enabled).

What it measures, per motor and per (mode, rate):
  rtt         kernel receive time of the reply minus time.time() just before send(): median/p99/max
  on-time     reply arrived BEFORE the next frame of the same kind was sent to that motor
  late        reply arrived AFTER the next send to that motor (the one-cycle-slip situation)
  lost        no reply within --expire-ms (50 ms)
  reply ids   histogram of the 11-bit arbitration id the replies came on (0xFD for all motors
              = the shared-id situation; per-motor host ids show up as separate ids)
  bus errors  `ip -details -statistics link show` + /sys/class/net/<ch>/statistics, before/after
  error frames  socket error frames (CAN_RAW_ERR_FILTER), counted by class
and the unanswered-frames-per-motor limit is respected: never more than --max-outstanding (<=3,
the 2026-09-09 measured motor buffer) unanswered frames to one motor; skipped sends are counted.

Modes
  sequential   send one frame, wait for ITS reply (or --timeout-ms), next motor
  pipelined    send all N frames back to back, then collect replies until the next cycle
               (this is what HUPHY does with send_many + collect)

Examples
  python3 bus_probe.py --channel can0 --ids 4 --model RS04 --mode both --rates 100,200,500,1000
  python3 bus_probe.py --channel canr --ids 7,8,9,10,11,12 --model-map 9:RS03,12:RS03 \\
         --mode pipelined --rates 100,200,300,500 --duration 10 --param vbus
  python3 bus_probe.py --fake --ids 7,8,9,10,11,12 --rates 100,200 --duration 1   # no hardware
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import canraw  # noqa: E402
import frames as fr  # noqa: E402

VERSION = "1.0"


# --------------------------------------------------------------------------- pure statistics
def pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def rtt_summary(rtts_s: List[float]) -> Dict[str, Optional[float]]:
    ms = [r * 1000.0 for r in rtts_s]
    return {"n": len(ms), "median": _r(statistics.median(ms) if ms else None), "p99": _r(pct(ms, 99)),
            "max": _r(max(ms) if ms else None)}


def _r(x, n=3):
    return None if x is None else round(x, n)


class Rec:
    """One frame we sent and its fate."""
    __slots__ = ("mid", "kind", "cycle", "t_send", "t_reply", "reply_id")

    def __init__(self, mid, kind, cycle, t_send):
        self.mid, self.kind, self.cycle, self.t_send = mid, kind, cycle, t_send
        self.t_reply = None
        self.reply_id = None


def classify(recs: List[Rec], expire_s: float):
    """Per (mid, kind): on-time / late / lost, from send + reply times.

    recs of one (mid, kind) in send order.  A reply is on-time when it arrived before the next
    send of the same kind to the same motor, late when after it, lost when it never came."""
    ontime = late = lost = 0
    rtts: List[float] = []
    for i, r in enumerate(recs):
        if r.t_reply is None:
            lost += 1
            continue
        rtts.append(r.t_reply - r.t_send)
        nxt = recs[i + 1].t_send if i + 1 < len(recs) else math.inf
        if r.t_reply < nxt:
            ontime += 1
        else:
            late += 1
    return ontime, late, lost, rtts


# --------------------------------------------------------------------------- the prober
class Prober:
    def __init__(self, can, ids, models, params, *, timeout_s, expire_s, max_outstanding):
        self.can = can
        self.ids = list(ids)
        self.models = models
        self.params = params                       # list of param names (frames.PARAMS)
        self.timeout_s = timeout_s
        self.expire_s = expire_s
        self.max_out = max_outstanding
        self.passive = {m: fr.passive_mit_frame(m, models[m]) for m in self.ids}

    # -- one run ------------------------------------------------------------------------
    def run(self, mode: str, rate_hz: float, duration_s: float) -> Dict:
        period = 1.0 / rate_hz
        self.recs: Dict[tuple, List[Rec]] = defaultdict(list)
        self.open: Dict[tuple, List[Rec]] = defaultdict(list)       # unanswered, oldest first
        self.out_per_motor: Dict[int, List[Rec]] = defaultdict(list)
        self.id_hist: Dict[int, Counter] = defaultdict(Counter)
        self.err = Counter()
        self.err_frames = 0
        self.foreign = Counter()
        self.unmatched = 0
        self.skipped = Counter()
        self.send_fail = 0
        sent_total = 0
        overruns = 0
        late_starts: List[float] = []

        self._drain(0.05)                                # clear anything left from before
        t0 = time.monotonic()
        t_end = t0 + duration_s
        cycle = 0
        while True:
            t_sched = t0 + cycle * period
            if t_sched >= t_end:
                break
            now = time.monotonic()
            if now < t_sched:
                self._pump(t_sched)                      # keep receiving while idle
                self._spin_until(t_sched)
            late_starts.append(max(0.0, time.monotonic() - t_sched))
            if mode == "pipelined":
                sent_total += self._send_burst(cycle)
            else:
                sent_total += self._sequential(cycle)
            if time.monotonic() > t0 + (cycle + 1) * period:
                overruns += 1
                # do not try to catch up with a burst of cycles: re-anchor the schedule
                t0 = time.monotonic() - (cycle + 1) * period
            cycle += 1
        # let the last replies in (nothing more is sent)
        self._pump(time.monotonic() + max(self.expire_s, 0.02))
        return self._summarise(mode, rate_hz, duration_s, cycle, sent_total, overruns, late_starts)

    # -- sending ------------------------------------------------------------------------
    def _allowed(self, mid: int, now_wall: float) -> bool:
        q = self.out_per_motor[mid]
        while q and now_wall - q[0].t_send > self.expire_s:
            q.pop(0)
        return len(q) < self.max_out

    def _send_one(self, mid: int, kind: str, can_id: int, data: bytes, cycle: int) -> bool:
        if not self._allowed(mid, time.time()):
            self.skipped[mid] += 1
            return False
        t = self.can.send(can_id, data)
        if t is None:
            self.send_fail += 1
            return False
        rec = Rec(mid, kind, cycle, t)
        self.recs[(mid, kind)].append(rec)
        self.open[(mid, kind)].append(rec)
        self.out_per_motor[mid].append(rec)
        return True

    def _frames_for(self, mid: int):
        yield ("state",) + self.passive[mid]
        for p in self.params:
            idx = fr.PARAMS[p][0]
            cid, data = fr.param_read_frame(mid, idx)
            yield (p, cid, data)

    def _send_burst(self, cycle: int) -> int:
        n = 0
        for mid in self.ids:
            for kind, cid, data in self._frames_for(mid):
                n += self._send_one(mid, kind, cid, data, cycle)
        self._pump(time.monotonic())                     # pick up what already arrived
        return n

    def _sequential(self, cycle: int) -> int:
        n = 0
        for mid in self.ids:
            for kind, cid, data in self._frames_for(mid):
                if not self._send_one(mid, kind, cid, data, cycle):
                    continue
                n += 1
                rec = self.recs[(mid, kind)][-1]
                self._pump(time.monotonic() + self.timeout_s, until=lambda r=rec: r.t_reply is not None)
        return n

    # -- receiving ----------------------------------------------------------------------
    def _spin_until(self, t_mono: float) -> None:
        while True:
            left = t_mono - time.monotonic()
            if left <= 0:
                return
            if left > 0.0007:
                time.sleep(left - 0.0005)

    def _drain(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.can.recv(0.005) is None:
                break

    def _pump(self, deadline_mono: float, until=None) -> None:
        while True:
            if until is not None and until():
                return
            left = deadline_mono - time.monotonic()
            if left <= 0:
                return
            r = self.can.recv(left)
            if r is None:
                return
            self._on_frame(*r)

    def _on_frame(self, ts: float, cid: int, data: bytes, is_err: bool) -> None:
        if is_err:
            self.err_frames += 1
            for name in canraw.decode_error_frame(cid, data):
                self.err[name] += 1
            return
        if cid & canraw.CAN_EFF_FLAG:
            self.foreign[f"ext:0x{cid & 0x1FFFFFFF:X}"] += 1
            return
        idx = fr.is_param_reply(data)
        if idx is not None:
            kind = fr.PARAM_BY_INDEX[idx]
            # a Command 14 reply carries the index, not the motor id: attribute to the motor whose
            # oldest unanswered read of that parameter is pending (one per motor per cycle).
            mid = self._owner_of_param(kind, ts)
            if mid is None:
                self.unmatched += 1
                return
        else:
            mid = fr.state_candidate(data, self.ids)
            kind = "state"
            if mid is None:
                self.foreign[f"id=0x{cid:X} d0={data[0] if data else -1}"] += 1
                return
        self.id_hist[mid][f"0x{cid:X}"] += 1
        q = self.open[(mid, kind)]
        while q and ts - q[0].t_send > self.expire_s:
            q.pop(0)                                     # expired -> stays 'lost'
        match = next((r for r in q if r.t_send <= ts), None)
        if match is None:
            self.unmatched += 1
            return
        q.remove(match)
        match.t_reply, match.reply_id = ts, cid
        try:
            self.out_per_motor[mid].remove(match)
        except ValueError:
            pass

    def _owner_of_param(self, kind: str, ts: float) -> Optional[int]:
        """Param replies do not name the motor.  Attribute to the motor with the OLDEST pending
        read of this parameter (replies come in the order the reads were accepted)."""
        best = None
        for mid in self.ids:
            for r in self.open[(mid, kind)]:
                if r.t_send <= ts and ts - r.t_send <= self.expire_s:
                    if best is None or r.t_send < best[1]:
                        best = (mid, r.t_send)
                    break
        return None if best is None else best[0]

    # -- results ------------------------------------------------------------------------
    def _summarise(self, mode, rate_hz, duration_s, cycles, sent_total, overruns, late_starts) -> Dict:
        per_motor = {}
        tot = Counter()
        all_rtts: List[float] = []
        for mid in self.ids:
            kinds = {}
            for kind in ["state"] + self.params:
                recs = self.recs.get((mid, kind), [])
                on, la, lo, rtts = classify(recs, self.expire_s)
                kinds[kind] = {"sent": len(recs), "ontime": on, "late": la, "lost": lo, "rtt_ms": rtt_summary(rtts)}
                if kind == "state":
                    tot.update(sent=len(recs), ontime=on, late=la, lost=lo)
                    all_rtts += rtts
            per_motor[str(mid)] = {
                **kinds["state"],
                "params": {k: v for k, v in kinds.items() if k != "state"},
                "reply_ids": dict(self.id_hist[mid]),
                "skipped_by_outstanding_limit": self.skipped[mid],
            }
        reply_ids = Counter()
        for h in self.id_hist.values():
            reply_ids.update(h)
        sent = tot["sent"] or 1
        res = {
            "mode": mode, "rate_hz": rate_hz, "duration_s": duration_s, "cycles": cycles,
            "achieved_cycle_hz": round(cycles / duration_s, 2), "overrun_cycles": overruns,
            "cycle_start_lateness_ms": {"median": _r(statistics.median(late_starts) * 1000 if late_starts else None),
                                        "p99": _r((pct(late_starts, 99) or 0) * 1000),
                                        "max": _r(max(late_starts) * 1000 if late_starts else None)},
            "frames_sent": sent_total, "send_failures": self.send_fail,
            "state": {"sent": tot["sent"], "ontime": tot["ontime"], "late": tot["late"], "lost": tot["lost"],
                      "ontime_pct": round(100 * tot["ontime"] / sent, 3), "late_pct": round(100 * tot["late"] / sent, 3),
                      "lost_pct": round(100 * tot["lost"] / sent, 3), "rtt_ms": rtt_summary(all_rtts)},
            "per_motor": per_motor,
            "reply_id_histogram": dict(reply_ids),
            "shared_reply_id": len(reply_ids) == 1 and len(self.ids) > 1,
            "foreign_frames": dict(self.foreign), "unmatched_replies": self.unmatched,
            "error_frames": {"total": self.err_frames, "by_class": dict(self.err)},
            "skipped_by_outstanding_limit": sum(self.skipped.values()),
        }
        return res


def summary_line(r: Dict, delta: Optional[Dict]) -> str:
    s = r["state"]
    ipd = (delta or {}).get("ip", {})
    be = ipd.get("bus-errors", "n/a")
    return (f"{r['mode']:<10} {r['rate_hz']:>6.0f} Hz  cycles={r['cycles']:<6} achieved={r['achieved_cycle_hz']:<7} "
            f"overrun={r['overrun_cycles']:<4} ontime={s['ontime_pct']:.2f}% late={s['late_pct']:.2f}% lost={s['lost_pct']:.2f}%  "
            f"rtt med/p99/max={s['rtt_ms']['median']}/{s['rtt_ms']['p99']}/{s['rtt_ms']['max']} ms  "
            f"errframes={r['error_frames']['total']} kernel bus-errors+={be} skipped={r['skipped_by_outstanding_limit']}")


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1] if "\n\n" in __doc__ else "")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--ids", default="4", help="comma list of motor CAN ids, e.g. 7,8,9,10,11,12")
    ap.add_argument("--model", default="RS04", help="default model for ids not in --model-map")
    ap.add_argument("--model-map", default="", help="e.g. 9:RS03,12:RS03 (RS00/RS02/RS03/RS04)")
    ap.add_argument("--mode", default="both", choices=["sequential", "pipelined", "both"])
    ap.add_argument("--rates", default="100,200,300,500", help="comma list of cycle rates in Hz")
    ap.add_argument("--duration", type=float, default=5.0, help="seconds per (mode, rate)")
    ap.add_argument("--param", default="none", help="none | vbus | vbus,iqf,mechpos,mechvel : one Command 14 read per motor per cycle")
    ap.add_argument("--timeout-ms", type=float, default=4.0, help="sequential: wait per frame for its reply")
    ap.add_argument("--expire-ms", type=float, default=50.0, help="a reply later than this counts as lost")
    ap.add_argument("--max-outstanding", type=int, default=3, help="max unanswered frames per motor (hard limit 3)")
    ap.add_argument("--pause", type=float, default=0.5, help="idle seconds between runs")
    ap.add_argument("--err-frames", dest="err_frames", action="store_true", default=True)
    ap.add_argument("--no-err-frames", dest="err_frames", action="store_false")
    ap.add_argument("--own-msgs", action="store_true", help="also receive our own sent frames (looped back)")
    ap.add_argument("--allow-enabled", action="store_true", help="continue even if a motor reports Motor mode (zero-gain frames release it!)")
    ap.add_argument("--force-sweep", action="store_true", help="run the sweep even when no motor answers the hello")
    ap.add_argument("--out", default=None, help="JSON output path (default results/probe_<channel>_<time>.json)")
    ap.add_argument("--fake", action="store_true", help="run against the in-process fake wire (no hardware)")
    ap.add_argument("--fake-collide", action="store_true", help="fake: model same-id simultaneous replies as collisions (hypothesis model)")
    return ap.parse_args(argv)


def hello(can, ids, models, timeout_s=0.05):
    """One zero-gain frame per motor, one at a time.  -> {mid: mode or None}"""
    modes = {}
    for mid in ids:
        cid, data = fr.passive_mit_frame(mid, models[mid])
        modes[mid] = None
        for _ in range(3):
            t = can.send(cid, data)
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                r = can.recv(end - time.monotonic())
                if r is None:
                    break
                ts, rid, rd, is_err = r
                if not is_err and fr.state_candidate(rd, [mid]) == mid:
                    modes[mid] = fr.state_mode(rd)
                    break
            if modes[mid] is not None:
                break
    return modes


def main(argv=None) -> int:
    a = parse_args(argv)
    ids = [int(x, 0) for x in a.ids.split(",") if x.strip()]
    if not 1 <= a.max_outstanding <= 3:
        print("--max-outstanding must be 1..3 (the motor holds 3 unanswered frames)", file=sys.stderr)
        return 2
    models = fr.parse_model_map(a.model_map, ids, a.model.upper())
    params = [] if a.param.lower() == "none" else [p.strip().lower() for p in a.param.split(",")]
    for p in params:
        if p not in fr.PARAMS or fr.PARAMS[p][0] < 0x7000:
            print(f"unknown/unsupported --param {p!r}; use vbus,iqf,mechpos,mechvel", file=sys.stderr)
            return 2
    if len(params) + 1 > a.max_outstanding and a.mode != "sequential":
        print(f"note: {1 + len(params)} frames per motor per cycle > --max-outstanding {a.max_outstanding}: "
              "some sends will be skipped by the limit", file=sys.stderr)
    rates = [float(x) for x in a.rates.split(",") if x.strip()]
    modes = ["sequential", "pipelined"] if a.mode == "both" else [a.mode]

    if a.fake:
        import fake_can as fc
        clock = fc.RealClock()
        wire = fc.FakeWire({m: fc.FakeMotor(m, proc_s=0.0004, jitter_s=0.0002) for m in ids}, clock=clock,
                           collide_same_id=a.fake_collide, seed=3)
        can = fc.RawTransport(wire)
        channel = "fake0"
    else:
        channel = a.channel
        try:
            can = canraw.RawCan(channel, err_frames=a.err_frames, own_msgs=a.own_msgs)
        except OSError as e:
            print(f"cannot open CAN channel {channel!r}: {e}  (ip -br link show type can ; is it up?)", file=sys.stderr)
            return 2

    out_path = a.out or os.path.join(HERE, "results", f"probe_{channel}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    result = {"tool": "bus_probe", "version": VERSION, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "host": platform.node(), "channel": channel, "fake": a.fake, "ids": ids, "models": models,
              "params": params, "args": vars(a), "sent_frame_kinds": ["zero-gain MIT (kp=kd=torque=0)"] +
              (["Command 14 read " + p for p in params]), "hello": {}, "runs": []}

    print(f"bus_probe {VERSION}  channel={channel}{' (FAKE)' if a.fake else ''}  motors={ids}  params={params or 'none'}")
    print("sends ONLY zero-gain MIT frames and Command 14 reads; never enable/stop/zero/id/protocol.")
    modes_seen = hello(can, ids, models)
    result["hello"] = {str(k): v for k, v in modes_seen.items()}
    answering = [m for m, v in modes_seen.items() if v is not None]
    print(f"hello: answering={answering} silent={[m for m in ids if m not in answering]} "
          f"mode(0=Reset,1=Cali,2=Motor)={ {m: v for m, v in modes_seen.items() if v is not None} }")
    enabled = [m for m, v in modes_seen.items() if v == 2]
    code = 0
    if enabled and not a.allow_enabled:
        print(f"REFUSING: motors {enabled} report Motor mode (enabled). A zero-gain frame would release them. "
              "Disable them first or pass --allow-enabled.", file=sys.stderr)
        result["aborted"] = f"enabled motors {enabled}"
        code = 3
    elif not answering and not a.force_sweep:
        print("No motor answered the hello (power? wiring? bitrate? wrong channel?). Nothing to measure; "
              "use --force-sweep to record the silence anyway.", file=sys.stderr)
        result["aborted"] = "no motor answered"
        before = canraw.snapshot(channel) if not a.fake else {"time": time.time(), "ip": {}, "sysfs": {}}
        result["adapter"] = before
        code = 2
    else:
        prober = Prober(can, ids, models, params, timeout_s=a.timeout_ms / 1000.0,
                        expire_s=a.expire_ms / 1000.0, max_outstanding=a.max_outstanding)
        snap = (lambda: {"time": time.time(), "ip": {}, "sysfs": {}}) if a.fake else (lambda: canraw.snapshot(channel))
        first = snap()
        result["adapter"] = {"before": first}
        for mode in modes:
            for rate in rates:
                b = snap()
                r = prober.run(mode, rate, a.duration)
                e = snap()
                r["kernel_counters_delta"] = canraw.snapshot_delta(b, e)
                r["summary"] = summary_line(r, r["kernel_counters_delta"])
                result["runs"].append(r)
                print(r["summary"], flush=True)
                time.sleep(a.pause)
        last = snap()
        result["adapter"]["after"] = last
        result["adapter"]["total_delta"] = canraw.snapshot_delta(first, last)
        if result["adapter"]["total_delta"].get("state_after") not in (None, "ERROR-ACTIVE"):
            print(f"WARNING: link state after the sweep = {result['adapter']['total_delta']['state_after']}")
    can.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"wrote {out_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
