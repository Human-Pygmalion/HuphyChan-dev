"""Scenario functions (virtual clock, deterministic).  Used by the unit tests and by
`run_all.py`, which writes the numbers quoted in the README to results/test_numbers.json."""

from __future__ import annotations

from typing import Dict

import harness as h

PERIOD = 0.010          # 100 Hz, HUPHY control_hz
DRAIN_S = 0.004         # the value in the archived robot logs (drain_ms=4)
LATE_EXTRA_S = 0.006    # reply arrives ~6.9 ms after the burst: after the 4 ms drain, before the next send


def late_once(pkg, *, late_motor: int = 9, glitch_cycle: int = 5, n: int = 60) -> Dict:
    """One reply (motor `late_motor`, cycle `glitch_cycle`) misses the drain window by 3 ms.

    Returns how many cycles from the glitch on contain a state that is one cycle old."""
    scripts = {late_motor: (lambda m, i: LATE_EXTRA_S if i == glitch_cycle else 0.0)}
    rig = h.Rig(pkg, scripts=scripts, drain_s=DRAIN_S)
    res = h.run_cycles(rig, n, PERIOD)
    shifted = [r["k"] for r in res if any((v or 0) >= 1 for v in r["lag"].values())]
    after = [k for k in shifted if k > glitch_cycle]
    behind = {m: sum(1 for r in res if r["k"] > glitch_cycle and (r["lag"][m] or 0) >= 1) for m in rig.ids}
    false_missing = sum(len(r["missing"]) for r in res if r["k"] > glitch_cycle)
    out = dict(
        late_motor=late_motor, cycles=n, glitch_cycle=glitch_cycle,
        shifted_cycles_after_glitch=len(after),
        cycles_after_glitch=n - glitch_cycle - 1,
        first_clean_cycle_after=(min((k for k in range(glitch_cycle + 1, n) if k not in shifted), default=None)),
        stale_cycles_per_motor=behind,
        missing_reports_after_glitch=false_missing,
    )
    rig.close()
    return out


def slow_motor(pkg, *, motor: int = 9, delay_s: float = 0.035, n: int = 60, **bus_kwargs) -> Dict:
    """One motor answers 35 ms late (3.5 cycles).  How many unanswered frames pile up at it?"""
    rig = h.Rig(pkg, proc_s=0.0004, drain_s=DRAIN_S, bus_kwargs=bus_kwargs,
                scripts={motor: (lambda m, i: delay_s)})
    res = h.run_cycles(rig, n, PERIOD)
    m = rig.motors[motor]
    sent_to_motor = sum(1 for _, cid, _ in rig.wire.tx_sent_log if cid == motor)
    others_ok = all(not (set(r["missing"]) - {motor}) for r in res)
    out = dict(motor=motor, delay_ms=delay_s * 1000, cycles=n, max_unanswered_at_motor=m.max_pending,
               frames_dropped_by_motor_buffer=m.n_dropped_by_buffer, frames_sent_to_motor=sent_to_motor,
               skipped_by_guard=getattr(getattr(rig.rs, "link", None), "skipped_commands", 0),
               other_motors_always_answered=others_ok)
    rig.close()
    return out


def all_numbers() -> Dict:
    o, p = h.orig(), h.patched()
    d = {"late_once": {"old": late_once(o), "new": late_once(p)},
         "late_once_by_motor": {"old": {}, "new": {}},
         "slow_motor": {"old": slow_motor(o), "new_K2": slow_motor(p),
                        "new_K0_guard_off": slow_motor(p, max_outstanding=0)}}
    for name, pkg in (("old", o), ("new", p)):
        for m in range(7, 13):
            r = late_once(pkg, late_motor=m)
            d["late_once_by_motor"][name][m] = r["shifted_cycles_after_glitch"]
    return d
