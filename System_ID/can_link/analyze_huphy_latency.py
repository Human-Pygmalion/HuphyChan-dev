#!/usr/bin/env python3
"""analyze_huphy_latency.py -- what do HUPHY's archived latency CSVs say about the CAN link?

Input: HUPHY `archivedlogs/*latency*.csv`  (written by telemetry/latency_log.py)
    # drain_ms=4 hz=200
    t,joint,latency_ms,status          status = ontime | late | lost
Row semantics (from latency_log.py / bus.py a1b67b1):
    t        seconds since the log object was made, taken in the CONTROL LOOP when collect() processed the
             reply -- i.e. it is the time of the collect() call, not of the frame's arrival.  Rows of one
             collect() call on one bus share (almost) the same t.  Lost rows have NO t (appended at the end
             of the file per joint, count = sent - received), so lost events cannot be placed in time.
    ontime   reply collected in the cycle of the command that caused it (arrival >= this cycle's send)
    late     reply whose arrival was BEFORE this cycle's send: it belongs to the previous command and was
             picked up one (or more) cycles late; latency is measured from that previous command
    lost     sent - received

What this tool adds
  * per joint: ontime / late / lost %, latency percentiles
  * cycles: rows are clustered into collect() calls per bus (gap > --gap-ms starts a new cycle), then per
    joint and cycle:  O = one on-time row, L = late row only, B = late + on-time in the same cycle,
    A = no row at all (the reply did not arrive in this collect)
  * RUN LENGTHS of consecutive cycles with a late row per joint.  Under the original drain(expect=N)
    the one-cycle slip persists until a reply is genuinely lost, so long runs = the slip; independent
    random lateness gives runs of 1-2.
  * absent -> late: P(late in cycle n+1 | absent in cycle n).  A reply that missed the drain window shows
    up late in the next cycle; this is the signature of the slip mechanism, not of a corrupted frame.
  * simultaneity: do several joints of the same bus go late/absent in the SAME cycle more often than
    independent joints would?  Observed vs exact independent expectation (Poisson-binomial), a
    circular-shift permutation test (keeps each joint's own run structure), mean pairwise phi, and the same
    between the two legs (system-wide hiccup vs bus-specific cause).
  * position in the burst: late% / lost% by the joint's index in the send order.

Usage:  python3 analyze_huphy_latency.py FILE.csv [FILE2.csv ...] [--out-dir results] [--gap-ms 1.0] [--shifts 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- loading
def load(path: str) -> Dict:
    meta = {}
    rows: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    lost: Counter = Counter()
    order: List[str] = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            meta[k] = float(v)
                        except ValueError:
                            meta[k] = v
                continue
            break
        f.seek(0)
        rd = csv.reader(l for l in f if not l.startswith("#"))
        header = next(rd)
        assert header[:4] == ["t", "joint", "latency_ms", "status"], header
        for r in rd:
            j = r[1]
            if j not in order:
                order.append(j)
            if r[3] == "lost":
                lost[j] += 1
            else:
                rows[j].append((float(r[0]), float(r[2]), r[3]))
    return {"path": path, "meta": meta, "rows": rows, "lost": lost, "joints": order}


def bus_of(joint: str) -> str:
    return joint.split("/")[0]


def pctile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def r3(x):
    return None if x is None else round(x, 3)


# --------------------------------------------------------------------------- per joint
def joint_stats(d: Dict) -> Dict[str, Dict]:
    out = {}
    for j in d["joints"]:
        rows = d["rows"][j]
        on = [lat for _, lat, s in rows if s == "ontime"]
        la = [lat for _, lat, s in rows if s == "late"]
        lost = d["lost"][j]
        sent = len(rows) + lost
        allr = on + la
        out[j] = {
            "sent": sent, "ontime": len(on), "late": len(la), "lost": lost,
            "ontime_pct": r3(100 * len(on) / sent), "late_pct": r3(100 * len(la) / sent), "lost_pct": r3(100 * lost / sent),
            "latency_ms_all": {"p50": r3(pctile(allr, 50)), "p95": r3(pctile(allr, 95)), "p99": r3(pctile(allr, 99)),
                               "max": r3(max(allr) if allr else None)},
            "latency_ms_ontime": {"p50": r3(pctile(on, 50)), "p99": r3(pctile(on, 99))},
            "latency_ms_late_rows": {"p50": r3(pctile(la, 50)), "p99": r3(pctile(la, 99))},
            # A late row whose reply took LESS than the drain window did not miss the window: it was left
            # behind because the collect ended early (frame COUNT reached by other frames).
            "late_rows_faster_than_drain_pct": (
                r3(100 * sum(1 for x in la if x < d["meta"]["drain_ms"]) / len(la))
                if la and "drain_ms" in d["meta"] else None),
        }
    return out


# --------------------------------------------------------------------------- cycles
def cluster_bus(d: Dict, bus: str, gap_s: float) -> Tuple[List[Dict], List[str]]:
    joints = [j for j in d["joints"] if bus_of(j) == bus]
    rows = sorted(((t, j, s) for j in joints for t, _, s in d["rows"][j]), key=lambda x: x[0])
    cycles: List[Dict] = []
    cur = None
    last_t = None
    for t, j, s in rows:
        if cur is None or t - last_t > gap_s:
            cur = {"t": t, "by": defaultdict(list)}
            cycles.append(cur)
        cur["by"][j].append(s)
        last_t = t
    for c in cycles:
        c["t_end"] = last_t if c is cycles[-1] else c["t"]
    return cycles, joints


def cell(statuses: Optional[List[str]]) -> str:
    if not statuses:
        return "A"
    la, on = "late" in statuses, "ontime" in statuses
    return "B" if la and on else ("L" if la else "O")


def runs_of(flags: List[bool]) -> List[Tuple[int, int]]:
    """[(start_index, length)] of maximal True runs."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def run_hist(lengths: List[int]) -> Dict[str, int]:
    bins = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 10 ** 9)]
    return {n: sum(1 for L in lengths if lo <= L <= hi) for n, lo, hi in bins}


def poisson_binomial(ps: List[float]) -> List[float]:
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - p)
            new[k + 1] += v * p
        dist = new
    return dist


def phi(a: List[int], b: List[int]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def simultaneity(ind: Dict[str, List[int]], shifts: int, rng: random.Random) -> Dict:
    """ind: joint -> 0/1 per cycle.  Observed vs independence for k = joints flagged in the same cycle."""
    joints = list(ind)
    n = len(ind[joints[0]])
    ks = [sum(ind[j][c] for j in joints) for c in range(n)]
    obs_hist = Counter(ks)
    ps = [sum(ind[j]) / n for j in joints]
    pb = poisson_binomial(ps)
    exp_hist = {k: pb[k] * n for k in range(len(pb))}

    def tail(k0):
        return sum(1 for k in ks if k >= k0)

    out = {"cycles": n, "marginal_rate_per_joint": {j: r3(p) for j, p in zip(joints, ps)},
           "observed_hist": {str(k): obs_hist.get(k, 0) for k in range(len(joints) + 1)},
           "expected_hist_if_independent": {str(k): round(v, 1) for k, v in exp_hist.items()}}
    for k0 in (2, 3, 4):
        if k0 > len(joints):
            continue
        e = sum(pb[k0:]) * n
        out[f"cycles_with_ge_{k0}"] = {"observed": tail(k0), "expected_independent": round(e, 1),
                                       "ratio": r3(tail(k0) / e) if e > 0 else None}
    # circular-shift permutation test on the >=3 statistic (keeps each joint's run structure)
    k0 = 3 if len(joints) >= 3 else 2
    obs = tail(k0)
    ge = 0
    for _ in range(shifts):
        sh = {j: ind[j][s:] + ind[j][:s] for j in joints for s in [rng.randrange(1, n)]}
        if sum(1 for c in range(n) if sum(sh[j][c] for j in joints) >= k0) >= obs:
            ge += 1
    out["permutation_test"] = {"statistic": f"cycles with >= {k0} joints flagged", "observed": obs,
                               "p_value": round((ge + 1) / (shifts + 1), 4), "shifts": shifts}
    phis = [phi(ind[a], ind[b]) for i, a in enumerate(joints) for b in joints[i + 1:]]
    phis = [x for x in phis if x is not None]
    out["mean_pairwise_phi"] = r3(statistics.mean(phis)) if phis else None
    return out


def analyse_cycles(d: Dict, gap_s: float, shifts: int, seed: int = 7) -> Dict:
    rng = random.Random(seed)
    res: Dict[str, Dict] = {}
    cyc_by_bus = {}
    buses = list(dict.fromkeys(bus_of(j) for j in d["joints"]))
    for bus in buses:
        cycles, joints = cluster_bus(d, bus, gap_s)
        cyc_by_bus[bus] = cycles
        states = {j: [cell(c["by"].get(j)) for c in cycles] for j in joints}
        late_flag = {j: [s in ("L", "B") for s in states[j]] for j in joints}
        abs_flag = {j: [s == "A" for s in states[j]] for j in joints}
        per_joint = {}
        all_lens: List[int] = []
        for j in joints:
            rl = runs_of(late_flag[j])
            lens = [L for _, L in rl]
            all_lens += lens
            nlate = sum(late_flag[j])
            longest = sorted(rl, key=lambda x: -x[1])[:3]
            # absent -> late next cycle
            a_idx = [i for i, f in enumerate(abs_flag[j][:-1]) if f]
            nxt_late = sum(1 for i in a_idx if late_flag[j][i + 1])
            base = nlate / max(1, len(cycles))
            per_joint[j] = {
                "cycles_seen": len(cycles), "late_cycles": nlate, "absent_cycles": sum(abs_flag[j]),
                "state_counts": dict(Counter(states[j])),
                "late_runs": {"n_runs": len(lens), "max": max(lens) if lens else 0,
                              "mean": r3(statistics.mean(lens)) if lens else 0, "hist": run_hist(lens),
                              "late_cycles_in_runs_ge3_pct": r3(100 * sum(L for L in lens if L >= 3) / nlate) if nlate else 0.0,
                              "longest": [{"start_t": r3(cycles[s]["t"]), "start_cycle": s, "length": L} for s, L in longest]},
                "absent_then_late_next": {"absent_events": len(a_idx), "followed_by_late": nxt_late,
                                          "pct": r3(100 * nxt_late / len(a_idx)) if a_idx else None,
                                          "baseline_late_pct": r3(100 * base)},
            }
        sim_late = simultaneity({j: [int(x) for x in late_flag[j]] for j in joints}, shifts, rng)
        sim_abs = simultaneity({j: [int(x) for x in abs_flag[j]] for j in joints}, shifts, rng)
        res[bus] = {"joints": joints, "cycles": len(cycles), "per_joint": per_joint,
                    "late_run_hist_all_joints": run_hist(all_lens), "late_run_max_all_joints": max(all_lens) if all_lens else 0,
                    "simultaneity_late": sim_late, "simultaneity_absent": sim_abs,
                    "by_send_order": [{"index": i, "joint": j,
                                       "late_cycles_pct": r3(100 * sum(late_flag[j]) / len(cycles)),
                                       "absent_cycles_pct": r3(100 * sum(abs_flag[j]) / len(cycles))}
                                      for i, j in enumerate(joints)]}
    if len(buses) == 2:
        res["between_buses"] = cross_bus(cyc_by_bus[buses[0]], cyc_by_bus[buses[1]], buses, d["meta"].get("hz"))
    return res


def cross_bus(ca: List[Dict], cb: List[Dict], names: List[str], hz) -> Dict:
    """Pair the collect() calls of the two buses that belong to the same control-loop iteration.

    HUPHY collects bus A, then bus B, in one loop iteration (biped.collect), so B's call comes
    after A's, by at most the drain time.  Pair each A call with the first unused B call that
    follows it within 0.9 periods; then ask whether lateness on the two buses coincides
    (a host-side hiccup hits both buses; a bus-specific cause does not)."""
    period = 1.0 / hz if hz else 0.01
    pairs = []
    j = 0
    for c in ca:
        while j < len(cb) and cb[j]["t"] < c["t"] - 1e-4:
            j += 1
        if j < len(cb) and cb[j]["t"] - c["t"] < 0.9 * period:
            pairs.append((c, cb[j]))
            j += 1

    def kcount(c):
        return sum(1 for st in c["by"].values() if "late" in st)

    ka = [kcount(a) for a, _ in pairs]
    kb = [kcount(b) for _, b in pairs]
    fa = [int(x > 0) for x in ka]
    fb = [int(x > 0) for x in kb]
    both = sum(1 for x, y in zip(fa, fb) if x and y)
    pa, pb_ = (sum(fa) / len(fa), sum(fb) / len(fb)) if pairs else (0, 0)
    return {"buses": names, "paired_cycles": len(pairs), "cycles_in_a": len(ca), "cycles_in_b": len(cb),
            "any_late_a_pct": r3(100 * pa), "any_late_b_pct": r3(100 * pb_),
            "both_any_late_observed": both, "both_expected_if_independent": round(pa * pb_ * len(pairs), 1),
            "pearson_late_counts": r3(phi(ka, kb)) if pairs else None}


# --------------------------------------------------------------------------- report
def report(d: Dict, js: Dict, cy: Dict) -> str:
    L = []
    m = d["meta"]
    L.append(f"== {os.path.basename(d['path'])}   drain_ms={m.get('drain_ms')} hz={m.get('hz')}")
    L.append(f"{'joint':<24}{'sent':>7}{'ontime%':>9}{'late%':>7}{'lost%':>7}{'p50':>7}{'p99':>7}{'max':>8}{'late<drain%':>12}   late-run max / hist(1,2,3,4-5,6-10,11-20,21+) / %late-cycles-in-runs>=3")
    for j in d["joints"]:
        s = js[j]
        cj = cy[bus_of(j)]["per_joint"][j]["late_runs"]
        h = cj["hist"]
        L.append(f"{j:<24}{s['sent']:>7}{s['ontime_pct']:>9.2f}{s['late_pct']:>7.2f}{s['lost_pct']:>7.2f}"
                 f"{s['latency_ms_all']['p50']:>7}{s['latency_ms_all']['p99']:>7}{s['latency_ms_all']['max']:>8}"
                 f"{s['late_rows_faster_than_drain_pct']:>12}"
                 f"   {cj['max']:>3} / {' '.join(str(h[k]) for k in h)} / {cj['late_cycles_in_runs_ge3_pct']}")
    for bus in [b for b in cy if b != "between_buses"]:
        c = cy[bus]
        L.append(f"-- {bus}: {c['cycles']} collect cycles; longest late run over all joints = {c['late_run_max_all_joints']} cycles")
        for tag, sim in (("late", c["simultaneity_late"]), ("absent", c["simultaneity_absent"])):
            g3 = sim.get("cycles_with_ge_3")
            L.append(f"   simultaneity ({tag}): cycles with >=3 joints {tag}: observed {g3['observed']} vs independent {g3['expected_independent']}"
                     f" (x{g3['ratio']}), permutation p={sim['permutation_test']['p_value']}, mean pairwise phi={sim['mean_pairwise_phi']}")
        an = [(j, v['absent_then_late_next']) for j, v in c["per_joint"].items() if v['absent_then_late_next']['pct'] is not None]
        L.append("   absent -> late next cycle: " + ", ".join(f"{j.split('/')[1]} {a['pct']}% (base {a['baseline_late_pct']}%)" for j, a in an))
    if "between_buses" in cy:
        b = cy["between_buses"]
        L.append(f"-- between {b['buses']}: paired cycles {b['paired_cycles']}; any-late both: observed {b['both_any_late_observed']} vs independent {b['both_expected_if_independent']}; "
                 f"pearson(late counts)={b['pearson_late_counts']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--gap-ms", type=float, default=1.0, help="rows closer than this belong to one collect() call")
    ap.add_argument("--shifts", type=int, default=2000)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    for path in a.files:
        d = load(path)
        js = joint_stats(d)
        cy = analyse_cycles(d, a.gap_ms / 1000.0, a.shifts)
        txt = report(d, js, cy)
        print(txt)
        print()
        base = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(a.out_dir, f"analysis_{base}.json"), "w") as f:
            json.dump({"file": os.path.basename(path), "meta": d["meta"], "joints": js, "cycles": cy}, f, indent=1)
        with open(os.path.join(a.out_dir, f"analysis_{base}.txt"), "w") as f:
            f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
