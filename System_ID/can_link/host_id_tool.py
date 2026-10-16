#!/usr/bin/env python3
"""host_id_tool.py -- read, and (guarded) change, the CAN id a RobStride motor ANSWERS on.

Why: every motor replies on the "host id" (parameter 0x200A CAN_MASTER, factory 0xFD).  Twelve
motors, two buses, six motors per bus, all answering on the same 11-bit id: replies cannot be
arbitrated against each other (docs/171 sections 5-7).  Giving each motor its own host id makes
the lowest id win arbitration and the others follow without a bit error.

Sub-commands
  read   Command 14 read of 0x200A (host id) and 0x2009 (own id) for the listed motors, ONE
         request at a time (a parameter reply does not carry the motor id).  Read-only.
  set    Command 9 for ONE motor.  DRY RUN unless --i-understand-f-cmd is given.
  save   Command 12 (write flash) for ONE motor.  DRY RUN unless --i-understand-flash-write.

Command 9 frame:   id = <motor id>        FF FF FF FF FF FF <HOST> 01
                                                           ^^^^^^ byte 6 = F_CMD = the NEW HOST ID.
!! Byte 6 is the F_CMD slot.  For Enable/Stop/Clear-fault it MUST be 0xFF; for Command 9 it is the
!! value you are setting.  Any other command byte with a non-0xFF byte 6 means a DIFFERENT command
!! (0xFD with F_CMD=2 switches the motor to the MIT protocol and takes effect after power-cycle; 12
!! motors were nearly lost to that once).  This tool never builds anything but 0x01 with a host id.

Safety rules enforced here (set):
  * exactly one motor per invocation (--motor takes one integer); no loops, no batch.
  * host id must be 0x10..0xFD, not 0x7F, not equal to ANY motor id on the bus (--bus-ids, default
    1..12 plus the motor): a reply sent with another motor's command id would be received by that
    motor as a COMMAND.  Selector ranges 0x100+ cannot occur (host id is one byte).
  * dry run is the default; the frame bytes are printed either way.
  * after sending, the tool listens on ALL ids and reads 0x200A back (replies may now come on the new id).
  * saving to flash is a separate, explicitly flagged step.

Verification after a change = power-cycle check, see `procedure()` below (also in README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import canraw  # noqa: E402
import frames as fr  # noqa: E402

HOST_MAX = 0xFD


def hexb(b: bytes) -> str:
    return " ".join(f"{x:02X}" for x in b)


def validate_host(motor: int, host: int, bus_ids: Sequence[int]) -> None:
    """Raise ValueError with the reason if this host id must not be used."""
    if not 1 <= motor <= 127:
        raise ValueError(f"motor id {motor} outside 1..127")
    if host < fr.HOST_ID_MIN:
        raise ValueError(f"host id 0x{host:02X} < 0x{fr.HOST_ID_MIN:02X}: overlaps small motor ids")
    if host > HOST_MAX or host in fr.HOST_ID_FORBIDDEN:
        raise ValueError(f"host id 0x{host:02X} is reserved (0x7F manual example target, 0xFE broadcast, 0xFF filler); "
                         f"allowed 0x{fr.HOST_ID_MIN:02X}..0x{HOST_MAX:02X}")
    ids = set(bus_ids) | {motor}
    if host in ids:
        raise ValueError(f"host id 0x{host:02X} equals motor id {host} on this bus: that motor would receive the "
                         "reply frame as a command")


def plan_set(motor: int, host: int, bus_ids: Sequence[int]) -> Tuple[int, bytes]:
    validate_host(motor, host, bus_ids)
    return fr.host_id_frame(motor, host)


def describe_set(motor: int, host: int, cid: int, data: bytes) -> str:
    return "\n".join([
        "Command 9 (Modify Host CAN ID) -- exact frame:",
        f"  11-bit id : 0x{cid:03X}   (= motor {motor}'s CAN id)",
        f"  data      : {hexb(data)}",
        f"              bytes 0-5 = FF x6   byte 6 = {data[6]:02X}  <-- F_CMD slot = NEW HOST ID 0x{host:02X} (NOT 0xFF!)",
        f"              byte 7    = {data[7]:02X}  (command 9)",
        "  WARNING: byte 6 is the F_CMD byte.  A non-FF value here is correct ONLY because byte 7 is 0x01.",
        f"  After this, motor {motor} answers on id 0x{host:03X} (was: its current host id, factory 0xFD).",
    ])


def procedure(motor: int, host: int) -> str:
    return f"""
Power-cycle verification (do this for the ONE motor you changed, before touching the next):
  1. Note the read-back above.  It is only RAM until saved ("lost in power failure" for parameter writes;
     the manual does not say whether Command 9 itself persists).
  2. If not yet saved:  host_id_tool.py save --channel <ch> --motor {motor} --i-understand-flash-write
  3. Cut the 48 V of that motor (or the whole leg) for >= 5 s, switch it on again, wait 3 s.
  4. host_id_tool.py read --channel <ch> --ids {motor}
       expect:  0x200A CAN_MASTER = 0x{host:02X}, reply arrived on id 0x{host:03X}
       if it says 0xFD again: the change did not persist -> repeat 2-4; if it still reverts, Command 9 is not
       persisted by Command 12 on this firmware -> write 0x200A with Command 15 + Command 12 instead (manual 6.18/6.15).
  5. bus_probe.py --channel <ch> --ids <all ids on the bus> : 'reply_id_histogram' must show the new id for
     motor {motor} and 0xFD for the not-yet-changed ones.  HUPHY side: RobStrideBus(..., reply_ids={{{motor}: 0x{host:02X}, ...}}).
  6. Only then the next motor.  Keep a paper list: motor -> host id.
"""


# --------------------------------------------------------------------------- bus operations
def listen(can, seconds: float) -> List[Tuple[float, int, bytes, bool]]:
    out = []
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        r = can.recv(left)
        if r is None:
            break
        out.append(r)
    return out


def read_param(can, motor: int, name: str, timeout_s: float = 0.15) -> Dict:
    """One Command 14 read.  Listens on ALL ids.  -> {'value','reply_id','rtt_ms'} or {'error':...}"""
    idx = fr.PARAMS[name][0]
    listen(can, 0.01)                                   # clear leftovers
    cid, data = fr.param_read_frame(motor, idx)
    t0 = can.send(cid, data)
    end = time.monotonic() + timeout_s
    seen = []
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        r = can.recv(left)
        if r is None:
            break
        ts, rid, rd, is_err = r
        seen.append((rid, rd, is_err))
        if not is_err and fr.is_param_reply(rd) == idx:
            return {"value": fr.parse_param_value(idx, rd), "reply_id": rid, "rtt_ms": round((ts - t0) * 1000, 3),
                    "raw": hexb(rd)}
    return {"error": "no reply", "other_frames": [(f"0x{i:X}", hexb(d), e) for i, d, e in seen[:6]]}


def do_read(can, ids: Sequence[int]) -> Dict[int, Dict]:
    res = {}
    for m in ids:
        host = read_param(can, m, "can_master")
        own = read_param(can, m, "can_id")
        res[m] = {"can_master_0x200A": host, "can_id_0x2009": own}
    return res


def print_read(res: Dict[int, Dict]) -> None:
    for m, r in res.items():
        h, o = r["can_master_0x200A"], r["can_id_0x2009"]
        if "value" in h:
            consistent = "" if h["reply_id"] == h["value"] else f"   !! reply came on 0x{h['reply_id']:X}, value says 0x{h['value']:X}"
            print(f"motor {m:>3}: CAN_MASTER(host id)=0x{h['value']:02X} ({h['value']})  reply arrived on id 0x{h['reply_id']:03X}  "
                  f"rtt {h['rtt_ms']} ms   own CAN_ID={o.get('value')}{consistent}")
        else:
            print(f"motor {m:>3}: no reply to the 0x200A read ({h.get('error')}) -- unpowered? firmware < 0.4.1.29 (Command 14 needs it)? wrong id?")


def do_set(can, motor: int, host: int, bus_ids: Sequence[int], *, execute: bool, save: bool) -> Dict:
    cid, data = plan_set(motor, host, bus_ids)
    log: Dict = {"motor": motor, "host": host, "frame": {"id": cid, "data": hexb(data)}, "executed": False}
    print(describe_set(motor, host, cid, data))
    if not execute:
        print("\nDRY RUN: nothing was sent.  To send exactly this frame add  --i-understand-f-cmd")
        return log
    before = read_param(can, motor, "can_master")
    print(f"\nbefore: {before}")
    if "value" not in before:
        raise SystemExit("the motor did not answer a 0x200A read -- not changing anything on a motor I cannot read back")
    log["before"] = before
    t = can.send(cid, data)
    log["executed"] = t is not None
    print(f"sent Command 9 to motor {motor} (new host id 0x{host:02X}); listening on ALL ids for 300 ms ...")
    frames = listen(can, 0.3)
    log["frames_after"] = [(f"0x{i:X}", hexb(d), e) for _, i, d, e in frames]
    for _, i, d, e in frames:
        print(f"  rx id=0x{i:03X} data={hexb(d)}{' ERROR-FRAME' if e else ''}")
    after = read_param(can, motor, "can_master")
    log["after"] = after
    print(f"read-back: {after}")
    if "value" in after and after["value"] == host:
        print(f"OK (RAM): motor {motor} reports CAN_MASTER=0x{host:02X}, reply came on id 0x{after['reply_id']:03X}")
    else:
        print("!! read-back does not show the new host id.  Do NOT save.  Re-read, then investigate.")
    if save:
        log["save"] = do_save(can, motor, execute=True)
    print(procedure(motor, host))
    return log


def do_save(can, motor: int, *, execute: bool) -> Dict:
    cid, data = fr.save_frame(motor)
    print(f"Command 12 (Save motor data) for motor {motor}:  id=0x{cid:03X}  data={hexb(data)}   (writes flash)")
    if not execute:
        print("DRY RUN: nothing sent.  Add  --i-understand-flash-write  to send.")
        return {"executed": False}
    can.send(cid, data)
    frames = listen(can, 0.3)
    for _, i, d, e in frames:
        print(f"  rx id=0x{i:03X} data={hexb(d)}{' ERROR-FRAME' if e else ''}")
    after = read_param(can, motor, "can_master")
    print(f"read-back after save: {after}")
    return {"executed": True, "frames": [(f"0x{i:X}", hexb(d)) for _, i, d, e in frames], "readback": after}


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--channel", default="can0")
        p.add_argument("--fake", action="store_true", help="in-process fake wire (demo/test, no hardware)")
        p.add_argument("--json", default=None, help="also write the result as JSON")

    r = sub.add_parser("read", help="read 0x200A and 0x2009 (read-only)")
    common(r)
    r.add_argument("--ids", required=True, help="comma list of motor ids, e.g. 4 or 7,8,9,10,11,12 (one request at a time)")

    s = sub.add_parser("set", help="Command 9 for ONE motor (dry run by default)")
    common(s)
    s.add_argument("--motor", type=lambda x: int(x, 0), required=True, help="ONE motor id")
    s.add_argument("--host", type=lambda x: int(x, 0), required=True, help="new host id, e.g. 0x70")
    s.add_argument("--bus-ids", default=",".join(str(i) for i in range(1, 13)),
                   help="all motor CAN ids on this bus (default 1..12); the host id may not equal any of them")
    s.add_argument("--i-understand-f-cmd", action="store_true",
                   help="actually send.  Byte 6 of this frame is the F_CMD slot and holds the new host id")
    s.add_argument("--save", action="store_true", help="after a successful read-back also send Command 12 (flash write)")

    v = sub.add_parser("save", help="Command 12 for ONE motor (dry run by default)")
    common(v)
    v.add_argument("--motor", type=lambda x: int(x, 0), required=True)
    v.add_argument("--i-understand-flash-write", action="store_true")
    return ap.parse_args(argv)


def open_can(a, fake_motors: Sequence[int]):
    if a.fake:
        import fake_can as fc
        wire = fc.FakeWire({m: fc.FakeMotor(m) for m in fake_motors}, clock=fc.RealClock())
        return fc.RawTransport(wire)
    try:
        return canraw.RawCan(a.channel, err_frames=True)
    except OSError as e:
        raise SystemExit(f"cannot open CAN channel {a.channel!r}: {e}  (ip -br link show type can ; is it up?)")


def main(argv=None) -> int:
    a = parse_args(argv)
    out = None
    if a.cmd == "read":
        ids = [int(x, 0) for x in a.ids.split(",") if x.strip()]
        can = open_can(a, ids)
        out = do_read(can, ids)
        print_read(out)
        can.close()
    elif a.cmd == "set":
        bus_ids = [int(x, 0) for x in a.bus_ids.split(",") if x.strip()]
        try:
            plan_set(a.motor, a.host, bus_ids)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        can = open_can(a, sorted(set(bus_ids) | {a.motor})) if (a.i_understand_f_cmd or a.fake) else None
        if can is None:
            out = do_set(None, a.motor, a.host, bus_ids, execute=False, save=False)
        else:
            out = do_set(can, a.motor, a.host, bus_ids, execute=a.i_understand_f_cmd, save=a.save)
            can.close()
    elif a.cmd == "save":
        can = open_can(a, [a.motor]) if (a.i_understand_flash_write or a.fake) else None
        out = do_save(can, a.motor, execute=a.i_understand_flash_write)
        if can:
            can.close()
    if a.json and out is not None:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
