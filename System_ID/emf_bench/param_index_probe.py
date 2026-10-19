"""Read-only probe: which parameter indices answer a MIT Command 14 read (motor id 4, can0).
0x70xx indices return values; the 0x30xx observation table (ibus, Uq, iq, id, ...) answers with 0.0 (2026-10-19)."""
import struct, time, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "can_link"))
import canraw
b = canraw.RawCan("can0")
idx = {0x701C:"VBUS(70xx)",0x701A:"iqf(70xx)",0x300C:"VBUS(30xx)",0x302B:"ibus",0x3021:"iq",0x3022:"id",0x3027:"Uq",0x301F:"iqf(30xx)",0x302C:"torque_fdb",0x302D:"rated_i"}
for ix, name in idx.items():
    while b.recv(0.002):
        pass
    b.send((3 << 8) | 4, struct.pack("<H6x", ix))
    got = None
    t = time.time() + 0.02
    while time.time() < t:
        r = b.recv(max(0, t - time.time()))
        if r is None:
            break
        ts, cid, data, err = r
        if not err and cid == 0xFD and (data[0] | data[1] << 8) == ix:
            got = data
            break
    if got:
        print("%#06x %-12s reply %s float=%.4f" % (ix, name, got.hex(" "), struct.unpack("<f", got[4:8])[0]))
    else:
        print("%#06x %-12s NO REPLY" % (ix, name))
