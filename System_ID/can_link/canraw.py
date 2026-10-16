"""Standard-library SocketCAN helpers shared by bus_probe.py and host_id_tool.py.

No python-can.  Python 3.10.  Everything here only OPENS, SENDS the frames the caller builds,
and READS; it never decides what a frame means.  The frame builders that matter for safety
(`passive_mit_frame`, `param_read_frame`, `host_id_frame`, `save_frame`) are in `frames.py`.

Kernel timestamps: the socket asks for SO_TIMESTAMPNS, so every received frame carries the
time the KERNEL queued it (CLOCK_REALTIME, ns resolution), not the time Python got round to
reading it.  Round-trip times use (kernel rx time) - (time.time() just before send()).
"""

from __future__ import annotations

import errno
import glob
import os
import re
import select
import socket
import struct
import subprocess
import time
from typing import Dict, List, Optional, Tuple

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x7FF
CAN_ERR_MASK = 0x1FFFFFFF

SOL_CAN_RAW = 101
CAN_RAW_ERR_FILTER = 2
CAN_RAW_RECV_OWN_MSGS = 4
SO_TIMESTAMPNS = 35
SCM_TIMESTAMPNS = 35

# linux/can/error.h
ERR_CLASSES = (
    (0x001, "tx_timeout"), (0x002, "lost_arbitration"), (0x004, "controller"),
    (0x008, "protocol"), (0x010, "transceiver"), (0x020, "no_ack"), (0x040, "bus_off"),
    (0x080, "bus_error"), (0x100, "restarted"), (0x200, "counters"),
)
ERR_CTRL_DETAIL = ((0x01, "rx_overflow"), (0x02, "tx_overflow"), (0x04, "rx_warning"),
                   (0x08, "tx_warning"), (0x10, "rx_passive"), (0x20, "tx_passive"), (0x40, "active"))
ERR_PROT_DETAIL = ((0x01, "bit"), (0x02, "form"), (0x04, "stuff"), (0x08, "bit0"),
                   (0x10, "bit1"), (0x20, "overload"), (0x40, "active"), (0x80, "tx"))


def decode_error_frame(can_id: int, data: bytes) -> List[str]:
    """Names of every error class/detail set in one error frame (one frame may have several)."""
    cls = can_id & 0x3FF
    data = data.ljust(8, b"\0")
    names = [n for bit, n in ERR_CLASSES if cls & bit] or ["unknown"]
    if cls & 0x004:
        names += [f"controller.{n}" for bit, n in ERR_CTRL_DETAIL if data[1] & bit]
    if cls & 0x008:
        names += [f"protocol.{n}" for bit, n in ERR_PROT_DETAIL if data[2] & bit]
    return names


class RawCan:
    """One CAN_RAW socket bound to one channel."""
    is_fake = False

    def __init__(self, channel: str, *, err_frames: bool = True, own_msgs: bool = False):
        self.channel = channel
        self.s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        self.s.setsockopt(SOL_CAN_RAW, CAN_RAW_ERR_FILTER, CAN_ERR_MASK if err_frames else 0)
        self.s.setsockopt(SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, 1 if own_msgs else 0)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.s.bind((channel,))
        self.tx_enobufs = 0

    def close(self) -> None:
        self.s.close()

    def now(self) -> float:
        return time.time()

    def send(self, can_id: int, data: bytes) -> Optional[float]:
        """Send one standard frame.  Returns the CLOCK_REALTIME time just before the call, or
        None if the kernel refused it (tx queue full for >20 ms)."""
        frame = struct.pack(CAN_FRAME_FMT, can_id & CAN_SFF_MASK, len(data), data.ljust(8, b"\0"))
        t = time.time()
        for _ in range(20):
            try:
                self.s.send(frame)
                return t
            except OSError as e:
                if e.errno not in (errno.ENOBUFS, errno.EAGAIN):
                    raise
                self.tx_enobufs += 1
                time.sleep(0.001)
        return None

    def recv(self, timeout: float) -> Optional[Tuple[float, int, bytes, bool]]:
        """-> (kernel_rx_epoch_s, can_id(11-bit or error class bits), data, is_error) or None."""
        r, _, _ = select.select([self.s], [], [], max(0.0, timeout))
        if not r:
            return None
        raw, anc, _flags, _addr = self.s.recvmsg(CAN_FRAME_SIZE, 64)
        ts = None
        for level, typ, cdata in anc:
            if level == socket.SOL_SOCKET and typ == SCM_TIMESTAMPNS and len(cdata) >= 16:
                sec, nsec = struct.unpack("@ll", cdata[:16])
                ts = sec + nsec * 1e-9
        if ts is None:
            ts = time.time()
        cid, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
        if cid & CAN_ERR_FLAG:
            return ts, cid & CAN_ERR_MASK, data[:dlc], True
        if cid & CAN_EFF_FLAG:
            return ts, cid, data[:dlc], False        # extended frames keep their flag (caller ignores)
        return ts, cid & CAN_SFF_MASK, data[:dlc], False


# --------------------------------------------------------------------------- kernel counters
def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def sysfs_stats(channel: str) -> Dict[str, int]:
    d = f"/sys/class/net/{channel}/statistics"
    out = {}
    for p in sorted(glob.glob(d + "/*")):
        v = _read_int(p)
        if v is not None:
            out[os.path.basename(p)] = v
    return out


def ip_stats(channel: str) -> Dict[str, object]:
    """Parse `ip -details -statistics link show <ch>`: state, bitrate, can error counters, RX/TX.

    Keys: state, bitrate, re-started, bus-errors, arbit-lost, error-warn, error-pass, bus-off,
    rx_*/tx_* (bytes packets errors dropped ...).  Anything unparsable is simply absent."""
    try:
        txt = subprocess.run(["ip", "-details", "-statistics", "link", "show", channel],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {"error": "ip command failed"}
    out: Dict[str, object] = {"raw_len": len(txt)}
    m = re.search(r"can state (\S+)", txt)
    if m:
        out["state"] = m.group(1)
    m = re.search(r"bitrate (\d+)", txt)
    if m:
        out["bitrate"] = int(m.group(1))
    m = re.search(r"berr-counter tx (\d+) rx (\d+)", txt)
    if m:
        out["berr_tx"], out["berr_rx"] = int(m.group(1)), int(m.group(2))
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if "re-started" in ln and "bus-errors" in ln:
            names = ln.split()
            for j in range(i + 1, min(i + 3, len(lines))):
                nums = re.findall(r"\d+", lines[j])
                if len(nums) >= len(names):
                    out.update({n: int(v) for n, v in zip(names, nums)})
                    break
        for tag in ("RX", "TX"):
            if ln.strip().startswith(tag + ":") and i + 1 < len(lines):
                cols = ln.split(":", 1)[1].split()
                nums = re.findall(r"\d+", lines[i + 1])
                if len(nums) >= len(cols):
                    out.update({f"{tag.lower()}_{n}": int(v) for n, v in zip(cols, nums)})
    return out


def snapshot(channel: str) -> Dict[str, object]:
    return {"time": time.time(), "ip": ip_stats(channel), "sysfs": sysfs_stats(channel)}


def snapshot_delta(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    """after - before for every integer counter present in both, plus the end state."""
    d: Dict[str, object] = {"seconds": round(b["time"] - a["time"], 3)}
    for grp in ("ip", "sysfs"):
        sub = {}
        for k, v in b[grp].items():
            av = a[grp].get(k)
            if isinstance(v, int) and isinstance(av, int) and k != "raw_len":
                sub[k] = v - av
        d[grp] = sub
    d["state_after"] = b["ip"].get("state")
    return d
