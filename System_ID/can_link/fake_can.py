"""Fake CAN wire with RobStride-like motors, for tests and for `bus_probe.py --fake`.

NOTHING HERE IS A MEASUREMENT.  The delays, the 3-slot motor buffer and the optional
same-id collision model are assumptions (the 3-slot limit is the 2026-09-09 bench result in
rs_direct_socket.py; the collision model is only the hypothesis of docs/171 section 6).
The point is to exercise the *code* (probe statistics, HUPHY collect logic, host-id tool)
deterministically without hardware.

Three front-ends share one `FakeWire`:

  * `RawTransport`   - what `bus_probe.py` / `host_id_tool.py` use (kernel-timestamped frames)
  * `install_fake_can_module()` - a stand-in for the `can` (python-can) package so the
    HUPHY `CanBus` runs unchanged: `can.Message`, `can.interface.Bus`
  * `VClock`         - optional virtual clock; with it a single-threaded test runs
    instantly and bit-for-bit repeatable (reader thread OFF).  Real clock for thread tests.

Wire model (lazy discrete-event): one shared 1 Mbit/s wire; at every free instant the
pending frame with the LOWEST 11-bit id wins arbitration, so host commands (ids 1..12) beat
motor replies (0xFD) exactly as on a real bus.  A frame occupies `frame_s` (115 us).
"""

from __future__ import annotations

import heapq
import random
import struct
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

FRAME_S = 0.000115          # 8-byte standard frame incl. 3-bit intermission at 1 Mbit/s
HOST_ID_DEFAULT = 0xFD
CAN_ERR_BUSERROR = 0x080


# --------------------------------------------------------------------------- clocks
class RealClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()

    def sleep(self, d: float) -> None:
        if d > 0:
            time.sleep(d)


class VClock:
    """Virtual clock.  `sleep` just advances time, so nothing really waits."""

    def __init__(self, t0: float = 1000.0, epoch0: float = 1.7e9):
        self.now = float(t0)
        self.epoch0 = float(epoch0)
        self.t0 = float(t0)

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.epoch0 + (self.now - self.t0)

    def sleep(self, d: float) -> None:
        if d > 0:
            self.now += d

    def advance_to(self, t: float) -> None:
        if t > self.now:
            self.now = t

    def epoch_of(self, mono: float) -> float:
        return self.epoch0 + (mono - self.t0)


# --------------------------------------------------------------------------- motors
@dataclass
class FakeMotor:
    mid: int
    host_id: int = HOST_ID_DEFAULT
    proc_s: float = 0.0004          # firmware turn-around before the reply is ready
    jitter_s: float = 0.0
    buffer: int = 3                 # frames the motor can hold unanswered (2026-09-09 bench)
    # scripted behaviour: f(motor, command_index) -> extra delay seconds, or None = drop the reply
    script: Optional[Callable[["FakeMotor", int], Optional[float]]] = None
    drop_p: float = 0.0
    mode: int = 0
    flash_host_id: int = HOST_ID_DEFAULT
    n_cmd: int = 0
    n_dropped_by_buffer: int = 0
    pending: int = 0                # received, reply not yet put on the wire
    max_pending: int = 0
    wire_tx_reply_log: List[Tuple[float, int]] = field(default_factory=list)

    def power_cycle(self) -> None:
        """Host id returns to the last SAVED value (assumption: Command 12 saves it)."""
        self.host_id = self.flash_host_id
        self.mode = 0
        self.pending = 0


@dataclass(order=True)
class _Reply:
    ready: float
    seq: int
    can_id: int = field(compare=False)
    data: bytes = field(compare=False)
    mid: int = field(compare=False)
    attempts: int = field(default=0, compare=False)


class Rx:
    """One frame as the host sees it."""
    __slots__ = ("can_id", "data", "t_mono", "is_error", "t_epoch")

    def __init__(self, can_id: int, data: bytes, t_mono: float, t_epoch: float, is_error: bool = False):
        self.can_id, self.data, self.t_mono, self.t_epoch, self.is_error = can_id, data, t_mono, t_epoch, is_error

    def __repr__(self) -> str:  # pragma: no cover
        return f"Rx(0x{self.can_id:X} {self.data.hex()} @{self.t_mono:.6f}{' ERR' if self.is_error else ''})"


class FakeWire:
    def __init__(self, motors: Dict[int, FakeMotor], *, clock=None, seed: int = 1,
                 collide_same_id: bool = False, collision_max_attempts: int = 2,
                 collision_drop_p: float = 0.0, frame_s: float = FRAME_S):
        self.motors = motors
        self.clock = clock or RealClock()
        self.rng = random.Random(seed)
        self.frame_s = frame_s
        self.collide_same_id = collide_same_id
        self.collision_max_attempts = collision_max_attempts
        self.collision_drop_p = collision_drop_p
        self._t_wire = self.clock.monotonic()        # wire free from here
        self._tx: List[Tuple[float, int, int, bytes]] = []   # (queued_at, seq, id, data)
        self._ready: List[_Reply] = []
        self._rx: List[Rx] = []
        self._seq = 0
        self._lock = threading.RLock()
        self.tx_sent_log: List[Tuple[float, int, bytes]] = []  # frames that actually left the host
        self.errors_emitted = 0
        self.collisions = 0
        self.reply_log: List[Tuple[float, int]] = []     # (arrival, mid) of delivered replies

    # ---- host side ------------------------------------------------------------------
    def _epoch(self, t: float) -> float:
        clk = self.clock
        return clk.epoch_of(t) if hasattr(clk, "epoch_of") else clk.time() - (clk.monotonic() - t)

    def send(self, can_id: int, data: bytes) -> None:
        with self._lock:
            self._seq += 1
            self._tx.append((self.clock.monotonic(), self._seq, can_id & 0x7FF, bytes(data)))

    def inject_error(self, after_s: float, cls: int, data: bytes = b"\0" * 8) -> None:
        """Make the host receive an error frame `after_s` from now (class bits as in linux can/error.h)."""
        with self._lock:
            t = self.clock.monotonic() + after_s
            self._rx.append(Rx(cls, bytes(data).ljust(8, b"\0"), t, self._epoch(t), True))
            self._rx.sort(key=lambda r: r.t_mono)

    def recv(self, timeout: float) -> Optional[Rx]:
        """Next frame for the host, waiting (on the clock's sleep) up to `timeout`.

        With the real clock the wait is sliced (0.2 ms) so that frames queued by another
        thread meanwhile are noticed, like a blocking socket woken by arrival."""
        clk = self.clock
        virtual = isinstance(clk, VClock)
        deadline = clk.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                now = clk.monotonic()
                self._advance(now)
                if self._rx and self._rx[0].t_mono <= now:
                    return self._rx.pop(0)
                nxt = self._next_event_time()
            if now >= deadline:
                return None
            target = deadline if (nxt is None or nxt > deadline) else nxt
            if virtual:
                clk.advance_to(target)
            else:
                clk.sleep(max(0.0, min(target - now, 0.0002)))

    # ---- wire engine ----------------------------------------------------------------
    def _next_event_time(self) -> Optional[float]:
        cands = []
        if self._rx:
            cands.append(self._rx[0].t_mono)
        if self._tx or self._ready:
            s = self._t_wire
            q = min((x[0] for x in self._tx), default=None)
            r = self._ready[0].ready if self._ready else None
            first = min(v for v in (q, r) if v is not None)
            start = max(s, first)
            cands.append(start + self.frame_s)
        return min(cands) if cands else None

    def _advance(self, upto: float) -> None:
        """Run the wire until `upto`: all frames that FINISH by then are delivered."""
        while True:
            q = min(self._tx, key=lambda x: (x[0], x[1]), default=None)
            r = self._ready[0] if self._ready else None
            if q is None and r is None:
                return
            first = min(v for v in (q[0] if q else None, r.ready if r else None) if v is not None)
            start = max(self._t_wire, first)
            end = start + self.frame_s
            if end > upto + 1e-12:
                return
            host_cands = [x for x in self._tx if x[0] <= start]
            rep_cands = [x for x in self._ready if x.ready <= start]
            best_host = min(host_cands, key=lambda x: (x[2], x[1])) if host_cands else None
            best_rep = min(rep_cands, key=lambda x: (x.can_id, x.seq)) if rep_cands else None
            if best_host and (not best_rep or best_host[2] <= best_rep.can_id):
                self._tx.remove(best_host)
                self._t_wire = end
                self.tx_sent_log.append((start, best_host[2], best_host[3]))
                self._motor_receives(best_host[2], best_host[3], end)
                continue
            # a reply goes out; same-id contenders?
            same = [x for x in rep_cands if x.can_id == best_rep.can_id]
            if self.collide_same_id and len(same) >= 2:
                self._collide(same, start, end)
                continue
            self._ready.remove(best_rep)
            heapq.heapify(self._ready)
            self._t_wire = end
            m = self.motors[best_rep.mid]
            m.pending = max(0, m.pending - 1)
            self._deliver(best_rep.can_id, best_rep.data, end)
            self.reply_log.append((end, best_rep.mid))

    def _deliver(self, can_id: int, data: bytes, t: float, is_error: bool = False) -> None:
        self._rx.append(Rx(can_id, data, t, self._epoch(t), is_error))
        self._rx.sort(key=lambda r: r.t_mono)

    def _collide(self, same: List[_Reply], start: float, end: float) -> None:
        """Hypothesis model: >=2 replies with the SAME id start together -> bit error."""
        self.collisions += 1
        self.errors_emitted += 1
        self._t_wire = end + 0.00005
        self._deliver(CAN_ERR_BUSERROR, bytes([0, 0, 0x01, 0, 0, 0, 0, 0]), start + self.frame_s / 2, True)
        same.sort(key=lambda x: x.mid)
        for k, rep in enumerate(same):
            rep.attempts += 1
            if rep.attempts >= self.collision_max_attempts:
                # error-passive suspend: the lowest-mid reply gets through, the rest wait longer
                rep.ready = self._t_wire + (0.0 if k == 0 else 8 * 1e-6 * (k + 1) + self.frame_s * k)
                if k > 0 and self.rng.random() < self.collision_drop_p:
                    self._ready.remove(rep)
                    self.motors[rep.mid].pending = max(0, self.motors[rep.mid].pending - 1)
            else:
                rep.ready = self._t_wire
        heapq.heapify(self._ready)

    # ---- motor firmware ----------------------------------------------------------------
    def _motor_receives(self, can_id: int, data: bytes, t_end: float) -> None:
        sel, mid = can_id >> 8, can_id & 0xFF
        m = self.motors.get(mid) if sel == 0 else self.motors.get(can_id & 0xFF)
        if m is None or len(data) != 8:
            return
        if sel == 0:
            self._handle_motor_frame(m, data, t_end)
        elif sel == 3:
            self._handle_param_read(m, data, t_end)

    def _queue_reply(self, m: FakeMotor, can_id: int, data: bytes, t_end: float, *, count_cmd: bool = True) -> None:
        if m.pending >= m.buffer:
            m.n_dropped_by_buffer += 1
            return
        idx = m.n_cmd
        m.n_cmd += 1
        extra = 0.0
        if m.script is not None:
            extra = m.script(m, idx)
            if extra is None:
                return                                   # scripted loss
        if m.drop_p and self.rng.random() < m.drop_p:
            return
        ready = t_end + m.proc_s + extra + (self.rng.uniform(0, m.jitter_s) if m.jitter_s else 0.0)
        self._seq += 1
        m.pending += 1
        m.max_pending = max(m.max_pending, m.pending)
        heapq.heappush(self._ready, _Reply(ready, self._seq, can_id, data, m.mid))

    @staticmethod
    def state_frame(mid: int, q_u: int, mode: int, temp_x10: int = 250) -> bytes:
        # dq_u = tau_u = 0x800 (mid-scale = zero); byte 6 low nibble + byte 7 = temperature
        return bytes([mid, (q_u >> 8) & 0xFF, q_u & 0xFF, 0x80, 0x08, 0x00,
                      ((mode & 3) << 6) | ((temp_x10 >> 8) & 0x0F), temp_x10 & 0xFF])

    def _handle_motor_frame(self, m: FakeMotor, d: bytes, t_end: float) -> None:
        if d[:6] == b"\xff" * 6:
            f_cmd, cmd = d[6], d[7]
            if cmd == 0xFC and f_cmd == 0xFF:
                m.mode = 2
            elif cmd == 0xFD and f_cmd == 0xFF:
                m.mode = 0
            elif cmd == 0x01:                              # Command 9: new host id = F_CMD byte
                m.host_id = f_cmd
                # manual: "Response Command 2" (id = motor id, MCU id); exact bytes unverified
                self._queue_reply(m, m.mid, bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]), t_end)
                return
            elif cmd == 0xF8 and f_cmd == 0xFF:            # Command 12: save
                m.flash_host_id = m.host_id
                self._queue_reply(m, m.mid, bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]), t_end)
                return
            self._queue_reply(m, m.host_id, self.state_frame(m.mid, 0x8000, m.mode), t_end)
            return
        q_u = (d[0] << 8) | d[1]                           # echo the commanded position
        self._queue_reply(m, m.host_id, self.state_frame(m.mid, q_u, m.mode), t_end)

    def _handle_param_read(self, m: FakeMotor, d: bytes, t_end: float) -> None:
        idx = d[0] | (d[1] << 8)
        if idx == 0x200A:
            val = bytes([m.host_id, 0, 0, 0])
        elif idx == 0x2009:
            val = bytes([m.mid, 0, 0, 0])
        elif idx == 0x701C:
            val = struct.pack("<f", 48.0)
        else:
            return
        self._queue_reply(m, m.host_id, struct.pack("<H2x", idx) + val, t_end)


# --------------------------------------------------------------------------- raw front-end
class RawTransport:
    """Same two calls `bus_probe.RawCan` offers: send(can_id, data), recv(timeout)."""
    is_fake = True

    def __init__(self, wire: FakeWire):
        self.wire = wire
        self.clock = wire.clock
        self.channel = "fake0"

    def send(self, can_id: int, data: bytes) -> float:
        t = self.clock.time()
        self.wire.send(can_id, data)
        return t

    def recv(self, timeout: float):
        """-> (epoch_ts, can_id, data, is_error) or None"""
        r = self.wire.recv(timeout)
        if r is None:
            return None
        return r.t_epoch, r.can_id, r.data, r.is_error

    def close(self) -> None:
        pass

    def now(self) -> float:
        return self.clock.time()


# --------------------------------------------------------------------------- python-can stand-in
class _FakeMsg:
    def __init__(self, arbitration_id=0, data=b"", is_extended_id=False, is_error_frame=False, timestamp=0.0):
        self.arbitration_id = arbitration_id
        self.data = bytearray(data)
        self.is_extended_id = is_extended_id
        self.is_error_frame = is_error_frame
        self.timestamp = timestamp


class _FakeSock:
    def __init__(self):
        self.opts: List[Tuple[int, int, int]] = []

    def setsockopt(self, level, opt, val):
        self.opts.append((level, opt, val))


class FakePythonCanBus:
    def __init__(self, wire: FakeWire, can_filters=None):
        self.wire = wire
        self.filters = can_filters
        self.socket = _FakeSock()
        self.closed = False
        self.sent: List[_FakeMsg] = []

    def send(self, msg, timeout=None):
        self.sent.append(msg)
        self.wire.send(msg.arbitration_id, bytes(msg.data))

    def set_filters(self, filters=None):
        self.filters = filters

    def _pass(self, can_id: int) -> bool:
        if not self.filters:
            return True
        return any((can_id & f["can_mask"]) == (f["can_id"] & f["can_mask"]) for f in self.filters)

    def recv(self, timeout=None):
        clk = self.wire.clock
        deadline = clk.monotonic() + (timeout or 0.0)
        while True:
            left = max(0.0, deadline - clk.monotonic())
            r = self.wire.recv(left)
            if r is None:
                return None
            if not r.is_error and not self._pass(r.can_id):
                if clk.monotonic() >= deadline:
                    return None
                continue
            return _FakeMsg(r.can_id, r.data, False, r.is_error, r.t_epoch)

    def shutdown(self):
        self.closed = True


def install_fake_can_module(wire: FakeWire) -> types.ModuleType:
    """Put a stand-in `can` package into sys.modules (HUPHY imports it lazily in connect())."""
    mod = types.ModuleType("can")
    holder = {}

    def Bus(channel=None, interface=None, can_filters=None, **kw):
        b = FakePythonCanBus(wire, can_filters)
        holder["bus"] = b
        return b

    mod.Message = _FakeMsg
    mod.interface = types.SimpleNamespace(Bus=Bus)
    mod._holder = holder
    sys.modules["can"] = mod
    return mod


def uninstall_fake_can_module() -> None:
    sys.modules.pop("can", None)
