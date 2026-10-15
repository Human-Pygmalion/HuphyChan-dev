"""In-process simulated RobStride motor for testing the bench without hardware.

It sits behind the same transport interface as a real CAN_RAW socket: `send(16 bytes)`,
`recv(16)`, `settimeout()`, `close()` with raw Linux `struct can_frame` bytes.  Every
request is DECODED from the raw frame here (written from the manual's bit layout, not by
calling the codec's pack function in reverse), and every reply is ENCODED with the vendored
codec's quantisers, so the real encode/decode path is exercised end to end.

Physics (all numbers are for the fake only, they are NOT motor specifications):
  * one inertia J = 0.07 kg m^2, viscous 0.01 N m s/rad, Coulomb 0.05 N m (smoothed)
  * firmware PD  tau = kp (q* - q) + kd (dq* - dq) + tau_ff, clamped to +-tmax of the model
    in the driving direction, tapered linearly to zero at 1.3 x vmax (crude torque-speed
    curve); braking torque is not tapered, so stopping produces regenerated power
  * bus voltage: first-order lag (20 ms) toward 48 V + gain * regenerated watts - sag * motoring
    watts.  Over 60 V sets the overvoltage fault (fault bit 3) and the motor drops to Reset.
  * the simulation advances in real time (wall clock) every time a frame arrives.

Frames in flight: like a real motor, a 4th unanswered frame is dropped (limit measured
2026-09-09, see rs_direct_socket.py).
"""

from __future__ import annotations

import math
import socket
import struct
import time
from collections import deque

from emf_frames import (PARAM_IQF, PARAM_MECH_POS, PARAM_MECH_VEL, PARAM_VBUS, CAN_EFF_FLAG,
                        CAN_SFF_MASK)
from rscodec import mit, tables

CAN_FRAME_FMT = "=IB3x8s"
HOST_ID = 0xFD
V_NOMINAL = 48.0


class FakeMotorTransport:
    def __init__(self, model: str = "RS04", motor_id: int = 4, *, answer_params: bool = True,
                 regen_v_per_w: float = 0.08, sag_v_per_w: float = 0.01, seed_pos_deg: float = 37.0,
                 clock=time.monotonic):
        self.model = model
        self.motor_id = motor_id
        self.enc = tables.MIT_ENCODING[tables.Model(model)]
        self.answer_params = answer_params
        self.regen_v_per_w = regen_v_per_w
        self.sag_v_per_w = sag_v_per_w
        self.clock = clock
        # physical state
        self.q = math.radians(seed_pos_deg)      # rad
        self.w = 0.0                             # rad/s
        self.tau = 0.0                           # N m actually applied
        self.vbus = V_NOMINAL
        self.temp_c = 25.0
        self.power_w = 0.0                       # tau * w
        # firmware state
        self.mode = 0                            # 0 Reset, 1 Cali, 2 Motor
        self.fault_word = 0
        self.warn = False
        # last MIT command (radians)
        self.cmd = dict(q=0.0, dq=0.0, kp=0.0, kd=0.0, tff=0.0)
        self._t_last = self.clock()
        self._rx: deque[bytes] = deque()
        self.dropped = 0
        self.closed = False
        self.log: list[tuple[int, bytes]] = []   # every request frame seen (for tests)

    # ---- socket-like interface -------------------------------------------------------
    def settimeout(self, _t):
        pass

    def close(self):
        self.closed = True

    def send(self, raw: bytes) -> int:
        cid, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
        data = data[:dlc]
        self.log.append((cid, data))
        self._advance()
        if len(self._rx) >= 3:                   # 4th frame in flight is dropped
            self.dropped += 1
            return len(raw)
        if cid & CAN_EFF_FLAG:
            return len(raw)                      # private protocol frames are ignored
        cid &= CAN_SFF_MASK
        if cid == self.motor_id:
            self._handle_motor_frame(data)
        elif cid == ((tables.SEL_PARAM_READ << 8) | self.motor_id):
            self._handle_param_read(data)
        return len(raw)

    def recv(self, _n: int = 16) -> bytes:
        if not self._rx:
            raise socket.timeout("fake: nothing queued")
        return self._rx.popleft()

    # ---- request handling ------------------------------------------------------------
    def _handle_motor_frame(self, d: bytes) -> None:
        if len(d) != 8:
            return
        if d[:6] == b"\xff" * 6:
            f_cmd, command = d[6], d[7]
            if f_cmd == 0xFF and command == tables.CMD_ENABLE:
                if self.fault_word == 0:
                    self.mode = 2
                self._reply_state()
            elif f_cmd == 0xFF and command == tables.CMD_STOP:
                self.mode = 0
                self.cmd = dict(q=0.0, dq=0.0, kp=0.0, kd=0.0, tff=0.0)
                self._reply_state()
            elif f_cmd == 0xFF and command == tables.CMD_FAULT:
                self.fault_word = 0
                self.warn = False
                self._reply_state()
            elif command == tables.CMD_FAULT:          # F_CMD != 0xFF: fault query
                self._queue(HOST_ID, bytes([self.motor_id]) + struct.pack("<I", self.fault_word) + b"\0\0\0")
            else:
                # Mode change / protocol change / anything else: the fake never obeys, and
                # the bench must never send these (test_frames.py asserts it).
                self.log.append((-1, b"UNEXPECTED_F_CMD"))
            return
        # MIT dynamic parameters, decoded from the manual's bit layout (Command 3, p.38).
        q_u = (d[0] << 8) | d[1]
        dq_u = (d[2] << 4) | (d[3] >> 4)
        kp_u = ((d[3] & 0x0F) << 8) | d[4]
        kd_u = (d[5] << 4) | (d[6] >> 4)
        t_u = ((d[6] & 0x0F) << 8) | d[7]
        e = self.enc
        self.cmd = dict(
            q=mit.uint_to_float(q_u, -e.pmax_rad, e.pmax_rad, e.pos_bits),
            dq=mit.uint_to_float(dq_u, -e.vmax_rad_s, e.vmax_rad_s, e.vel_bits),
            kp=mit.uint_to_float(kp_u, 0.0, e.kp_max, e.gain_bits),
            kd=mit.uint_to_float(kd_u, 0.0, e.kd_max, e.gain_bits),
            tff=mit.uint_to_float(t_u, -e.tmax_nm, e.tmax_nm, e.tau_bits),
        )
        self._reply_state()

    def _handle_param_read(self, d: bytes) -> None:
        if not self.answer_params or len(d) < 2:
            return
        index = d[0] | (d[1] << 8)
        kt = 2.1 if self.model == "RS04" else 1.8        # fake only
        values = {PARAM_VBUS: self.vbus, PARAM_IQF: self.tau / kt,
                  PARAM_MECH_POS: self.q, PARAM_MECH_VEL: self.w}
        if index not in values:
            return
        # The real RS04 replies on the host id 0xFD (measured 2026-10-15), not 0x300|id.
        self._queue(HOST_ID, struct.pack("<H2xf", index, values[index]))

    # ---- replies ---------------------------------------------------------------------
    def _queue(self, cid: int, data: bytes) -> None:
        self._rx.append(struct.pack(CAN_FRAME_FMT, cid, len(data), data.ljust(8, b"\0")))

    def _reply_state(self) -> None:
        e = self.enc
        q = self.q
        # the motor reports position in [-4 pi, 4 pi): wrap like the real one
        span = 2 * e.pmax_rad
        q = (q + e.pmax_rad) % span - e.pmax_rad
        q_u = mit.float_to_uint(q, -e.pmax_rad, e.pmax_rad, e.pos_bits)
        dq_u = mit.float_to_uint(self.w, -e.vmax_rad_s, e.vmax_rad_s, e.vel_bits)
        t_u = mit.float_to_uint(self.tau, -e.tmax_nm, e.tmax_nm, e.tau_bits)
        temp_u = int(self.temp_c * 10) & 0x0FFF
        b6 = (self.mode << 6) | (0x20 if self.fault_word else 0) | (0x10 if self.warn else 0) | (temp_u >> 8)
        data = bytes([self.motor_id, q_u >> 8, q_u & 0xFF, dq_u >> 4,
                      ((dq_u & 0x0F) << 4) | (t_u >> 8), t_u & 0xFF, b6, temp_u & 0xFF])
        self._queue(HOST_ID, data)

    # ---- physics ---------------------------------------------------------------------
    def _advance(self) -> None:
        now = self.clock()
        dt_total = min(now - self._t_last, 0.2)
        self._t_last = now
        e = self.enc
        h = 0.0005
        n = max(1, int(dt_total / h)) if dt_total > 0 else 0
        for _ in range(n):
            self._step(h, e)

    def _step(self, h: float, e) -> None:
        c = self.cmd
        tau_cmd = 0.0
        if self.mode == 2:
            tau_cmd = c["kp"] * (c["q"] - self._wrapped_q()) + c["kd"] * (c["dq"] - self.w) + c["tff"]
            drive = tau_cmd * self.w > 0
            lim = e.tmax_nm * (max(0.0, 1.0 - abs(self.w) / (1.3 * e.vmax_rad_s)) if drive else 1.0)
            tau_cmd = max(-lim, min(lim, tau_cmd))
        self.tau = tau_cmd
        friction = 0.01 * self.w + 0.05 * math.tanh(self.w / 0.02)
        acc = (tau_cmd - friction) / 0.07
        self.w += acc * h
        self.q += self.w * h
        self.power_w = tau_cmd * self.w
        # bus voltage: negative tau*w = energy pushed back to the supply
        target_v = V_NOMINAL - self.regen_v_per_w * min(self.power_w, 0.0) - self.sag_v_per_w * max(self.power_w, 0.0)
        self.vbus += (target_v - self.vbus) * (h / 0.02)
        self.temp_c += (0.002 * tau_cmd * tau_cmd - (self.temp_c - 25.0) / 600.0) * h
        if self.vbus > 60.0 and self.fault_word == 0:
            self.fault_word |= 1 << tables.FAULT_BITS["overvoltage"]
            self.mode = 0
        self.warn = self.vbus > 55.0

    def _wrapped_q(self) -> float:
        span = 2 * self.enc.pmax_rad
        return (self.q + self.enc.pmax_rad) % span - self.enc.pmax_rad
