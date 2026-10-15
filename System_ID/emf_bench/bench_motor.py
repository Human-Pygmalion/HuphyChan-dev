"""Motor wrapper for the bench: reuses System_ID/rs_direct_socket.Motor, adds
  * an injectable transport (real CAN_RAW socket or the fake),
  * a receive that tells state frames from Command 14 replies by CAN id,
  * Command 14 parameter read and the documented fault query.

`rs_direct_socket.Motor.recv()` decodes ANY 8-byte frame as a state frame.  That is fine on
a bus where only state frames come back, but a Command 14 reply (id 0x300 | motor_id) would
be mis-decoded as a bogus state, so this subclass classifies by CAN id first.

Import order matters: `rscodec` lives next to this file, `rs_direct_socket.py` one folder up
(System_ID/), and rs_direct_socket does `from rscodec import ...`.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from typing import Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
  if p not in sys.path:
    sys.path.insert(0, p)

import emf_frames as fr  # noqa: E402
from rs_direct_socket import CAN_FRAME_FMT, CAN_FRAME_SIZE, Motor, State  # noqa: E402
from rscodec import mit  # noqa: E402

Frame = Tuple[int, bytes]


class BenchMotor(Motor):
  def __init__(self, transport=None, channel: str = "can0", motor_id: int = 4, model: str = "RS04"):
    if transport is None:
      super().__init__(channel=channel, motor_id=motor_id, model=model)   # opens CAN_RAW
    else:
      from rscodec import tables
      self.channel = channel
      self.motor_id = int(motor_id)
      self.model = model
      self.enc = tables.MIT_ENCODING[tables.Model(model)]
      self._s = transport
      self._enabled = False

  # ---- sending ---------------------------------------------------------------------
  def send_frame(self, can_id: int, data: bytes, tries: int = 40) -> bool:
    """Like Motor._send but with an arbitrary 11-bit id (Command 14 needs 0x300|id)."""
    frame = struct.pack(CAN_FRAME_FMT, can_id, len(data), data)
    for _ in range(tries):
      try:
        self._s.send(frame)
        return True
      except OSError:
        time.sleep(0.002)
    return False

  def send_param_read(self, index: int) -> bool:
    return self.send_frame(*fr.param_read_frame(self.motor_id, index))

  def send_fault_query(self) -> bool:
    return self.send_frame(*fr.fault_query_frame(self.motor_id))

  def send_passive(self) -> bool:
    """Zero-gain MIT frame: asks for state, commands no torque."""
    return self.send_frame(self.motor_id, mit.pack_command(
        position_deg=0.0, velocity_deg_s=0.0, kp=0.0, kd=0.0, torque_nm=0.0, enc=self.enc))

  # ---- receiving -------------------------------------------------------------------
  def recv_frame(self, timeout_s: float) -> Optional[Frame]:
    deadline = time.monotonic() + timeout_s
    while True:
      left = deadline - time.monotonic()
      if left < 0:
        return None
      self._s.settimeout(max(left, 1e-4))
      try:
        raw = self._s.recv(CAN_FRAME_SIZE)
      except (OSError, TimeoutError):
        return None
      cid, dlc, data = struct.unpack(CAN_FRAME_FMT, raw)
      return cid, data[:dlc]

  def classify(self, frame: Frame) -> Optional[Union[State, fr.ParamReply]]:
    cid, data = frame
    pr = fr.parse_param_reply(cid, data, self.motor_id)
    if pr is not None:
      return pr
    if fr.is_state_candidate(cid, data, self.motor_id):
      mid, pos, vel, tau, temp = mit.decode_state(data, enc=self.enc)
      fl = mit.decode_flags(data)
      return State(motor_id=mid, position_deg=pos, velocity_deg_s=vel, torque_nm=tau,
                   temp_c=temp, stamp=time.monotonic(), mode=fl.mode,
                   fault=bool(fl.fault), warn=bool(fl.warning))
    return None

  def drain(self) -> int:
    n = 0
    while self.recv_frame(0.0005) is not None:
      n += 1
    return n

  def query_fault_word(self, timeout_s: float = 0.03) -> Optional[int]:
    """Send the documented fault query and take the first state-id frame as the answer
    (the reply shares the state frame's CAN id, so it can only be told by timing)."""
    self.drain()
    if not self.send_fault_query():
      return None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      f = self.recv_frame(deadline - time.monotonic())
      if f is None:
        return None
      cid, data = f
      if fr.is_state_candidate(cid, data, self.motor_id):
        return mit.decode_fault(data)[1]
    return None
