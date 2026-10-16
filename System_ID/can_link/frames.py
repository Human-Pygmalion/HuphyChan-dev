"""Frame builders for the can_link tools.  Pure functions, no I/O, unit-tested (tests/test_tools.py).

SAFETY MAP -- which frames this workspace can ever build
  passive_mit_frame     MIT Command 3 with kp = kd = torque = 0.  Commands zero torque.
                        (If the motor is ENABLED this releases it: it goes limp.  The probe
                        refuses to run on a motor that reports Motor mode unless told to.)
  param_read_frame      Command 14, selector 3, id = 0x300 | motor.  Read-only.
  host_id_frame         Command 9: id = motor, FF FF FF FF FF FF <HOST> 01.
                        !! byte 6 is "F_CMD" = the new host id, NOT 0xFF.  Only host_id_tool.py
                        builds it, only for ONE motor, only after explicit flags.
  save_frame            Command 12: id = motor, FF FF FF FF FF FF FF F8.  Writes flash.
There is deliberately NO builder for enable (0xFC), stop (0xFD), set-zero (0xFE), set-CAN-id
(0xFA), protocol switch, or any F_CMD != 0xFF on 0xFC/0xFD/0xFE/0xFA.
"""

from __future__ import annotations

import math
import os
import struct
import sys
from typing import Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from huphy_motors_biped_a1b67b1.robstride import tables  # noqa: E402
from huphy_motors_biped_a1b67b1.robstride.codec import mit  # noqa: E402

Frame = Tuple[int, bytes]

PARAMS = {                      # name -> (index, struct format of the 4 value bytes, unit)
    "vbus": (0x701C, "<f", "V"),
    "iqf": (0x701A, "<f", "A"),
    "mechpos": (0x7019, "<f", "rad"),
    "mechvel": (0x701B, "<f", "rad/s"),
    "can_master": (0x200A, "B", ""),     # host id the motor answers on (uint8)
    "can_id": (0x2009, "B", ""),         # the motor's own id (uint8)
}
PARAM_BY_INDEX = {v[0]: k for k, v in PARAMS.items()}

HOST_ID_DEFAULT = 0xFD
HOST_ID_MIN = 0x10                       # below this we refuse (overlaps small motor ids / selector space)
HOST_ID_FORBIDDEN = {0x00, 0x7F, 0xFE, 0xFF}   # 0x7F manual's example target, 0xFE private-protocol broadcast, 0xFF = "no change" filler


def parse_model_map(spec: Optional[str], ids, default_model: str) -> Dict[int, str]:
    """'7:RS04,8:RS04,9:RS03'  ->  {7: 'RS04', ...}; ids not listed get default_model."""
    out = {int(i): default_model for i in ids}
    if spec:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition(":")
            out[int(k, 0)] = v.strip().upper()
    for m, name in out.items():
        tables.Model(name)               # raises ValueError for typos
    return out


def passive_mit_frame(motor_id: int, model: str) -> Frame:
    enc = tables.encoding_for(tables.Model(model))
    data = mit.pack_command(position_deg=0.0, velocity_deg_s=0.0, kp=0.0, kd=0.0, torque_nm=0.0, enc=enc)
    assert_passive(motor_id, data, model)
    return motor_id, bytes(data)


def decode_mit_gains(data: bytes, model: str) -> Tuple[float, float, float]:
    """(kp, kd, torque) straight from the bit layout of Command 3 -- independent of pack_command."""
    enc = tables.encoding_for(tables.Model(model))
    kp_u = ((data[3] & 0x0F) << 8) | data[4]
    kd_u = (data[5] << 4) | (data[6] >> 4)
    t_u = ((data[6] & 0x0F) << 8) | data[7]
    kp = kp_u / 4095.0 * enc.kp_max
    kd = kd_u / 4095.0 * enc.kd_max
    tau = -enc.tmax_nm + t_u / 4095.0 * 2 * enc.tmax_nm
    return kp, kd, tau


def assert_passive(motor_id: int, data: bytes, model: str) -> None:
    """Hard check on every zero-gain frame the probe sends: gains 0, torque ~0, not a control frame."""
    enc = tables.encoding_for(tables.Model(model))
    if len(data) != 8 or data[:6] == b"\xff" * 6:
        raise AssertionError(f"not a MIT motion frame: {data.hex()}")
    kp, kd, tau = decode_mit_gains(data, model)
    if kp != 0.0 or kd != 0.0 or abs(tau) > 0.001 * enc.tmax_nm:
        raise AssertionError(f"frame for motor {motor_id} is not zero-gain: kp={kp} kd={kd} tau={tau}")


def param_read_frame(motor_id: int, index: int) -> Frame:
    return (tables.SEL_PARAM_READ << 8) | motor_id, struct.pack("<H6x", index)


def is_param_reply(data: bytes) -> Optional[int]:
    """index if this 8-byte frame is a Command 14 reply (measured: same host id as state frames)."""
    if len(data) < 8 or data[2] or data[3]:
        return None
    idx = data[0] | (data[1] << 8)
    return idx if idx in PARAM_BY_INDEX else None


def parse_param_value(index: int, data: bytes):
    name = PARAM_BY_INDEX[index]
    fmt = PARAMS[name][1]
    if fmt == "B":
        return data[4]
    v = struct.unpack(fmt, data[4:8])[0]
    return None if (isinstance(v, float) and math.isnan(v)) else v


def host_id_frame(motor_id: int, host_id: int) -> Frame:
    """Command 9.  Byte 6 = new host id.  Validation of the host id is host_id_tool's job."""
    return motor_id, bytes([0xFF] * 6 + [host_id & 0xFF, 0x01])


def save_frame(motor_id: int) -> Frame:
    return motor_id, bytes([0xFF] * 7 + [0xF8])


def state_candidate(data: bytes, motor_ids) -> Optional[int]:
    """motor id if this looks like an MIT state reply (data[0] = motor id, not a param reply)."""
    if len(data) < 8 or is_param_reply(data) is not None:
        return None
    return data[0] if data[0] in motor_ids else None


def state_mode(data: bytes) -> int:
    """Byte 6 bits 7-6: 0 Reset, 1 Cali, 2 Motor."""
    return (data[6] >> 6) & 0x03
