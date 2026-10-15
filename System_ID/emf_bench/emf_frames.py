"""Frame builders and parsers for the back-EMF bench (no hardware access here).

Everything that decides the *bytes of a frame* lives in this file so it can be unit
tested without a CAN adapter (see test_frames.py).  MIT state/command frames themselves
come from the vendored HUPHY codec (`rscodec/`); this file adds what that codec does not
have: the MIT parameter-read (Command 14) request/reply and the guarded control frames.

Manual references (RobStride user manuals, GitHub RobStride/Product_Information):
  RS04  RS04User Manual260713.pdf   printed page 82-83  Section 6.17 "Command 14"
        parameter table             printed page 50-51  (0x7019 mechPos, 0x701A iqf,
                                                          0x701B mechVel, 0x701C VBUS)
  RS03  RS03User Manual261013.pdf   printed page 79-80  Section 6.17 "Command 14"
        parameter table             printed page 48-49
  The copies in Motor_Spec/manuals/*_251112.pdf are an OLDER edition: their MIT chapter
  stops at Command 11 (p.38-41) and has no Command 14.  Only the parameter table
  (p.26 of 251112, same indices) and the private-protocol "Type 17" read example (p.27)
  exist there.
"""

from __future__ import annotations

import math
import struct
from typing import NamedTuple, Optional

from rscodec import tables

# --------------------------------------------------------------------------------------
# F_CMD (data[6]) -- READ THIS BEFORE BUILDING ANY NEW FRAME.
#
# For the "all 0xFF" control frames
#     data = FF FF FF FF FF FF  F_CMD  COMMAND
# the byte F_CMD must be 0xFF for the ordinary action:
#     COMMAND 0xFC + F_CMD 0xFF  = enable
#     COMMAND 0xFD + F_CMD 0xFF  = disable (stop)
#     COMMAND 0xFB + F_CMD 0xFF  = clear fault
# ANY OTHER F_CMD value turns the same command byte into a different command:
#     0xFC + F_CMD=n  -> change control mode
#     0xFD + F_CMD=n  -> change COMMUNICATION PROTOCOL (private/CANopen/MIT); it takes
#                        effect after a power cycle -- this once nearly bricked 12 motors
#     0xFB + F_CMD=n  -> fault *query* instead of fault clear (harmless, documented)
# So this module builds control frames only through `control_frame()`, which pins
# F_CMD = 0xFF, and the one exception is `fault_query_frame()` (0xFB with F_CMD 0x00).
# Nothing here ever builds 0xFC/0xFD with F_CMD != 0xFF.
# --------------------------------------------------------------------------------------
F_CMD_NORMAL = tables.F_CMD_DEFAULT          # 0xFF
assert F_CMD_NORMAL == 0xFF

_CONTROL_COMMANDS = {
    "enable": tables.CMD_ENABLE,        # 0xFC
    "disable": tables.CMD_STOP,         # 0xFD
    "clear_fault": tables.CMD_FAULT,    # 0xFB
}

# --------------------------------------------------------------------------------------
# Command 14 "Read motor parameters" (MIT protocol, 11-bit standard frame)
#   request  11-bit ID = (3 << 8) | motor_id      (selector 3 = tables.SEL_PARAM_READ)
#            Byte0~1 = parameter index, LITTLE endian (low byte first)
#            Byte2~3 = 00, Byte4~7 = 00
#   reply    11-bit ID = (3 << 8) | motor_id      (same as request -- manual shows
#            "3 | 7F" for both), Byte0~1 = index, Byte4~7 = value (float32, little endian
#            -- the manual's reply table only says "low byte first"; the byte position
#            Byte4~7 follows the private-protocol Type 17 example in the manual
#            (251112 p.27: reply "1E 70 00 00 00 00 F0 41" = loc_kp 30.0) and
#            the Command 15 layout.  NOT yet confirmed on a real motor with this code.)
#   Firmware requirement printed in the manual headings of 6.15-6.18:
#            RS04 >= 0.4.1.29     RS03 >= 0.3.1.41     (older firmware: no reply)
# --------------------------------------------------------------------------------------
SEL_PARAM_READ = tables.SEL_PARAM_READ      # 3
assert SEL_PARAM_READ == 3

PARAM_FIRMWARE_MIN = {"RS04": "0.4.1.29", "RS03": "0.3.1.41"}

# Parameter indices (read-only, float32). Same index in RS03 and RS04 tables.
PARAM_MECH_POS = 0x7019   # mechPos  rad   (RS04 p.50, RS03 p.48)
PARAM_IQF = 0x701A        # iqf      A     filtered q-axis current (RS04 -90..90 A, RS03 -43..43 A)
PARAM_MECH_VEL = 0x701B   # mechVel  rad/s
PARAM_VBUS = 0x701C       # VBUS     V     bus voltage (RS04 p.51, RS03 p.49)

PARAM_NAMES = {
    PARAM_MECH_POS: "mechPos",
    PARAM_IQF: "iqf",
    PARAM_MECH_VEL: "mechVel",
    PARAM_VBUS: "VBUS",
}

# Host id used by the motor's state frames when the CAN arbitration id is the host id
# (manual: Command 9 "Modify Host CAN ID"; factory default 0xFD, see manual Type 17 example).
CAN_SFF_MASK = 0x7FF
CAN_EFF_FLAG = 0x80000000


def control_frame(motor_id: int, action: str) -> tuple[int, bytes]:
    """(can_id, data) for enable / disable / clear_fault.  F_CMD is always 0xFF."""
    command = _CONTROL_COMMANDS[action]
    data = bytes([0xFF] * 6 + [F_CMD_NORMAL, command])
    return motor_id, data


def fault_query_frame(motor_id: int) -> tuple[int, bytes]:
    """Command 5 with F_CMD != 0xFF = read the 32-bit fault word (documented, read only)."""
    assert tables.F_CMD_FAULT_QUERY != 0xFF
    data = bytes([0xFF] * 6 + [tables.F_CMD_FAULT_QUERY, tables.CMD_FAULT])
    return motor_id, data


def param_read_frame(motor_id: int, index: int) -> tuple[int, bytes]:
    """Command 14 request."""
    return (SEL_PARAM_READ << 8) | motor_id, struct.pack("<H6x", index)


class ParamReply(NamedTuple):
    index: int
    value: float


def param_reply_id(motor_id: int) -> int:
    return (SEL_PARAM_READ << 8) | motor_id


def parse_param_reply(can_id: int, data: bytes, motor_id: int) -> Optional[ParamReply]:
    """Decode a Command 14 reply, or None if this frame is not one for `motor_id`."""
    if can_id & CAN_EFF_FLAG:
        return None
    if (can_id & CAN_SFF_MASK) != param_reply_id(motor_id) or len(data) < 8:
        return None
    index = data[0] | (data[1] << 8)
    (value,) = struct.unpack("<f", data[4:8])
    if math.isnan(value):
        return None
    return ParamReply(index, value)


def is_state_candidate(can_id: int, data: bytes, motor_id: int) -> bool:
    """A state frame has an 11-bit id below 0x100 (host id) and data[0] = motor id."""
    if can_id & CAN_EFF_FLAG:
        return False
    return (can_id & CAN_SFF_MASK) < 0x100 and len(data) >= 8 and data[0] == motor_id


def decode_fault_names(word: int) -> list[str]:
    return [name for name, bit in tables.FAULT_BITS.items() if (word >> bit) & 1]
