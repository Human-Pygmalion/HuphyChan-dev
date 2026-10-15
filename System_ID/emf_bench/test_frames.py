#!/usr/bin/env python3
"""Frame round-trip checks, no hardware.  Run:  python3 test_frames.py"""

import math
import os
import struct
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emf_frames as fr  # noqa: E402
from bench_motor import BenchMotor, State  # noqa: E402
from fake_motor import FakeMotorTransport  # noqa: E402
from rscodec import mit, tables  # noqa: E402


class FrameTests(unittest.TestCase):
  def test_param_request_layout(self):
    cid, data = fr.param_read_frame(4, fr.PARAM_VBUS)
    self.assertEqual(cid, 0x304)                       # (3 << 8) | 4
    self.assertEqual(data, bytes([0x1C, 0x70, 0, 0, 0, 0, 0, 0]))   # index little endian
    cid, data = fr.param_read_frame(0x7F, 0x7005)      # manual example: "3 | 7F", "05 70 00 ..."
    self.assertEqual((cid, data), (0x37F, bytes([0x05, 0x70, 0, 0, 0, 0, 0, 0])))

  def test_param_reply_parse(self):
    data = struct.pack("<H2xf", fr.PARAM_VBUS, 48.25)
    r = fr.parse_param_reply(0x304, data, 4)
    self.assertEqual((r.index, round(r.value, 3)), (0x701C, 48.25))
    self.assertIsNone(fr.parse_param_reply(0x305, data, 4))       # other motor
    self.assertIsNone(fr.parse_param_reply(0x304 | fr.CAN_EFF_FLAG, data, 4))
    # real RS04 id 4 reply, captured 2026-10-15 on the host id 0xFD
    real = bytes.fromhex("1c70000071bd4242")
    r = fr.parse_param_reply(0xFD, real, 4)
    self.assertEqual((r.index, round(r.value, 2)), (0x701C, 48.69))
    # a state frame on the same id must not parse as a parameter reply
    self.assertIsNone(fr.parse_param_reply(0xFD, bytes.fromhex("0493f47fe7ff0104"), 4))

  def test_f_cmd_rule(self):
    for action in ("enable", "disable", "clear_fault"):
      cid, d = fr.control_frame(4, action)
      self.assertEqual(d[:7], b"\xff" * 7, action)               # data[6] == 0xFF
    self.assertEqual(fr.control_frame(4, "enable")[1][7], 0xFC)
    self.assertEqual(fr.control_frame(4, "disable")[1][7], 0xFD)
    self.assertEqual(fr.control_frame(4, "clear_fault")[1][7], 0xFB)
    q = fr.fault_query_frame(4)[1]
    self.assertEqual((q[6], q[7]), (0x00, 0xFB))                 # the only F_CMD != 0xFF frame
    # nothing the bench sends may be 0xFC/0xFD with F_CMD != 0xFF
    tr = FakeMotorTransport("RS04", 4)
    m = BenchMotor(transport=tr, motor_id=4, model="RS04")
    m.enable(); m.send_passive(); m.send_param_read(fr.PARAM_VBUS); m.send_fault_query()
    m.send(position_deg=10, kp=5, kd=1); m.clear_fault(); m.disable()
    for _cid, d in tr.log:
      if d[:6] == b"\xff" * 6 and d[7] in (0xFC, 0xFD):
        self.assertEqual(d[6], 0xFF)
    self.assertNotIn((-1, b"UNEXPECTED_F_CMD"), tr.log)

  def test_mit_roundtrip_through_fake(self):
    for model in ("RS03", "RS04"):
      tr = FakeMotorTransport(model, 4)
      m = BenchMotor(transport=tr, motor_id=4, model=model)
      raw = mit.pack_command(position_deg=-123.4, velocity_deg_s=0.0, kp=37.5, kd=2.25, torque_nm=1.5, enc=m.enc)
      m.send_frame(4, raw)
      c = tr.cmd
      self.assertAlmostEqual(math.degrees(c["q"]), -123.4, delta=0.03)
      self.assertAlmostEqual(c["kp"], 37.5, delta=2 * m.enc.kp_max / 4095)
      self.assertAlmostEqual(c["kd"], 2.25, delta=2 * m.enc.kd_max / 4095)
      self.assertAlmostEqual(c["tff"], 1.5, delta=2 * m.enc.tmax_nm / 4095)
      # reply decodes back to the fake's true state
      f = m.recv_frame(0.1)
      st = m.classify(f)
      self.assertIsInstance(st, State)
      self.assertAlmostEqual(st.position_deg, math.degrees(tr.q), delta=0.03)
      self.assertEqual(st.mode, 0)

  def test_param_read_via_fake(self):
    tr = FakeMotorTransport("RS04", 4)
    m = BenchMotor(transport=tr, motor_id=4, model="RS04")
    m.send_param_read(fr.PARAM_VBUS)
    m.send_param_read(fr.PARAM_IQF)
    r1 = m.classify(m.recv_frame(0.1)); r2 = m.classify(m.recv_frame(0.1))
    self.assertEqual((r1.index, r2.index), (0x701C, 0x701A))
    self.assertAlmostEqual(r1.value, 48.0, delta=0.01)
    tr2 = FakeMotorTransport("RS04", 4, answer_params=False)
    m2 = BenchMotor(transport=tr2, motor_id=4, model="RS04")
    m2.send_param_read(fr.PARAM_VBUS)
    self.assertIsNone(m2.recv_frame(0.01))                       # old firmware: silence

  def test_fault_word_roundtrip(self):
    tr = FakeMotorTransport("RS04", 4)
    tr.fault_word = 1 << tables.FAULT_BITS["overvoltage"]
    m = BenchMotor(transport=tr, motor_id=4, model="RS04")
    w = m.query_fault_word()
    self.assertEqual(w, 8)
    self.assertEqual(fr.decode_fault_names(w), ["overvoltage"])

  def test_closed_loop_and_regen(self):
    tr = FakeMotorTransport("RS04", 4)
    m = BenchMotor(transport=tr, motor_id=4, model="RS04")
    m.send_passive(); st0 = m.classify(m.recv_frame(0.1))
    m.enable(); m.recv_frame(0.1)
    target = st0.position_deg + 60.0
    vmax = 0.0; pmin = 0.0; last = None
    t_end = time.monotonic() + 2.5
    while time.monotonic() < t_end:
      m.send_frame(4, mit.pack_command(position_deg=target, velocity_deg_s=0, kp=20, kd=2, torque_nm=0, enc=m.enc))
      m.send_param_read(fr.PARAM_VBUS)
      got = {}
      for _ in range(2):
        o = m.classify(m.recv_frame(0.02))
        got[type(o).__name__] = o
      st = got.get("State"); pr = got.get("ParamReply")
      if st: last = st; pmin = min(pmin, st.torque_nm * math.radians(st.velocity_deg_s)); vmax = max(vmax, pr.value if pr else 0)
      time.sleep(0.005)
    self.assertAlmostEqual(last.position_deg, target, delta=1.0)
    self.assertLess(pmin, -1.0, "braking must show negative power")
    self.assertGreater(vmax, 48.2, "bus voltage must rise during braking")
    self.assertEqual(last.mode, 2)

  def test_exit_sends_disable(self):
    import argparse
    from bench_server import Controller
    a = argparse.Namespace(model="RS04", id=4, channel="can0", rate=200.0, kp=10.0, kd=1.0, fake=True,
                           fake_no_param=False, data_dir="/tmp")
    ctl = Controller(a)
    ctl.start()
    self.assertTrue(ctl.submit("connect", {})["ok"])
    time.sleep(0.2)
    self.assertTrue(ctl.submit("enable", {})["ok"])
    time.sleep(0.2)
    tr = ctl.motor._s
    self.assertEqual(tr.mode, 2)
    n_before = sum(1 for c, d in tr.log if d[7:8] == b"\xfd" and d[6:7] == b"\xff")
    ctl.shutdown()
    n_after = sum(1 for c, d in tr.log if d[7:8] == b"\xfd" and d[6:7] == b"\xff")
    self.assertGreaterEqual(n_after - n_before, 5)
    self.assertEqual(tr.mode, 0)
    self.assertTrue(tr.closed)


if __name__ == "__main__":
  unittest.main(verbosity=2)
