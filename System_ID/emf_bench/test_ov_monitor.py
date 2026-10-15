#!/usr/bin/env python3
"""Over-voltage monitor checks, no hardware.  Run:  python3 test_ov_monitor.py"""

import argparse
import csv
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ov_monitor as ov  # noqa: E402
from ov_monitor import OvMonitor, classify_fault  # noqa: E402


def feed(m, t0, n, dt, vbus, p=0.0, vel=0.0, tau=0.0):
  for k in range(n):
    m.on_sample(t0 + k * dt, vbus(k) if callable(vbus) else vbus, p, vel, tau)
  return t0 + n * dt


class ClassifyTests(unittest.TestCase):
  def test_classes(self):
    self.assertEqual(classify_fault(1 << 3), "overvoltage")
    self.assertEqual(classify_fault(1 << 2), "undervoltage")
    self.assertEqual(classify_fault(1 << 0), "other")            # over-temperature
    self.assertEqual(classify_fault(1 << 16 | 1 << 14), "other")
    self.assertEqual(classify_fault(0), "other")                 # flag set but word empty: not hidden as 'unread'
    self.assertEqual(classify_fault(None), "unread")             # fault query got no answer
    self.assertEqual(classify_fault((1 << 3) | (1 << 0)), "overvoltage")   # several bits: OV wins, word kept
    self.assertEqual((ov.BIT_OV, ov.BIT_UV), (3, 2))             # manual: fault word bit 3 / bit 2


class EventTests(unittest.TestCase):
  def make(self):
    m = OvMonitor()
    m.reset(10.0)
    return m

  def test_event_record_fields(self):
    m = self.make()
    # 2 s of samples at 200 Hz: VBUS ramps 48 -> 59.5
    t = feed(m, 10.0, 400, 0.005, lambda k: 48.0 + 11.5 * k / 399, p=-5.0, vel=100.0, tau=-0.05)
    m.on_sample(t, 60.4, 0.0, 0.0, 0.0)          # the sample that first shows the fault flag (motor already in Reset)
    ev = m.on_fault_edge(t, 1_700_000_000.0, 1 << 3, kp=20.0, kd=1.5, sine_on=True, sine_amp=30.0, sine_period=0.6)
    self.assertEqual(ev["cls"], "overvoltage")
    self.assertEqual(ev["fault_hex"], "0x00000008")
    self.assertEqual(ev["fault_names"], ["overvoltage"])
    self.assertAlmostEqual(ev["since_reset_s"], t - 10.0)
    self.assertEqual((ev["kp"], ev["kd"]), (20.0, 1.5))
    self.assertEqual((ev["sine_amp_deg"], ev["sine_period_s"]), (30.0, 0.6))
    self.assertEqual(ev["vbus_at_event"], 60.4)
    self.assertAlmostEqual(ev["vbus_max_1s"], 60.4)
    # the 1 s window holds only samples with t >= t_event - 1 s: VBUS min is from ~1 s before the end
    self.assertGreater(ev["vbus_min_1s"], 48.0 + 11.5 * 0.45)
    self.assertLess(ev["vbus_min_1s"], 48.0 + 11.5 * 0.55)
    # vel / torque / power are from the LAST sample BEFORE the flag sample, not from the flag sample
    self.assertEqual((ev["vel_deg_s"], ev["tau_nm"], ev["power_w"]), (100.0, -0.05, -5.0))
    self.assertEqual(ev["power_min_1s_w"], -5.0)
    s = m.summary(t)
    self.assertEqual(s["counts"], {"overvoltage": 1, "undervoltage": 0, "other": 0, "unread": 0})
    self.assertIsNotNone(s["since_last_ov_s"])

  def test_power_min_and_sine_off(self):
    m = self.make()
    t = feed(m, 10.0, 100, 0.005, 48.0, p=-1.0)
    m.on_sample(t, 48.0, -33.0, 0.0, 0.0)
    t += 0.005
    m.on_sample(t, 61.0, 0.0, 0.0, 0.0)
    ev = m.on_fault_edge(t, 1.0, 1 << 2, 5.0, 1.0, False, 10.0, 2.0)
    self.assertEqual(ev["cls"], "undervoltage")
    self.assertEqual(ev["power_min_1s_w"], -33.0)
    self.assertIsNone(ev["sine_amp_deg"])
    self.assertIsNone(ev["sine_period_s"])

  def test_classes_counted_separately_and_warnings_not_overvoltage(self):
    m = self.make()
    t = feed(m, 10.0, 10, 0.005, 48.0)
    for w in (1 << 3, 1 << 3, 1 << 2, 1 << 0, 1 << 14, None):
      m.on_fault_edge(t, 1.0, w, 1, 1, False, 0, 1)
    for _ in range(4):
      m.on_warn_edge()
    s = m.summary(t)
    self.assertEqual(s["counts"], {"overvoltage": 2, "undervoltage": 1, "other": 2, "unread": 1})
    self.assertEqual((s["warn"], s["faults"]), (4, 6))
    # newest first
    self.assertEqual([e["cls"] for e in m.latest_events(3)], ["unread", "other", "other"])

  def test_reset_clears_everything_but_keeps_csv(self):
    d = tempfile.mkdtemp()
    m = OvMonitor()
    path = m.open_csv(d, 1_700_000_000.0)
    t = feed(m, 10.0, 10, 0.005, 56.0)
    m.on_fault_edge(t, 1.0, 1 << 3, 1, 1, False, 0, 1)
    m.on_warn_edge()
    m.reset(t + 1)
    s = m.summary(t + 2)
    self.assertEqual((s["faults"], s["warn"], s["vbus"]["max"], s["vbus"]["n"], m.latest_events()), (0, 0, None, 0, []))
    self.assertIsNone(s["since_last_ov_s"])
    self.assertEqual(s["csv"], path)
    m.on_fault_edge(t + 3, 2.0, 1 << 3, 1, 1, False, 0, 1)           # logging continues in the same file
    m.close_csv()
    with open(path) as fh:
      rows = list(csv.reader(fh))
    self.assertEqual(len(rows), 3)                                      # header + 2 events
    self.assertEqual(rows[1][4], "1"); self.assertEqual(rows[2][4], "2")   # reset_n column tells them apart


class ThresholdTests(unittest.TestCase):
  def test_threshold_counts_and_time(self):
    m = OvMonitor(); m.reset(0.0)
    # 100 samples at 100 Hz: 40 at 51 V, 20 at 53 V, 20 at 56 V, 10 at 59 V, 10 at 47 V
    vals = [51.0] * 40 + [53.0] * 20 + [56.0] * 20 + [59.0] * 10 + [47.0] * 10
    feed(m, 0.0, 100, 0.01, lambda k: vals[k])
    s = m.summary(1.0)["vbus"]
    got = {a["v"]: (a["n"], round(a["s"], 3)) for a in s["above"]}
    self.assertEqual(got[52.0][0], 50)       # 20 + 20 + 10
    self.assertEqual(got[55.0][0], 30)       # 20 + 10
    self.assertEqual(got[58.0][0], 10)
    self.assertAlmostEqual(got[52.0][1], 0.5, places=2)
    self.assertAlmostEqual(got[58.0][1], 0.1, places=2)
    self.assertEqual((s["max"], s["min"], s["n"]), (59.0, 47.0, 100))
    self.assertAlmostEqual(s["margin_v"], 1.0)            # 60 - 59
    # strictly greater than: exactly on the threshold does not count
    m.reset(0.0); feed(m, 0.0, 5, 0.01, 52.0)
    self.assertEqual(m.summary(1.0)["vbus"]["above"][0]["n"], 0)

  def test_missing_vbus_samples_are_skipped(self):
    m = OvMonitor(); m.reset(0.0)
    feed(m, 0.0, 10, 0.01, lambda k: None if k % 2 else 57.0)
    s = m.summary(1.0)["vbus"]
    self.assertEqual((s["n"], s["above"][1]["n"]), (5, 5))

  def test_long_gap_is_capped(self):
    m = OvMonitor(); m.reset(0.0)
    m.on_sample(0.0, 57.0, 0, 0, 0); m.on_sample(5.0, 57.0, 0, 0, 0)    # 5 s without VBUS must not count as 5 s above
    self.assertLessEqual(m.summary(6.0)["vbus"]["above"][1]["s"], 0.1 + 1e-9)


class RateTests(unittest.TestCase):
  def test_rates(self):
    m = OvMonitor(); m.reset(0.0)
    t = feed(m, 0.0, 10, 0.01, 48.0)
    for tt in (100.0, 200.0, 290.0):                      # three over-voltage events
      m.on_fault_edge(tt, 1.0, 1 << 3, 1, 1, False, 0, 1)
    m.on_fault_edge(295.0, 1.0, 1 << 0, 1, 1, False, 0, 1)
    s = m.summary(300.0)["rate"]
    self.assertEqual((s["ov_5m"], s["faults_5m"]), (3, 4))
    self.assertAlmostEqual(s["ov_per_min_5m"], 3 / 5.0)   # 3 events in a 300 s window
    self.assertAlmostEqual(s["ov_per_min_all"], 3 / 5.0)
    s = m.summary(600.0)["rate"]                          # events older than 5 min drop out of the 5 min rate only
    self.assertEqual(s["ov_5m"], 0)
    self.assertAlmostEqual(s["ov_per_min_all"], 3 / 10.0)
    m.stop(300.0)                                         # disconnect freezes the elapsed time
    self.assertAlmostEqual(m.summary(900.0)["elapsed_s"], 300.0)


class ControllerIntegrationTests(unittest.TestCase):
  """Real Controller loop against the fake motor with a high regen gain: a fast sine must trip the
  60 V over-voltage fault, the monitor must classify it from the queried fault word, write the CSV
  row, and the monitor must NOT clear the fault or re-enable by itself."""

  def test_fake_trips_and_monitor_counts(self):
    from bench_server import Controller
    d = tempfile.mkdtemp()
    a = argparse.Namespace(model="RS04", id=4, channel="can0", rate=200.0, kp=20.0, kd=1.0, fake=True,
                           fake_no_param=False, fake_ov_gain=2.0, data_dir=d, vbus_priority=True)
    ctl = Controller(a)
    ctl.start()
    try:
      self.assertTrue(ctl.submit("connect", {})["ok"])
      time.sleep(0.3)
      self.assertEqual(ctl.status()["ov"]["counts"]["overvoltage"], 0)
      self.assertTrue(ctl.submit("enable", {})["ok"])
      ctl.set_params({"sine_amp_deg": 40.0, "sine_period_s": 0.5, "sine": "on"})
      t_end = time.monotonic() + 8.0
      while time.monotonic() < t_end and ctl.status()["ov"]["counts"]["overvoltage"] == 0:
        time.sleep(0.05)
      st = ctl.status()
      self.assertEqual(st["ov"]["counts"]["overvoltage"], 1, st["ov"])
      self.assertEqual(st["ov"]["counts"]["unread"], 0)
      ev = st["ov_events"][0]
      self.assertEqual((ev["cls"], ev["fault_hex"]), ("overvoltage", "0x00000008"))
      self.assertTrue(ev["sine_on"])
      self.assertGreater(ev["vbus_max_1s"], 50.0)
      self.assertLess(ev["power_min_1s_w"], 0.0)               # it was regenerating
      self.assertGreater(st["ov"]["vbus"]["above"][0]["n"], 0)  # samples above 52 V were counted
      # no protective automation: the fault stays, the motor stays in Reset, the count stays 1
      time.sleep(1.0)
      st2 = ctl.status()
      self.assertTrue(st2["fault"])
      self.assertEqual((st2["mode"], st2["ov"]["counts"]["overvoltage"]), (0, 1))
      self.assertEqual(st2["vbus_priority"], True)
      # the CSV has the same event, flushed already (file still open)
      with open(st["ov"]["csv"]) as fh:
        rows = list(csv.reader(fh))
      self.assertEqual((len(rows), rows[1][5], rows[1][6]), (2, "overvoltage", "0x00000008"))
      # Reset button
      ctl.ov_reset()
      self.assertEqual(ctl.status()["ov"]["faults"], 0)
      # clearing the fault stops the sine so that the next Enable cannot jump
      ctl.submit("clear_fault", {})
      self.assertFalse(ctl.status()["sine_on"])
    finally:
      ctl.shutdown()


if __name__ == "__main__":
  unittest.main(verbosity=2)
