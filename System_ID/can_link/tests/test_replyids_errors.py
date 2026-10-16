"""(4) per-motor reply ids map correctly, (5) error frames are counted by class, plus kernel
timestamps, kernel filters, legacy collect(expect=) and the latency-recorder contract."""

import os
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h  # noqa: E402
import scenarios as sc  # noqa: E402

HOSTS = {7: 0x70, 8: 0x71, 9: 0x72, 10: 0x73, 11: 0x74, 12: 0x75}


class ReplyIds(unittest.TestCase):
    def setUp(self):
        self.pkg = h.patched()

    def tearDown(self):
        h.fc.uninstall_fake_can_module()

    def test_each_motor_answers_on_its_own_id_and_is_mapped_back(self):
        rig = h.Rig(self.pkg, host_ids=HOSTS, bus_kwargs=dict(reply_ids=HOSTS))
        res = h.run_cycles(rig, 10, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        self.assertTrue(all(all(v == 0 for v in r["lag"].values()) for r in res))
        rig.close()

    def test_reply_arbitration_ids_on_the_wire_really_differ(self):
        rig = h.Rig(self.pkg, host_ids=HOSTS, bus_kwargs=dict(reply_ids=HOSTS))
        got = {}
        orig_deliver = rig.wire._deliver

        def spy(can_id, data, t, is_error=False):
            if not is_error:
                got.setdefault(can_id, set()).add(data[0])
            orig_deliver(can_id, data, t, is_error)
        rig.wire._deliver = spy
        h.run_cycles(rig, 5, sc.PERIOD)
        self.assertEqual({k: sorted(v) for k, v in got.items()}, {HOSTS[m]: [m] for m in HOSTS})
        rig.close()

    def test_frame_whose_id_and_data0_disagree_is_dropped_and_counted(self):
        rig = h.Rig(self.pkg, host_ids=HOSTS, bus_kwargs=dict(reply_ids=HOSTS))
        frame = self.pkg.canbus.CanFrame(can_id=0x72, data=bytes([8, 0x80, 0, 0x80, 8, 0, 0, 0xFA]), stamp=rig.clock.monotonic())
        rig.rs.send_mit({7: self.pkg.bus.MitCommand(position_deg=1.0)})
        before = rig.rs.state(8).stamp
        rig.rs._ingest([frame], {}, {}, {}, set(), awaited=(7,))
        self.assertEqual(rig.rs.link.id_mismatch, 1)
        self.assertEqual(rig.rs.state(8).stamp, before)     # nothing was applied to motor 8 or 9
        rig.close()

    def test_unconfigured_motors_still_work_on_the_shared_id(self):
        half = {7: 0x70, 8: 0x71, 9: 0x72}                   # 10..12 still on 0xFD
        rig = h.Rig(self.pkg, host_ids=half, bus_kwargs=dict(reply_ids=half))
        res = h.run_cycles(rig, 10, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        rig.close()

    def test_motors_sharing_one_id_fall_back_to_data0(self):
        share = {7: 0x70, 8: 0x70, 9: 0x72, 10: 0x72, 11: 0x74, 12: 0x75}
        rig = h.Rig(self.pkg, host_ids=share, bus_kwargs=dict(reply_ids=share))
        self.assertEqual(rig.rs._mid_by_reply_id, {0x74: 11, 0x75: 12})
        res = h.run_cycles(rig, 10, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        rig.close()

    def test_a_reply_id_equal_to_a_motor_id_is_refused(self):
        with self.assertRaises(ValueError):
            h.Rig(self.pkg, bus_kwargs=dict(reply_ids={7: 8}))     # 8 is motor 8's command id
        with self.assertRaises(ValueError):
            h.Rig(self.pkg, bus_kwargs=dict(reply_ids={99: 0x70}))  # not on this bus

    def test_kernel_filter_passes_configured_ids_and_0xFD_for_unconfigured(self):
        half = {7: 0x70, 8: 0x71, 9: 0x72}
        rig = h.Rig(self.pkg, host_ids=half, bus_kwargs=dict(reply_ids=half, kernel_filter=True))
        flt = rig.canmod._holder["bus"].filters
        self.assertEqual(sorted(f["can_id"] for f in flt), [0x70, 0x71, 0x72, 0xFD])
        self.assertTrue(all(f["can_mask"] == 0x7FF and not f["extended"] for f in flt))
        res = h.run_cycles(rig, 5, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        rig.close()

    def test_filter_blocks_other_ids(self):
        rig = h.Rig(self.pkg, host_ids=HOSTS, canbus_kwargs=dict(reply_ids=[0x70]))
        rig.wire._deliver(0x123, b"12345678", rig.clock.monotonic() - 0.001)
        rig.wire._deliver(0x70, b"12345678", rig.clock.monotonic() - 0.0005)
        got = rig.can.poll(0.001)
        self.assertEqual([f.can_id for f in got], [0x70])
        rig.close()


class ErrorFrames(unittest.TestCase):
    def setUp(self):
        self.pkg = h.patched()
        self.rig = h.Rig(self.pkg)

    def tearDown(self):
        self.rig.close()

    def _inject(self, cls, data=b"\0" * 8):
        self.rig.wire.inject_error(0.0, cls, data)
        self.rig.clock.sleep(0.0001)

    def test_classes_are_counted_and_not_queued(self):
        self._inject(0x080, bytes([0, 0, 0x01, 0, 0, 0, 0, 0]))        # bus error, bit error
        self._inject(0x080, bytes([0, 0, 0x04, 0, 0, 0, 0, 0]))        # bus error, stuff error
        self._inject(0x002, bytes([5, 0, 0, 0, 0, 0, 0, 0]))           # lost arbitration at bit 5
        self._inject(0x004, bytes([0, 0x01, 0, 0, 0, 0, 0, 0]))        # controller rx overflow
        self._inject(0x040)                                           # bus off
        got = self.rig.can.poll(0.001)
        self.assertEqual(got, [], "error frames must never reach the frame queue")
        c = self.rig.can.counters
        self.assertEqual(c.error_frames, 5)
        ec = c.error_classes
        self.assertEqual(ec["bus_error"], 2)
        self.assertEqual(ec["lost_arbitration"], 1)
        self.assertEqual(ec["controller"], 1)
        self.assertEqual(ec["controller.rx_overflow"], 1)
        self.assertEqual(ec["bus_off"], 1)
        self.assertEqual(ec.get("protocol.bit", 0) + ec.get("protocol.stuff", 0), 0)  # class 0x08 not set

    def test_protocol_class_details(self):
        self._inject(0x008, bytes([0, 0, 0x01 | 0x04, 0, 0, 0, 0, 0]))
        self.rig.can.poll(0.001)
        ec = self.rig.can.counters.error_classes
        self.assertEqual((ec["protocol"], ec["protocol.bit"], ec["protocol.stuff"]), (1, 1, 1))

    def test_error_frames_do_not_hide_a_reply_in_the_same_collect(self):
        self.rig.wire.inject_error(0.0002, 0x080, bytes([0, 0, 0x01, 0, 0, 0, 0, 0]))
        res = h.run_cycles(self.rig, 3, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        self.assertEqual(self.rig.can.counters.error_classes["bus_error"], 1)

    def test_error_fields_carry_every_class_key_and_as_fields_is_unchanged(self):
        c = self.rig.can.counters
        f = c.error_fields()
        for n in self.pkg.canbus.ERR_CLASS_NAMES:
            self.assertIn(f"can.err_{n}", f)
        self.assertIn("can.stamp_fallbacks", f)
        self.assertEqual(set(c.as_fields()), set(h.orig().canbus.CanCounters().as_fields()))   # upstream key set

    def test_socket_gets_the_error_mask(self):
        opts = self.rig.canmod._holder["bus"].socket.opts
        self.assertIn((101, 2, 0x1FFFFFFF), opts)

    def test_reset_clears_the_class_counts(self):
        self._inject(0x080)
        self.rig.can.poll(0.001)
        self.rig.can.counters.reset()
        self.assertEqual(self.rig.can.counters.error_classes, {})
        self.assertEqual(self.rig.can.counters.error_frames, 0)

    def test_original_counts_only_a_total(self):
        o = h.orig()
        rig = h.Rig(o)
        rig.wire.inject_error(0.0, 0x080)
        rig.clock.sleep(0.0001)
        rig.can.drain(timeout_s=0.001)
        self.assertEqual(rig.can.counters.error_frames, 1)
        self.assertFalse(hasattr(rig.can.counters, "error_classes"))
        rig.close()


class StampsAndCompat(unittest.TestCase):
    def tearDown(self):
        h.fc.uninstall_fake_can_module()

    def test_patched_stamp_is_kernel_arrival_not_read_time(self):
        pkg = h.patched()
        rig = h.Rig(pkg, virtual=False, canbus_kwargs=dict(reader=False))
        t_send = time.monotonic()
        rig.can.send(pkg.canbus.CanFrame(can_id=7, data=bytes([0x80, 0, 0, 0x80, 0, 0, 0, 0])))
        time.sleep(0.030)                                   # nobody reads for 30 ms
        t_read = time.monotonic()
        got = rig.can.poll(0.001)
        self.assertEqual(len(got), 1)
        self.assertLess(got[0].stamp - t_send, 0.004, "stamp must be the arrival time (~1 ms), not the read time")
        self.assertGreater(t_read - got[0].stamp, 0.020)
        self.assertEqual(rig.can.counters.stamp_fallbacks, 0)
        rig.close()

    def test_original_stamp_is_read_time(self):
        o = h.orig()
        rig = h.Rig(o, virtual=False)
        t_send = time.monotonic()
        rig.can.send(o.canbus.CanFrame(can_id=7, data=bytes([0x80, 0, 0, 0x80, 0, 0, 0, 0])))
        time.sleep(0.030)
        got = rig.can.drain(timeout_s=0.001)
        self.assertGreater(got[0].stamp - t_send, 0.025)    # this is why the original cannot tell "late"
        rig.close()

    def test_missing_kernel_timestamp_falls_back_and_is_counted(self):
        pkg = h.patched()
        rig = h.Rig(pkg, virtual=False, canbus_kwargs=dict(reader=False))
        msg = h.fc._FakeMsg(7, bytes(8), False, False, 0.0)
        f = rig.can._frame_of(msg)
        self.assertAlmostEqual(f.stamp, time.monotonic(), delta=0.01)
        self.assertEqual(rig.can.counters.stamp_fallbacks, 1)
        rig.close()

    def test_reader_is_on_by_default_and_off_when_asked(self):
        pkg = h.patched()
        rig = h.Rig(pkg, virtual=False, canbus_kwargs={})
        self.assertTrue(rig.can.reader_running)
        rig.close()
        rig = h.Rig(pkg, virtual=False, canbus_kwargs=dict(reader=False))
        self.assertFalse(rig.can.reader_running)
        rig.close()

    def test_import_without_python_can(self):
        pkg = h.patched()
        root = os.path.dirname(os.path.dirname(pkg.pkg.__file__))
        code = ("import sys; sys.path.insert(0, %r); import %s.robstride.bus, %s.canbus; "
                "assert 'can' not in sys.modules; print('ok')" % (root, pkg.name, pkg.name))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "ok", out.stderr)

    def test_legacy_collect_without_send_mit_counts_frames_like_before(self):
        pkg = h.patched()
        rig = h.Rig(pkg)
        rig.rs.enable_torque()
        rig.clock.sleep(0.003)
        missing = rig.rs.collect(expect=6)
        self.assertEqual(missing, [])
        rig.close()

    def test_refresh_states_works_and_clears_the_guard_records(self):
        pkg = h.patched()
        rig = h.Rig(pkg)
        self.assertEqual(rig.rs.refresh_states(), [])
        self.assertTrue(all(len(q) == 0 for q in rig.rs._outstanding.values()))
        rig.close()

    def test_latency_recorder_sees_the_same_ontime_and_late_events_as_the_original(self):
        class Rec:
            def __init__(self):
                self.sent_n = 0
                self.ev = []

            def sent(self, mid):
                self.sent_n += 1

            def received(self, mid, s, late):
                self.ev.append((mid, bool(late)))

        out = {}
        for name, pkg in (("old", h.orig()), ("new", h.patched())):
            rig = h.Rig(pkg, drain_s=sc.DRAIN_S, scripts={9: (lambda m, i: sc.LATE_EXTRA_S if i == 5 else 0.0)})
            rec = Rec()
            rig.rs.latency = rec
            h.run_cycles(rig, 12, sc.PERIOD)
            out[name] = (rec.sent_n, sum(1 for _, l in rec.ev if l), sum(1 for _, l in rec.ev if not l))
            rig.close()
        self.assertEqual(out["new"][0], out["old"][0])
        self.assertEqual(out["new"][1], 1)                   # the one late reply
        self.assertGreaterEqual(out["new"][2], out["old"][2])


class ReaderThreadRealTime(unittest.TestCase):
    """Real clock, real reader thread, fake wire in the middle.  Timing margins >= 3 ms."""

    def tearDown(self):
        h.fc.uninstall_fake_can_module()

    def _run(self, pkg, **canbus_kwargs):
        period, drain, extra = 0.020, 0.006, 0.010          # reply 11 ms after send: after drain, before next send
        rig = h.Rig(pkg, virtual=False, drain_s=drain, canbus_kwargs=canbus_kwargs,
                    scripts={9: (lambda m, i: extra if i == 3 else 0.0)})
        t0 = time.monotonic() + 0.02
        lags = []
        for k in range(25):
            while time.monotonic() < t0 + k * period:
                time.sleep(0.0005)
            rig.rs.send_mit({m: pkg.bus.MitCommand(position_deg=float(k)) for m in rig.ids})
            rig.rs.collect(expect=6)
            lags.append([k - round(rig.rs.state(m).position_deg) for m in rig.ids])
        rig.close()
        return sum(1 for k, row in enumerate(lags) if k > 3 and any(v >= 1 for v in row)), len(lags) - 4

    def test_old_with_reader_slips_forever_new_recovers(self):
        o_slip, n = self._run(h.orig(), reader=True)
        n_slip, _ = self._run(h.patched())
        self.assertGreaterEqual(o_slip, n - 2, f"old: {o_slip}/{n}")
        self.assertLessEqual(n_slip, 1, f"new: {n_slip}/{n}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
