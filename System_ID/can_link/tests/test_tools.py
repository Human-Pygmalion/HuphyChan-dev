"""Tests for frames.py, bus_probe.py, host_id_tool.py, analyze_huphy_latency.py, canraw parsers.
No hardware: the probe/tool run against fake_can.FakeWire with the real clock."""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import analyze_huphy_latency as an  # noqa: E402
import bus_probe as bp  # noqa: E402
import canraw  # noqa: E402
import fake_can as fc  # noqa: E402
import frames as fr  # noqa: E402
import host_id_tool as hid  # noqa: E402


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class Frames(unittest.TestCase):
    def test_passive_frames_are_zero_gain_for_every_model(self):
        for model in ("RS00", "RS02", "RS03", "RS04"):
            cid, data = fr.passive_mit_frame(7, model)
            self.assertEqual(cid, 7)
            self.assertEqual(len(data), 8)
            self.assertNotEqual(data[:6], b"\xff" * 6)
            kp, kd, tau = fr.decode_mit_gains(data, model)
            self.assertEqual((kp, kd), (0.0, 0.0))
            self.assertLess(abs(tau), 0.001 * fr.tables.encoding_for(fr.tables.Model(model)).tmax_nm)

    def test_assert_passive_rejects_gains_and_control_frames(self):
        enc = fr.tables.encoding_for(fr.tables.Model("RS04"))
        bad = fr.mit.pack_command(position_deg=0, velocity_deg_s=0, kp=10.0, kd=0.0, torque_nm=0.0, enc=enc)
        with self.assertRaises(AssertionError):
            fr.assert_passive(7, bad, "RS04")
        bad = fr.mit.pack_command(position_deg=0, velocity_deg_s=0, kp=0.0, kd=0.0, torque_nm=5.0, enc=enc)
        with self.assertRaises(AssertionError):
            fr.assert_passive(7, bad, "RS04")
        with self.assertRaises(AssertionError):
            fr.assert_passive(7, bytes([0xFF] * 7 + [0xFC]), "RS04")          # enable frame

    def test_param_read_frame(self):
        cid, data = fr.param_read_frame(4, 0x701C)
        self.assertEqual((cid, data.hex()), (0x304, "1c70000000000000"))
        cid, data = fr.param_read_frame(7, 0x200A)
        self.assertEqual((cid, data.hex()), (0x307, "0a20000000000000"))

    def test_host_id_and_save_frames(self):
        self.assertEqual(fr.host_id_frame(7, 0x70), (7, bytes([0xFF] * 6 + [0x70, 0x01])))
        self.assertEqual(fr.save_frame(7), (7, bytes([0xFF] * 7 + [0xF8])))

    def test_param_reply_parse_and_state_not_confused(self):
        reply = bytes([0x1C, 0x70, 0, 0]) + bytes.fromhex("7 1 BD 42 42".replace(" ", "").replace("7", "71", 1)[:8])
        reply = bytes.fromhex("1c700000" "71bd4242")
        self.assertEqual(fr.is_param_reply(reply), 0x701C)
        self.assertAlmostEqual(fr.parse_param_value(0x701C, reply), 48.685, places=2)
        state = fc.FakeWire.state_frame(7, 0x8000, 0)
        self.assertIsNone(fr.is_param_reply(state))
        self.assertEqual(fr.state_candidate(state, [7, 8]), 7)
        self.assertIsNone(fr.state_candidate(reply, [7, 8, 0x1C]))

    def test_model_map(self):
        m = fr.parse_model_map("9:RS03,12:rs03", [7, 8, 9, 10, 11, 12], "RS04")
        self.assertEqual(m, {7: "RS04", 8: "RS04", 9: "RS03", 10: "RS04", 11: "RS04", 12: "RS03"})
        with self.assertRaises(ValueError):
            fr.parse_model_map("9:RS99", [9], "RS04")

    def test_read_only_tools_cannot_build_dangerous_frames(self):
        """AST-level guard: bus_probe.py / canraw.py never reference the dangerous builders or command bytes."""
        import ast
        banned_names = {"CMD_ENABLE", "CMD_STOP", "CMD_SET_ZERO", "CMD_SET_CAN_ID", "CMD_SET_PROTOCOL",
                        "CMD_SET_MODE", "CMD_FAULT", "CMD_SET_HOST_ID", "CMD_SAVE", "host_id_frame", "save_frame"}
        banned_ints = {0xFC, 0xFD, 0xFE, 0xFA, 0xFB, 0xF8}
        for fn in ("bus_probe.py", "canraw.py"):
            with open(os.path.join(ROOT, fn)) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, banned_names, f"{fn}: {node.id}")
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, banned_names, f"{fn}: .{node.attr}")
                if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                    self.assertNotIn(node.value, banned_ints, f"{fn}: constant {node.value:#x} line {node.lineno}")
        # every frame the workspace builders emit with an all-FF prefix is Command 9 or Command 12 only
        for mid in (1, 7, 12):
            for fnc in (lambda: fr.passive_mit_frame(mid, "RS04"), lambda: fr.param_read_frame(mid, 0x701C),
                        lambda: fr.save_frame(mid), lambda: fr.host_id_frame(mid, 0x70)):
                cid, data = fnc()
                if data[:6] == b"\xff" * 6:
                    self.assertIn(data[7], (0x01, 0xF8))


class HostIdValidation(unittest.TestCase):
    def test_refusals(self):
        bus = list(range(1, 13))
        for host in (0x00, 0x05, 0x0C, 0x0F, 0x7F, 0xFE, 0xFF, 0x100, 8):
            with self.assertRaises(ValueError, msg=hex(host)):
                hid.validate_host(7, host, bus)
        hid.validate_host(7, 0x70, bus)
        hid.validate_host(7, 0xFD, bus)
        hid.validate_host(7, 0x10, bus)
        with self.assertRaises(ValueError):
            hid.validate_host(7, 0x20, [0x20])                   # equals a motor id on the bus
        with self.assertRaises(ValueError):
            hid.validate_host(200, 0x70, bus)

    def test_cli_refuses_batch_and_bad_host(self):
        for argv in (["set", "--motor", "7,8", "--host", "0x70"],):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    hid.parse_args(argv)
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = hid.main(["set", "--motor", "7", "--host", "0x08"])
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED", buf.getvalue())


class HostIdFlow(unittest.TestCase):
    def make(self, ids=(7, 8, 9)):
        wire = fc.FakeWire({m: fc.FakeMotor(m) for m in ids}, clock=fc.RealClock())
        return wire, fc.RawTransport(wire)

    def test_dry_run_sends_nothing(self):
        wire, can = self.make()
        out = quiet(hid.do_set, can, 7, 0x70, list(range(1, 13)), execute=False, save=False)
        self.assertFalse(out["executed"])
        self.assertEqual(wire.tx_sent_log, [])
        self.assertEqual(out["frame"]["data"], "FF FF FF FF FF FF 70 01")

    def test_read_reports_default_host_and_own_id(self):
        wire, can = self.make()
        res = hid.do_read(can, [7, 8])
        for m in (7, 8):
            self.assertEqual(res[m]["can_master_0x200A"]["value"], 0xFD)
            self.assertEqual(res[m]["can_master_0x200A"]["reply_id"], 0xFD)
            self.assertEqual(res[m]["can_id_0x2009"]["value"], m)

    def test_set_changes_only_that_motor_and_reply_moves_to_new_id(self):
        wire, can = self.make()
        out = quiet(hid.do_set, can, 7, 0x70, list(range(1, 13)), execute=True, save=False)
        self.assertEqual(out["after"]["value"], 0x70)
        self.assertEqual(out["after"]["reply_id"], 0x70)
        self.assertEqual(wire.motors[7].host_id, 0x70)
        self.assertEqual(wire.motors[8].host_id, 0xFD)
        self.assertEqual(wire.motors[9].host_id, 0xFD)
        sent_to_8 = [d for _, cid, d in wire.tx_sent_log if cid == 8]
        self.assertTrue(all(d[6:] != b"\xff\x01" and d[7] != 0x01 for d in sent_to_8))

    def test_unsaved_change_reverts_on_power_cycle_saved_one_survives(self):
        wire, can = self.make()
        quiet(hid.do_set, can, 7, 0x70, list(range(1, 13)), execute=True, save=False)
        quiet(hid.do_set, can, 8, 0x71, list(range(1, 13)), execute=True, save=True)
        for m in wire.motors.values():
            m.power_cycle()
        res = hid.do_read(can, [7, 8, 9])
        self.assertEqual(res[7]["can_master_0x200A"]["value"], 0xFD)      # not saved -> back to default (fake's model)
        self.assertEqual(res[8]["can_master_0x200A"]["value"], 0x71)      # saved
        self.assertEqual(res[9]["can_master_0x200A"]["value"], 0xFD)

    def test_set_refuses_to_change_a_motor_that_does_not_read_back(self):
        wire, can = self.make(ids=(7,))
        with self.assertRaises(SystemExit):
            quiet(hid.do_set, can, 9, 0x70, list(range(1, 13)), execute=True, save=False)
        self.assertEqual([d for _, cid, d in wire.tx_sent_log if d[7:] == b"\x01"], [])

    def test_save_is_dry_run_by_default(self):
        wire, can = self.make()
        out = quiet(hid.do_save, can, 7, execute=False)
        self.assertFalse(out["executed"])
        self.assertEqual(wire.tx_sent_log, [])


class ProbeStats(unittest.TestCase):
    def test_classify_ontime_late_lost(self):
        R = bp.Rec
        a = [R(1, "state", 0, 0.00), R(1, "state", 1, 0.01), R(1, "state", 2, 0.02), R(1, "state", 3, 0.03)]
        a[0].t_reply = 0.001      # on time
        a[1].t_reply = 0.021      # arrives after the next send (0.02) -> late
        a[2].t_reply = None       # lost
        a[3].t_reply = 0.0315
        on, late, lost, rtts = bp.classify(a, 0.05)
        self.assertEqual((on, late, lost), (2, 1, 1))
        self.assertEqual(len(rtts), 3)

    def test_pct_and_summary(self):
        self.assertEqual(bp.pct([1, 2, 3, 4, 5], 50), 3)
        s = bp.rtt_summary([0.001, 0.002, 0.003])
        self.assertEqual((s["n"], s["median"], s["max"]), (3, 2.0, 3.0))

    def _run(self, motors, mode, rate, dur=0.6, params=(), max_out=3, **wire_kw):
        wire = fc.FakeWire(motors, clock=fc.RealClock(), **wire_kw)
        can = fc.RawTransport(wire)
        ids = sorted(motors)
        models = {m: "RS04" for m in ids}
        p = bp.Prober(can, ids, models, list(params), timeout_s=0.004, expire_s=0.05, max_outstanding=max_out)
        return p.run(mode, rate, dur), wire

    def test_clean_bus_has_no_loss_and_rtt_is_plausible(self):
        motors = {m: fc.FakeMotor(m) for m in (7, 8, 9)}
        for mode in ("sequential", "pipelined"):
            r, _ = self._run(motors, mode, 100)
            self.assertEqual(r["state"]["lost"], 0, mode)
            self.assertEqual(r["state"]["late"], 0, mode)
            self.assertGreater(r["state"]["sent"], 150)
            self.assertTrue(0.3 < r["state"]["rtt_ms"]["median"] < 3.0)
            self.assertTrue(r["shared_reply_id"])
            self.assertEqual(r["reply_id_histogram"].keys(), {"0xFD"})

    def test_distinct_host_ids_show_up_in_the_histogram(self):
        motors = {m: fc.FakeMotor(m, host_id=0x60 + m) for m in (1, 2, 3)}
        r, _ = self._run(motors, "pipelined", 100, dur=0.3)
        self.assertEqual(set(r["reply_id_histogram"]), {"0x61", "0x62", "0x63"})
        self.assertFalse(r["shared_reply_id"])

    def test_scripted_loss_is_counted_per_motor(self):
        motors = {m: fc.FakeMotor(m) for m in (7, 8)}
        motors[8].script = lambda m, i: None if i % 5 == 0 else 0.0     # every 5th reply never comes
        r, _ = self._run(motors, "pipelined", 100, dur=0.8)
        m8, m7 = r["per_motor"]["8"], r["per_motor"]["7"]
        self.assertEqual(m7["lost"], 0)
        self.assertAlmostEqual(m8["lost"] / m8["sent"], 0.2, delta=0.04)

    def test_late_reply_after_next_send_is_classified_late(self):
        motors = {m: fc.FakeMotor(m) for m in (7,)}
        motors[7].script = lambda m, i: 0.012 if i % 4 == 1 else 0.0      # 12 ms > 10 ms period
        r, _ = self._run(motors, "pipelined", 100, dur=0.8)
        m7 = r["per_motor"]["7"]
        self.assertGreater(m7["late"], 5)
        self.assertEqual(m7["lost"], 0)

    def test_error_frames_counted_by_class(self):
        motors = {m: fc.FakeMotor(m) for m in (7,)}
        wire = fc.FakeWire(motors, clock=fc.RealClock())
        can = fc.RawTransport(wire)
        p = bp.Prober(can, [7], {7: "RS04"}, [], timeout_s=0.004, expire_s=0.05, max_outstanding=3)
        wire.inject_error(0.05, 0x088, bytes([0, 0, 0x01, 0, 0, 0, 0, 0]))   # real bus errors set PROT|BUSERROR
        wire.inject_error(0.06, 0x002)
        r = p.run("pipelined", 100, 0.3)
        self.assertEqual(r["error_frames"]["total"], 2)
        self.assertEqual(r["error_frames"]["by_class"]["bus_error"], 1)
        self.assertEqual(r["error_frames"]["by_class"]["protocol.bit"], 1)
        self.assertEqual(r["error_frames"]["by_class"]["lost_arbitration"], 1)

    def test_outstanding_limit_is_respected_at_the_motor(self):
        motors = {7: fc.FakeMotor(7, proc_s=0.035)}                       # answers 3.5 cycles late
        r, wire = self._run(motors, "pipelined", 100, dur=0.6, max_out=2)
        self.assertLessEqual(motors[7].max_pending, 2)
        self.assertEqual(motors[7].n_dropped_by_buffer, 0)
        self.assertGreater(r["skipped_by_outstanding_limit"], 0)

    def test_param_reads_are_attributed_and_do_not_pollute_state(self):
        motors = {m: fc.FakeMotor(m) for m in (7, 8)}
        r, _ = self._run(motors, "pipelined", 100, dur=0.5, params=("vbus",))
        for m in ("7", "8"):
            self.assertEqual(r["per_motor"][m]["params"]["vbus"]["lost"], 0)
            self.assertGreater(r["per_motor"][m]["params"]["vbus"]["sent"], 20)
            self.assertEqual(r["per_motor"][m]["lost"], 0)

    def test_main_runs_fake_end_to_end_and_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "p.json")
            rc = quiet(bp.main, ["--fake", "--ids", "7,8,9", "--rates", "100", "--duration", "0.4", "--mode", "both", "--out", out])
            self.assertEqual(rc, 0)
            with open(out) as fh:
                j = json.load(fh)
            self.assertEqual([r["mode"] for r in j["runs"]], ["sequential", "pipelined"])
            self.assertIn("ontime=", j["runs"][0]["summary"])

    def test_probe_refuses_an_enabled_motor(self):
        motors = {7: fc.FakeMotor(7)}
        motors[7].mode = 2
        wire = fc.FakeWire(motors, clock=fc.RealClock())
        can = fc.RawTransport(wire)
        modes = bp.hello(can, [7], {7: "RS04"})
        self.assertEqual(modes[7], 2)


class Parsers(unittest.TestCase):
    SAMPLE = """6: can0: <NOARP,UP,LOWER_UP> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can  promiscuity 0 minmtu 0 maxmtu 0
    can state ERROR-ACTIVE restart-ms 0
\t  bitrate 1000000 sample-point 0.875
\t  re-started bus-errors arbit-lost error-warn error-pass bus-off
\t  1          2          3          4          5          6         numtxqueues 1 numrxqueues 1 gso_max_size 65536 gso_max_segs 65535
    RX:  bytes  packets errors dropped  missed   mcast
       1580688   197586      7       8       9       0
    TX:  bytes  packets errors dropped carrier collsns
     186436256 23304532     10      11      12       0
"""

    def test_ip_stats_parse(self):
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(stdout=self.SAMPLE)):
            st = canraw.ip_stats("can0")
        self.assertEqual(st["state"], "ERROR-ACTIVE")
        self.assertEqual(st["bitrate"], 1000000)
        self.assertEqual([st[k] for k in ("re-started", "bus-errors", "arbit-lost", "error-warn", "error-pass", "bus-off")], [1, 2, 3, 4, 5, 6])
        self.assertEqual((st["rx_packets"], st["rx_errors"], st["tx_packets"], st["tx_errors"]), (197586, 7, 23304532, 10))

    def test_snapshot_delta(self):
        a = {"time": 1.0, "ip": {"bus-errors": 2, "state": "X"}, "sysfs": {"rx_packets": 10}}
        b = {"time": 3.5, "ip": {"bus-errors": 7, "state": "ERROR-ACTIVE"}, "sysfs": {"rx_packets": 40}}
        d = canraw.snapshot_delta(a, b)
        self.assertEqual((d["ip"]["bus-errors"], d["sysfs"]["rx_packets"], d["seconds"]), (5, 30, 2.5))

    def test_error_frame_decode(self):
        self.assertEqual(canraw.decode_error_frame(0x080, bytes([0, 0, 0x04, 0, 0, 0, 0, 0])), ["bus_error"])
        self.assertEqual(canraw.decode_error_frame(0x088, bytes([0, 0, 0x04, 0, 0, 0, 0, 0])),
                         ["protocol", "bus_error", "protocol.stuff"])
        self.assertEqual(canraw.decode_error_frame(0x004, bytes([0, 0x10, 0, 0, 0, 0, 0, 0])), ["controller", "controller.rx_passive"])

    @unittest.skipUnless(os.path.exists("/sys/class/net/lo"), "no sysfs")
    def test_sysfs_stats_reads_something(self):
        self.assertIn("rx_packets", canraw.sysfs_stats("lo"))


class Analyzer(unittest.TestCase):
    def write_csv(self, rows, header="# drain_ms=4 hz=100\n"):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(header + "t,joint,latency_ms,status\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def synth(self):
        """bus A with joints a0,a1,a2 over 60 cycles (10 ms apart):
        a0: late in cycles 10..16 (run of 7, replies fast = left behind), late once at 40
        a1: late at cycle 30 and 40 together with a0 and a2 (simultaneous), absent at 20
        a2: late at 40; 2 lost.  bus B: b0 clean."""
        rows = []
        late = {"A/a0": set(range(10, 17)) | {40}, "A/a1": {30, 40}, "A/a2": {40}}
        absent = {"A/a1": {20}, "A/a0": {5}}
        for c in range(60):
            t = 1.0 + 0.010 * c
            for j in ("A/a0", "A/a1", "A/a2"):
                if c in absent.get(j, ()):
                    continue
                if c in late[j]:
                    rows.append((f"{t:.4f}", j, 1.2, "late"))
                else:
                    rows.append((f"{t:.4f}", j, 1.0, "ontime"))
            rows.append((f"{t + 0.0003:.4f}", "B/b0", 1.0, "ontime"))
        rows += [("", "A/a2", "", "lost")] * 2
        return rows

    def test_runs_simultaneity_and_counts(self):
        path = self.write_csv(self.synth())
        d = an.load(path)
        js = an.joint_stats(d)
        self.assertEqual(js["A/a2"]["lost"], 2)
        self.assertEqual(js["A/a0"]["late"], 8)
        self.assertEqual(js["A/a0"]["late_rows_faster_than_drain_pct"], 100.0)
        cy = an.analyse_cycles(d, 0.001, shifts=200)
        a0 = cy["A"]["per_joint"]["A/a0"]
        self.assertEqual(a0["late_runs"]["max"], 7)
        self.assertEqual(a0["late_runs"]["n_runs"], 2)
        self.assertEqual(a0["late_runs"]["hist"]["6-10"], 1)
        self.assertEqual(cy["A"]["cycles"], 60)
        self.assertEqual(cy["A"]["per_joint"]["A/a1"]["state_counts"]["A"], 1)
        sim = cy["A"]["simultaneity_late"]
        self.assertEqual(sim["observed_hist"]["3"], 1)               # cycle 40: all three joints late
        self.assertEqual(sim["cycles_with_ge_3"]["observed"], 1)
        self.assertEqual(cy["B"]["cycles"], 60)

    def test_absent_then_late_next(self):
        rows = self.synth()
        # make a1 late right after its absent cycle 20
        rows = [r for r in rows if not (r[1] == "A/a1" and r[0] == f"{1.0 + 0.010 * 21:.4f}")]
        rows.append((f"{1.0 + 0.010 * 21:.4f}", "A/a1", 1.1, "late"))
        d = an.load(self.write_csv(rows))
        cy = an.analyse_cycles(d, 0.001, shifts=50)
        self.assertEqual(cy["A"]["per_joint"]["A/a1"]["absent_then_late_next"]["followed_by_late"], 1)

    def test_run_helpers(self):
        self.assertEqual(an.runs_of([1, 1, 0, 1, 0, 0, 1, 1, 1]), [(0, 2), (3, 1), (6, 3)])
        pb = an.poisson_binomial([0.5, 0.5])
        self.assertEqual([round(x, 2) for x in pb], [0.25, 0.5, 0.25])
        self.assertAlmostEqual(an.phi([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
