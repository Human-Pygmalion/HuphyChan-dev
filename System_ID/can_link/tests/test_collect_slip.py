"""(1) old collect keeps a one-cycle slip after ONE late reply, (2) new collect recovers next cycle,
(3) the outstanding guard keeps unanswered frames per motor <= K.   No hardware, virtual clock."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as h  # noqa: E402
import scenarios as sc  # noqa: E402


class SlipOld(unittest.TestCase):
    def test_old_collect_slip_never_recovers(self):
        for late_motor in range(7, 13):
            r = sc.late_once(h.orig(), late_motor=late_motor)
            self.assertEqual(r["shifted_cycles_after_glitch"], r["cycles_after_glitch"],
                             f"old HUPHY collect should stay shifted to the end (late motor {late_motor}): {r}")
            self.assertIsNone(r["first_clean_cycle_after"])

    def test_old_slip_is_one_cycle_on_the_last_motor_of_the_burst(self):
        r = sc.late_once(h.orig())
        self.assertEqual(r["stale_cycles_per_motor"][12], r["cycles_after_glitch"])
        self.assertTrue(all(r["stale_cycles_per_motor"][m] == 0 for m in range(7, 12)))


class SlipNew(unittest.TestCase):
    def test_new_collect_recovers_in_the_next_cycle(self):
        for late_motor in range(7, 13):
            r = sc.late_once(h.patched(), late_motor=late_motor)
            self.assertEqual(r["shifted_cycles_after_glitch"], 0, r)
            self.assertEqual(r["first_clean_cycle_after"], r["glitch_cycle"] + 1)

    def test_late_reply_is_flagged_with_its_true_age_not_counted_as_fresh(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S,
                    scripts={9: (lambda m, i: sc.LATE_EXTRA_S if i == 5 else 0.0)})
        res = h.run_cycles(rig, 8, sc.PERIOD)
        st5, st6 = res[5]["stats"], res[6]["stats"]
        self.assertEqual(st5.missing, (9,))
        self.assertTrue(st5.timed_out)
        self.assertEqual(st6.late, (9,))
        self.assertEqual(st6.missing, ())
        # the late frame arrived ~6.9 ms after cycle 5's burst; collect 6 ends ~4.5 ms after it
        self.assertGreater(st6.late_age_s[9], 0.003)
        self.assertLess(st6.late_age_s[9], 0.008)
        self.assertEqual(rig.rs.link.late_replies, 1)
        rig.close()

    def test_latest_reply_per_motor_wins_when_several_arrive(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S,
                    scripts={9: (lambda m, i: sc.LATE_EXTRA_S if i == 5 else 0.0)})
        res = h.run_cycles(rig, 8, sc.PERIOD)
        # cycle 6 saw m9's stale reply (pos 5) AND the fresh one (pos 6); state must be the fresh one
        self.assertEqual(res[6]["lag"][9], 0)
        rig.close()

    def test_clean_run_has_no_timeouts_and_returns_as_soon_as_all_arrived(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S)
        res = h.run_cycles(rig, 30, sc.PERIOD)
        self.assertTrue(all(not r["missing"] for r in res))
        self.assertEqual(rig.rs.link.timeouts, 0)
        self.assertTrue(all(r["stats"].waited_s < 0.002 for r in res))
        rig.close()

    def test_dead_motor_costs_one_drain_per_cycle_when_guard_is_off(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S, scripts={8: (lambda m, i: None)},
                    bus_kwargs=dict(max_outstanding=0))
        res = h.run_cycles(rig, 10, sc.PERIOD)
        self.assertTrue(all(r["missing"] == [8] for r in res))
        self.assertTrue(all(abs(r["stats"].waited_s - sc.DRAIN_S) < 1e-4 for r in res))
        rig.close()

    def test_dead_motor_with_guard_is_still_reported_missing_every_cycle_but_waited_for_less(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S, scripts={8: (lambda m, i: None)})
        res = h.run_cycles(rig, 40, sc.PERIOD)
        self.assertTrue(all(r["missing"] == [8] for r in res))      # link-loss counting is unchanged
        waits = [r["stats"].waited_s for r in res]
        self.assertLess(sum(waits), 0.6 * sc.DRAIN_S * len(res))    # guard skips the dead motor most cycles
        rig.close()


class OutstandingGuard(unittest.TestCase):
    def test_original_piles_up_and_the_motor_drops_frames(self):
        r = sc.slow_motor(h.orig())
        self.assertEqual(r["max_unanswered_at_motor"], 3)
        self.assertGreater(r["frames_dropped_by_motor_buffer"], 0)

    def test_guard_keeps_unanswered_at_or_below_K(self):
        for K in (1, 2, 3):
            r = sc.slow_motor(h.patched(), max_outstanding=K)
            self.assertLessEqual(r["max_unanswered_at_motor"], K, r)
            self.assertEqual(r["frames_dropped_by_motor_buffer"], 0, r)
            self.assertGreater(r["skipped_by_guard"], 0 if K < 3 else -1)
            self.assertTrue(r["other_motors_always_answered"])

    def test_guard_off_reproduces_the_original_pile_up(self):
        r = sc.slow_motor(h.patched(), max_outstanding=0)
        self.assertEqual(r["max_unanswered_at_motor"], 3)
        self.assertEqual(r["frames_sent_to_motor"], 60)

    def test_K_above_the_motor_buffer_is_refused(self):
        with self.assertRaises(ValueError):
            h.Rig(h.patched(), bus_kwargs=dict(max_outstanding=4))
        h.fc.uninstall_fake_can_module()

    def test_disable_torque_is_never_skipped_by_the_guard(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S, scripts={9: (lambda m, i: 0.5)})
        h.run_cycles(rig, 6, sc.PERIOD)                      # motor 9 now has 2 unanswered, guard closed
        before = len([1 for _, cid, d in rig.wire.tx_sent_log if cid == 9])
        rig.rs.disable_torque([9])
        rig.clock.sleep(0.01)
        rig.wire.recv(0.01)
        stops = [d for _, cid, d in rig.wire.tx_sent_log if cid == 9 and d[6:] == b"\xff\xfd"]
        self.assertEqual(len(stops), 1)
        self.assertGreater(len([1 for _, cid, d in rig.wire.tx_sent_log if cid == 9]), before)
        rig.close()

    def test_stale_records_expire_so_a_recovered_motor_is_commanded_again(self):
        rig = h.Rig(h.patched(), drain_s=sc.DRAIN_S,
                    scripts={9: (lambda m, i: None if i < 2 else 0.0)})   # first 2 replies lost, then healthy
        res = h.run_cycles(rig, 30, sc.PERIOD)
        late_cycles = [r["k"] for r in res if 9 not in r["missing"]]
        self.assertTrue(len(late_cycles) >= 20, "motor 9 should be answering again after the ttl")
        rig.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
