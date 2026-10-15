#!/usr/bin/env python3
"""Log-file helper checks (names, uniqueness, path safety, columns, zip), no hardware.  Run:  python3 test_logs.py"""

import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emf_logs as lg  # noqa: E402


class NameTests(unittest.TestCase):
  def test_default_stamp_format(self):
    s = lg.default_stamp()
    self.assertRegex(s, r"^\d{8}_\d{6}$")

  def test_sanitize(self):
    f = lg.sanitize_log_name
    self.assertEqual(f("run1", "S"), "run1.csv")
    self.assertEqual(f("run1.csv", "S"), "run1.csv")
    self.assertEqual(f("RUN1.CSV", "S"), "RUN1.csv")
    self.assertEqual(f("my test/1", "S"), "my_test_1.csv")
    self.assertEqual(f("../../etc/passwd", "S"), "_._etc_passwd.csv")
    self.assertNotIn("..", f("a..b", "S"))
    self.assertEqual(f("", "20261015_101010"), "20261015_101010.csv")
    self.assertEqual(f(None, "20261015_101010"), "20261015_101010.csv")
    self.assertEqual(f("   ", "20261015_101010"), "20261015_101010.csv")
    self.assertEqual(f("한글", "20261015_101010"), "20261015_101010.csv")      # nothing allowed left
    self.assertEqual(f(".hidden", "S"), "hidden.csv")
    self.assertEqual(f("...", "S"), "S.csv")
    self.assertTrue(f("x" * 500, "S").endswith(".csv"))
    self.assertLessEqual(len(f("x" * 500, "S")), lg.MAX_NAME + 4)
    for raw in ["a b", "a/b", "a\\b", "a\0b", "a;rm -rf", "é", "a:b"]:
      self.assertRegex(f(raw, "S"), r"^[A-Za-z0-9._-]+\.csv$")

  def test_unique_never_overwrites(self):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as legacy:
      names = []
      for _ in range(3):
        fh, p = lg.open_unique(d, "run.csv", (legacy,))
        fh.write("x")
        fh.close()
        names.append(os.path.basename(p))
      self.assertEqual(names, ["run.csv", "run_1.csv", "run_2.csv"])
      open(os.path.join(legacy, "emf_old.csv"), "w").close()                 # same name in the legacy folder
      fh, p = lg.open_unique(d, "emf_old.csv", (legacy,))
      fh.close()
      self.assertEqual(os.path.basename(p), "emf_old_1.csv")
      open(os.path.join(d, "side.meta.json"), "w").close()                  # a stray sidecar blocks its name too
      fh, p = lg.open_unique(d, "side.csv")
      fh.close()
      self.assertEqual(os.path.basename(p), "side_1.csv")
      self.assertEqual(open(os.path.join(d, "run.csv")).read(), "x")


class SafetyTests(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = self.tmp.name
    self.log = os.path.join(self.root, "emf_logs")
    os.makedirs(self.log)
    for p, body in [(os.path.join(self.log, "a.csv"), "h\n1\n2\n"), (os.path.join(self.log, "a.meta.json"), "{}\n"),
                    (os.path.join(self.root, "emf_old.csv"), "h\n1\n"), (os.path.join(self.root, "secret.csv"), "no\n"),
                    (os.path.join(self.root, "notes.txt"), "no\n")]:
      with open(p, "w") as fh:
        fh.write(body)

  def tearDown(self):
    self.tmp.cleanup()

  def test_resolve(self):
    r = lambda n: lg.resolve_download(n, self.log, self.root)  # noqa: E731
    self.assertEqual(r("a.csv"), os.path.join(self.log, "a.csv"))
    self.assertEqual(r("a.meta.json"), os.path.join(self.log, "a.meta.json"))
    self.assertEqual(r("emf_old.csv"), os.path.join(self.root, "emf_old.csv"))     # legacy
    self.assertIsNone(r("secret.csv"))            # in the data dir but not an emf_ file
    self.assertIsNone(r("notes.txt"))
    self.assertIsNone(r("missing.csv"))
    for bad in ["../secret.csv", "..", "../emf_logs/a.csv", "emf_logs/a.csv", "/etc/passwd", "a/../a.csv", "a.csv/",
                "..\\a.csv", "a.csv\0", "", ".hidden.csv", "a..csv", "~/a.csv", "%2e%2e/a.csv"]:
      self.assertIsNone(r(bad), bad)
      self.assertFalse(lg.is_safe_name(bad), bad)

  def test_list_and_rows(self):
    rows = lg.list_logs(self.log, self.root, {"a.csv": 99})
    by = {r["name"]: r for r in rows}
    self.assertEqual(set(by), {"a.csv", "emf_old.csv"})                        # secret.csv / txt / meta not listed
    self.assertEqual(by["a.csv"]["rows"], 99)                                  # live count for the open file
    self.assertTrue(by["a.csv"]["open"] and not by["emf_old.csv"]["open"])
    self.assertTrue(by["emf_old.csv"]["legacy"] and not by["a.csv"]["legacy"])
    self.assertEqual(by["emf_old.csv"]["rows"], 1)
    self.assertEqual(by["a.csv"]["meta"], "a.meta.json")
    self.assertEqual(lg.count_rows(os.path.join(self.log, "a.csv")), 2)

  def test_partial_row_is_cut(self):
    p = os.path.join(self.log, "p.csv")
    with open(p, "w") as fh:
      fh.write("h\n1,2\n3,")
    n = lg.complete_length(p, os.path.getsize(p))
    with open(p) as fh:
      self.assertEqual(fh.read()[:n], "h\n1,2\n")
    z, size = lg.build_zip([p, os.path.join(self.log, "a.csv")])
    with zipfile.ZipFile(io.BytesIO(z.read())) as zf:
      self.assertEqual(zf.namelist(), ["p.csv", "a.csv"])
      self.assertEqual(zf.read("p.csv").decode(), "h\n1,2\n")
      self.assertEqual(zf.read("a.csv").decode(), "h\n1\n2\n")


class ColumnTests(unittest.TestCase):
  def test_columns_unique_and_described(self):
    names = lg.COLUMN_NAMES
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(len(names), len(lg.LOG_COLUMNS))
    need = ["t_s", "unix_s", "wall_time", "cycle", "cycle_dt_ms", "loop_period_ms", "connected", "enabled_req", "mode",
            "fault", "warn", "fault_word", "fault_class", "temp_c", "pos_deg", "pos_rel_deg", "vel_deg_s", "tau_nm",
            "target_deg", "target_rel_deg", "track_err_deg", "kp", "kd", "sine_on", "sine_amp_deg", "sine_period_s",
            "sine_center_deg", "sine_phase_rad", "vbus_v", "iqf_a", "param_req", "power_w", "vbus_priority", "rate_hz",
            "ov_count", "uv_count", "other_count", "warn_count", "lost_state", "lost_vbus", "lost_iqf"]
    for n in need:
      self.assertIn(n, names)
    for _, d in lg.LOG_COLUMNS:
      self.assertTrue(d)


class ServerLogTests(unittest.TestCase):
  """Start the fake server, log for a moment, check the file, the sidecar and the HTTP endpoints."""

  def test_end_to_end(self):
    port = 8197
    with tempfile.TemporaryDirectory() as d:
      p = subprocess.Popen([sys.executable, os.path.join(HERE, "bench_server.py"), "--fake", "--port", str(port),
                            "--data-dir", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      base = f"http://127.0.0.1:{port}"

      def post(obj):
        rq = urllib.request.Request(base + "/cmd", json.dumps(obj).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(rq, timeout=5).read())

      try:
        for _ in range(60):
          try:
            urllib.request.urlopen(base + "/status", timeout=0.5).read()
            break
          except Exception:  # noqa: BLE001
            time.sleep(0.1)
        post({"action": "connect"})
        time.sleep(0.5)
        r = post({"action": "log_start", "name": "e2e test"})
        self.assertTrue(r["ok"])
        self.assertTrue(r["status"]["logging"] and r["status"]["log_file"] == "e2e_test.csv")
        r2 = post({"action": "log_start", "name": "x"})                         # second start refused
        self.assertFalse(r2["ok"])
        time.sleep(1.5)
        lst = json.loads(urllib.request.urlopen(base + "/logs").read())
        row = next(x for x in lst["files"] if x["name"] == "e2e_test.csv")
        self.assertTrue(row["open"] and row["rows"] > 100)
        r = post({"action": "log_stop"})
        self.assertFalse(r["status"]["logging"])
        r = post({"action": "log_start", "name": "e2e_test"})                   # same name -> _1, original untouched
        self.assertEqual(r["status"]["log_file"], "e2e_test_1.csv")
        self.assertIn("already exists", r["note"])
        post({"action": "log_stop"})
        path = os.path.join(d, "emf_logs", "e2e_test.csv")
        with open(path, newline="") as fh:
          rows = list(csv.reader(fh))
        self.assertEqual(rows[0], lg.COLUMN_NAMES)
        self.assertTrue(all(len(x) == len(rows[0]) for x in rows))
        self.assertGreater(len(rows), 100)
        ts = [float(x[0]) for x in rows[1:]]
        self.assertEqual(ts, sorted(ts))
        with open(os.path.join(d, "emf_logs", "e2e_test.meta.json")) as fh:
          meta = json.load(fh)
        self.assertTrue(meta["complete"] and meta["rows"] == len(rows) - 1 and meta["motor"]["model"] == "RS04")
        self.assertEqual([c["name"] for c in meta["columns"]], lg.COLUMN_NAMES)
        # download endpoints
        body = urllib.request.urlopen(base + "/logs/download?files=e2e_test.csv").read().decode()
        self.assertEqual(body.splitlines()[0], ",".join(lg.COLUMN_NAMES))
        z = urllib.request.urlopen(base + "/logs/download?files=e2e_test.csv,e2e_test_1.csv").read()
        self.assertEqual(sorted(zipfile.ZipFile(io.BytesIO(z)).namelist()), ["e2e_test.csv", "e2e_test_1.csv"])
        for bad in ["../x.csv", "emf_logs/e2e_test.csv", "e2e_test.csv,../x", ""]:
          with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(base + "/logs/download?files=" + urllib.request.quote(bad, safe=","))
          self.assertIn(cm.exception.code, (400, 404))
        with self.assertRaises(urllib.error.HTTPError) as cm:
          urllib.request.urlopen(base + "/logs/download?files=nope.csv")
        self.assertEqual(cm.exception.code, 404)
        # the event CSV of the connection lives in emf_logs too
        self.assertTrue([n for n in os.listdir(os.path.join(d, "emf_logs")) if n.startswith("emf_events_")])
      finally:
        p.terminate()
        p.wait(timeout=5)


if __name__ == "__main__":
  unittest.main(verbosity=2)
