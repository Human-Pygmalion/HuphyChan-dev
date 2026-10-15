#!/usr/bin/env python3
"""Drive the bench web page in a real browser against the FAKE motor and check what a user sees.

Adapted from tools/pygviewer/scripts/ux_check.py (same idea: do not read the code, press the
buttons).  Needs playwright + chromium:

    mujoco-sim/mjlab/.venv/bin/python3 System_ID/emf_bench/ux_check_bench.py --shots /some/dir

It starts its own `bench_server.py --fake` processes (never touches hardware), so nothing real
can be moved.  Two servers: one normal fake, one with `--fake-no-param` (old firmware: no reply
to Command 14) to check that the page says "no reply" instead of inventing values.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


class Checks:
  def __init__(self):
    self.rows: list[tuple[bool, str, str]] = []

  def add(self, ok, name, detail=""):
    self.rows.append((bool(ok), name, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""), flush=True)
    return bool(ok)

  def summary(self):
    bad = [r for r in self.rows if not r[0]]
    print(f"\n{len(self.rows) - len(bad)}/{len(self.rows)} passed")
    return 1 if bad else 0


def start_server(port: int, data_dir: str, extra: list[str]):
  p = subprocess.Popen([sys.executable, os.path.join(HERE, "bench_server.py"), "--fake", "--port", str(port),
                        "--data-dir", data_dir] + extra, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  for _ in range(50):
    try:
      urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.5).read()
      return p
    except Exception:  # noqa: BLE001
      time.sleep(0.1)
  raise RuntimeError("server did not start")


async def run(args) -> int:
  from playwright.async_api import async_playwright

  c = Checks()
  os.makedirs(args.shots, exist_ok=True)
  data_dir = tempfile.mkdtemp(prefix="emf_ux_")
  srv = start_server(args.port, data_dir, [])
  srv2 = start_server(args.port + 1, data_dir, ["--fake-no-param"])
  ov_dir = tempfile.mkdtemp(prefix="emf_ux_ov_")
  srv3 = start_server(args.port + 2, ov_dir, ["--fake-ov-gain", "2.0"])
  try:
    async with async_playwright() as p:
      b = await p.chromium.launch()
      pg = await b.new_page(viewport={"width": 1400, "height": 1500})
      errs: list[str] = []
      pg.on("pageerror", lambda e: errs.append(str(e)))
      pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
      await pg.goto(f"http://127.0.0.1:{args.port}/", wait_until="load")
      await pg.wait_for_timeout(800)

      async def st():
        return await pg.evaluate("fetch('/status').then(r => r.json())")

      async def txt(i):
        return (await pg.text_content(i) or "").strip()

      c.add("RS04" in await txt("#hdrModel") and "id 4" in await txt("#hdrModel"), "header shows model / id / channel", await txt("#hdrModel"))
      c.add(not await pg.is_hidden("#pFake"), "fake badge visible (simulated motor is labeled)")
      c.add("Disconnected" in await txt("#pConn"), "starts disconnected")

      # ---- connect
      await pg.click("#btnConnect")
      await pg.wait_for_timeout(1500)
      s = await st()
      c.add(s["connected"] and s["alive"] and "Connected" in await txt("#pConn"), "connect: pill says connected, motor alive", await txt("#pConn"))
      nbuf = await pg.evaluate("buf.length")
      c.add(nbuf > 150, "plots receive samples (~200 Hz)", f"{nbuf} samples in buffer after 1.5 s")
      c.add(190 < s["loop_hz"] < 210, "loop rate achieved ~200 Hz", f"{s['loop_hz']:.1f} Hz")
      c.add(s["vbus_ok"] and s["iqf_ok"] and s["vbus_pct"] > 90 and s["iqf_pct"] > 90, "VBUS and iqf replies arrive", f"vbus {s['vbus_pct']:.0f}% iqf {s['iqf_pct']:.0f}%")
      vb = await pg.evaluate("buf.filter(x => x.vbus !== null).map(x => x.vbus).slice(-5)")
      c.add(vb and all(47.5 < v < 48.5 for v in vb), "bus voltage ~48 V at rest", str([round(v, 2) for v in vb]))
      await pg.screenshot(path=os.path.join(args.shots, "1_connected.png"), full_page=True)

      # ---- gains
      await pg.fill("#kpNum", "20"); await pg.fill("#kdNum", "2")
      await pg.wait_for_timeout(300)
      s = await st()
      c.add(abs(s["kp"] - 20) < 1e-6 and abs(s["kd"] - 2) < 1e-6, "Kp / Kd number entry reaches the server", f"kp={s['kp']} kd={s['kd']}")
      await pg.fill("#kpRange", "30")                       # slider (input event)
      await pg.wait_for_timeout(300)
      s = await st()
      c.add(abs(s["kp"] - 30) < 1e-6 and await pg.input_value("#kpNum") == "30", "Kp slider moves number box and server", f"kp={s['kp']}")
      await pg.fill("#kpNum", "20"); await pg.wait_for_timeout(200)

      # ---- enable: no jump
      pos_before = (await st())["pos_deg"]
      await pg.click("#btnEnable")
      await pg.wait_for_timeout(600)
      s = await st()
      c.add(s["mode"] == 2 and "Torque on" in await txt("#pTorque"), "enable: motor reports Motor mode", await txt("#pTorque"))
      c.add(abs(s["pos_deg"] - pos_before) < 0.5 and abs(s["target_rel_deg"]) < 0.5, "enable does not jump (target := measured position)",
            f"pos {pos_before:.2f}->{s['pos_deg']:.2f} deg, target_rel {s['target_rel_deg']:.2f}")

      # ---- move target with the slider, watch tracking and regen
      await pg.fill("#tgtRange", "60")
      await pg.wait_for_timeout(2500)
      await pg.screenshot(path=os.path.join(args.shots, "2_after_step_60deg.png"), full_page=True)
      s = await st()
      posrel = await pg.evaluate("buf[buf.length-1].posRel")
      c.add(abs(posrel - 60) < 3, "slider target 60 deg: actual position follows", f"actual {posrel:.1f} deg relative")
      mins = await pg.evaluate("Math.min(...buf.filter(x => x.t > buf[buf.length-1].t - 3).map(x => x.p))")
      vmax = await pg.evaluate("Math.max(...buf.filter(x => x.vbus !== null && x.t > buf[buf.length-1].t - 3).map(x => x.vbus))")
      c.add(mins < -1.0, "regenerated power goes negative while braking", f"min {mins:.1f} W")
      c.add(vmax > 48.2, "bus voltage rises above 48 V during braking", f"max {vmax:.2f} V")
      c.add("window min" in await txt("#n3"), "voltage panel shows window min/max numbers", await txt("#n3"))
      await pg.click("button[data-step='30']")
      await pg.wait_for_timeout(300)
      c.add(abs(float(await pg.input_value("#tgtNum")) - 90) < 1e-6 and abs((await st())["target_rel_deg"] - 90) < 1e-6, "+30 deg step button", await pg.input_value("#tgtNum"))
      await pg.fill("#kpNum", "0"); await pg.wait_for_timeout(300)   # relax so the next steps are calm
      await pg.fill("#kpNum", "20")

      # ---- logging panel (default timestamp name, sine for a few seconds, stop, list, download, ZIP)
      dflt = await pg.input_value("#logName")
      c.add(re.fullmatch(r"\d{8}_\d{6}", dflt) is not None and not await pg.is_disabled("#btnLogStart") and await pg.is_disabled("#btnLogStop"),
            "logging panel: file name defaults to a YYYYMMDD_HHMMSS timestamp, Stop disabled", dflt)
      await pg.wait_for_timeout(1300)
      c.add(await pg.input_value("#logName") != dflt, "default name refreshes while not logging", await pg.input_value("#logName"))
      dflt = await pg.input_value("#logName")
      await pg.click("#btnLogStart")
      await pg.wait_for_timeout(300)
      s = await st()
      c.add(s["logging"] and s["log_file"] == dflt + ".csv" and await pg.is_disabled("#btnLogStart") and not await pg.is_disabled("#btnLogStop"),
            "Start logging: file named after the timestamp, buttons swap", f"{s['log_file']}")
      await pg.fill("#sineAmp", "10"); await pg.fill("#sinePer", "1"); await pg.wait_for_timeout(200)
      await pg.click("#btnSine")
      await pg.wait_for_timeout(3000)
      s = await st()
      stat = await txt("#logKv")
      c.add(s["sine_on"] and s["log_rows"] > 400 and "Rows" in stat and "Elapsed" in stat and "File size" in stat,
            "live row count / elapsed / size shown while logging with a sine running", f"rows={s['log_rows']} {stat[:80]}")
      await pg.screenshot(path=os.path.join(args.shots, "10_english_logging.png"), full_page=True)
      await pg.click("#btnSine")
      await pg.click("#btnLogStop")
      await pg.wait_for_timeout(600)
      s = await st()
      nrows_srv = s["log_rows"]
      c.add(not s["logging"] and await pg.is_disabled("#btnLogStop") and await pg.input_value("#logName") != dflt,
            "Stop logging: buttons swap back, name field offers a fresh timestamp", await pg.input_value("#logName"))
      await pg.click("#btnLogsRefresh"); await pg.wait_for_timeout(500)
      lst = json.loads(await pg.evaluate("fetch('/logs').then(r => r.text())"))
      mine = [x for x in lst["files"] if x["name"] == dflt + ".csv"]
      c.add(len(mine) == 1 and mine[0]["rows"] == nrows_srv > 0 and not mine[0]["open"] and dflt + ".csv" in await txt("#logsRows"),
            "file appears in Saved logs with rows > 0", str(mine))
      logdir = os.path.join(data_dir, "emf_logs")
      c.add(os.path.isfile(os.path.join(logdir, dflt + ".csv")) and os.path.isfile(os.path.join(logdir, dflt + ".meta.json")),
            "CSV and meta.json sidecar are in <data_dir>/emf_logs/")
      async with pg.expect_download() as dl_info:
        await pg.click(f"#logsRows a[href*='files={dflt}.csv']")
      dl = await dl_info.value
      got = open(await dl.path(), newline="").read()
      grows = list(csv.reader(io.StringIO(got)))
      want = ["t_s", "unix_s", "wall_time", "cycle", "cycle_dt_ms", "loop_period_ms", "rate_hz", "connected", "enabled_req", "state_ok", "mode",
              "mode_name", "fault", "warn", "fault_word", "fault_class", "fault_names", "temp_c", "pos_deg", "pos_rel_deg", "origin_deg",
              "vel_deg_s", "tau_nm", "target_deg", "target_rel_deg", "target_clamped", "track_err_deg", "kp", "kd", "sine_on",
              "sine_amp_deg", "sine_period_s", "sine_center_deg", "sine_phase_rad", "vbus_v", "iqf_a", "param_req", "power_w",
              "vbus_priority", "ov_count", "uv_count", "other_count", "warn_count", "lost_state", "lost_vbus", "lost_iqf",
              "send_fail", "overruns"]
      c.add(dl.suggested_filename == dflt + ".csv" and all(w in grows[0] for w in want) and len(grows) - 1 == nrows_srv,
            "downloaded CSV: file name, all expected columns, row count equals the server's", f"{dl.suggested_filename} cols={len(grows[0])} rows={len(grows) - 1}/{nrows_srv}")
      col = {n: i for i, n in enumerate(grows[0])}
      data = grows[1:]
      sine_rows = [r for r in data if r[col["sine_on"]] == "1"]
      both = sum(1 for r in data if r[col["vbus_v"]] != "" and r[col["iqf_a"]] != "")
      neither = sum(1 for r in data if r[col["vbus_v"]] == "" and r[col["iqf_a"]] == "")
      c.add(all(len(r) == len(grows[0]) for r in grows) and len(sine_rows) > 300 and
            sum(1 for r in sine_rows if r[col["mode"]] == "2") >= len(sine_rows) - 2 and
            max(abs(float(r[col["track_err_deg"]])) for r in sine_rows if r[col["track_err_deg"]] != "") > 0.01,
            "values: sine rows are in Motor mode and carry a tracking error",
            f"{len(sine_rows)} sine rows of {len(data)}")
      c.add(sum(1 for r in data if r[col["vbus_v"]] != "") > 100 and sum(1 for r in data if r[col["iqf_a"]] != "") > 100 and
            both <= 2 and neither <= 0.05 * len(data),
            "VBUS and iqf alternate: one value per cycle, not both, rarely neither",
            f"both={both} neither={neither} of {len(data)}; mid row: " + ",".join(f"{k}={data[len(data) // 2][i]}" for k, i in list(col.items())[:12]))
      ts_ = [float(r[col["t_s"]]) for r in data]
      dts = [float(r[col["cycle_dt_ms"]]) for r in data if r[col["cycle_dt_ms"]] != ""]
      c.add(ts_ == sorted(ts_) and 3.0 < sum(dts) / len(dts) < 7.0, "time monotonic, mean cycle_dt ~5 ms", f"mean {sum(dts) / len(dts):.2f} ms")
      # second (short) log so two files can be selected
      await pg.fill("#logName", "second one")
      await pg.click("#btnLogStart"); await pg.wait_for_timeout(800)
      c.add((await st())["log_file"] == "second_one.csv", "typed name is sanitized (space -> _) and .csv added", (await st())["log_file"])
      await pg.click("#btnLogStop"); await pg.wait_for_timeout(500)
      await pg.fill("#logName", "second one")
      await pg.click("#btnLogStart"); await pg.wait_for_timeout(300)
      s = await st()
      c.add(s["log_file"] == "second_one_1.csv", "existing name is never overwritten (_1 appended)", s["log_file"])
      await pg.click("#btnLogStop"); await pg.wait_for_timeout(500)
      await pg.click("#btnLogsRefresh"); await pg.wait_for_timeout(500)
      c.add(await pg.is_disabled("#btnDlSel"), "Download selected disabled with nothing selected")
      await pg.check(f"#logsRows input[data-name='{dflt}.csv']")
      await pg.check("#logsRows input[data-name='second_one.csv']")
      c.add("ZIP" in await txt("#btnDlSel"), "two selected: button announces a ZIP", await txt("#btnDlSel"))
      async with pg.expect_download() as dl_info:
        await pg.click("#btnDlSel")
      dl = await dl_info.value
      zf = zipfile.ZipFile(await dl.path())
      names = sorted(zf.namelist())
      zc = list(csv.reader(io.StringIO(zf.read(dflt + ".csv").decode())))
      c.add(dl.suggested_filename.endswith(".zip") and names == sorted([dflt + ".csv", "second_one.csv"]) and zc[0] == grows[0] and len(zc) == len(grows),
            "ZIP contains both selected files, content identical", f"{dl.suggested_filename} {names}")
      # path safety over HTTP
      codes = []
      for bad in ["..%2Fsecret.csv", "emf_logs%2F" + dflt + ".csv", "%2Fetc%2Fpasswd", "a.csv,..%2Fb.csv"]:
        try:
          urllib.request.urlopen(f"http://127.0.0.1:{args.port}/logs/download?files={bad}", timeout=3)
          codes.append(200)
        except urllib.error.HTTPError as e_:
          codes.append(e_.code)
      c.add(all(x in (400, 404) for x in codes), "path traversal in /logs/download is refused", str(codes))
      c.add(not (await st())["logging"], "logging off")

      # ---- Esc disables (focus inside a number box on purpose)
      await pg.focus("#kpNum")
      await pg.keyboard.press("Escape")
      await pg.wait_for_timeout(600)
      s = await st()
      c.add(s["mode"] == 0 and not s["enabled"] and "Torque off" in await txt("#pTorque"), "Esc disables torque (mode back to Reset)", await txt("#pTorque"))

      # ---- fault: hard gains trip the fake's 60 V over-voltage protection, then clear
      await pg.click("#btnEnable"); await pg.wait_for_timeout(400)
      await pg.fill("#kpNum", "60"); await pg.fill("#kdNum", "3"); await pg.fill("#tgtNum", "-150")
      await pg.wait_for_timeout(2500)
      s = await st()
      c.add(s["fault"] and "overvoltage" in s["fault_names"] and "Fault" in await txt("#pFault"), "fault shown and decoded (overvoltage)", f"{s['fault_names']} {await txt('#pFault')}")
      await pg.screenshot(path=os.path.join(args.shots, "3_overvoltage_fault.png"), full_page=True)
      await pg.click("#btnClear"); await pg.wait_for_timeout(600)
      s = await st()
      c.add(not s["fault"] and "No fault" in await txt("#pFault"), "clear-fault button clears the fault")
      await pg.click("#btnDisable")

      # ---- disconnect
      await pg.click("#btnConnect"); await pg.wait_for_timeout(500)
      c.add(not (await st())["connected"] and "Disconnected" in await txt("#pConn"), "disconnect")
      c.add(not errs, "no page errors / console errors", "; ".join(errs[:3]))
      kor = re.compile("[가-힣]")
      src = await pg.evaluate("fetch('/').then(r => r.text())")
      c.add(not kor.search(await pg.evaluate("document.body.innerText")) and not kor.search(src),
            "no Korean characters in the visible page text or the page source", str(kor.findall(src)[:5]))

      # ---- old firmware: no Command 14 reply -> "no reply", no invented numbers
      pg2 = await b.new_page(viewport={"width": 1400, "height": 1500})
      await pg2.goto(f"http://127.0.0.1:{args.port + 1}/", wait_until="load")
      await pg2.click("#btnConnect"); await pg2.wait_for_timeout(2200)
      s2 = await pg2.evaluate("fetch('/status').then(r => r.json())")
      kv = await pg2.text_content("#kv")
      nvb = await pg2.evaluate("buf.filter(x => x.vbus !== null || x.iqf !== null).length")
      c.add(not s2["vbus_ok"] and not s2["iqf_ok"] and "no reply" in kv and nvb == 0, "no Command 14 reply: page says no reply, plots hold no fabricated values", f"{nvb} param samples, status vbus_ok={s2['vbus_ok']}")
      c.add("No reply" in (await pg2.text_content("#n3")) and "0.4.1.29" in kv, "voltage panel and hint (firmware 0.4.1.29) shown")
      await pg2.screenshot(path=os.path.join(args.shots, "4_no_param_reply.png"), full_page=True)

      # ---- over-voltage monitor (fake with a high regen gain: a fast sine trips the 60 V fault)
      pg3 = await b.new_page(viewport={"width": 1400, "height": 1900})
      errs3: list[str] = []
      pg3.on("pageerror", lambda e: errs3.append(str(e)))
      pg3.on("console", lambda m: errs3.append(m.text) if m.type == "error" else None)
      await pg3.goto(f"http://127.0.0.1:{args.port + 2}/", wait_until="load")
      await pg3.wait_for_timeout(500)

      async def st3():
        return await pg3.evaluate("fetch('/status').then(r => r.json())")

      async def t3(i):
        return (await pg3.text_content(i) or "").strip()

      c.add(await pg3.is_visible("#ovCard") and "Over-voltage monitor" in (await pg3.text_content("#ovCard")), "over-voltage monitor card is on the page")
      await pg3.click("#btnConnect"); await pg3.wait_for_timeout(2600)      # rates are measured over 2 s
      s3 = await st3()
      c.add(s3["ov"]["counts"]["overvoltage"] == 0 and await t3("#cOv") == "0" and s3["ov"]["csv"] and "emf_events_" in s3["ov"]["csv"],
            "monitor starts at 0 and an events file is created at connect", str(s3["ov"]["csv"]))
      hz_alt = s3["vbus_hz"]
      c.add("1 ms" in await t3("#ovHint") and "oscilloscope" in await t3("#ovHint") and "Hz" in await t3("#ovHint"),
            "sampling-rate caveat (1 ms spikes can be missed) is shown", await t3("#ovHint"))
      # VBUS priority toggle
      await pg3.click("#chkVbusPri"); await pg3.wait_for_timeout(2600)
      s3 = await st3()
      c.add(s3["vbus_priority"] and s3["vbus_hz"] > 1.7 * hz_alt and "not read" in await t3("#kv") and await pg3.is_checked("#chkVbusPri"),
            "VBUS-priority checkbox: VBUS read every cycle, iqf off", f"vbus {hz_alt:.0f} Hz -> {s3['vbus_hz']:.0f} Hz, iqf_ok={s3['iqf_ok']}")
      iq = await pg3.evaluate("buf.filter(x => x.iqf !== null && x.t > buf[buf.length-1].t - 1).length")
      c.add(iq == 0 and "current read off" in await t3("#n2"), "no iqf samples are drawn while VBUS priority is on", await t3("#n2"))

      # enable, fast sine -> over-voltage fault
      await pg3.fill("#kpNum", "20"); await pg3.wait_for_timeout(200)
      await pg3.click("#btnEnable"); await pg3.wait_for_timeout(500)
      await pg3.fill("#sineAmp", "40"); await pg3.fill("#sinePer", "0.5"); await pg3.wait_for_timeout(200)
      await pg3.click("#btnSine")
      for _ in range(80):
        if (await st3())["ov"]["counts"]["overvoltage"] >= 1:
          break
        await pg3.wait_for_timeout(100)
      await pg3.wait_for_timeout(800)
      s3 = await st3()
      c.add(s3["ov"]["counts"]["overvoltage"] == 1 and await t3("#cOv") == "1", "over-voltage counter increases after a fast sine", f"counts {s3['ov']['counts']}")
      rows_txt = await t3("#ovRows")
      c.add("Over-voltage" in rows_txt and "0x00000008" in rows_txt, "event row appears in the table (class + fault word)", rows_txt[:120])
      ev = s3["ov_events"][0]
      c.add(ev["vbus_max_1s"] > 52 and ev["power_min_1s_w"] < 0 and ev["sine_on"] and ev["kp"] > 0 and ev["wall"],
            "event record carries VBUS max/min, regen power, gains and sine settings",
            f"vbus {ev['vbus_min_1s']:.1f}..{ev['vbus_max_1s']:.1f} V, pmin {ev['power_min_1s_w']:.1f} W, sine {ev['sine_amp_deg']}/{ev['sine_period_s']}")
      c.add(s3["ov"]["vbus"]["above"][0]["n"] > 0 and "52" in await t3("#ovKv"), "threshold counts (>52 V) shown", await t3("#ovKv"))
      c.add("Margin to 60 V" in await t3("#ovKv") and "margin to 60 V" in await t3("#n3"), "margin to 60 V shown in the card and on the VBUS plot", await t3("#n3"))
      c.add(s3["mode"] == 0 and s3["fault"], "no auto-clear / auto-enable: motor stays faulted in Reset")
      files = glob.glob(os.path.join(ov_dir, "emf_logs", "emf_events_*.csv"))
      erows = list(csv.reader(open(files[0]))) if files else []
      c.add(len(files) == 1 and len(erows) == 2 and erows[1][5] == "overvoltage" and erows[1][6] == "0x00000008",
            "event persisted to emf_events_*.csv (flushed)", f"{[os.path.basename(f) for f in files]} {erows[1][:7] if len(erows) > 1 else ''}")
      # reload: the counters live on the server
      await pg3.reload(wait_until="load"); await pg3.wait_for_timeout(900)
      c.add(await t3("#cOv") == "1" and "Over-voltage" in await t3("#ovRows"), "page reload keeps the counter and the table (server side state)")
      await pg3.screenshot(path=os.path.join(args.shots, "6_ov_monitor.png"), full_page=True)
      c.add(not re.search("[가-힣]", await pg3.evaluate("document.body.innerText")), "no Korean characters on the over-voltage page after a fault (events table, labels)")

      # second event after clear + enable + sine, then Reset clears the counters (the file keeps both)
      await pg3.click("#btnClear"); await pg3.wait_for_timeout(500)
      await pg3.click("#btnEnable"); await pg3.wait_for_timeout(500)
      await pg3.click("#btnSine")
      for _ in range(80):
        if (await st3())["ov"]["counts"]["overvoltage"] >= 2:
          break
        await pg3.wait_for_timeout(100)
      s3 = await st3()
      c.add(s3["ov"]["counts"]["overvoltage"] == 2 and await t3("#cOv") == "2", "second over-voltage counted", f"counts {s3['ov']['counts']}")
      await pg3.click("#btnOvReset"); await pg3.wait_for_timeout(500)
      s3 = await st3()
      c.add(s3["ov"]["faults"] == 0 and await t3("#cOv") == "0" and "No fault events yet" in await t3("#ovRows") and (s3["ov"]["vbus"]["max"] or 0) < 55,
            "Reset clears counters, table and VBUS statistics (new samples keep arriving)", f"{s3['ov']['counts']} vbus max {s3['ov']['vbus']['max']}")
      with open(files[0]) as fh:
        erows = list(csv.reader(fh))
      c.add(len(erows) == 3, "Reset keeps the events file (2 events, header)", f"{len(erows)} lines")
      await pg3.click("#btnDisable")
      c.add(not errs3, "no page errors in the monitor page", "; ".join(errs3[:3]))
      await b.close()
  finally:
    for pr in (srv, srv2, srv3):
      pr.terminate()
    for pr in (srv, srv2, srv3):
      try:
        pr.wait(timeout=5)
      except subprocess.TimeoutExpired:
        pr.kill()
  return c.summary()


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=8191)
  ap.add_argument("--shots", default=os.path.join(HERE, "ux_shots"))
  sys.exit(asyncio.run(run(ap.parse_args())))
