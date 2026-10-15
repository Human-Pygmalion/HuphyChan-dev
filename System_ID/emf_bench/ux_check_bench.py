#!/usr/bin/env python3
"""Drive the bench web page in a real browser against the FAKE motor and check what a user sees.

Adapted from tools/pygviewer/scripts/ux_check.py (same idea: do not read the code, press the
buttons).  Needs playwright + chromium:

    mujoco-sim/mjlab/.venv/bin/python3 System_ID/emf_bench/ux_check_bench.py --shots /some/dir

It starts its own `bench_server.py --fake` processes (never touches hardware), so nothing real
can be moved.  Two servers: one normal fake, one with `--fake-no-param` (old firmware: no reply
to Command 14) to check that the page says "응답 없음" instead of inventing values.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

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
      c.add("연결 안 됨" in await txt("#pConn"), "starts disconnected")

      # ---- connect
      await pg.click("#btnConnect")
      await pg.wait_for_timeout(1500)
      s = await st()
      c.add(s["connected"] and s["alive"] and "연결됨" in await txt("#pConn"), "connect: pill says connected, motor alive", await txt("#pConn"))
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
      c.add(s["mode"] == 2 and "토크 켜짐" in await txt("#pTorque"), "enable: motor reports Motor mode", await txt("#pTorque"))
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
      c.add("구간 최소" in await txt("#n3"), "voltage panel shows window min/max numbers", await txt("#n3"))
      await pg.click("button[data-step='30']")
      await pg.wait_for_timeout(300)
      c.add(abs(float(await pg.input_value("#tgtNum")) - 90) < 1e-6 and abs((await st())["target_rel_deg"] - 90) < 1e-6, "+30 deg step button", await pg.input_value("#tgtNum"))
      await pg.fill("#kpNum", "0"); await pg.wait_for_timeout(300)   # relax so the next steps are calm
      await pg.fill("#kpNum", "20")

      # ---- record
      await pg.click("#btnRecord")
      await pg.wait_for_timeout(1500)
      s = await st()
      c.add(s["recording"] and s["rec_rows"] > 150 and "행" in await txt("#kv"), "recording on: file name and row count shown", f"{s['rec_file']} rows={s['rec_rows']}")
      await pg.click("#btnRecord")
      await pg.wait_for_timeout(300)
      files = glob.glob(os.path.join(data_dir, "emf_*.csv"))
      rows = list(csv.reader(open(files[0]))) if files else []
      ts = [float(r[0]) for r in rows[1:]]
      gap = max((b_ - a_ for a_, b_ in zip(ts, ts[1:], strict=False)), default=0.0)
      c.add(len(files) == 1 and len(rows) > 150 and rows[0][0] == "t_s" and "vbus_v" in rows[0], "CSV written with header and rows", f"{files} rows={len(rows)} largest time gap {gap * 1000:.0f} ms")
      c.add(not (await st())["recording"], "recording off")

      # ---- Esc disables (focus inside a number box on purpose)
      await pg.focus("#kpNum")
      await pg.keyboard.press("Escape")
      await pg.wait_for_timeout(600)
      s = await st()
      c.add(s["mode"] == 0 and not s["enabled"] and "꺼짐" in await txt("#pTorque"), "Esc disables torque (mode back to Reset)", await txt("#pTorque"))

      # ---- fault: hard gains trip the fake's 60 V over-voltage protection, then clear
      await pg.click("#btnEnable"); await pg.wait_for_timeout(400)
      await pg.fill("#kpNum", "60"); await pg.fill("#kdNum", "3"); await pg.fill("#tgtNum", "-150")
      await pg.wait_for_timeout(2500)
      s = await st()
      c.add(s["fault"] and "overvoltage" in s["fault_names"] and "고장" in await txt("#pFault"), "fault shown and decoded (overvoltage)", f"{s['fault_names']} {await txt('#pFault')}")
      await pg.screenshot(path=os.path.join(args.shots, "3_overvoltage_fault.png"), full_page=True)
      await pg.click("#btnClear"); await pg.wait_for_timeout(600)
      s = await st()
      c.add(not s["fault"] and "고장 없음" in await txt("#pFault"), "clear-fault button clears the fault")
      await pg.click("#btnDisable")

      # ---- disconnect
      await pg.click("#btnConnect"); await pg.wait_for_timeout(500)
      c.add(not (await st())["connected"] and "연결 안 됨" in await txt("#pConn"), "disconnect")
      c.add(not errs, "no page errors / console errors", "; ".join(errs[:3]))

      # ---- old firmware: no Command 14 reply -> "응답 없음", no invented numbers
      pg2 = await b.new_page(viewport={"width": 1400, "height": 1500})
      await pg2.goto(f"http://127.0.0.1:{args.port + 1}/", wait_until="load")
      await pg2.click("#btnConnect"); await pg2.wait_for_timeout(2200)
      s2 = await pg2.evaluate("fetch('/status').then(r => r.json())")
      kv = await pg2.text_content("#kv")
      nvb = await pg2.evaluate("buf.filter(x => x.vbus !== null || x.iqf !== null).length")
      c.add(not s2["vbus_ok"] and not s2["iqf_ok"] and "응답 없음" in kv and nvb == 0, "no Command 14 reply: page says 응답 없음, plots hold no fabricated values", f"{nvb} param samples, status vbus_ok={s2['vbus_ok']}")
      c.add("응답 없음" in (await pg2.text_content("#n3")) and "0.4.1.29" in kv, "voltage panel and hint (firmware 0.4.1.29) shown")
      await pg2.screenshot(path=os.path.join(args.shots, "4_no_param_reply.png"), full_page=True)
      await b.close()
  finally:
    for pr in (srv, srv2):
      pr.terminate()
    for pr in (srv, srv2):
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
