#!/usr/bin/env python3
"""Click-through check of the trail teleop page (rule: verify UIs on screen, not in code).

  1. open http://<host>:8095/, time until the loading box disappears (terrain + robot drawn)
  2. keyboard ON THE PAGE: W x4 -> command 0.40 m/s; 'walked' must grow over 10 s; contact panel must name the heightfield
  3. screenshot <out>; switch on 'LiDAR terrain map' + 'LiDAR beams', wait, screenshot <out>_lidar.png; Space -> stop
  4. the same page must also answer on :8096
usage: ux_check_teleop.py [host] [out.png]
"""
import re
import sys
import time

from playwright.sync_api import sync_playwright

host = sys.argv[1] if len(sys.argv) > 1 else "10.8.0.14"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/teleop_ux.png"
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    page = b.new_page(viewport={"width": 1400, "height": 820})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    t0 = time.time()
    page.goto(f"http://{host}:8095/")
    page.wait_for_selector("#load", state="detached", timeout=120000)
    print(f"page ready (terrain + robot drawn) in {time.time() - t0:.1f} s")
    page.wait_for_timeout(3000)
    page.mouse.click(400, 400)  # focus the 3D view, as a user would
    for _ in range(4):
        page.keyboard.press("w")
        page.wait_for_timeout(250)

    def walked():
        m = re.search(r"walked ([\d.]+) m", page.inner_text("#status"))
        return float(m.group(1)) if m else None

    print("command after W x4:", re.search(r"command [^\n]*", page.inner_text("#status")).group(0))
    w0 = walked()
    page.wait_for_timeout(10000)
    w1 = walked()
    print(f"walked: {w0} -> {w1} m in 10 s", "OK" if (w0 is not None and w1 is not None and w1 - w0 > 2.0) else "FAIL")
    print("status:", page.inner_text("#status").splitlines()[0])
    print("contact:", " | ".join(page.inner_text("#contact").splitlines()))
    page.screenshot(path=out)
    page.check("#s_elev")
    page.check("#s_beams")
    page.wait_for_timeout(8000)
    print("lidar:", " | ".join(page.inner_text("#lidar").splitlines()))
    page.screenshot(path=out.replace(".png", "_lidar.png"))
    page.keyboard.press(" ")
    page.wait_for_timeout(800)
    print("after Space:", re.search(r"command [^\n]*", page.inner_text("#status")).group(0))
    p2 = b.new_page()
    r = p2.goto(f"http://{host}:8096/")
    print("port 8096:", r.status, p2.title())
    print("page errors:", errors or "none")
    b.close()
