#!/usr/bin/env python3
"""Click-through check of the trail teleop (rule: verify UIs on screen, not in code).
Keyboard page: W x4 -> command 0.4 m/s; 3D page: status text 'walked X m' must grow; screenshot; Space -> stop.
usage: ux_check_teleop.py [host] [out.png]"""
import re
import sys
import time

from playwright.sync_api import sync_playwright

host = sys.argv[1] if len(sys.argv) > 1 else "10.8.0.14"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/teleop_ux.png"
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    view = b.new_page(viewport={"width": 1280, "height": 760})
    view.goto(f"http://{host}:8095/")
    view.wait_for_timeout(15000)
    keys = b.new_page()
    keys.goto(f"http://{host}:8096/")
    for _ in range(4):
        keys.keyboard.press("w")
        keys.wait_for_timeout(300)
    print("keyboard page says:", keys.inner_text("#s"))

    def walked():
        m = re.search(r"walked ([\d.]+) m", view.inner_text("body"))
        return float(m.group(1)) if m else None

    w0 = walked()
    view.wait_for_timeout(10000)
    w1 = walked()
    print(f"walked: {w0} -> {w1} m in 10 s", "OK" if (w0 is not None and w1 is not None and w1 - w0 > 2.0) else "FAIL")
    print("status:", re.search(r"(walking[^\n]*|FALLEN[^\n]*)", view.inner_text("body")).group(1))
    view.screenshot(path=out)
    keys.keyboard.press(" ")
    keys.wait_for_timeout(500)
    print("after Space:", keys.inner_text("#s"))
    b.close()
