#!/usr/bin/env python3
"""Joystick check of the trail teleop page (rule: verify UIs on screen, not in code).

  1. https page (self-signed, warning accepted): secure context, so the Gamepad API is available.
     A fake standard gamepad is injected (no real controller in the test machine): left stick fully up for 6 s ->
     command must be 1.20 m/s and the robot must walk; stick back to centre -> 0; button B (stop), A (get up).
  2. http page: the on-screen stick is dragged up with the mouse -> command > 0 and walking; release -> 0.
usage: ux_check_joystick.py [host] [out.png]
"""
import re
import sys

from playwright.sync_api import sync_playwright

host = sys.argv[1] if len(sys.argv) > 1 else "10.8.0.14"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/joystick_ux.png"
FAKE_PAD = """
window.__pad = { id: 'test gamepad (injected)', index: 0, connected: true, mapping: 'standard',
  axes: [0, 0, 0, 0], buttons: Array.from({ length: 17 }, () => ({ pressed: false, value: 0 })) };
navigator.getGamepads = () => [window.__pad];
"""


def cmd_of(page):
    return re.search(r"command (-?[\d.]+) m/s, turn (-?[\d.]+)", page.inner_text("#status")).groups()


def walked(page):
    return float(re.search(r"walked ([\d.]+) m", page.inner_text("#status")).group(1))


with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = b.new_context(ignore_https_errors=True, viewport={"width": 1400, "height": 820})
    ctx.add_init_script(FAKE_PAD)
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"https://{host}:8097/")
    page.wait_for_selector("#load", state="detached", timeout=120000)
    print("https page, secure context:", page.evaluate("window.isSecureContext"))
    page.evaluate("fetch('/cmd?start=1')")
    page.wait_for_timeout(3000)
    print("pad panel:", page.inner_text("#pad").replace("\n", " | "))
    w0 = walked(page)
    page.evaluate("window.__pad.axes[1] = -1.0")  # left stick fully up
    page.wait_for_timeout(6000)
    print("left stick up 6 s -> command", cmd_of(page), f"walked {w0} -> {walked(page)} m",
          "OK" if cmd_of(page)[0] == "1.20" and walked(page) - w0 > 2 else "FAIL")
    print("pad panel:", page.inner_text("#pad").replace("\n", " | "))
    page.evaluate("window.__pad.axes[1] = 0; window.__pad.axes[2] = 1.0")  # right stick right
    page.wait_for_timeout(800)
    print("right stick right -> command", cmd_of(page), "OK" if cmd_of(page) == ("0.00", "-0.60") else "FAIL")
    page.screenshot(path=out)
    page.evaluate("window.__pad.axes[2] = 0")
    page.wait_for_timeout(800)
    print("sticks centred -> command", cmd_of(page), "OK" if cmd_of(page) == ("0.00", "0.00") else "FAIL")
    page.evaluate("window.__pad.axes[1] = -0.5")
    page.wait_for_timeout(500)
    page.evaluate("window.__pad.axes[1] = 0; window.__pad.buttons[1].pressed = true")  # B = stop
    page.wait_for_timeout(500)
    page.evaluate("window.__pad.buttons[1].pressed = false")
    print("button B -> command", cmd_of(page), "OK" if cmd_of(page)[0] == "0.00" else "FAIL")

    page2 = b.new_page(viewport={"width": 1400, "height": 820})
    page2.goto(f"http://{host}:8095/")
    page2.wait_for_selector("#load", state="detached", timeout=120000)
    print("http page pad panel:", page2.inner_text("#pad")[:90])
    box = page2.locator("#vstick").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    w0 = walked(page2)
    page2.mouse.move(cx, cy)
    page2.mouse.down()
    page2.mouse.move(cx, cy - box["height"] * 0.35, steps=5)
    page2.wait_for_timeout(5000)
    c = cmd_of(page2)
    print("on-screen stick dragged up 5 s -> command", c, f"walked {w0} -> {walked(page2)} m",
          "OK" if float(c[0]) > 0.5 and walked(page2) - w0 > 1 else "FAIL")
    page2.screenshot(path=out.replace(".png", "_vstick.png"))
    page2.mouse.up()
    page2.wait_for_timeout(800)
    print("released -> command", cmd_of(page2), "OK" if cmd_of(page2) == ("0.00", "0.00") else "FAIL")
    print("page errors:", errors or "none")
    b.close()
