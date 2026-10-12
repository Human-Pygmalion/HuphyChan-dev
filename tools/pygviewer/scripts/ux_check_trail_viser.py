#!/usr/bin/env python3
"""Open the viser viewer (trail model) in a real browser, load a policy, drive it, screenshot.

2026-10-12: the user drives the latest rough-terrain policy on the mountain-trail heightfield in
viser.  This presses the same panel controls a person would: "policy" dropdown -> "load policy"
-> "mode" = policy_sim -> types a forward speed into "vx [m/s]" -> waits -> screenshots.

  mujoco-sim/mjlab/.venv/bin/python3 tools/pygviewer/scripts/ux_check_trail_viser.py \
      --url http://10.8.0.14:8098 --policy roughmain --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import os

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://10.8.0.14:8098")
ap.add_argument("--policy", default="roughmain")
ap.add_argument("--vx", default="0.6")
ap.add_argument("--walk-s", type=float, default=10.0)
ap.add_argument("--out", default=".")
a = ap.parse_args()


async def pick(page, label: str, option: str, nth: int = 0) -> None:
  """viser dropdowns are Mantine selects (click, then the option); two panel folders both have a
  "mode" (base link / policy) - nth picks which."""
  box = page.get_by_label(label, exact=True).nth(nth)
  await box.scroll_into_view_if_needed()
  await box.click()
  await page.wait_for_timeout(400)
  await page.get_by_role("option", name=option, exact=True).click()


async def main() -> None:
  from playwright.async_api import async_playwright

  async with async_playwright() as p:
    b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    page = await b.new_page(viewport={"width": 1600, "height": 900})
    await page.goto(a.url)
    await page.wait_for_timeout(15000)
    await page.screenshot(path=os.path.join(a.out, "trail_viser_0_open.png"))
    await pick(page, "policy", a.policy)
    btn = page.get_by_role("button", name="load policy")
    await btn.scroll_into_view_if_needed()
    await btn.click()
    await page.wait_for_timeout(2500)
    await pick(page, "mode", "policy_sim", nth=1)
    await page.wait_for_timeout(1500)
    # the number box beside the slider has no label of its own: take the last input in the row
    row = page.get_by_text("vx [m/s]", exact=True).locator("xpath=ancestor::*[count(.//input)>=2][1]")
    box = row.locator("input").last
    await box.scroll_into_view_if_needed()
    await box.fill(a.vx)
    await box.press("Enter")
    await page.wait_for_timeout(500)
    print("vx box now:", await box.input_value(), flush=True)
    await page.screenshot(path=os.path.join(a.out, "trail_viser_1_start.png"))
    await page.wait_for_timeout(int(a.walk_s * 1000))
    await page.screenshot(path=os.path.join(a.out, "trail_viser_2_walked.png"))
    await b.close()


asyncio.run(main())
