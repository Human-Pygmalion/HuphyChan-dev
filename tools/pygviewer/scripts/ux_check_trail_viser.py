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
ap.add_argument("--gain-scale", default="",
                help="if set (e.g. 0.30): move the 'Action scale (Kp·Kd ×)' slider there after the walk and check the readout")
ap.add_argument("--terrain-test", action="store_true",
                help="switch the 'Terrain' dropdown to flat, walk ~10 s reading the Status base pose, switch back to rough")
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


async def joint_states(page) -> list[str]:
  """The Joints folder's per-joint state boxes ("q .. tgt .. tau .."): they refresh ~10 Hz only
  while the sim runs, so two reads that differ prove the robot is still running."""
  return await page.eval_on_selector_all("input", "els=>els.map(e=>e.value).filter(x=>/ tau /.test(x))")


async def base_pos(page) -> tuple[float, float, float]:
  """Base position from the Status folder ("base `free` pos (x, y, z) ...")."""
  import re

  t = await page.inner_text("body")
  m = re.search(r"base\s+\S+\s+pos \(([-+0-9.]+), ([-+0-9.]+), ([-+0-9.]+)\)", t)
  return tuple(float(v) for v in m.groups()) if m else (float("nan"),) * 3


async def set_vx(page, value: str) -> str:
  """Type into the vx number box and wait until the Status line echoes the command the sim holds
  (policy cmd (vx, vy, wz)); retried, because a box typed into while a dropdown closes can drop it."""
  import re

  want = float(value)
  row = page.get_by_text("vx [m/s]", exact=True).locator("xpath=ancestor::*[count(.//input)>=2][1]")
  box = row.locator("input").last
  got = ""
  for _ in range(4):
    await box.scroll_into_view_if_needed()
    await box.click()
    await box.fill(value)
    await box.press("Enter")
    await page.wait_for_timeout(2500)
    m = re.search(r"cmd \(([-+0-9.]+),", await page.inner_text("body"))
    got = m.group(1) if m else ""
    if got and abs(float(got) - want) < 1e-6:
      break
  print(f"vx set to {value}: box {await box.input_value()}, sim holds cmd vx {got}", flush=True)
  return got


async def terrain_check(page, _vx_box) -> None:
  """Terrain dropdown like a person: flat -> walk at vx 0.6 for ~10 s (base x and z logged each
  second) -> stop -> back to rough (the trail must be drawn again and carry the robot)."""
  await pick(page, "Terrain", "flat")
  await page.wait_for_timeout(4000)  # the switch resets the robot onto the flat ground
  print("terrain -> flat; standing base:", await base_pos(page), flush=True)
  await set_vx(page, "0.6")
  track = []
  for k in range(11):
    p = await base_pos(page)
    track.append(p)
    print(f"flat t+{k:2d}s base x {p[0]:+.2f} y {p[1]:+.2f} z {p[2]:.3f}", flush=True)
    await page.wait_for_timeout(1000)
  zs = [p[2] for p in track]
  print(f"FLAT walk: min z {min(zs):.3f} (>0.7 ok: {min(zs) > 0.7}), x {track[0][0]:+.2f} -> {track[-1][0]:+.2f} "
        f"(increasing: {track[-1][0] > track[0][0] + 1.0})", flush=True)
  await page.screenshot(path=os.path.join(a.out, "trail_viser_4_flat.png"), timeout=180000)
  await set_vx(page, "0")
  await pick(page, "Terrain", "rough (trail)")
  await page.wait_for_timeout(4000)
  for k in range(3):
    p = await base_pos(page)
    print(f"rough standing t+{2 * k}s base x {p[0]:+.2f} y {p[1]:+.2f} z {p[2]:.3f}", flush=True)
    await page.wait_for_timeout(2000)
  await page.screenshot(path=os.path.join(a.out, "trail_viser_5_rough_back.png"), timeout=180000)


async def gain_scale_check(page, value: str) -> None:
  """Use the 'Action scale (Kp·Kd ×)' slider like a person: type a value into its number box, then
  read the readout beneath it (it echoes ×s and knee/hip/crank Kp, Kd -> scaled) and prove the sim
  keeps running (the joint state boxes keep changing) while the policy is walking."""
  label = page.get_by_text("Action scale (Kp·Kd ×)", exact=True)
  await label.scroll_into_view_if_needed()
  print("slider label found:", await label.count(), flush=True)
  row = label.locator("xpath=ancestor::*[count(.//input)>=2][1]")
  box = row.locator("input").last
  print("slider box before:", await box.input_value(), flush=True)
  t0 = await joint_states(page)
  await box.fill(value)
  await box.press("Enter")
  await page.wait_for_timeout(1500)
  print("slider box after :", await box.input_value(), flush=True)
  md = page.get_by_text("on every joint's Kp and Kd").first
  print("readout:", (await md.inner_text()).replace("\n", " | "), flush=True)
  await page.wait_for_timeout(8000)
  t1 = await joint_states(page)
  await page.wait_for_timeout(1000)
  t3 = await joint_states(page)
  print("joint 0 before  :", t0[:1], "\\njoint 0 +10s    :", t1[:1], "\\njoint 0 +11s    :", t3[:1], flush=True)
  print("sim running (state changes over time):", t1 != t3 and t0 != t1, flush=True)
  await page.screenshot(path=os.path.join(a.out, "trail_viser_3_gain_scale.png"), timeout=180000)


async def main() -> None:
  from playwright.async_api import async_playwright

  async with async_playwright() as p:
    b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    page = await b.new_page(viewport={"width": 1600, "height": 900})
    await page.goto(a.url)
    await page.wait_for_timeout(15000)
    await page.screenshot(path=os.path.join(a.out, "trail_viser_0_open.png"), timeout=180000)
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
    await page.screenshot(path=os.path.join(a.out, "trail_viser_1_start.png"), timeout=180000)
    await page.wait_for_timeout(int(a.walk_s * 1000))
    await page.screenshot(path=os.path.join(a.out, "trail_viser_2_walked.png"), timeout=180000)
    if a.terrain_test:
      await terrain_check(page, box)
    if a.gain_scale:
      await gain_scale_check(page, a.gain_scale)
    await b.close()


asyncio.run(main())
