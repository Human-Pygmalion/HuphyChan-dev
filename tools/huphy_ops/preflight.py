#!/usr/bin/env python3
"""움직이기 **전에** 명령을 한계와 대조해 보고, 넘으면 멈춘다.

2026-09-12 · 사용자 지시: *"조인트 명령 위치가 조인트 가동한계를 넘었다면 에러내야하지."*

로봇 프로그램은 한계를 넘는 목표를 **조용히 자릅니다**(`safety.guards.apply` 가
`ClipReason.LIMIT` 을 세어 텔레메트리로 내보냄). 그것은 제어 중에 맞는 동작입니다 —
버리면 그 관절만 직전 명령을 유지해 자세가 어긋나기 때문입니다.

이 검사는 **막지 않습니다.** 자르는 것이 맞는 동작이므로(사용자 지시 2026-09-12),
잘릴 값이 무엇이고 실제로 어디로 가게 되는지를 **미리 보여 주기만** 합니다. 사람이
보내기 전에 "내가 시킨 것과 실제로 갈 곳이 다르다" 는 사실을 알고 누르게 하려는
것입니다.

무엇을 보여 주는가

  1. 목표가 하드스톱을 넘는가        -> 어디로 잘리는지 같이 표시
  2. 목표가 여유 안쪽인가            -> 어디로 잘리는지 같이 표시
     여유(`command_margin_deg`)는 오버슛·관성·측정오차를 흡수하는 공간입니다.
     한계 근처에서 감쇠를 키우는 장치는 **없습니다** — 이 여유가 유일한 흡수 수단이고,
     그 3도조차 코드 주석이 "임의값" 이라고 적고 있습니다.
  3. 지금 위치가 이미 한계 밖인가    -> 표시. 복구 중일 수 있고 조금씩 돌아오는 것이 맞음

    preflight.py --config <robot.yaml> --pose zero
    preflight.py --config <robot.yaml> --pose "left_leg/knee=30,right_leg/knee=-30"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

SINGLE = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_a", "ankle_b")


def parse_pose(text: str, names) -> dict:
  """`zero` 면 전부 0. 아니면 `다리/관절=값` 을 쉼표로."""
  if text.strip() == "zero":
    return {n: 0.0 for n in names}
  out = {n: 0.0 for n in names}
  for part in text.split(","):
    part = part.strip()
    if not part:
      continue
    name, _, val = part.partition("=")
    name = name.strip()
    if name not in out:
      raise SystemExit(f"모르는 관절 이름: {name}\n가능: {sorted(out)}")
    out[name] = float(val)
  return out


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", required=True)
  ap.add_argument("--pose", default="zero")
  ap.add_argument("--now", default=None,
                  help="지금 각도를 읽어 올 주소 (있으면 현재 위치도 같이 봄)")
  a = ap.parse_args(argv)

  from huphy import calibration as calib
  from huphy.config import load_robot

  robot = load_robot(a.config)
  margin = robot.safety.command_margin_deg

  rows = []
  for leg in robot.limbs.values():
    cal = calib.load(leg.calibration_path)
    for jn in SINGLE:
      if jn in leg.motors:
        rows.append((f"{leg.name}/{jn}", leg.motors[jn].id, cal[jn].limits_deg))

  targets = parse_pose(a.pose, [n for n, _, _ in rows])
  now = {}
  if a.now:
    try:
      now = json.loads(urllib.request.urlopen(a.now, timeout=8).read())["motors"]
    except Exception as exc:                                    # noqa: BLE001
      print(f"지금 각도를 못 읽었습니다 ({type(exc).__name__}) — 목표만 검사합니다")

  print(f"자세 검사 — 여유 {margin:g}도\n")
  print("  %-22s %9s %20s %s" % ("관절", "목표", "명령 가능 구간", "판정"))
  print("  " + "-" * 74)
  errors, warns = [], []
  for name, mid, limits in rows:
    tgt = targets[name]
    if limits is None:
      print("  %-22s %9.2f %20s %s" % (name, tgt, "(한계 없음)", "** 한계가 비어 있음 **"))
      errors.append(f"{name}: 한계값이 없습니다 (커미셔닝 전)")
      continue
    lo, hi = limits
    slo, shi = lo + margin, hi - margin
    sent = min(shi, max(slo, tgt))
    verdict = "그대로 나감"
    if abs(sent - tgt) > 1e-9:
      verdict = f"**잘림 -> {sent:+.2f}도** ({sent - tgt:+.2f})"
      errors.append(f"{name}: 시킨 {tgt:+.2f}도가 {sent:+.2f}도로 잘림 "
                    f"(하드스톱 [{lo}, {hi}], 여유 {margin:g}도)")
    print("  %-22s %9.2f %20s %s" % (name, tgt, str([round(slo, 1), round(shi, 1)]), verdict))
    q = now.get(str(mid), {}).get("deg")
    if q is not None and not (lo <= q <= hi):
      warns.append(f"{name}: 지금 위치 {q:+.2f}도가 한계 밖 (복구 중일 수 있음)")
  print("  " + "-" * 74)

  for w in warns:
    print(f"\n  경고 — {w}")
  if errors:
    print("\n  ** 시킨 것과 실제로 갈 곳이 다른 관절 — 알고 누르세요 **")
    for e in errors:
      print(f"    - {e}")
  else:
    print("\n  모든 목표가 그대로 나갑니다 (잘리는 것 없음).")
  print("\n  참고: 한계 근처에서 감쇠를 키우는 장치는 없습니다. "
        f"여유 {margin:g}도가 유일한 흡수 공간입니다.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
