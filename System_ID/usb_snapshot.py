#!/usr/bin/env python3
"""라즈베리파이에 **무엇이 꽂혀 있는지** 찍어 두고, 나중에 **차분**을 본다.

2026-09-15 · 사용자 지시: *"imu 빼뒀어. 기록해. 다음에 imu 꽂고 다시말할테니
차분확인해."*

왜 필요한가
-----------
IMU 가 어느 자리로 잡히는지 말로만 주고받으면 틀립니다. 실제로 이번에 한 번 틀렸습니다 —
설정은 `/dev/ebimu` 를 가리키는데 그런 링크가 없었고, `ttyACM0` 일 거라고 했지만 실제로는
`ttyACM2`/`ttyACM3` 두 개(둘 다 CAN 어댑터)뿐이었습니다. **꽂기 전과 후를 각각 찍어
빼면** 새로 생긴 자리가 무엇인지 추측 없이 나옵니다.

무엇을 찍는가
-------------
장치 목록 세 가지입니다. 사람이 읽을 수 있게 그대로 둡니다.

    /dev/ttyACM*, /dev/ttyUSB*      시리얼 자리
    /dev/serial/by-id/              같은 자리를 **일련번호로** 부른 이름 (뽑았다 꽂아도 안 변함)
    lsusb                           USB 장치 목록

`by-id` 가 가장 중요합니다. `ttyACM` 번호는 꽂는 순서에 따라 바뀌지만 `by-id` 는 안
바뀝니다 — `can_up.sh` 가 이미 그 이유로 `by-id` 를 씁니다.

    usb_snapshot.py --save before_imu          # 지금 상태를 찍는다
    usb_snapshot.py --save after_imu           # IMU 꽂은 뒤 다시 찍는다
    usb_snapshot.py --diff before_imu after_imu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

HOST = "maido@10.8.0.17"
STORE = Path(__file__).resolve().parent / "data" / "usb_snapshots"

PROBES = {
  "tty": "ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null",
  "by_id": "ls -1 /dev/serial/by-id/ 2>/dev/null",
  "lsusb": "lsusb 2>/dev/null",
}


def capture(host: str) -> dict:
  """로봇에서 세 가지를 읽어 온다. **읽기만 한다.**"""
  script = " ; echo '@@' ; ".join(PROBES[k] for k in PROBES)
  r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, script],
                     capture_output=True, text=True, timeout=60)
  if r.returncode and not r.stdout.strip():
    raise SystemExit(f"로봇에서 못 읽었습니다: {r.stderr.strip()[:200]}")
  parts = r.stdout.split("@@")
  out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host}
  for (name, _), chunk in zip(PROBES.items(), parts, strict=False):
    out[name] = sorted(x.strip() for x in chunk.splitlines() if x.strip())
  return out


def show(snap: dict, title: str) -> None:
  print(f"\n=== {title}  ({snap['when']})")
  for k in PROBES:
    print(f"  [{k}]")
    for line in snap.get(k, []) or ["(없음)"]:
      print(f"    {line}")


def diff(a: dict, b: dict) -> int:
  """b 에만 있는 것 = 새로 꽂힌 것. a 에만 있는 것 = 빠진 것."""
  found = 0
  print(f"\n=== 차분   {a['when']}  ->  {b['when']}")
  for k in PROBES:
    added = [x for x in b.get(k, []) if x not in a.get(k, [])]
    gone = [x for x in a.get(k, []) if x not in b.get(k, [])]
    if not added and not gone:
      print(f"  [{k}] 그대로")
      continue
    print(f"  [{k}]")
    for x in added:
      print(f"    + {x}")
      found += 1
    for x in gone:
      print(f"    - {x}")
  if found:
    print("\n  `+` 로 새로 생긴 자리가 IMU 입니다. `by_id` 이름을 robot.yaml 의 port 에\n"
          "  쓰거나, 그 이름으로 udev 링크(/dev/ebimu)를 만드십시오 — ttyACM 번호는 변합니다.")
  else:
    print("\n  새로 생긴 자리가 없습니다. 아직 안 잡힌 것입니다 (케이블·전원 확인).")
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--save", metavar="이름", help="지금 상태를 이 이름으로 찍는다")
  ap.add_argument("--diff", nargs=2, metavar=("이전", "이후"))
  ap.add_argument("--list", action="store_true", help="찍어 둔 것 목록")
  ap.add_argument("--host", default=HOST)
  a = ap.parse_args(argv)

  STORE.mkdir(parents=True, exist_ok=True)
  if a.list:
    for p in sorted(STORE.glob("*.json")):
      print(f"  {p.stem:<20} {json.loads(p.read_text())['when']}")
    return 0
  if a.diff:
    x, y = (STORE / f"{n}.json" for n in a.diff)
    for p in (x, y):
      if not p.exists():
        raise SystemExit(f"{p.stem} 을 찍어 둔 적이 없습니다. --list 로 보십시오.")
    return diff(json.loads(x.read_text()), json.loads(y.read_text()))
  if not a.save:
    show(capture(a.host), "지금")
    return 0
  snap = capture(a.host)
  path = STORE / f"{a.save}.json"
  path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
  show(snap, a.save)
  print(f"\n  저장: {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
