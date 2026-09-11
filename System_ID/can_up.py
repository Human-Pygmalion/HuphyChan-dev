#!/usr/bin/env python3
"""두 다리의 통신선을 올리고, **어느 선에 어느 다리가 붙었는지 직접 확인한다.**

2026-09-11 · 왜 이렇게까지 하는가
---------------------------------
어댑터를 `/dev/ttyACM0`, `ttyACM1` 로 부르면 **꽂는 순서대로 번호가 붙습니다.** 다시
꽂거나 재부팅하면 뒤바뀔 수 있습니다. 그러면 왼다리로 갈 명령이 오른다리로 갑니다.

실제로 오늘 두 가지가 함께 드러났습니다.

  * 설정 파일(`robot_v1.0.yaml`)은 오른다리를 `can1`, 왼다리를 `can0` 이라고 적어
    두었는데, **실제 배선은 그 반대**였습니다.
  * USB 어댑터를 하나 더 꽂자 허브가 다시 잡히면서 카메라의 장치 번호까지 바뀌었습니다.

그래서 두 겹으로 막습니다.

  1. 어댑터를 **고유 이름**(`/dev/serial/by-id/...`, 제조 일련번호가 들어 있음)으로
     엽니다. 꽂는 순서에 흔들리지 않습니다.
  2. 그래도 **케이블을 바꿔 꽂으면** 고유 이름과 다리의 짝이 달라집니다. 그래서 올린
     뒤에 **모터 번호를 물어봐서** 어느 다리인지 직접 확인합니다. 1~6 번이 답하면
     왼다리, 7~12 번이 답하면 오른다리입니다.

확인 결과를 `~/System_ID/can_map.json` 에 남깁니다. 다른 프로그램은 `can0`/`can1` 을
짐작하지 말고 **이 파일을 읽어야 합니다.**

무엇을 보내는가
---------------
**고장값 조회뿐**입니다 (`FF FF FF FF FF FF 00 FB`). 모터의 기록을 읽어 달라는
질의라서 제어 경로를 지나지 않습니다 — **축이 움직일 수 없습니다.** 켜기(0xFC)·
끄기(0xFD)·영점(0xFE) 은 이 파일에 들어 있지 않습니다.

    can_up.py              # 올리고 확인하고 지도를 남김
    can_up.py --check      # 이미 올라와 있는 것을 확인만 함
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import struct
import subprocess
import sys
import time

CAN_FRAME = "=IB3x8s"
FAULT_QUERY = bytes([0xFF] * 6 + [0x00, 0xFB])       # 명령 5 의 '조회' 형태
BITRATE_FLAG = "-s8"                                 # 초당 100만 비트
MAP_PATH = os.path.expanduser("~/System_ID/can_map.json")

LEGS = {"left_leg": (1, 2, 3, 4, 5, 6), "right_leg": (7, 8, 9, 10, 11, 12)}


def run(cmd, **kw):
  return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def adapters():
  """꽂혀 있는 CAN 어댑터를 **고유 이름**으로. 일련번호 순이라 순서가 일정합니다."""
  found = sorted(glob.glob("/dev/serial/by-id/*CANable*"))
  return [(p, os.path.basename(p).split("_")[-1].replace("-if00", "")) for p in found]


def clean_slate():
  """붙어 있던 것을 **전부** 떼어 낸다.

  옛 `slcand` 가 어댑터를 붙잡고 있으면 새로 붙일 때 `Input/output error` 가 납니다
  (2026-09-11 실제로 겪음). 이름을 맞춰 골라 죽이려 하면, 앞서 다른 이름으로 붙여
  놓았을 때 놓칩니다. 그래서 그냥 다 정리하고 시작합니다.
  """
  for iface in glob.glob("/sys/class/net/can*"):
    run(f"sudo ip link set {os.path.basename(iface)} down")
  run("sudo pkill -x slcand")
  time.sleep(1.0)


def attach(by_id_path, iface):
  run(f"sudo ip link set {iface} down")
  time.sleep(0.2)
  r = run(f"sudo slcand -o -c {BITRATE_FLAG} {by_id_path} {iface}")
  if r.returncode != 0:
    return f"slcand 실패: {r.stderr.strip() or r.stdout.strip()}"
  time.sleep(1.0)
  r = run(f"sudo ip link set {iface} up")
  if r.returncode != 0:
    return f"올리기 실패: {r.stderr.strip()}"
  time.sleep(0.3)
  return None


def who_is_there(iface, wait_s=0.05):
  """이 선에 어떤 모터 번호가 답하는지. 고장값 조회만 보냅니다."""
  sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  sock.bind((iface,))
  found, faults = [], {}
  for motor_id in range(1, 13):
    sock.settimeout(0.0)
    try:
      while True:
        sock.recv(16)
    except (BlockingIOError, TimeoutError, socket.timeout):
      pass
    sock.send(struct.pack(CAN_FRAME, motor_id, 8, FAULT_QUERY))
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
      sock.settimeout(max(0.0, end - time.monotonic()))
      try:
        raw = sock.recv(16)
      except (TimeoutError, socket.timeout):
        break
      _, dlc, data = struct.unpack(CAN_FRAME, raw)
      if dlc >= 5 and data[0] == motor_id:
        found.append(motor_id)
        faults[motor_id] = data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24)
        break
  sock.close()
  return found, faults


def name_leg(found):
  """답한 번호로 어느 다리인지 판정. 애매하면 그렇게 말한다."""
  if not found:
    return None, "아무도 답하지 않음"
  hits = {leg: [m for m in found if m in ids] for leg, ids in LEGS.items()}
  live = {leg: ms for leg, ms in hits.items() if ms}
  if len(live) != 1:
    return None, f"한 선에 두 다리가 섞여 있음 {live}"
  leg, ms = next(iter(live.items()))
  missing = [m for m in LEGS[leg] if m not in ms]
  note = "여섯 개 다 있음" if not missing else f"**{missing} 번이 없음**"
  return leg, note


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--check", action="store_true", help="올리지 않고 확인만")
  a = ap.parse_args(argv)

  devs = adapters()
  if not devs:
    print("CAN 어댑터가 안 보입니다 (/dev/serial/by-id 에 CANable 없음)")
    return 1

  if not a.check:
    clean_slate()
  print(f"어댑터 {len(devs)}개를 고유 이름으로 찾았습니다\n")
  table, problems = [], []
  for i, (path, serial) in enumerate(devs):
    iface = f"can{i}"
    if not a.check:
      err = attach(path, iface)
      if err:
        problems.append(f"{iface}: {err}")
        print(f"  {iface}  {serial}  -> {err}")
        continue
    found, faults = who_is_there(iface)
    leg, note = name_leg(found)
    bad = [m for m, w in faults.items() if w]
    table.append(dict(iface=iface, serial=serial, by_id=path, leg=leg,
                      motors=found, note=note, faulted=bad))
    if leg is None:
      problems.append(f"{iface}: {note}")
    if bad:
      problems.append(f"{iface}: 고장 기록이 있는 모터 {bad}")

  print(f"  {'선':<6} {'어댑터 일련번호':<26} {'다리':<11} 모터")
  print("  " + "-" * 72)
  for row in table:
    leg = row["leg"] or "??"
    print(f"  {row['iface']:<6} {row['serial']:<26} {leg:<11} "
          f"{row['motors']}  {row['note']}")
    if row["faulted"]:
      print(f"  {'':<6} {'':<26} {'':<11} ** 고장: {row['faulted']} **")
  print("  " + "-" * 72)

  with open(MAP_PATH, "w") as f:
    json.dump(dict(checked_at=time.strftime("%Y-%m-%d %H:%M:%S"), buses=table), f,
              ensure_ascii=False, indent=2)
  print(f"\n  지도를 남겼습니다: {MAP_PATH}")
  print("  다른 프로그램은 can0/can1 을 짐작하지 말고 이 파일을 읽을 것.")

  if problems:
    print("\n  ** 확인할 것 **")
    for p in problems:
      print(f"    - {p}")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
