#!/usr/bin/env python3
"""제어 루프가 뿌리는 상태를 **파일로 받아 적는다.** 보내지 않는다.

2026-09-12. 슬라이더로 관절을 움직이면서 "목표를 줬을 때 실제로 어디로 갔는가" 를
남기려고 만들었습니다. 제어 루프는 초당 스무 번 상태를 뿌리는데, 그걸 화면만 보고
넘기면 나중에 숫자로 따질 것이 없습니다.

한 줄에 한 시점, 모터마다 네 칸씩 적습니다.

    t          프로그램 시작부터의 초
    <번호>_deg  지금 각도 (도)
    <번호>_tgt  제어 루프가 들고 있는 목표 (도)
    <번호>_nm   토크 (뉴턴미터)
    <번호>_temp 권선 온도 (도)

★ 이 파일은 **읽기만 합니다.** 모터로 나가는 길이 없습니다.

    huphy_log.py --listen 9874 --out data/slider_2026-09-12.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--listen", type=int, default=9874)
  ap.add_argument("--out", required=True)
  ap.add_argument("--seconds", type=float, default=1800.0)
  a = ap.parse_args(argv)

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(("0.0.0.0", a.listen))
  sock.settimeout(1.0)

  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fh = out.open("w", newline="")
  writer = None
  ids: list = []
  t0 = time.monotonic()
  rows = 0
  print(f"  {a.listen} 번 포트에서 받아 {out} 에 적습니다. Ctrl-C 로 멈춥니다.",
        flush=True)
  try:
    while time.monotonic() - t0 < a.seconds:
      try:
        raw, _ = sock.recvfrom(65535)
      except (TimeoutError, socket.timeout):
        continue
      try:
        msg = json.loads(raw)
      except (ValueError, TypeError):
        continue
      motors = msg.get("motors") or {}
      if writer is None:
        # 열 순서는 **모터 번호 순**으로 고정합니다. 사전 순서에 기대면 판마다
        # 열이 바뀌어 나중에 붙여 읽을 수 없습니다.
        ids = sorted(motors, key=lambda x: int(x))
        head = ["t", "note"]
        for mid in ids:
          head += [f"{mid}_deg", f"{mid}_tgt", f"{mid}_nm", f"{mid}_temp"]
        writer = csv.writer(fh)
        writer.writerow(head)
        names = {mid: motors[mid].get("joint", mid) for mid in ids}
        print("  모터 -> 관절: " + ", ".join(f"{k}={v}" for k, v in names.items()),
              flush=True)
      row = [round(time.monotonic() - t0, 3), msg.get("note", "")]
      for mid in ids:
        m = motors.get(mid) or {}
        row += [m.get("deg"), m.get("tgt"), m.get("nm"), m.get("temp")]
      writer.writerow(row)
      rows += 1
      if rows % 200 == 0:
        fh.flush()
  except KeyboardInterrupt:
    print("\n  사람이 멈춤")
  finally:
    fh.close()
    sock.close()
  print(f"  {rows}줄 적었습니다 -> {out}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
