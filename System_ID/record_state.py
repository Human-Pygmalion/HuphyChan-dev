#!/usr/bin/env python3
"""관절 상태를 **받아서 파일에 쌓는다.** 보내는 길이 없다.

2026-09-15 · 사용자 지시: *"이제 로봇 동작범위 내에서 손으로 돌려볼테니 데이터 기록해."*

`huphy_live.py --observe` 가 `--state-to` 로 뿌리는 꾸러미를 그대로 받아 CSV 한 줄씩
씁니다. 카메라 녹화(`record_cam.py`)와 **같은 규칙**을 따릅니다:

    1) 기록을 켠다
    2) 줄 수가 **실제로 느는지** 확인한다
    3) 그 다음에 움직인다

시각은 `t` (기록 시작부터의 초)와 `wall` (벽시계)을 둘 다 남깁니다 — 영상과 맞추려면
벽시계가 필요하고, 그래프를 그리려면 상대 시각이 편합니다.

★ 이 파일은 **받기만 합니다.** 모터로 나가는 길이 없습니다.

    record_state.py --out ~/pyg_rec/2026-09-15_handcheck.csv --port 9878
    record_state.py --out ... --seconds 600        # 저절로 멈추게
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time

FIELDS = ("deg", "nm", "temp", "vel")
"""각도·토크·온도·속도. 나머지(한계·게인)는 안 변하므로 첫 줄에 주석으로만 남깁니다."""


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", required=True, help="쓸 CSV 경로")
  ap.add_argument("--port", type=int, default=9878, help="받을 자리 (기본 9878)")
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--seconds", type=float, default=3600.0)
  ap.add_argument("--quiet-every", type=float, default=10.0,
                  help="이 초마다 한 줄씩 살아 있다고 알림")
  a = ap.parse_args(argv)

  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  try:
    s.bind((a.host, a.port))
  except OSError as e:
    print(f"{a.port} 번을 못 엽니다: {e}\n  이미 쓰는 프로그램이 있는지 보십시오: "
          f"ss -lunp | grep {a.port}", file=sys.stderr)
    return 2
  s.settimeout(1.0)

  t0 = time.monotonic()
  writer = None
  rows = 0
  last_note = 0.0
  print(f"  {a.out} 에 기록합니다 · {a.port} 번에서 받는 중 · 멈추려면 Ctrl-C")
  with open(a.out, "w", newline="", encoding="utf-8") as fh:
    try:
      while time.monotonic() - t0 < a.seconds:
        try:
          data, _ = s.recvfrom(65535)
        except socket.timeout:
          continue
        try:
          pkt = json.loads(data.decode("utf-8"))
        except Exception:                                   # noqa: BLE001
          continue
        motors = pkt.get("motors")
        if not isinstance(motors, dict):
          continue                       # 납작한 모양(3차원 뷰어용)은 여기서 안 씀
        now = time.monotonic()
        row = {"t": round(now - t0, 3), "wall": round(time.time(), 3)}
        for v in motors.values():
          name = str(v.get("joint", "")).replace("_leg/", "_")
          for f in FIELDS:
            row[f"{name}_{f}"] = v.get(f)
        if writer is None:
          # 안 변하는 것(한계·게인)은 맨 위에 주석으로 한 번만 남깁니다.
          fh.write("# 관절 한계 [도] (로봇 설정, 여유 뺀 값)\n")
          for v in sorted(motors.values(), key=lambda x: str(x.get("joint"))):
            fh.write(f"#   {v.get('joint')}: {v.get('lo')} ~ {v.get('hi')}"
                     f"  kp={v.get('kp')} kd={v.get('kd')}\n")
          writer = csv.DictWriter(fh, fieldnames=list(row))
          writer.writeheader()
        writer.writerow(row)
        rows += 1
        if rows % 20 == 0:
          fh.flush()
        if now - last_note >= a.quiet_every:
          last_note = now
          print(f"  {now - t0:7.1f}초 · {rows}줄 · 모터 {len(motors)}개", flush=True)
    except KeyboardInterrupt:
      print("\n  사람이 멈춤")
    finally:
      fh.flush()
  print(f"  끝. {rows}줄 · {time.monotonic() - t0:.1f}초 · {a.out}")
  return 0 if rows else 1


if __name__ == "__main__":
  raise SystemExit(main())
