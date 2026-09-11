#!/usr/bin/env python3
"""실물 관절을 **3D 화면(viser)에 그대로 비춘다.** 명령을 보내는 길이 없다.

2026-09-11 · 사용자 지시: *"viewer (viser 쓴거) 에도 연결해줘"* 와
*"실수로라도 모터 안움직이게 조심해"*.

왜 따로 만드는가
----------------
`run.py` 로 띄우는 평소 뷰어는 웹 조종판(FastAPI)을 함께 엽니다. 거기에는 무장·전송
단추가 있고, 지금은 **실수로도 모터가 움직이면 안 되는 단계**입니다. 그리고
`run.py bridge huphy` 는 반대로 화면 없이 계산만 돌립니다 — 볼 수가 없습니다.

이 파일은 그 둘 사이입니다.

    받는다   HUPHY 형식 UDP 텔레메트리 (`HuphyUdpReceiver`)  -> 시뮬 관절에 반영
    보여준다 viser 3D 화면 (`build_ui`)
    보낸다   **없음**

명령을 만드는 부분(`tx`)을 아예 만들지 않습니다. 웹 조종판도 열지 않습니다. 그래서
이 프로그램이 켜져 있는 동안 **모터로 나갈 수 있는 것이 하나도 없습니다.**

더 근본적인 안전장치가 하나 더 있습니다: 뷰어의 명령은 로봇 쪽 브리지
(`huphy_remote_motion --listen 0.0.0.0:9872`)만 받습니다. 그것을 띄우지 않으면
명령은 받는 사람이 없어 버려집니다. 두 겹으로 막혀 있습니다.

받침대는 `fixed` 로 둡니다 — 실물이 크레인에 매달려 있고, 우리가 보려는 것은
관절이 어느 쪽으로 도는가이지 로봇이 서 있는가가 아닙니다.

    huphy_viser.py --variant LegOnly-AB --port 8094 --udp-port 9871
    http://<이 컴퓨터>:8094/
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  from .. import CACHE_DIR, VARIANTS
  ap.add_argument("--variant", default="LegOnly-AB", choices=list(VARIANTS))
  ap.add_argument("--cache", default=CACHE_DIR)
  ap.add_argument("--port", type=int, default=8094, help="3D 화면 (viser)")
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--udp-port", type=int, default=9871, help="텔레메트리를 받을 포트")
  ap.add_argument("--map", default=None, help="관절 대응표. 기본은 biped 판")
  ap.add_argument("--keyframe", default="knees_bent", choices=("home", "knees_bent"))
  ap.add_argument("--stale-ok", action="store_true")
  a = ap.parse_args(argv)

  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # GPU 는 학습기 것임

  from ..contract import load_contract
  from ..sim_core import SimCore
  from ..ui import build_ui
  from .huphy_udp import HuphyUdpReceiver

  c = load_contract(a.cache, a.variant)
  fresh = c.freshness()
  if fresh["stale"] and not a.stale_ok:
    print(f"이 변형의 계약이 낡았습니다: {fresh['checks']}\n"
          f"다시 구우거나 --stale-ok 로 그냥 돌리세요", file=sys.stderr)
    return 3

  core = SimCore(c, realtime=True)
  core.start()
  core.reset(a.keyframe)

  recv = HuphyUdpReceiver(core, port=a.udp_port, host="0.0.0.0", jmap_path=a.map)
  recv.start()

  server = build_ui(core, host=a.host, port=a.port, freshness=fresh, base="fixed")
  print(f"\n  3D 화면   http://{a.host}:{a.port}/")
  print(f"  텔레메트리 {a.udp_port} 번으로 받는 중 ({a.variant})")
  print("  ** 이 프로그램에는 모터로 명령을 보내는 길이 없습니다 **")
  print("  Ctrl-C 로 멈춤\n", flush=True)
  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    print("\n멈춥니다")
  finally:
    recv.stop()
    try:
      server.stop()
    except Exception:                                   # noqa: BLE001
      pass
    core.stop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
