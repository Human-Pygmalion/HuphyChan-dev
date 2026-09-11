#!/usr/bin/env python3
"""이 컴퓨터를 거쳐 파이의 화면을 **랜(192.168.20.x)에서도** 볼 수 있게 한다.

왜 필요한가
-----------
파이는 WireGuard 주소(10.8.0.17)로만 닿습니다. 랜에 있는 다른 컴퓨터는 그 주소를
모릅니다. 그런데 이 컴퓨터는 **양쪽에 다 있습니다**(192.168.20.177 과 10.8.0.16).
그래서 이 컴퓨터가 가운데서 바이트를 옮겨 주면 랜에서도 보입니다.

    랜의 브라우저 --> 192.168.20.177:8098 --> (이 프로그램) --> 10.8.0.17:8098

중계만 합니다 — 내용을 읽지도, 바꾸지도, 저장하지도 않습니다. 그래서 영상도
원본 그대로 흐르고 지연이 거의 늘지 않습니다(왕복 11밀리초짜리 터널 한 번).

    portfwd.py 8098:10.8.0.17:8098 8099:10.8.0.17:8099
"""

from __future__ import annotations

import socket
import sys
import threading


def pipe(a: socket.socket, b: socket.socket) -> None:
  try:
    while True:
      data = a.recv(65536)
      if not data:
        break
      b.sendall(data)
  except OSError:
    pass
  finally:
    for s in (a, b):
      try:
        s.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass
      s.close()


def serve(listen_port: int, host: str, port: int) -> None:
  srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  srv.bind(("0.0.0.0", listen_port))
  srv.listen(64)
  print(f"  0.0.0.0:{listen_port}  ->  {host}:{port}", flush=True)
  while True:
    c, _ = srv.accept()
    try:
      u = socket.create_connection((host, port), timeout=8)
    except OSError as exc:
      print(f"  {host}:{port} 에 못 붙음: {exc}", flush=True)
      c.close()
      continue
    # 영상이 끊기지 않게 작은 꾸러미도 바로 보낸다
    for s in (c, u):
      try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      except OSError:
        pass
    threading.Thread(target=pipe, args=(c, u), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c), daemon=True).start()


def main(argv=None) -> int:
  args = (argv if argv is not None else sys.argv[1:])
  if not args:
    print(__doc__)
    return 2
  print("중계 시작")
  for spec in args:
    lp, host, port = spec.split(":")
    threading.Thread(target=serve, args=(int(lp), host, int(port)), daemon=True).start()
  try:
    threading.Event().wait()
  except KeyboardInterrupt:
    print("\n멈춥니다")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
