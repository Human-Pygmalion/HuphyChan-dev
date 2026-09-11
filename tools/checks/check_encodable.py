#!/usr/bin/env python3
"""파일 안의 문자열 상수가 전부 utf-8 로 나갈 수 있는지 확인한다.

2026-09-12. 화면 코드에 자물쇠 그림(🔒)을 `\\uD83D\\uDD12` 로 적었다가 페이지가
통째로 죽었습니다. 파이썬은 그 두 조각을 **합쳐 주지 않습니다** — 짝 잃은 반쪽
두 개가 그대로 남고, 내보낼 때 `surrogates not allowed` 로 터집니다. 그런데 이건
**파일을 불러올 때는 안 터집니다.** 그 글자가 든 문자열을 실제로 내보내는
순간에만 터지므로, 여기서는 `python3 -c "import ..."` 로도 안 잡혔습니다.

기본 문자 바깥(그림 문자 대부분)은 `\\U0001F512` 처럼 **여덟 자리**로 적습니다.

    python3 tools/checks/check_encodable.py System_ID/joint_web.py
"""
import ast
import sys


def check(path: str) -> list[tuple[int, str]]:
  bad = []
  tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
  for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      try:
        node.value.encode("utf-8")
      except UnicodeEncodeError as e:
        start = max(0, e.start - 20)
        bad.append((node.lineno, repr(node.value[start:e.end + 20])))
  return bad


def main(argv=None) -> int:
  paths = (argv or sys.argv[1:])
  if not paths:
    print(__doc__)
    return 2
  hurt = 0
  for p in paths:
    bad = check(p)
    for line, frag in bad:
      print(f"{p}:{line}: utf-8 로 못 내보내는 글자 — {frag}")
      print(f"  기본 문자 바깥은 \\U 여덟 자리로 적으세요 (예: \\U0001F512).")
    hurt += len(bad)
    if not bad:
      print(f"{p}: 문자열 상수 전부 utf-8 로 나감")
  return 1 if hurt else 0


if __name__ == "__main__":
  raise SystemExit(main())
