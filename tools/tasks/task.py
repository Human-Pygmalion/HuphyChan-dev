#!/usr/bin/env python3
"""사용자가 시킨 일을 **파일에 적어 두고**, 끝날 때까지 안 잊는다.

2026-09-16 · 사용자 질책: *"내가 이걸 시켰고, 충분히 백그라운드에서 이걸 하면서 다른
일도 할수 있었잖아. 왜 내가 시킨거 안하고 멈췄지? 원인 분석해서 무슨 일이 있어도
재발안하도록 해결해."*

## 무엇이 잘못됐었나

사용자가 *"외력·초기자세 강건화 학습해줘"* 를 시켰는데, 곧이어 *"그전에 값부터 뽑아줘"*
라고 끼어들었습니다. 저는 값을 뽑고 **거기서 멈췄습니다.** 앞의 지시가 사라진 이유는
단순합니다 — **그 지시가 제 머릿속에만 있었고, 어디에도 안 적혀 있었습니다.**

문맥이 압축돼서 잃은 게 아닙니다. **끼어들기로 인한 전환**이고, 전환 뒤에 원래 일을
다시 꺼내 주는 장치가 하나도 없었습니다. 이 리포의 훅은 전부
`Stop`(세션 끝) 과 `PostToolUse` 에만 걸려 있어서, 턴 중간에는 아무도 안 깨웁니다.

## 이 파일이 하는 일

열린 일을 `docs/.open_tasks.json` 에 적어 둡니다. 그리고 `open_tasks.sh` 가
**`UserPromptSubmit` 훅**으로 매 턴 그 목록을 앞에 꽂습니다. 그래서 무엇에 끼어들든
다음 턴에 열린 일이 다시 눈앞에 옵니다.

    task.py add "외력/초기자세 강건화 학습"        # 시킨 일을 받자마자
    task.py list
    task.py done 3
    task.py block 2 "루트 권한 필요"              # 못 하는 이유를 붙여 둔다

## 규칙

* 사용자가 일을 시키면 **먼저 적고** 시작한다. 끼어들기가 와도 목록에 남는다.
* 끝났으면 `done`. **사용자가 됐다고 하기 전에는 지우지 않는다.**
* 막히면 `block` 에 이유를 적는다 — 조용히 사라지는 것이 가장 나쁘다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

STORE = Path(__file__).resolve().parents[2] / "docs" / ".open_tasks.json"


def load() -> list:
  if not STORE.exists():
    return []
  try:
    return json.loads(STORE.read_text(encoding="utf-8"))
  except Exception:                                         # noqa: BLE001
    return []


def save(rows) -> None:
  STORE.parent.mkdir(parents=True, exist_ok=True)
  STORE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def show(rows, only_open=True) -> None:
  live = [r for r in rows if r["state"] != "done"] if only_open else rows
  if not live:
    print("  (열린 일 없음)")
    return
  for r in live:
    mark = {"open": "[ ]", "doing": "[~]", "blocked": "[!]", "done": "[x]"}[r["state"]]
    age = (time.time() - r["added"]) / 3600.0
    line = f"  {mark} #{r['id']:<3} {r['text']}"
    if r.get("why"):
      line += f"   <- {r['why']}"
    if age > 1:
      line += f"   ({age:.0f}시간 전)"
    print(line)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  a = sub.add_parser("add"); a.add_argument("text", nargs="+")
  for name in ("done", "doing"):
    s = sub.add_parser(name); s.add_argument("id", type=int)
  b = sub.add_parser("block"); b.add_argument("id", type=int); b.add_argument("why", nargs="+")
  sub.add_parser("list"); sub.add_parser("all")
  d = sub.add_parser("drop"); d.add_argument("id", type=int)
  n = ap.parse_args(argv)

  rows = load()
  if n.cmd == "add":
    nid = max((r["id"] for r in rows), default=0) + 1
    rows.append({"id": nid, "text": " ".join(n.text), "state": "open",
                 "added": time.time(), "why": ""})
    save(rows)
    print(f"  #{nid} 적음: {' '.join(n.text)}")
  elif n.cmd in ("done", "doing"):
    for r in rows:
      if r["id"] == n.id:
        r["state"] = n.cmd
        if n.cmd == "done":
          r["why"] = ""
    save(rows)
    print(f"  #{n.id} -> {n.cmd}")
  elif n.cmd == "block":
    for r in rows:
      if r["id"] == n.id:
        r["state"] = "blocked"
        r["why"] = " ".join(n.why)
    save(rows)
    print(f"  #{n.id} 막힘: {' '.join(n.why)}")
  elif n.cmd == "drop":
    rows = [r for r in rows if r["id"] != n.id]
    save(rows)
    print(f"  #{n.id} 지움")
  else:
    show(rows, only_open=(n.cmd == "list"))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
