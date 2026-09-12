#!/usr/bin/env python3
"""새 통신 두절 규칙을 **흉내만 내 본다.** 아무것도 멈추지 않는다.

2026-09-13 · `System_ID/huphy_live.py` 에서 옮겨왔습니다 (`docs/138`).

`docs/138` 에는 이 파일을 `tools/huphy_audit/` 에 두겠다고 적었는데, **`bridge/` 로
바꿨습니다.** 제어 루프가 `--link-probe` 로 켜는 물건이라 브리지가 불러야 하고,
검사 하네스가 브리지를 불러오는 방향이 그 반대보다 자연스럽기 때문입니다.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict


class LinkProbe:
  """새 두절 판정을 **실물 위에서 흉내만 내 본다.** 아무것도 멈추지 않는다.

  ## 왜 이게 필요한가

  두절 판정 방식을 바꿨습니다 (`docs/132`). 예전에는 **연속** 무응답 주기를 셌고,
  새 방식은 **최근 열 번의 명령 중 몇 번 답이 없었나**를 셉니다. 책상 위 가짜
  모터로는 다 통과했지만, 실물 배선이 얼마나 자주 답을 빠뜨리는지는 **재 보기
  전에는 모릅니다.**

  새 규칙이 지금 배선에서 헛되이 걸린다면, 그것을 로봇에 올리는 순간 멀쩡한
  로봇이 몇 분마다 섭니다. 그래서 **먼저 흉내만 내 봅니다** — 로봇 프로그램은
  예전 규칙 그대로 돌고, 여기서는 "새 규칙이었다면 걸렸을까" 만 세어 기록합니다.

  ## 무엇을 세는가

  매 주기 `link_status()` 를 읽습니다. `ack` 가 1 이면 답한 것, 0 이면 이번 주기에
  명령했는데 답이 없는 것, -1 이면 명령하지 않은 것(세지 않음)입니다.

      lost      최근 `window` 번 명령 중 답이 없었던 횟수  <- 새 규칙이 보는 값
      lost_max  그 값이 이번 운용에서 도달한 최대
      miss_max  연속 무응답이 도달한 최대                  <- 예전 규칙이 보는 값
      asked     명령한 주기 수
      missed    그중 답이 없던 주기 수

  `lost_max` 가 문턱(기본 5)에 가까우면 **올리면 안 됩니다.**
  """

  WINDOW = 20
  """연속 문턱의 네 배. `Leg._link_window` 와 같은 값이어야 뜻이 같습니다."""

  THRESHOLD = 15
  """창 안에서 이만큼 잃으면 "샘" 으로 봅니다 = 75%.

  실측으로 고른 값입니다 (2026-09-12, 12개 모터 60초 6001주기). 지금 배선은
  모터마다 응답의 17.8~27.5% 를 잃고, 창 20 안에서 최악 12번입니다. 절반(10번)
  기준이면 전체 주기의 15.5% 에서 두절로 판정해 **로봇이 못 돕니다.**
  자세한 표는 HUPHY `safety/link.py` 문서에 있습니다."""

  def __init__(self, inner, robot, *, window: int = WINDOW):
    self.inner = inner
    self.robot = robot
    self.window = int(window)
    self.recent: Dict[str, deque] = {}
    self.lost_max: Dict[str, int] = {}
    self.miss_max: Dict[str, int] = {}
    self.asked: Dict[str, int] = {}
    self.missed: Dict[str, int] = {}
    self.cycles = 0
    self.would_trip = 0
    """새 규칙이었다면 두절로 판정했을 주기 수. **0 이어야 올릴 수 있습니다.**"""

  def _status(self) -> Dict[str, Any]:
    getter = getattr(self.robot, "link_status", None)
    if getter is None:
      return {}
    try:
      return getter() or {}
    except Exception:                                       # noqa: BLE001
      return {}

  def sample(self) -> None:
    self.cycles += 1
    tripped = False
    for name, row in self._status().items():
      ack = float(row.get("ack", -1.0))
      if ack < 0.0:
        continue                                   # 명령 안 한 모터는 안 셈
      silent = ack == 0.0
      buf = self.recent.setdefault(name, deque(maxlen=self.window))
      buf.append(silent)
      self.asked[name] = self.asked.get(name, 0) + 1
      if silent:
        self.missed[name] = self.missed.get(name, 0) + 1
      # 0 이어도 **자리를 만들어 둡니다.** 안 그러면 "한 번도 안 샜다" 와
      # "그 모터를 아예 못 봤다" 가 둘 다 빈칸이라 구분이 안 됩니다.
      lost = sum(buf)
      self.lost_max[name] = max(self.lost_max.get(name, 0), lost)
      self.miss_max[name] = max(self.miss_max.get(name, 0), int(row.get("miss", 0)))
      if lost >= self.THRESHOLD:
        tripped = True
    if tripped:
      self.would_trip += 1

  def rows(self) -> Dict[str, Dict[str, int]]:
    """관절 이름 -> 지금 값. 상태 전송에 실어 화면·기록으로 보냅니다."""
    return {
      name: dict(lost=int(sum(buf)), lost_max=int(self.lost_max.get(name, 0)),
                 miss_max=int(self.miss_max.get(name, 0)),
                 asked=int(self.asked.get(name, 0)),
                 missed=int(self.missed.get(name, 0)))
      for name, buf in self.recent.items()
    }

  def report(self) -> str:
    if not self.recent:
      return "  통신 흉내내기: 잰 것이 없습니다 (link_status 를 못 읽었습니다)"
    lines = [f"  '샘' 규칙 흉내내기 — {self.cycles}주기 "
             f"(최근 {self.window}번 중 {self.THRESHOLD}번 무응답이면 두절)"]
    lines.append(f"    {'관절':<22}{'명령':>8}{'무응답':>8}{'손실률':>8}"
                 f"{'창최대':>8}{'연속최대':>9}")
    for name in sorted(self.recent):
      asked = max(1, self.asked.get(name, 0))
      lines.append(f"    {name:<22}{self.asked.get(name, 0):>8}"
                   f"{self.missed.get(name, 0):>8}"
                   f"{100.0 * self.missed.get(name, 0) / asked:>7.1f}%"
                   f"{self.lost_max.get(name, 0):>8}{self.miss_max.get(name, 0):>9}")
    worst = max(self.lost_max.values()) if self.lost_max else 0
    if self.would_trip:
      lines.append(f"    ★ '샘' 규칙이었다면 {self.would_trip}주기에서 두절로 "
                   f"판정했습니다 (창 최대 {worst}). **올리면 안 됩니다.**")
    else:
      lines.append(f"    '샘' 규칙으로는 한 번도 안 걸렸습니다 (창 최대 {worst} "
                   f"< 문턱 {self.THRESHOLD}). 여유 {self.THRESHOLD - worst}번.")
    return "\n".join(lines)

  def __call__(self, t: float, observation: Dict[str, Any]) -> Dict[str, float]:
    self.sample()
    return self.inner(t, observation)
