#!/usr/bin/env python3
"""발목 크랭크 명령을 **안전한 쌍으로 바꾸는 한 곳.** 화면도 검사도 여기를 부른다.

2026-09-13 · 설계는 `docs/139`. 이 파일은 그 설계의 구현이고, **입출력이 좁습니다** --
받는 것은 명령 크랭크각 두 개, 내는 것은 나갈 크랭크각 두 개와 그때의 발판 자세,
그리고 **어느 코드 블럭이 반응했는지**입니다.

    r = clip_ab(cmd_a, cmd_b)
    r.a_out, r.b_out      실제로 나갈 크랭크각
    r.pitch, r.roll       그때 발판이 서는 자세
    r.stage               마지막으로 값을 바꾼 블럭 이름
    r.stages              거쳐 간 블럭 전부 (순서대로)
    r.inside              최종 결과가 정말 허용 영역 안인가  <- False 면 못 막은 것

## 왜 이렇게 자르는가 (한 문단)

발목은 크랭크 두 개가 발판 하나를 미는 닫힌 기구라, 두 각을 **따로** 잘라서는
조합이 만드는 발판 자세를 막을 수 없습니다. 제약이 **네모로 보이는 공간**이 둘인데
서로 다릅니다 -- 모터 한계는 크랭크 공간에서 네모, 발판 ROM 은 자세 공간에서
네모입니다. 그래서 **번갈아** 자릅니다. 실측으로 두 번이면 수렴합니다.

## 블럭 이름 (그대로 화면에 뜹니다)

    motor_box      크랭크 자체 한계로 잘림
    a_reach        그 A 로는 어떤 B 도 ROM 안에 못 서서 A 를 당김
    rom_box        발판 자세를 ROM 으로 잘라 되돌아옴
    table_fallback 기구학이 안 풀려 표로만 잘림
    hold_last      두 번 돌려도 못 들어와 직전 통과값을 유지
    none           손댈 것이 없었음
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
HUPHY_SRC = Path("/home/syaro/external_repos/HUPHY/src")
if HUPHY_SRC.exists() and str(HUPHY_SRC) not in sys.path:
  sys.path.insert(0, str(HUPHY_SRC))

ROM_PITCH = (-50.0, 30.0)
ROM_ROLL = (-20.0, 20.0)
MOTOR_LIMIT = 68.75
MAX_ROUNDS = 3
POSE_SANE_DEG = 90.0
"""발판 자세가 이보다 크면 **푼 값이 틀린 것**입니다 (다중회전). 기구가 그렇게 못 돕니다."""


@dataclass
class ClipResult:
  a_in: float
  b_in: float
  a_out: float
  b_out: float
  pitch: float = float("nan")
  roll: float = float("nan")
  stage: str = "none"
  stages: List[str] = field(default_factory=list)
  rounds: int = 0
  inside: bool = True
  notes: List[str] = field(default_factory=list)

  @property
  def changed(self) -> bool:
    return abs(self.a_out - self.a_in) > 1e-9 or abs(self.b_out - self.b_in) > 1e-9

  def as_dict(self) -> dict:
    return dict(a_in=self.a_in, b_in=self.b_in, a_out=self.a_out, b_out=self.b_out,
                pitch=self.pitch, roll=self.roll, stage=self.stage, stages=self.stages,
                rounds=self.rounds, inside=self.inside, changed=self.changed,
                notes=self.notes)


class AnkleClipper:
  """한 다리의 발목 클리퍼. 표를 한 번 읽어 두고 매 주기 쓴다.

  표(`ankle_ab_envelope_<leg>.json`)가 없으면 **정확 기구학만으로** 돕니다 -- 느리지만
  결과는 같습니다. 표는 속도(뉴턴 반복 223us -> 보간 18us)와 대비책을 위한 것입니다.
  """

  def __init__(self, leg: str = "left", *, rom_pitch=ROM_PITCH, rom_roll=ROM_ROLL,
               motor_limit: float = MOTOR_LIMIT, table: Optional[dict] = None,
               kinematics=None):
    self.leg = leg
    self.rom_pitch = tuple(rom_pitch)
    self.rom_roll = tuple(rom_roll)
    self.limit = float(motor_limit)
    self.kin = kinematics if kinematics is not None else self._default_kinematics(leg)
    self.table = table if table is not None else self._load_table(leg)
    self._axis = self.table["axis"] if self.table else None
    self._blo = self.table["b_lo"] if self.table else None
    self._bhi = self.table["b_hi"] if self.table else None
    # 표에서 "B 가 하나라도 갈 수 있는 A" 의 양 끝. 모터 한계(±68.75)보다 좁습니다 --
    # A 를 그 밖으로 두면 어떤 B 를 골라도 발판이 ROM 안에 못 섭니다 (실측 -55.2~+66.8).
    self._a_reach: Optional[Tuple[float, float]] = None
    if self._axis and self._blo:
      have = [self._axis[i] for i, v in enumerate(self._blo) if v is not None]
      if have:
        self._a_reach = (float(have[0]), float(have[-1]))
    self._last_ok: Optional[Tuple[float, float]] = None
    self._guess = (0.0, 0.0)
    self.counts = dict(none=0, motor_box=0, a_reach=0, rom_box=0, table_fallback=0,
                       hold_last=0, not_converged=0, escaped=0)

  # ---- 준비물 -------------------------------------------------------------
  @staticmethod
  def _default_kinematics(leg: str):
    from huphy.kinematics.ankle import AnkleGeometry, AnkleKinematics
    geom = AnkleGeometry()
    if leg == "left":
      geom = geom.mirrored()
    return AnkleKinematics(geom)

  @staticmethod
  def _load_table(leg: str) -> Optional[dict]:
    p = HERE / f"ankle_ab_envelope_{leg}.json"
    if not p.exists():
      return None
    return json.loads(p.read_text(encoding="utf-8"))

  # ---- 기구학 --------------------------------------------------------------
  def fk(self, a: float, b: float,
         guess: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """크랭크각 -> 발판 자세. 못 풀면 `None`.

    ★ **추정값을 조심해야 합니다.** 뉴턴 반복이라 추정이 가까우면 빠르지만, 멀면
    **다른 자세로 수렴**합니다 (`kinematics/ankle.py` 머리말). 그래서

      * 추정은 **한 번의 `clip()` 안에서만** 이어 씁니다. 클리퍼가 오래 들고 있는
        값은 **마지막으로 통과한 자세** 하나뿐입니다.
      * 그 추정으로 푼 답은 **IK 로 되짚어 확인**합니다. 어긋나면 추정 없이(0,0)
        다시 풉니다.

    이 확인이 없을 때 무작위 명령 800개 중 778개가 "못 막음" 으로 잘못 보고됐습니다
    (2026-09-13). 자르기가 틀린 게 아니라 **확인하는 눈이 틀렸던** 것입니다.
    """
    g = self._guess if guess is None else guess
    for attempt in (g, (0.0, 0.0)):
      try:
        p, r = self.kin.solve_fk(a, b, guess_pitch_deg=attempt[0],
                                 guess_roll_deg=attempt[1])
      except Exception:                                     # noqa: BLE001
        continue
      # ★ 다중회전 값을 먼저 버립니다. 추정이 멀면 뉴턴이 +5840도 같은 답으로
      # 수렴하는데, 그 각을 IK 로 되짚으면 360도 배수라 **같은 크랭크 쌍이 나와**
      # 되짚기 검사를 통과합니다 (2026-09-13 실측). 발판은 그렇게 못 돕니다.
      if abs(p) > POSE_SANE_DEG or abs(r) > POSE_SANE_DEG:
        continue
      back = self.ik(p, r)
      if back is None:
        continue
      if abs(back[0] - a) < 0.05 and abs(back[1] - b) < 0.05:
        return p, r
    return None

  def ik(self, pitch: float, roll: float) -> Optional[Tuple[float, float]]:
    try:
      return self.kin.solve_ik(pitch, roll, enforce_envelope=False)
    except Exception:                                       # noqa: BLE001
      return None

  # ---- 표 -----------------------------------------------------------------
  def table_span(self, a: float) -> Optional[Tuple[float, float]]:
    """그 A 에서 B 가 갈 수 있는 구간. 표가 없거나 범위 밖이면 `None`."""
    if not self._axis:
      return None
    lo_a, hi_a = self._axis[0], self._axis[-1]
    if not (lo_a <= a <= hi_a):
      return None
    step = self._axis[1] - self._axis[0]
    i = int(round((a - lo_a) / step))
    i = max(0, min(len(self._axis) - 1, i))
    # 그 칸이 비어 있으면(경계에 딱 걸린 경우) 가까운 칸을 찾습니다. 안 그러면
    # 마지막 그물이 조용히 건너뛰어집니다 (2026-09-13 실측).
    for d in range(0, 8):
      for j in (i - d, i + d):
        if 0 <= j < len(self._axis) and self._blo[j] is not None:
          return float(self._blo[j]), float(self._bhi[j])
    return None

  def inside(self, a: float, b: float) -> bool:
    """이 쌍이 허용 영역 안인가. **기구학으로** 판정한다 (표는 근사)."""
    f = self.fk(a, b)
    if f is None:
      return False
    p, r = f
    return (self.rom_pitch[0] - 1e-6 <= p <= self.rom_pitch[1] + 1e-6
            and self.rom_roll[0] - 1e-6 <= r <= self.rom_roll[1] + 1e-6)

  # ---- 본체 ---------------------------------------------------------------
  def clip(self, a_cmd: float, b_cmd: float) -> ClipResult:
    """명령 한 쌍을 안전한 쌍으로. **이 함수 하나가 전부입니다.**"""
    res = ClipResult(a_in=float(a_cmd), b_in=float(b_cmd),
                     a_out=float(a_cmd), b_out=float(b_cmd))

    # (1) 모터 상자 -- 크랭크 자체 한계
    a = _clamp(res.a_out, -self.limit, self.limit)
    b = _clamp(res.b_out, -self.limit, self.limit)
    if (a, b) != (res.a_out, res.b_out):
      res.stages.append("motor_box")
      res.notes.append(f"모터 한계 ±{self.limit:g}도로 자름")
    res.a_out, res.b_out = a, b

    # (1b) A 가 **어떤 B 로도 못 가는** 자리면 갈 수 있는 A 로 먼저 당긴다.
    # 이것이 없으면 아래 번갈아 자르기가 수렴할 곳이 아예 없습니다 (실측: 800개 중
    # 3개가 여기서 빠져나갔음, 셋 다 A 가 ±68.75 로 잘린 뒤의 자리였음).
    if self._a_reach is not None:
      a2 = _clamp(res.a_out, self._a_reach[0], self._a_reach[1])
      if abs(a2 - res.a_out) > 1e-9:
        res.stages.append("a_reach")
        res.notes.append(
          f"A {res.a_out:+.1f}도에서는 어떤 B 도 ROM 안에 못 섬 → "
          f"{a2:+.1f}도로 당김 (갈 수 있는 A {self._a_reach[0]:+.1f}~{self._a_reach[1]:+.1f})")
        res.a_out = a2

    for rnd in range(MAX_ROUNDS):
      res.rounds = rnd + 1
      f = self.fk(res.a_out, res.b_out)

      # (3) 기구학이 안 풀림 -> 표로만 자른다
      if f is None:
        span = self.table_span(res.a_out)
        if span is None:
          res.stages.append("hold_last")
          res.notes.append("기구학도 표도 못 씀 — 직전 통과값 유지")
          if self._last_ok:
            res.a_out, res.b_out = self._last_ok
          break
        nb = _clamp(res.b_out, span[0], span[1])
        res.stages.append("table_fallback")
        res.notes.append(f"기구학이 안 풀려 표로 자름: B {span[0]:+.1f}~{span[1]:+.1f}")
        if abs(nb - res.b_out) < 1e-9:
          break
        res.b_out = nb
        continue

      p, r = f
      pc = _clamp(p, *self.rom_pitch)
      rc = _clamp(r, *self.rom_roll)
      res.pitch, res.roll = p, r
      if (pc, rc) == (p, r):                    # (2) 이미 안. 손대지 않는다
        break

      # (2) ROM 상자 -- 발판 자세를 네모로 자르고 되돌아온다
      s = self.ik(pc, rc)
      if s is None:
        span = self.table_span(res.a_out)
        if span is not None:
          nb = _clamp(res.b_out, span[0], span[1])
          res.stages.append("table_fallback")
          res.notes.append("되돌리기(IK)가 안 풀려 표로 자름")
          if abs(nb - res.b_out) < 1e-9:
            break
          res.b_out = nb
          continue
        res.stages.append("hold_last")
        if self._last_ok:
          res.a_out, res.b_out = self._last_ok
        break

      a2 = _clamp(s[0], -self.limit, self.limit)
      b2 = _clamp(s[1], -self.limit, self.limit)
      res.stages.append("rom_box")
      res.notes.append(
        f"발판 {p:+.1f}/{r:+.1f}도가 ROM 밖 → {pc:+.1f}/{rc:+.1f}도로 자름")
      if abs(a2 - res.a_out) < 1e-3 and abs(b2 - res.b_out) < 1e-3:
        res.a_out, res.b_out = a2, b2
        break
      res.a_out, res.b_out = a2, b2
    else:
      res.stages.append("not_converged")

    # (4) 마지막 그물: 아직도 밖이면 **표로** 자른다. 표는 정의상 영역 안이라
    # 반드시 안에 듭니다 -- 자세는 최선이 아닐 수 있어도(설계 4절) 안전이 먼저입니다.
    f = self.fk(res.a_out, res.b_out)
    if f is not None:
      p_, r_ = f
      out = not (self.rom_pitch[0] - 1e-3 <= p_ <= self.rom_pitch[1] + 1e-3
                 and self.rom_roll[0] - 1e-3 <= r_ <= self.rom_roll[1] + 1e-3)
    else:
      out = True
    if out:
      span = self.table_span(res.a_out)
      if span is not None:
        nb = _clamp(res.b_out, span[0], span[1])
        if abs(nb - res.b_out) > 1e-9:
          res.b_out = nb
          res.stages.append("table_fallback")
          res.notes.append(
            f"번갈아 자르기로 못 들어와 표로 마무리: B {span[0]:+.1f}~{span[1]:+.1f}")

    # 최종 확인 -- 못 막았으면 그렇게 적는다. 조용히 넘기지 않는다.
    f = self.fk(res.a_out, res.b_out)
    if f is None:
      res.inside = False
      res.notes.append("최종 자세를 못 구함")
    else:
      res.pitch, res.roll = f
      res.inside = (self.rom_pitch[0] - 1e-3 <= res.pitch <= self.rom_pitch[1] + 1e-3
                    and self.rom_roll[0] - 1e-3 <= res.roll <= self.rom_roll[1] + 1e-3)
    if res.inside:
      self._last_ok = (res.a_out, res.b_out)
      self._guess = (res.pitch, res.roll)      # 통과한 자세만 다음 주기의 추정으로
    else:
      self.counts["escaped"] += 1
      res.notes.append("★ 못 막았습니다 — 결과가 여전히 허용 영역 밖입니다")

    res.stage = res.stages[-1] if res.stages else "none"
    self.counts[res.stage] = self.counts.get(res.stage, 0) + 1
    return res


def _clamp(v: float, lo: float, hi: float) -> float:
  return lo if v < lo else (hi if v > hi else v)


def clip_ab(a: float, b: float, leg: str = "left") -> ClipResult:
  """한 번만 쓸 때. 되풀이해 쓸 것이면 `AnkleClipper` 를 들고 있을 것."""
  return _shared(leg).clip(a, b)


_CACHE: dict = {}


def _shared(leg: str) -> AnkleClipper:
  if leg not in _CACHE:
    _CACHE[leg] = AnkleClipper(leg)
  return _CACHE[leg]


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("a", type=float, nargs="?", help="크랭크 A 명령 (도)")
  ap.add_argument("b", type=float, nargs="?", help="크랭크 B 명령 (도)")
  ap.add_argument("--leg", default="left", choices=("left", "right"))
  ap.add_argument("--sweep", type=int, default=0,
                  help="무작위 명령을 이만큼 넣어 통계만 낸다 (헤드리스 점검)")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  c = AnkleClipper(args.leg)
  if args.sweep:
    import random
    rng = random.Random(0)
    escaped = []
    for _ in range(args.sweep):
      a, b = rng.uniform(-90, 90), rng.uniform(-90, 90)
      r = c.clip(a, b)
      if not r.inside:
        escaped.append((round(a, 2), round(b, 2)))
    print(f"  {args.sweep}개 · 막음 {args.sweep - len(escaped)} · **못 막음 {len(escaped)}**")
    print(f"  블럭별: {c.counts}")
    if escaped:
      print(f"  못 막은 입력 예: {escaped[:5]}")
    return 1 if escaped else 0

  if args.a is None or args.b is None:
    ap.error("a 와 b 를 주거나 --sweep 을 쓸 것")
  r = c.clip(args.a, args.b)
  if args.json:
    print(json.dumps(r.as_dict(), ensure_ascii=False))
  else:
    print(f"  명령 A{r.a_in:+.2f} B{r.b_in:+.2f}  ->  A{r.a_out:+.2f} B{r.b_out:+.2f}")
    print(f"  발판 숙임 {r.pitch:+.2f}도 · 기울임 {r.roll:+.2f}도")
    print(f"  반응한 블럭: {' -> '.join(r.stages) or 'none'} · 반복 {r.rounds}회 · "
          f"{'허용 안' if r.inside else '★ 허용 밖'}")
    for n in r.notes:
      print(f"    - {n}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
