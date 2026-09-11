"""Naming the viewer's current setup, and saying plainly when it has no name.

A "scenario" is NOT a mode the UI owns and switches. It is a NAME computed from state that
already exists and can be changed from elsewhere: the viewer's run mode, whether hardware
transmit is armed, and which program the robot is running. Two of those three have their own
controls (the mode dropdown, the TX arm button), so if either moves the name has to
recompute - and if the combination no longer matches a recipe EXACTLY, the name has to fall
away rather than linger and mislead.

Ported from the mockup (``mockups/scenario_common.js``, 2026-09-04) with one substantive
change. The mockup could only ever BELIEVE what the robot was running: it switched the
program over SSH and marked it confirmed on a zero exit code, with the caveat surfaced in the
UI. Here the robot program reports its own identity in the telemetry it already sends, so
``confirmed`` is a measurement with an age, and it goes stale on its own when the robot stops
talking. Nothing else about the decision changed: an inexact match is still custom, and
"unknown" still counts as not matching.

Pure - no DOM, no HTTP, no clock of its own. Every caller passes the state and the time.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

# ---------------------------------------------------------------- robot-side programs
PROGRAM_REMOTE_MOTION = 1
"""``bridge/huphy_remote_motion.py`` - takes targets from the viewer, torque ON."""

PROGRAM_OBS_STREAMER = 2
"""Reserved: streams observations with torque OFF. Not written yet."""

PROGRAM_POLICY_EXEC = 3
"""Reserved: runs a policy on the robot and streams its observations. Not written yet."""

PROGRAM_JOINT_WEB_READ = 4
"""``System_ID/joint_web.py`` with no ``--command`` - torque cannot leave, by construction.

This is the observation streamer :data:`PROGRAM_OBS_STREAMER` was reserved for, arrived from
the other direction: it was written on the robot side as a joint-angle page for a human
standing at the bench, and it happens to be exactly what ``mirror-hardware`` needs - all 12
joints reported at ~50 Hz with the motors free. It sends only the zero-gain frame, so there
is no mode in which it can push."""

PROGRAM_JOINT_WEB_DRY = 5
"""Same program with ``--command`` but not ``--live``: it computes what it would send and
sends nothing. Torque still cannot leave."""

PROGRAM_JOINT_WEB_LIVE = 6
"""Same program with ``--live``: pressing a button on that page puts torque into a motor.

Deliberately a SEPARATE id from the dry variant rather than one id plus a flag. An id is what
this module turns into the words "힘 꺼짐"/"힘 켜짐" on screen, and a flag that has to be read
alongside the id is a flag that will one day be read late. See ``System_ID/joint_web.py``
(``PROG_READ_ONLY``/``PROG_SLIDER_DRY``/``PROG_SLIDER_LIVE``) - **the numbers must match on
both sides**; change one and this module silently falls back to "unknown"."""

PROGRAMS: dict[int, dict] = {
  PROGRAM_REMOTE_MOTION: {"id": PROGRAM_REMOTE_MOTION, "name": "huphy_remote_motion",
                          "label": "화면에서 준 목표를 모터로 보내는 프로그램",
                          "torque": "ON", "exists": True},
  PROGRAM_OBS_STREAMER: {"id": PROGRAM_OBS_STREAMER, "name": "obs_streamer",
                         "label": "관측만 내보내는 프로그램 (힘 꺼짐)",
                         "torque": "OFF", "exists": False},
  PROGRAM_POLICY_EXEC: {"id": PROGRAM_POLICY_EXEC, "name": "policy_exec",
                        "label": "로봇이 정책을 돌리며 관측을 내보내는 프로그램",
                        "torque": "ON", "exists": False},
  PROGRAM_JOINT_WEB_READ: {"id": PROGRAM_JOINT_WEB_READ, "name": "joint_web(read-only)",
                           "label": "관절 화면 — 읽기만 함 (힘이 나갈 수 없음)",
                           "torque": "OFF", "exists": True},
  PROGRAM_JOINT_WEB_DRY: {"id": PROGRAM_JOINT_WEB_DRY, "name": "joint_web(dry-run)",
                          "label": "관절 화면 — 시험 모드 (계산만 하고 안 보냄)",
                          "torque": "OFF", "exists": True},
  PROGRAM_JOINT_WEB_LIVE: {"id": PROGRAM_JOINT_WEB_LIVE, "name": "joint_web(live)",
                           "label": "관절 화면 — 사람이 슬라이더로 모터를 움직임",
                           "torque": "ON", "exists": True},
}

PROGRAM_STALE_S = 2.0
"""s - a program identity older than this is 'unknown' again.

The robot repeats its identity with every diagnostic packet (a few times a second), so two
seconds of silence already means the link is gone - and a name derived from a dead link is
exactly the kind of stale confidence this module exists to prevent.
"""


# ---------------------------------------------------------------- the named combinations
@dataclasses.dataclass(frozen=True)
class Scenario:
  key: str
  name_ko: str
  summary: str
  action: str
  mode: str
  tx_armed: bool
  program: int
  arms_torque: bool
  """True if reaching this combination turns real torque ON. Only one does, and that has to
  be legible from the name and the button, not only from a colour."""
  available: bool = True
  unavailable_reason: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
  Scenario(
    key="drive-both", name_ko="실물 동시 구동",
    summary="같은 목표를 화면 속 모형과 실물에 동시에 주고, 둘의 응답을 비교합니다.",
    action="목표가 실물로 나갑니다. 모터에 힘이 들어갑니다.",
    mode="manual", tx_armed=True, program=PROGRAM_REMOTE_MOTION, arms_torque=True,
  ),
  Scenario(
    key="mirror-hardware", name_ko="손으로 미러링",
    summary="사람이 손으로 움직인 각도를 물리 계산 없이 화면에 그대로 그립니다.",
    action="아무것도 전송하지 않습니다. 받아서 그리기만 합니다.",
    mode="real_replay", tx_armed=False, program=PROGRAM_JOINT_WEB_READ, arms_torque=False,
  ),
  Scenario(
    key="shadow-policy", name_ko="로봇 정책 관측",
    summary="로봇이 자기 정책을 돌리고, 그 관측을 받아 화면 쪽 정책과 비교합니다.",
    action="아무것도 전송하지 않습니다. 정책 출력은 실물로 절대 나가지 않습니다.",
    mode="policy_shadow", tx_armed=False, program=PROGRAM_POLICY_EXEC, arms_torque=False,
    available=False,
    unavailable_reason="로봇 쪽 정책 실행기가 아직 없고, 받은 관측을 쓰는 경로도 아직 없습니다",
  ),
  Scenario(
    key="policy-drive", name_ko="정책으로 실물 일부 구동",
    summary="화면 속 모형이 걷고, 그 정책이 낸 목표가 지금 붙어 있는 모터로 그대로 나갑니다. "
            "정책이 보는 관측은 화면 속 모형의 것이지만, 항목별로 실물 값으로 바꿔 넣을 수 "
            "있습니다 (지금 쓸 수 있는 것은 실물 자세계).",
    action="정책 출력이 실물로 나갑니다. 모터에 힘이 들어갑니다.",
    mode="policy_sim", tx_armed=True, program=PROGRAM_REMOTE_MOTION, arms_torque=True,
  ),
)
"""``policy-drive`` (2026-09-08, 사용자 요청) 가 지금까지와 다른 점 하나만 말하면: 목표를
사람이 아니라 **정책이** 만든다는 것이다. 나가는 경로도, 그 경로를 지키는 관문도 전부
``drive-both`` 와 같다.

``shadow-policy`` 와 헷갈리기 쉬워 갈라 적는다. 그쪽은 **로봇이** 자기 정책을 돌리고 화면은
관측만 받는다(전송 없음). 이쪽은 **화면이** 정책을 돌리고 그 출력을 실물로 보낸다. 이름이
비슷하지만 힘이 들어가는 쪽은 이쪽 하나다.

관측을 실물 값으로 바꾸는 것은 이 시나리오의 **선택 사항**이지 조건이 아니다. 관절 12개 중
2개만 붙어 있는 지금, 관절 각도를 실물에서 받는 것은 의미가 없다(나머지 10개는 값이 없다).
자세계는 로봇 전체에 하나라 그 문제가 없어서, 지금 실물로 바꿔 넣을 수 있는 유일한 항목이다."""

BY_KEY = {s.key: s for s in SCENARIOS}

ProgramConfirm = Literal["confirmed", "unknown", "mismatch"]


def program_state(reported_id: int | None, age_s: float | None) -> tuple[ProgramConfirm, int | None]:
  """What the robot says it is running, and whether that is current.

  ``confirmed`` needs BOTH a name and a fresh one. A program id with no age, or an age past
  :data:`PROGRAM_STALE_S`, is ``unknown`` - not a guess, and not the last thing we heard.
  ``mismatch`` is reserved for a program we can name but do not recognise.
  """
  if reported_id is None or age_s is None or age_s > PROGRAM_STALE_S:
    return "unknown", None
  if reported_id not in PROGRAMS:
    return "mismatch", reported_id
  return "confirmed", reported_id


def derive(mode: str, tx_armed: bool, confirm: ProgramConfirm,
           program_id: int | None) -> str | None:
  """The name the UI is allowed to show, or ``None`` for "no name" (custom).

  Matching is exact on purpose - all three axes, with the program actually confirmed. A
  close-but-not-exact match is still custom. Naming a near-miss is how an operator ends up
  believing torque is off when it is on.
  """
  if confirm != "confirmed":
    return None
  for s in SCENARIOS:
    if mode == s.mode and bool(tx_armed) == s.tx_armed and program_id == s.program:
      return s.key
  return None


def differences(mode: str, tx_armed: bool, confirm: ProgramConfirm,
                program_id: int | None, key: str) -> list[str]:
  """Plain-language list of what stands between the current state and one named combination."""
  s = BY_KEY[key]
  out: list[str] = []
  if mode != s.mode:
    out.append(f"화면 모드가 '{mode}' 입니다 (필요: '{s.mode}')")
  if bool(tx_armed) != s.tx_armed:
    out.append("전송이 무장 상태입니다 (필요: 해제)" if tx_armed
               else "전송이 해제 상태입니다 (필요: 무장)")
  if confirm == "unknown":
    out.append("로봇이 어떤 프로그램을 돌리는지 지금 알 수 없습니다 (응답 없음)")
  elif confirm == "mismatch":
    out.append(f"로봇이 모르는 프로그램을 돌리고 있습니다 (번호 {program_id})")
  elif program_id != s.program:
    here = PROGRAMS.get(program_id, {}).get("label", f"번호 {program_id}")
    need = PROGRAMS[s.program]["label"]
    out.append(f"로봇이 '{here}' 를 돌리고 있습니다 (필요: '{need}')")
  return out


def nearest(mode: str, tx_armed: bool, confirm: ProgramConfirm,
            program_id: int | None) -> dict:
  """The named combination fewest steps away, for the "no name" banner.

  Only ever consulted when there IS no name, so it explains rather than labels.
  """
  best = None
  for s in SCENARIOS:
    d = differences(mode, tx_armed, confirm, program_id, s.key)
    if best is None or len(d) < len(best["differences"]):
      best = {"key": s.key, "name_ko": s.name_ko, "differences": d}
  return best


def would_arm_torque(tx_armed: bool, key: str) -> bool:
  """True if moving to ``key`` from here turns real torque on. Drives the extra confirmation."""
  s = BY_KEY.get(key)
  return bool(s and s.arms_torque and not tx_armed)


def status(mode: str, tx_armed: bool, reported_id: int | None, age_s: float | None) -> dict:
  """Everything the panel needs, in one shape."""
  confirm, pid = program_state(reported_id, age_s)
  key = derive(mode, tx_armed, confirm, pid)
  s = BY_KEY[key] if key else None
  out = {
    "key": key,
    "name_ko": s.name_ko if s else None,
    "summary": s.summary if s else None,
    "arms_torque": bool(s.arms_torque) if s else None,
    "mode": mode,
    "tx_armed": bool(tx_armed),
    "program": {
      "confirm": confirm,
      "id": pid,
      "name": PROGRAMS.get(pid, {}).get("name") if pid else None,
      "label": PROGRAMS.get(pid, {}).get("label") if pid else None,
      "torque": PROGRAMS.get(pid, {}).get("torque") if pid else None,
      "age_s": age_s,
    },
    "choices": [
      {
        "key": s2.key, "name_ko": s2.name_ko, "summary": s2.summary, "action": s2.action,
        "arms_torque": s2.arms_torque, "available": s2.available,
        "unavailable_reason": s2.unavailable_reason,
        "differences": differences(mode, tx_armed, confirm, pid, s2.key),
        "would_arm_torque": would_arm_torque(tx_armed, s2.key),
        "is_current": s2.key == key,
      }
      for s2 in SCENARIOS
    ],
  }
  if key is None:
    out["nearest"] = nearest(mode, tx_armed, confirm, pid)
  return out
