#!/usr/bin/env python3
"""**실물 관절을 손으로 움직이면서 보는 화면.** 모터로 나가는 길이 없다.

2026-09-14 · 사용자 지시: *"지금 최종으로 할당된 mjcf 를 기반으로 viser 와, 실제
값들을 모니터링하는 UI 를 띄워 달라. 그러면 내가 실제 로봇을 조작하면서, 상 / 하한에
걸리는지, 지금 어디 있는지 실제로 만져보면서 검토해보겠다."*

    mujoco-sim/mjlab/.venv/bin/python3 tools/robot_model/live_monitor.py
    http://192.168.20.177:8113/

## 안전

* **명령을 만드는 코드가 이 파일에 없습니다.** 슬라이더도, 전송 단추도, 목표도
  없습니다. UDP 를 **받기만** 합니다.
* 로봇 쪽도 관찰 모드로 돌립니다 (`huphy_live.py --observe`). 그 모드는 켜기(0xFC)를
  한 번도 안 보내므로 모터에 힘이 들어가지 않습니다. 그래서 손으로 움직일 수 있습니다.

## 무엇을 재사용했나 (새로 만든 것이 아닙니다)

| 남의 것 / 우리 것 | 어디 | 무엇을 |
|---|---|---|
| HUPHY | `robots/leg.py:722` `Leg.refresh()` | 명령 없이 상태만 읽기 |
| 우리 | `System_ID/huphy_live.py` `StateSender` | 상태를 JSON 한 줄로 UDP 에 뿌리기 |
| 우리 | `tools/viewer/mjcf_joint_viewer.py` `build_model` | **최종 MJCF** + 크랭크 위치 서보 + 중력 끔 |
| 우리 | `tools/pygviewer/.../joint_map_huphy.json` | 모터 이름 → 모형 관절 이름 대응 |
| 우리 | `tools/robot_model/ankle_clip.py` | 크랭크 A·B → 발판 숙임·기울임 |

`tools/pygviewer/.../bridge/huphy_viser.py` 도 같은 일을 하는 도구입니다만, 그쪽은
**구워 둔 모형(2026-09-04)** 을 씁니다. 좌우 이름 고침이 2026-09-12 이라 그 안의
모형은 **고치기 전 것**입니다. 사용자가 "지금 최종으로 할당된 mjcf" 를 요구했으므로
여기서는 MJCF 를 직접 엽니다. (그쪽을 쓰려면 계약을 다시 구워야 합니다.)

## 화면에 무엇이 뜨나

관절마다 한 줄로

    지금 각도 · 로봇이 아는 한계(lo~hi) · **모형(MJCF)이 아는 한계** · 어느 쪽에 몇 도 남았나

두 한계를 **나란히** 보여 주는 것이 요점입니다 — 둘이 다르면 그 자체가 찾던 문제입니다.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for extra in (HERE, REPO / "tools" / "viewer"):
  if str(extra) not in sys.path:
    sys.path.insert(0, str(extra))

JMAP_PATH = REPO / "tools" / "pygviewer" / "pygviewer" / "bridge" / "joint_map_huphy.json"

NEAR_DEG = 5.0
"""한계에 이만큼 안쪽으로 들어오면 노랗게, 넘으면 빨갛게 칠합니다."""
STALE_S = 1.5
"""이 시간 넘게 새 소식이 없으면 "끊겼다" 고 크게 적습니다."""


def _fmt(v, w=7, p=2):
  return " " * w if v is None else f"{v:{w}.{p}f}"


class Telemetry:
  """`StateSender` 가 뿌리는 JSON 한 줄을 받아 그대로 들고 있는다. 보내지 않는다."""

  def __init__(self, port: int, host: str = "0.0.0.0"):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.bind((host, port))
    self.sock.settimeout(0.5)
    self.rows: dict = {}
    self.stamp = 0.0
    self.packets = 0
    self.note = ""
    self._stop = threading.Event()
    self._t = threading.Thread(target=self._loop, daemon=True)

  def start(self):
    self._t.start()
    return self

  def stop(self):
    self._stop.set()

  def _loop(self):
    while not self._stop.is_set():
      try:
        data, _ = self.sock.recvfrom(65535)
      except socket.timeout:
        continue
      except OSError:
        break
      try:
        pkt = json.loads(data.decode("utf-8"))
      except Exception:                                     # noqa: BLE001
        continue
      motors = pkt.get("motors")
      if not isinstance(motors, dict):
        continue
      self.rows = motors
      self.note = str(pkt.get("note") or "")
      self.stamp = time.monotonic()
      self.packets += 1

  @property
  def age(self) -> float:
    return float("inf") if not self.stamp else time.monotonic() - self.stamp


def load_map() -> dict:
  """`(limb, motor)` → 모형 관절 이름. 없는 짝은 **추측하지 않고 버립니다.**"""
  raw = json.loads(JMAP_PATH.read_text(encoding="utf-8"))
  out = {}
  for row in raw.get("motors", []):
    out[(row["limb"], row["motor"])] = row
  return dict(rows=out, verified=bool(raw.get("side_mapping_verified")))


def split_joint(full: str):
  """`left_leg/hip_pitch` → `("left", "hip_pitch")`. 대응표는 `left` 를 씁니다."""
  limb, _, motor = str(full).partition("/")
  return limb.replace("_leg", ""), motor


def run(host: str, port: int, udp_port: int) -> int:
  import mujoco
  import viser
  import mjcf_joint_viewer as JV
  from mjviser import ViserMujocoScene
  from ankle_clip import AnkleClipper

  jmap = load_map()
  model, crank_actuators = JV.build_model("legonly", "loop")
  data = mujoco.MjData(model)

  def jid(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

  act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
         for n, a in crank_actuators.items()}
  cranks = set(JV.CRANK_JOINTS)

  # 크랭크·로드·발목만 물리로 풀고 나머지는 우리가 직접 넣습니다.
  free_names = JV.LOOP_HIDDEN + JV.CRANK_JOINTS + JV.LOOP_PASSIVE_READOUT
  free_q = {int(model.jnt_qposadr[jid(n)]) for n in free_names}
  free_v = {int(model.jnt_dofadr[jid(n)]) for n in free_names}

  # 직접 넣을 관절(엉덩이·무릎)의 자리
  driven = {}
  for key, row in jmap["rows"].items():
    sj = row["sim_joint"]
    if sj in cranks or jid(sj) < 0:
      continue
    driven[key] = (int(model.jnt_qposadr[jid(sj)]), int(row.get("sign", 1)),
                   float(row.get("offset_rad", 0.0)), sj)
  pin_q = sorted(set(range(model.nq)) - free_q - {a for a, *_ in driven.values()})
  pin_v = sorted(set(range(model.nv)) - free_v)

  data.qpos[:] = 0.0
  data.qpos[0:3] = [0, 0, 1.0]
  data.qpos[3:7] = [1, 0, 0, 0]
  mujoco.mj_forward(model, data)
  home = data.qpos.copy()

  # MJCF 가 아는 한계 [도]
  mjcf_lim = {}
  for key, row in jmap["rows"].items():
    j = jid(row["sim_joint"])
    if j < 0:
      continue
    lo, hi = model.jnt_range[j]
    mjcf_lim[key] = (math.degrees(lo), math.degrees(hi)) if (lo or hi) else None

  clippers = {s: AnkleClipper(s) for s in ("left", "right")}

  tel = Telemetry(udp_port).start()
  server = viser.ViserServer(host=host, port=port, label="실물 관절 보기 (읽기 전용)")
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False        # 켜 두면 몸통이 원점으로 끌려옵니다

  gui = server.gui
  with gui.add_folder("상태"):
    head = gui.add_markdown("소식을 기다리는 중…")
  with gui.add_folder("관절 (실시간)"):
    table = gui.add_markdown(" ")
  with gui.add_folder("한계 대조 (로봇 설정 vs 최종 MJCF)", expand_by_default=False):
    # ★ 이 표는 **안 변합니다.** 첫 꾸러미가 오면 한 번 그리고 그대로 둡니다.
    # 실시간 표에 같이 넣었더니 칸이 일곱 개가 되어 좁은 창에서 잘렸습니다
    # (2026-09-14 화면으로 확인).
    limits_md = gui.add_markdown("소식이 오면 채웁니다.")
  with gui.add_folder("발목 (크랭크 → 발판)"):
    ankle_md = gui.add_markdown(" ")
  with gui.add_folder("읽어 주세요"):
    gui.add_markdown(
      "* 이 화면에는 **모터로 나가는 길이 없습니다.** 받기만 합니다.\n"
      "* 로봇 쪽은 관찰 모드(`--observe`)라 **켜기를 한 번도 안 보냅니다.**\n"
      "* `모형` 칸은 지금 최종 MJCF 의 관절 한계입니다. `로봇` 칸은 로봇이 "
      "설정으로 들고 있는 한계(여유 3도를 이미 뺀 값)입니다. "
      "**둘이 다르면 그것이 찾던 문제입니다.**\n"
      + ("" if jmap["verified"] else
         "* ⚠ **좌우·부호가 아직 실측으로 확인되지 않았습니다** "
         "(`joint_map_huphy.json` 의 `side_mapping_verified: false`). "
         "지금 손으로 움직여 보시는 것이 바로 그 확인입니다 — 왼쪽을 굽혔는데 "
         "화면에서 오른쪽이 움직이거나 반대로 돌면 말씀해 주십시오."))

  def physics():
    """받은 값을 모형에 넣고 닫힌 고리를 계속 풀어 준다. **읽기만 한다.**"""
    while True:
      rows = tel.rows
      for key, (adr, sign, off, _sj) in driven.items():
        r = _find(rows, key)
        if r is not None and r.get("deg") is not None:
          data.qpos[adr] = sign * math.radians(float(r["deg"])) + off
      for full, aname in crank_actuators.items():
        side = "left" if full.startswith("L_") else "right"
        tag = "ankle_a" if "_A_" in full else "ankle_b"
        r = _find(rows, (side, tag))
        if r is not None and r.get("deg") is not None:
          data.ctrl[act[full]] = math.radians(float(r["deg"]))
      for _ in range(8):
        data.qpos[pin_q] = home[pin_q]
        data.qvel[pin_v] = 0.0
        mujoco.mj_step(model, data)
      mujoco.mj_forward(model, data)
      scene.update_from_mjdata(data)
      time.sleep(0.02)

  def _find(rows: dict, key):
    for r in rows.values():
      if split_joint(r.get("joint", "")) == key:
        return r
    return None

  def refresh():
    while True:
      rows = tel.rows
      age = tel.age
      if not rows:
        head.content = (f"### ⏳ 아직 아무 소식이 없습니다\n"
                        f"`{udp_port}` 번으로 기다리는 중. 로봇에서 관찰 모드를 "
                        f"띄우셨는지 확인해 주십시오.")
      elif age > STALE_S:
        head.content = (f"### 🔴 끊겼습니다 — 마지막 소식이 **{age:.1f}초 전**\n"
                        f"받은 꾸러미 {tel.packets}개")
      else:
        head.content = (f"### 🟢 받는 중 · 마지막 소식 {age*1000:.0f}밀리초 전\n"
                        f"꾸러미 {tel.packets}개 · 모터 {len(rows)}개"
                        + (f"\n\n{tel.note}" if tel.note else ""))

      # 좁은 창에 맞추려고 **고정폭 글자**로, 그리고 **한 줄 34칸 안으로** 씁니다.
      # 칸이 넷만 넘어도 오른쪽이 잘려 안 보였습니다 (2026-09-14 화면으로 두 번 확인).
      # 그래서 **안 변하는 것(한계 값)은 아래 접힌 서랍으로 옮겼습니다** -- 여기에는
      # 움직이는 것만 둡니다.
      lines = ["```",
               "관절          지금    아래   위",
               "---------------------------------"]
      for r in sorted(rows.values(), key=lambda x: str(x.get("joint", ""))):
        key = split_joint(str(r.get("joint", "?")))
        short = f"{key[0][0].upper()} {key[1]}"
        deg, lo, hi = r.get("deg"), r.get("lo"), r.get("hi")
        mark, room = "", "             "
        if deg is not None and lo is not None and hi is not None:
          d_lo, d_hi = float(deg) - float(lo), float(hi) - float(deg)
          room = f"{d_lo:+6.1f}{d_hi:+7.1f}"
          if d_lo < 0 or d_hi < 0:
            mark = " <<"                      # 한계 밖
          elif min(d_lo, d_hi) < NEAR_DEG:
            mark = " ! "                      # 한계가 코앞
        lines.append(f"{short:<12}{_fmt(deg, 7, 1)}{room}{mark}")
      lines += ["```",
                "`아래`·`위` 는 그 한계까지 **몇 도 남았나** 입니다. "
                "`!` 5도 안쪽 · `<<` **넘었음**",
                "",
                "```",
                "관절           힘    온도",
                "-------------------------",
                *[f"{split_joint(str(r.get('joint','?')))[0][0].upper()} "
                  f"{split_joint(str(r.get('joint','?')))[1]:<10}"
                  f"{_fmt(r.get('nm'), 6, 2)}{_fmt(r.get('temp'), 7, 1)}"
                  for r in sorted(rows.values(), key=lambda x: str(x.get("joint", "")))],
                "```"]

      if rows and limits_md.content.startswith("소식"):
        cmp_lines = ["```",
                     "관절          로봇 설정",
                     "                 최종 MJCF",
                     "---------------------------------"]
        for r in sorted(rows.values(), key=lambda x: str(x.get("joint", ""))):
          key = split_joint(str(r.get("joint", "?")))
          lo, hi = r.get("lo"), r.get("hi")
          m = mjcf_lim.get(key)
          a_ = "      --     " if lo is None else f"{lo:+6.1f} ~{hi:+6.1f}"
          b_ = "      --     " if m is None else f"{m[0]:+6.1f} ~{m[1]:+6.1f}"
          flag = ""
          if m and lo is not None and (abs(m[0] - lo) > 5.0 or abs(m[1] - hi) > 5.0):
            flag = " <--"
          cmp_lines.append(f"{key[0][0].upper()} {key[1]:<11}{a_}")
          cmp_lines.append(f"{'':<13}{b_}{flag}")
        cmp_lines += ["```",
                      "로봇 값은 여유 3도를 이미 뺀 것입니다. **다름** 이 붙은 줄은 "
                      "로봇이 아는 한계와 모형이 아는 한계가 5도 넘게 벌어진 것입니다 — "
                      "부호 규약이 다르거나, 한쪽이 낡은 것입니다."]
        limits_md.content = "\n".join(cmp_lines)

      table.content = "\n".join(lines) if rows else " "

      ank = []
      for side in ("left", "right"):
        ra = _find(rows, (side, "ankle_a"))
        rb = _find(rows, (side, "ankle_b"))
        if not (ra and rb) or ra.get("deg") is None or rb.get("deg") is None:
          continue
        a, b = float(ra["deg"]), float(rb["deg"])
        f = clippers[side].fk(a, b)
        name = "왼발" if side == "left" else "오른발"
        if f is None:
          ank.append(f"**{name}** · A {a:+.1f} B {b:+.1f} → 기구학이 안 풀립니다")
          continue
        p, r_ = f
        inb = clippers[side].inside(a, b)
        ank.append(
          f"**{name}** · 크랭크 A `{a:+.1f}°` B `{b:+.1f}°` → "
          f"발판 숙임 `{p:+.1f}°` 기울임 `{r_:+.1f}°` "
          + ("✅ 허용 안" if inb else "🔴 **허용 밖**"))
      ank.append("\n_숙임 허용 −50~+30 · 기울임 허용 ±20 (모형 기준). 기울임 값은 "
                 "우리 기구학이 실제보다 작게 봅니다 — `docs/141` 3절._")
      ankle_md.content = "\n\n".join(ank)
      time.sleep(0.1)

  threading.Thread(target=physics, daemon=True).start()
  threading.Thread(target=refresh, daemon=True).start()

  @server.on_client_connect
  def _(client) -> None:
    client.camera.up_direction = (0.0, 0.0, 1.0)
    client.camera.position = (1.5, 1.2, 1.05)
    client.camera.look_at = (0.0, 0.0, 0.55)

  print(f"\n  화면       http://{host}:{port}/")
  print(f"  텔레메트리 udp {udp_port} 번으로 받는 중")
  print("  ** 이 프로그램에는 모터로 명령을 보내는 길이 없습니다 **")
  print("  Ctrl-C 로 멈춤\n", flush=True)
  try:
    while True:
      time.sleep(1.0)
  except KeyboardInterrupt:
    print("\n멈춥니다")
  finally:
    tel.stop()
  return 0


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=8113, help="3차원 화면")
  ap.add_argument("--udp-port", type=int, default=9871, help="상태를 받을 자리")
  a = ap.parse_args(argv)
  return run(a.host, a.port, a.udp_port)


if __name__ == "__main__":
  raise SystemExit(main())
