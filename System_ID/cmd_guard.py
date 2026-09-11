#!/usr/bin/env python3
"""**모터를 움직일 수 있는 유일한 파일.** 경계를 좁게 잡고, 그 안에서만 움직인다.

2026-09-11 · 사용자가 원격지에 있어 **손을 댈 수 없는 상태**에서 엉덩이 회전
관절 하나를 화면 슬라이더로 움직이려 합니다. 사람이 못 잡는다는 것이 이 파일의
설계를 전부 정합니다.

왜 이 파일 하나에 모아 두는가
-----------------------------
`joint_web.py` 와 `can_probe.py` 는 **힘을 넣는 바이트가 아예 없는** 파일로 두었습니다.
그 성질을 지키려고, 켜기(0xFC)·끄기(0xFD)·세기가 0이 아닌 프레임을 만드는 일을
여기 한 곳에 모읍니다. 무엇이 모터를 움직일 수 있는지 알고 싶으면 이 파일만 보면
됩니다.

보내는 곳도 하나여야 합니다. 관절 화면이 세기 0 프레임을 초당 110번 보내는 중에
다른 프로그램이 세기 10을 보내면 **둘이 번갈아 나가 힘이 반으로 깎이고 떨립니다.**
그래서 이 파일은 스스로 보내지 않고, 읽는 쪽(`joint_web.Poller`)에게 **"이번 주기에
이 모터로는 이 바이트를 보내라"** 고 알려 주기만 합니다.

막아 둔 것 일곱 가지
--------------------
1. **관절 하나만**    허락한 모터 번호 말고는 어떤 이름이 와도 무시합니다.
2. **좁은 구간**      힘을 넣는 순간의 각도를 0으로 잡고 거기서 ±`span_deg` 안에서만
                      움직입니다. 실측 가동범위가 아직 비어 있어(`limits_deg` 가 전부
                      null) 로봇 프로그램의 범위 검사는 통과만 합니다 — 그래서 이
                      구간이 **사실상 유일한 각도 보호**입니다.
3. **느린 속도**      한 주기에 갈 수 있는 거리를 `max_vel_deg_s` 로 묶습니다.
4. **약한 세기**      `kp`/`kd` 에 상한이 있습니다. 세기 10 이면 구간 끝(10도)에서도
                      토크가 1.75 N·m 입니다 (RS03 정격 20 N·m 의 9%).
5. **손 떼면 멈춤**   `deadman_s` 안에 새 명령이 없으면 자세를 붙잡고 힘을 끊습니다.
                      화면이 닫히거나 인터넷이 끊겨도 그렇게 됩니다.
6. **뜨거우면 끊음**  `max_temp_c` 를 넘으면 힘을 끊고, 5도 내려가야 다시 받습니다.
7. **암호**           `arm_token` 이 다르면 그 꾸러미는 버립니다.

그리고 `dry_run=True` 면 **힘을 아예 넣지 않습니다** — 받은 명령을 해석해 무엇을
보낼지 계산하고 기록만 합니다. 길이 제대로 뚫렸는지 확인할 때 씁니다.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time

CAN_FRAME = "=IB3x8s"

# ---- 힘이 들어가는 바이트. 이 파일 바깥에는 없습니다. ----------------------
CMD_ENABLE = 0xFC
CMD_STOP = 0xFD
_CTRL_HEAD = bytes([0xFF] * 6 + [0xFF])          # data[0:6]=FF, data[6]=F_CMD


def control_frame(cmd: int) -> bytes:
  """켜기/끄기 프레임의 **내용 8바이트**. `mit_frame` 과 모양을 맞춰 둡니다 —
  보내는 쪽이 둘을 똑같이 다루면 실수할 자리가 줄어듭니다.

  영점(0xFE) 은 **일부러 만들지 않습니다** — 제어 기준이 옮겨가 수백 도 이동
  지시가 되는 것을 2026-09-11 에 확인했습니다."""
  if cmd not in (CMD_ENABLE, CMD_STOP):
    raise ValueError(f"이 파일은 켜기/끄기만 만듭니다 (받은 값 0x{cmd:02X})")
  return _CTRL_HEAD + bytes([cmd])


def _f2u(x: float, lo: float, hi: float, bits: int) -> int:
  x = max(lo, min(hi, float(x)))
  span = hi - lo
  return int(((x - lo) / span if span > 0 else 0.0) * ((1 << bits) - 1))


def mit_frame(pos_deg: float, kp: float, kd: float, enc: dict) -> bytes:
  """MIT 동작 프레임 8바이트. 목표속도와 앞먹임 토크는 0 으로 둡니다."""
  q = _f2u(math.radians(pos_deg), -enc["pmax"], enc["pmax"], 16)
  dq = _f2u(0.0, -enc["vmax"], enc["vmax"], 12)
  kp_u = _f2u(kp, 0.0, enc["kp_max"], 12)
  kd_u = _f2u(kd, 0.0, enc["kd_max"], 12)
  tau = _f2u(0.0, -enc["tmax"], enc["tmax"], 12)
  return bytes([
    (q >> 8) & 0xFF, q & 0xFF,
    (dq >> 4) & 0xFF, ((dq & 0x0F) << 4) | ((kp_u >> 8) & 0x0F), kp_u & 0xFF,
    (kd_u >> 4) & 0xFF, ((kd_u & 0x0F) << 4) | ((tau >> 8) & 0x0F), tau & 0xFF,
  ])


ENC_RS03 = dict(pmax=12.57, vmax=20.0, tmax=60.0, kp_max=5000.0, kd_max=100.0)

# 허락할 수 있는 관절. **엉덩이 회전 둘뿐입니다.** 다른 관절을 움직이려면 여기에
# 적어야 하고, 그 순간이 사람이 다시 생각해 보는 자리입니다.
ALLOWED = {
  "L_hip_yaw_joint": dict(motor=3, iface="can0", enc=ENC_RS03, ko="왼 엉덩이회전"),
  "R_hip_yaw_joint": dict(motor=9, iface="can1", enc=ENC_RS03, ko="오른 엉덩이회전"),
}


class CommandGuard:
  """받은 명령을 경계 안으로 줄여, 읽는 쪽이 보낼 바이트를 알려 준다."""

  def __init__(self, *, allow, arm_token, port=9872, kp=10.0, kd=1.0,
               kp_cap=20.0, kd_cap=5.0, span_deg=10.0, max_vel_deg_s=20.0,
               deadman_s=0.3, max_temp_c=50.0, dry_run=True):
    bad = [n for n in allow if n not in ALLOWED]
    if bad:
      raise SystemExit(f"허락되지 않은 관절 {bad}. 가능한 것: {sorted(ALLOWED)}")
    self.allow = {ALLOWED[n]["motor"]: dict(ALLOWED[n], sim=n) for n in allow}
    self.arm_token = str(arm_token)
    self.port = int(port)
    self.kp = min(float(kp), float(kp_cap))
    self.kd = min(float(kd), float(kd_cap))
    self.span = abs(float(span_deg))
    self.max_vel = abs(float(max_vel_deg_s))
    self.deadman = float(deadman_s)
    self.max_temp = float(max_temp_c)
    self.dry_run = bool(dry_run)

    self.lock = threading.Lock()
    self.wanted = {}          # 모터 -> 받은 목표(도, 이미 부호/영점 적용)
    self.last_rx = 0.0
    self.q0 = {}              # 모터 -> 힘을 넣은 순간의 각도
    self.cmd = {}             # 모터 -> 지금 실제로 보내는 목표
    self.torque_on = {}       # 모터 -> 힘이 켜져 있다고 우리가 아는가
    self.cut_temp = {}        # 모터 -> 뜨거워서 끊었는가
    self.pending_ctrl = {}    # 모터 -> 다음에 보낼 켜기/끄기 프레임
    self.stats = dict(rx=0, bad_token=0, unknown=0, clipped=0, stale=0)
    self.note = "아직 명령 없음"
    self._stop = threading.Event()

  # ---- 받기 -------------------------------------------------------------
  def start(self):
    threading.Thread(target=self._listen, daemon=True).start()
    return self

  def stop(self):
    self._stop.set()

  def _listen(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", self.port))
    s.settimeout(0.5)
    while not self._stop.is_set():
      try:
        data, _ = s.recvfrom(65535)
      except (TimeoutError, socket.timeout):
        continue
      for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
          self._take(line)
    s.close()

  def _take(self, line: str):
    try:
      m = json.loads(line)
    except ValueError:
      return
    if m.get("type") != "JointTarget":
      return
    with self.lock:
      self.stats["rx"] += 1
      if str(m.get("arm_token", "")) != self.arm_token or not self.arm_token:
        self.stats["bad_token"] += 1
        self.note = "암호가 달라 버림"
        return
      names = m.get("joint_names") or []
      q = m.get("q_target") or []
      took = 0
      for name, val in zip(names, q):
        row = ALLOWED.get(name)
        if row is None or row["motor"] not in self.allow:
          self.stats["unknown"] += 1
          continue
        self.wanted[row["motor"]] = math.degrees(float(val))
        took += 1
      if took:
        self.last_rx = time.monotonic()
        self.note = f"{took}개 관절 목표 받음"

  # ---- 읽는 쪽이 매 주기 물어보는 것 -------------------------------------
  def frame_for(self, motor_id: int, deg_now: float, temp_c: float, dt: float):
    """이 모터로 이번 주기에 보낼 8바이트. `None` 이면 세기 0 프레임을 보내라는 뜻.

    켜기/끄기 프레임이 먼저 나가야 하면 `("ctrl", 바이트)` 를 돌려준다.
    """
    row = self.allow.get(motor_id)
    if row is None:
      return None

    with self.lock:
      pend = self.pending_ctrl.pop(motor_id, None)
      if pend is not None:
        return ("ctrl", pend)

      fresh = (time.monotonic() - self.last_rx) <= self.deadman if self.last_rx else False
      hot = temp_c >= self.max_temp
      if hot:
        self.cut_temp[motor_id] = True
      elif self.cut_temp.get(motor_id) and temp_c <= self.max_temp - 5.0:
        self.cut_temp[motor_id] = False
      blocked = self.cut_temp.get(motor_id, False)

      # -- 힘을 끊어야 하는 경우 --------------------------------------
      if not fresh or blocked or self.dry_run:
        if self.torque_on.get(motor_id):
          self.torque_on[motor_id] = False
          self.q0.pop(motor_id, None)
          self.cmd.pop(motor_id, None)
          self.note = ("뜨거워서 끊음" if blocked else
                       "명령이 끊겨 힘을 뺌" if not fresh else "시험 모드")
          self.pending_ctrl[motor_id] = control_frame(CMD_STOP)
          if not fresh:
            self.stats["stale"] += 1
        if self.dry_run and fresh and not blocked:
          # 보내지는 않지만 **무엇을 보낼지는 계산해 둔다** — 길이 뚫렸는지 보려고.
          self.cmd[motor_id] = self._bounded(motor_id, deg_now, dt, preview=True)
        return None

      # -- 여기부터는 실제로 힘이 들어간다 ----------------------------
      if not self.torque_on.get(motor_id):
        self.torque_on[motor_id] = True
        self.q0[motor_id] = deg_now           # 켜는 순간의 자리를 0 으로
        self.cmd[motor_id] = deg_now
        self.note = f"힘을 넣음 (기준 {deg_now:.2f}도, 구간 ±{self.span:g}도)"
        return ("ctrl", control_frame(CMD_ENABLE))

      target = self._bounded(motor_id, deg_now, dt)
      self.cmd[motor_id] = target
      return mit_frame(target, self.kp, self.kd, row["enc"])

  def _bounded(self, motor_id, deg_now, dt, preview=False):
    """받은 목표를 **구간**과 **속도**로 줄인다."""
    q0 = self.q0.get(motor_id, deg_now)
    want = self.wanted.get(motor_id, q0)
    lo, hi = q0 - self.span, q0 + self.span
    clipped = min(hi, max(lo, want))
    if clipped != want and not preview:
      self.stats["clipped"] += 1
    prev = self.cmd.get(motor_id, deg_now)
    step = self.max_vel * max(0.0, min(dt, 0.05))      # 루프가 멈췄다 오면 한 주기 몫만
    return min(prev + step, max(prev - step, clipped))

  def release(self, motor_id: int):
    """끝낼 때 부르는 것. 다음 주기에 끄기 프레임이 나간다."""
    with self.lock:
      if self.torque_on.get(motor_id):
        self.torque_on[motor_id] = False
        self.pending_ctrl[motor_id] = control_frame(CMD_STOP)

  def status(self):
    with self.lock:
      age = (time.monotonic() - self.last_rx) if self.last_rx else None
      return dict(
        dry_run=self.dry_run, kp=self.kp, kd=self.kd, span_deg=self.span,
        max_vel_deg_s=self.max_vel, deadman_s=self.deadman, max_temp_c=self.max_temp,
        note=self.note, rx_age_s=round(age, 2) if age is not None else None,
        stats=dict(self.stats), port=self.port,
        motors={str(m): dict(ko=r["ko"], sim=r["sim"], iface=r["iface"],
                             torque_on=self.torque_on.get(m, False),
                             q0=self.q0.get(m), cmd=self.cmd.get(m),
                             temp_cut=self.cut_temp.get(m, False))
                for m, r in self.allow.items()},
      )
