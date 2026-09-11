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
6b. **절대 한계**     `limit_by` 로 준 관절은 **모터 각도 기준의 고정 구간** 밖으로
                      못 나갑니다. 위의 "좁은 구간"은 힘을 넣은 자리에서 재는 상대값이라,
                      **한쪽 끝이 하드스톱인 관절에는 그것만으로 부족합니다** — 무릎이
                      그렇습니다(한쪽이 완전히 편 자세이고 그 너머는 없습니다).
                      2026-09-12 에 무릎을 상대 구간만으로 열었다가 하드스톱 쪽으로
                      9.5도까지 밀 수 있는 상태를 만들었습니다. 그 관절은 절대 한계를
                      주지 않으면 **열지 않습니다**.
7. **암호(선택)**     `arm_token` 을 주면 다른 꾸러미는 버립니다. **비워 두면 검사하지
                      않습니다** — 화면에 닿을 수 있는 사람은 누구나 움직일 수 있다는
                      뜻이므로, 바깥에 열지 않은 망 안에서만 그렇게 쓸 것.

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
ENC_RS04 = dict(pmax=12.57, vmax=15.0, tmax=120.0, kp_max=5000.0, kd_max=100.0)

# 허락할 수 있는 관절. **엉덩이 회전 둘뿐입니다.** 다른 관절을 움직이려면 여기에
# 적어야 하고, 그 순간이 사람이 다시 생각해 보는 자리입니다.
ALLOWED = {
  "L_hip_yaw_joint": dict(motor=3, iface="can0", enc=ENC_RS03, ko="왼 엉덩이회전"),
  "R_hip_yaw_joint": dict(motor=9, iface="can1", enc=ENC_RS03, ko="오른 엉덩이회전"),
  # 발목 크랭크. **둘은 하나의 기구다** — 아래 PAIR 와 마스크를 같이 볼 것.
  "L_crank_A_joint": dict(motor=5,  iface="can0", enc=ENC_RS03, ko="왼 발목 A", leg="left"),
  "L_crank_B_joint": dict(motor=6,  iface="can0", enc=ENC_RS03, ko="왼 발목 B", leg="left"),
  "R_crank_A_joint": dict(motor=11, iface="can1", enc=ENC_RS03, ko="오른 발목 A", leg="right"),
  "R_crank_B_joint": dict(motor=12, iface="can1", enc=ENC_RS03, ko="오른 발목 B", leg="right"),
  # 무릎. **중력이 걸리는 유일한 관절이다** — 다른 것들은 세로축이거나 발끝이라
  # 거의 안 걸린다. 매달린 다리에서 무릎 아래 3.80 kg 이 만드는 토크가 0도에서
  # 0.88, 30도에서 6.45, 60도에서 10.29 N·m 다. 약한 세기로는 들어 올리지 못하고
  # 처지므로, 세기와 구간을 관절마다 따로 줄 수 있게 해 두고 여기는 낮게 잡는다.
  "L_knee_joint": dict(motor=4,  iface="can0", enc=ENC_RS04, ko="왼 무릎"),
  "R_knee_joint": dict(motor=10, iface="can1", enc=ENC_RS04, ko="오른 무릎"),
  # 엉덩이 앞뒤·좌우. 2026-09-12 추가 (사용자 지시: "다른 모터들도 제어시 위치
  # 타겟줬으면 홀드하도록"). **여기가 사람이 다시 생각해 보는 자리이므로 적어 둔다.**
  #
  # 이 둘은 **다리 전체를 든다.** 매달린 다리 한 짝이 약 7.6 kg 이고 무게중심이
  # 관절에서 약 0.30 m 떨어져 있으므로, 수평으로 드는 최악에서 22 N·m 가 걸린다
  # (무릎의 10.3 N·m 보다 두 배 이상). 세기 10 으로는 **들어 올리지 못하고
  # 처집니다** -- 그것이 정상이고, 세기를 올려 억지로 들면 안 됩니다. 낮은 세기로
  # 처지는 것을 보고 한계를 아는 것이 이 단계의 목적입니다.
  #
  # 그래서 절대 한계 없이는 열지 않습니다 (`NEEDS_LIMIT`). 처지다가 하드스톱에
  # 부딪히는 것을 막을 방법이 한계뿐이기 때문입니다.
  "L_hip_pitch_joint": dict(motor=1, iface="can0", enc=ENC_RS04, ko="왼 엉덩이앞뒤"),
  "R_hip_pitch_joint": dict(motor=7, iface="can1", enc=ENC_RS04, ko="오른 엉덩이앞뒤"),
  "L_hip_roll_joint":  dict(motor=2, iface="can0", enc=ENC_RS04, ko="왼 엉덩이좌우"),
  "R_hip_roll_joint":  dict(motor=8, iface="can1", enc=ENC_RS04, ko="오른 엉덩이좌우"),
}

HOME_SETTLE_S = 1.5
"""영자세를 명령한 채 이만큼 기다리면, 관절이 못 따라왔어도 끝난 것으로 본다.

약한 세기는 **일부러 고른 것**이다 -- 다리 무게를 억지로 들지 않으려고 그렇게
두었다. 그러니 "못 따라옴" 은 고장이 아니라 예상된 결과이고, 그걸 이유로
슬라이더를 영영 잠가 두면 안 된다. 대신 화면에 **몇 도 처졌는지 숫자로** 적는다."""


PAIR = {5: 6, 6: 5, 11: 12, 12: 11}
"""발목 크랭크의 짝. 한쪽만 움직여도 **다른 쪽이 갈 수 있는 범위가 바뀐다.**

발목은 모터 두 개가 로드 두 개로 발판 하나를 밀고 당기는 닫힌 기구다. 그래서 두
각도를 따로 자르면 안 된다 — 각각은 범위 안이어도 **그 조합이 갈 수 없는 자리**일 수
있고, 실물에서 그것은 기구가 버티는 상태다.

`ankle_pair_mask.json` 은 HUPHY 의 발목 기구학으로 미리 계산한 **갈 수 있는 조합**의
지도다. 순진한 사각형의 12.1% 만 실제로 갈 수 있다.

★ 전제: 지금 실측 영점이 없어서 **모터 각도 = 크랭크 각도**로 본다(sign 1, offset 0).
그것이 어긋나면 이 지도도 어긋나므로, 지도만 믿지 않고 **켠 자리에서의 좁은 구간**을
함께 건다. 실측 영점이 생기면 그때 이 전제를 지운다."""


class CommandGuard:
  """받은 명령을 경계 안으로 줄여, 읽는 쪽이 보낼 바이트를 알려 준다."""

  def __init__(self, *, allow, arm_token, port=9872, kp=10.0, kd=1.0,
               kp_cap=20.0, kd_cap=5.0, span_deg=10.0, max_vel_deg_s=20.0,
               deadman_s=0.3, max_temp_c=50.0, dry_run=True, pair_mask=None,
               kp_by=None, span_by=None, limit_by=None, needs_limit=(),
               home_margin_deg=3.0):
    bad = [n for n in allow if n not in ALLOWED]
    if bad:
      raise SystemExit(f"허락되지 않은 관절 {bad}. 가능한 것: {sorted(ALLOWED)}")
    self.allow = {ALLOWED[n]["motor"]: dict(ALLOWED[n], sim=n) for n in allow}
    self.arm_token = str(arm_token)
    self.port = int(port)
    self.kp = min(float(kp), float(kp_cap))
    self.kd = min(float(kd), float(kd_cap))
    self.span = abs(float(span_deg))
    # 관절마다 다르게 줄 수 있다. 무릎처럼 중력이 걸리는 곳은 낮게 잡는다.
    self.kp_by = {ALLOWED[n]["motor"]: min(float(v), float(kp_cap))
                  for n, v in (kp_by or {}).items() if n in ALLOWED}
    self.span_by = {ALLOWED[n]["motor"]: abs(float(v))
                    for n, v in (span_by or {}).items() if n in ALLOWED}
    # 모터 각도 기준의 **고정** 구간. 상대 구간과 달리 자리가 옮겨 다니지 않는다.
    self.limit_by = {ALLOWED[n]["motor"]: (float(a), float(b))
                     for n, (a, b) in (limit_by or {}).items() if n in ALLOWED}
    for n in needs_limit:
      row = ALLOWED.get(n)
      if row and row["motor"] in self.allow and row["motor"] not in self.limit_by:
        raise SystemExit(
          f"{n} 은 한쪽 끝이 하드스톱인 관절입니다. 절대 한계를 주지 않으면 열 수 "
          f"없습니다 — --limit-map \"{n}=<낮은쪽>:<높은쪽>\" (모터 각도 기준)")
    self.max_vel = abs(float(max_vel_deg_s))
    self.deadman = float(deadman_s)
    self.max_temp = float(max_temp_c)
    self.dry_run = bool(dry_run)

    self.lock = threading.Lock()
    self.wanted = {}          # 모터 -> 받은 목표(도, 이미 부호/영점 적용)
    self.last_rx = {}         # ★ 모터별 마지막 명령 시각
    """왜 모터별인가 (2026-09-11 실측으로 드러난 결함).

    처음에는 시각을 **하나만** 들고 있었다. 그랬더니 9번에 명령이 오는 동안 3번도
    '명령이 살아 있다' 고 판단해 **켜진 채 제자리를 붙잡았다.** 움직이지는 않았지만
    시키지도 않은 관절에 힘이 들어간 것이고, 쉬고 있어야 할 모터가 토크 0.25 N·m 를
    보고한 것으로 드러났다(쉴 때 정상값은 0.01).

    손을 뗀다는 것은 **그 관절에서** 손을 뗀다는 뜻이므로 시각도 관절마다 따로 센다."""
    self.last_rx_any = 0.0    # 화면 표시용(가장 최근 아무 명령)
    self.q0 = {}              # 모터 -> 힘을 넣은 순간의 각도
    self.cmd = {}             # 모터 -> 지금 실제로 보내는 목표
    self.torque_on = {}       # 모터 -> 힘이 켜져 있다고 우리가 아는가
    self.cut_temp = {}        # 모터 -> 뜨거워서 끊었는가
    self.pending_ctrl = {}    # 모터 -> 다음에 보낼 켜기/끄기 프레임
    self.meas = {}            # 모터 -> 가장 최근 측정 각도 (짝을 볼 때 씀)
    self.mask = pair_mask     # 발목 크랭크 조합 지도
    self.margin = abs(float(home_margin_deg))
    self.homing = {}          # ★ 모터 -> 아직 영자세로 가는 중인가
    self._home_since = {}     # 모터 -> 명령이 영자세에 닿은 시각
    """왜 힘을 넣자마자 영자세부터 가는가 (사용자 지시 2026-09-12).

    *"이제 어떤 하드웨어 명령을 보내더라도, 먼저 이 영점 위치로 복귀하고, 그 다음
    관절을 움직일 수 있도록 한다."*

    그전에는 힘을 넣은 **그 자리**를 0 으로 삼았다(`q0 = deg_now`). 그래서 조작할
    수 있는 구간이 켤 때마다 다른 자리에 생겼고, 껐다 켜기를 되풀이하면 구간이
    한쪽으로 걸어갔다 — 엉덩이회전이 0도 근처에서 ±43.4도까지 밀려난 적이 있다.

    이제는 힘을 넣으면 **먼저 영자세로 돌아가고**, 도착할 때까지 화면에서 온 목표를
    받아만 두고 쓰지 않는다. 도착한 뒤에야 슬라이더가 살아나고, 그때 구간의 기준도
    영자세로 고정된다. 로봇 프로그램(HUPHY)의 `via_zero`/`staged` 와 같은 규칙이다.

    되돌아가는 속도는 따로 두지 않는다 — 이미 있는 속도 제한(`max_vel`)이 그대로
    걸린다. 하드스톱이 한쪽 끝인 관절은 `home_margin_deg` 만큼 안쪽까지만 간다."""
    self.stats = dict(rx=0, bad_token=0, unknown=0, clipped=0, stale=0, pair_clipped=0,
                      homed=0)
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
      if self.arm_token and str(m.get("arm_token", "")) != self.arm_token:
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
        now = time.monotonic()
        for name, _ in zip(names, q):
          row = ALLOWED.get(name)
          if row is not None and row["motor"] in self.allow:
            self.last_rx[row["motor"]] = now
        self.last_rx_any = now
        self.note = f"{took}개 관절 목표 받음"

  def local_target(self, name: str, deg: float, token: str) -> dict:
    """화면에서 바로 들어오는 목표. UDP 로 오는 것과 **똑같은 검사**를 거친다.

    화면이 인터넷에 열려 있으므로 암호를 반드시 본다. 암호가 없으면 이 길은 닫힌
    것과 같다.
    """
    if self.arm_token and str(token) != self.arm_token:
      with self.lock:
        self.stats["bad_token"] += 1
      return dict(ok=False, why="암호가 다릅니다")
    row = ALLOWED.get(name)
    if row is None or row["motor"] not in self.allow:
      with self.lock:
        self.stats["unknown"] += 1
      return dict(ok=False, why=f"허락되지 않은 관절입니다: {name}")
    with self.lock:
      self.stats["rx"] += 1
      self.wanted[row["motor"]] = float(deg)
      self.last_rx[row["motor"]] = time.monotonic()
      self.last_rx_any = self.last_rx[row["motor"]]
      if self.homing.get(row["motor"]):
        # 받아는 둔다 -- 도착하면 바로 이 값부터 쓴다.
        self.note = f"{row['ko']} 영자세로 가는 중 (받은 목표 {deg:+.2f}도는 대기)"
        return dict(ok=True, motor=row["motor"], homing=True,
                    why="영자세로 돌아가는 중입니다. 도착하면 듣습니다")
      self.note = f"{row['ko']} 목표 {deg:+.2f}도"
    return dict(ok=True, motor=row["motor"], homing=False)

  # ---- 읽는 쪽이 매 주기 물어보는 것 -------------------------------------
  def frame_for(self, motor_id: int, deg_now: float, temp_c: float, dt: float):
    """이 모터로 이번 주기에 보낼 8바이트. `None` 이면 세기 0 프레임을 보내라는 뜻.

    켜기/끄기 프레임이 먼저 나가야 하면 `("ctrl", 바이트)` 를 돌려준다.
    """
    row = self.allow.get(motor_id)
    if row is None:
      return None

    with self.lock:
      self.meas[motor_id] = deg_now
      pend = self.pending_ctrl.pop(motor_id, None)
      if pend is not None:
        return ("ctrl", pend)

      # ★ 2026-09-12 사용자 지적으로 뜻이 바뀐 자리.
      #
      #   *"영자세로 가게 한 다음 힘이 왜 빠지냐고. 자세 유지하라고."*
      #   *"지금 내가 슬라이더로 넣는건 q target 이라고. 끄면 중력이던 외력이던
      #     작용해서 싱크 박살나잖아."*
      #
      # 슬라이더는 **목표 위치**다. 목표를 받은 관절은 그 자리를 **계속 잡고
      # 있어야** 화면의 값과 실물이 같은 것을 가리킨다. 손을 떼었다고 힘을 빼면
      # 중력이 관절을 끌어내리고 그 순간 둘이 어긋난다.
      #
      # 그래서 명령이 끊긴 것은 이제 **힘을 빼는 이유가 아니다.** 마지막 목표를
      # 그대로 잡는다. 힘이 빠지는 길은 셋뿐:
      #   * 사람이 힘 빼기를 누름 (`release`)
      #   * 권선이 너무 뜨거움 (`cut_temp`)
      #   * 시험 모드 (`dry_run`)
      seen = self.last_rx.get(motor_id)
      fresh = (time.monotonic() - seen) <= self.deadman if seen else False
      holding = seen is not None          # 목표를 한 번이라도 받았는가
      if seen is not None and not fresh:
        self.stats["stale"] += 1          # 세기만 한다. 끊지 않는다
      hot = temp_c >= self.max_temp
      if hot:
        self.cut_temp[motor_id] = True
      elif self.cut_temp.get(motor_id) and temp_c <= self.max_temp - 5.0:
        self.cut_temp[motor_id] = False
      blocked = self.cut_temp.get(motor_id, False)

      # -- 힘을 끊어야 하는 경우 --------------------------------------
      if not holding or blocked or self.dry_run:
        if self.torque_on.get(motor_id):
          self.torque_on[motor_id] = False
          self.q0.pop(motor_id, None)
          self.cmd.pop(motor_id, None)
          self.homing.pop(motor_id, None)
          self.note = ("뜨거워서 끊음" if blocked else
                       "시험 모드" if self.dry_run else "힘을 뺌")
          self.pending_ctrl[motor_id] = control_frame(CMD_STOP)
        if self.dry_run and holding and not blocked:
          # 보내지는 않지만 **무엇을 보낼지는 계산해 둔다** — 길이 뚫렸는지 보려고.
          self.cmd[motor_id] = self._bounded(motor_id, deg_now, dt, preview=True)
        return None

      # -- 여기부터는 실제로 힘이 들어간다 ----------------------------
      if not self.torque_on.get(motor_id):
        self.torque_on[motor_id] = True
        self.homing[motor_id] = True          # ★ 슬라이더보다 영자세 복귀가 먼저
        self.q0[motor_id] = self._home_target(motor_id)
        self.cmd[motor_id] = deg_now          # 명령은 지금 자리에서 이어 나감
        self.note = (f"{row['ko']} 힘을 넣음 — 먼저 영자세"
                     f"({self.q0[motor_id]:+.2f}도)로 돌아갑니다 "
                     f"(지금 {deg_now:+.2f}도)")
        return ("ctrl", control_frame(CMD_ENABLE))

      target = self._bounded(motor_id, deg_now, dt)
      self.cmd[motor_id] = target
      return mit_frame(target, self.kp_by.get(motor_id, self.kp), self.kd, row["enc"])

  def _pair_ok(self, motor_id, value) -> bool:
    """이 값과 짝의 지금 값이 **같이 갈 수 있는 조합인가.**"""
    if self.mask is None or motor_id not in PAIR:
      return True
    row = self.allow.get(motor_id)
    partner = PAIR[motor_id]
    other = self.cmd.get(partner, self.meas.get(partner, 0.0))
    a, b = (value, other) if motor_id in (5, 11) else (other, value)
    g = self.mask
    lim, step = g["lim_deg"], g["step_deg"]
    i, j = int(round((a + lim) / step)), int(round((b + lim) / step))
    rows = g["mask"].get(row.get("leg", "right"), [])
    if not (0 <= i < len(rows) and 0 <= j < len(rows[0])):
      return False
    return rows[i][j] == "1"

  def _inset_limit(self, motor_id):
    """고정 한계에서 **여유만큼 안쪽**으로 물린 구간. 없으면 `None`.

    한계는 하드스톱을 잰 값이라 그대로 명령하면 스톱에 대고 미는 것이 된다.
    HUPHY 의 `command_margin_deg` 와 같은 생각이고, 영자세로 돌아갈 때만이 아니라
    **슬라이더로 오는 목표에도 똑같이** 건다 — 2026-09-12 시험에서, 영자세에
    도착한 뒤에는 슬라이더가 하드스톱 위로 그대로 갈 수 있는 것이 드러났다.
    """
    abs_lim = self.limit_by.get(motor_id)
    if abs_lim is None:
      return None
    lo, hi = abs_lim
    if hi - lo <= 2.0 * self.margin:           # 물릴 폭이 없으면 한가운데로 모은다
      middle = 0.5 * (lo + hi)
      return (middle, middle)
    return (lo + self.margin, hi - self.margin)

  def _slider_span(self, motor_id):
    """화면 슬라이더가 보여 줄 범위. **가드가 실제로 허락하는 것과 같아야 한다.**

    한쪽만 보여 주면 사람이 갈 수 없는 자리를 고르고, 왜 안 가는지 모른다.
    고정 한계가 있으면 여유를 문 그 값이고, 없으면 영자세 ± 상대 구간이다.
    """
    span = self.span_by.get(motor_id, self.span)
    centre = self.q0.get(motor_id, self._home_target(motor_id))
    lo, hi = centre - span, centre + span
    inset = self._inset_limit(motor_id)
    if inset is not None:
      lo, hi = max(lo, inset[0]), min(hi, inset[1])
    return (lo, hi)

  def _home_target(self, motor_id) -> float:
    """이 관절의 **영자세**. 하드스톱이 한쪽 끝이면 그만큼 안쪽으로 물린다.

    무릎처럼 0도가 곧 하드스톱인 관절이 있다. 0 을 그대로 명령하면 스톱에 대고
    미는 것이 되므로 여유만큼 안쪽을 영자세로 삼는다. HUPHY 의
    `command_margin_deg` 와 같은 생각이다.
    """
    inset = self._inset_limit(motor_id)
    if inset is None:
      return 0.0
    lo, hi = inset
    return min(hi, max(lo, 0.0))

  def _bounded(self, motor_id, deg_now, dt, preview=False):
    """받은 목표를 **구간**과 **속도**로 줄이고, 발목이면 **짝까지 보고** 더 줄인다.

    **영자세로 가는 중이면 화면에서 온 목표를 쓰지 않는다** — 받아만 두고, 도착한
    뒤에 쓴다 (`self.homing`).
    """
    q0 = self.q0.get(motor_id, deg_now)
    home = self._home_target(motor_id)
    if self.homing.get(motor_id):
      want = home
    else:
      want = self.wanted.get(motor_id, q0)
    span = self.span_by.get(motor_id, self.span)
    lo, hi = q0 - span, q0 + span
    abs_lim = self._inset_limit(motor_id)
    if abs_lim is not None:          # 고정 한계가 있으면 **그것이 우선**
      lo, hi = max(lo, abs_lim[0]), min(hi, abs_lim[1])
    clipped = min(hi, max(lo, want))
    if clipped != want:
      self.stats["clipped"] += 1
    prev = self.cmd.get(motor_id, deg_now)
    step = self.max_vel * max(0.0, min(dt, 0.05))      # 루프가 멈췄다 오면 한 주기 몫만
    value = min(prev + step, max(prev - step, clipped))

    if motor_id in PAIR and self.mask is not None and not self._pair_ok(motor_id, value):
      # 갈 수 없는 조합이면 **직전 자리 쪽으로 물러난다.** 한 번에 뛰지 않고 반씩.
      lo_v, hi_v = prev, value
      for _ in range(12):
        mid = (lo_v + hi_v) / 2
        if self._pair_ok(motor_id, mid):
          lo_v = mid
        else:
          hi_v = mid
      value = lo_v
      self.stats["pair_clipped"] += 1     # 시험 모드에서도 센다. 안 그러면
                                          # 걸렸는데 안 걸린 것처럼 보인다

    if self.homing.get(motor_id) and not preview:
      # 끝났다고 보는 조건이 둘이다. 하나만 두면 안 된다.
      #
      #  (가) 명령이 영자세에 닿았고 **관절도 따라왔다** -- 제일 깔끔한 경우.
      #  (나) 명령이 영자세에 닿은 채 `HOME_SETTLE_S` 가 지났다 -- 관절이 못
      #       따라온 경우. **이것도 끝난 것으로 본다.**
      #
      # (나) 가 없으면 슬라이더가 영영 안 열린다. 2026-09-12 실제로 그랬다:
      # 엉덩이회전을 세기 10 으로 켜 두었더니 정지마찰을 못 이겨 1.29도에서
      # 멈춰 있었고, "영자세로 돌아가는 중" 이 12초 넘게 안 끝났다. 약한 세기로
      # 처지는 것은 **우리가 일부러 고른 것**이므로, 그걸 고장으로 보고 길을
      # 막으면 안 된다.
      if abs(value - home) <= 0.05:
        t = self._home_since.get(motor_id)
        if t is None:
          self._home_since[motor_id] = time.monotonic()
          t = self._home_since[motor_id]
        close = abs(deg_now - home) <= 1.0
        waited = (time.monotonic() - t) >= HOME_SETTLE_S
        if close or waited:
          self.homing[motor_id] = False
          self._home_since.pop(motor_id, None)
          self.stats["homed"] += 1
          gap = deg_now - home
          self.note = (
            f"{self.allow[motor_id]['ko']} 영자세 도착 — 이제 슬라이더가 듣습니다"
            if close else
            f"{self.allow[motor_id]['ko']} 영자세에서 {gap:+.1f}도 처진 채 멈춤 "
            f"(세기가 무게를 못 이김) — 슬라이더를 엽니다")
      else:
        self._home_since.pop(motor_id, None)
    return value

  def release(self, motor_id: int):
    """끝낼 때 부르는 것. 다음 주기에 끄기 프레임이 나간다.

    **바로 끊는 길이다.** 데드맨(응답이 끊기면 끄는 것)은 사람이 사라졌을 때를 위한
    뒷문이라 넉넉하게 잡아 두는데, 사람이 **직접 멈춤을 눌렀을 때**까지 그만큼
    기다리면 안 된다. 그래서 화면의 멈춤은 이 길로 온다.

    `last_rx` 도 같이 지운다 -- 안 지우면 아직 신선한 명령이 남아 있다고 보고
    다음 주기에 **혼자 다시 켜진다.**
    """
    with self.lock:
      self.last_rx.pop(motor_id, None)
      self.wanted.pop(motor_id, None)
      self.homing.pop(motor_id, None)
      self._home_since.pop(motor_id, None)
      if self.torque_on.get(motor_id):
        self.torque_on[motor_id] = False
        self.q0.pop(motor_id, None)
        self.cmd.pop(motor_id, None)
        self.note = f"{self.allow[motor_id]['ko']} 사람이 멈춤 — 바로 힘을 뺌"
        self.pending_ctrl[motor_id] = control_frame(CMD_STOP)

  def release_by_name(self, name: str, token: str) -> dict:
    """화면에서 이름으로 부르는 멈춤. 암호는 **보지 않는다.**

    멈추는 길에 암호를 걸면, 암호가 틀린 사람은 **켤 수는 없는데 끌 수도 없게**
    된다. 그건 안전장치가 아니다."""
    del token
    row = ALLOWED.get(name)
    if row is None or row["motor"] not in self.allow:
      return dict(ok=False, why=f"허락되지 않은 관절입니다: {name}")
    self.release(row["motor"])
    return dict(ok=True, motor=row["motor"], released=True)

  def status(self):
    with self.lock:
      now = time.monotonic()
      age = (now - self.last_rx_any) if self.last_rx_any else None
      return dict(
        dry_run=self.dry_run, kp=self.kp, kd=self.kd, span_deg=self.span,
        max_vel_deg_s=self.max_vel, deadman_s=self.deadman, max_temp_c=self.max_temp,
        note=self.note, rx_age_s=round(age, 2) if age is not None else None,
        needs_token=bool(self.arm_token),
        stats=dict(self.stats), port=self.port,
        has_pair_mask=self.mask is not None,
        motors={str(m): dict(ko=r["ko"], sim=r["sim"], iface=r["iface"],
                             kp=self.kp_by.get(m, self.kp),
                             span=self.span_by.get(m, self.span),
                             torque_on=self.torque_on.get(m, False),
                             age_s=(round(now - self.last_rx[m], 2)
                                    if self.last_rx.get(m) else None),
                             q0=self.q0.get(m), cmd=self.cmd.get(m),
                             homing=self.homing.get(m, False),
                             home=round(self._home_target(m), 2),
                             lo=round(self._slider_span(m)[0], 2),
                             hi=round(self._slider_span(m)[1], 2),
                             temp_cut=self.cut_temp.get(m, False))
                for m, r in self.allow.items()},
      )
