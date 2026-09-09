#!/usr/bin/env python3
"""모터를 움직이고, 그 순간을 카메라로 같이 남긴다. **로봇 위에서 돌린다.**

2026-09-09. 두 가지 움직임을 낸다. 성격이 정반대라 쓰임도 다르다.

``pd``
  위치 명령으로 정해진 각도까지 갔다가 돌아온다. **끝이 있는 움직임**이라 폭주할 수 없다.
  파라미터를 재는 데는 못 쓰지만(내부 PD 가 물리를 가린다), 다음 세 가지를 확인하는 데는
  이것 말고 안전한 방법이 없다 — 어느 모터가 움직이는가, 몸통이 책상에 고정돼 있는가,
  카메라가 그 움직임을 보는가.

``sweep``
  다중사인 **순수 토크**(kp=0, kd=0). 사전연구가 쓴 방법이고 파라미터를 재는 것은 이쪽이다.
  위치 되먹임이 없어 모터는 자기가 어디 있는지 신경쓰지 않는다 — 그래서 아래 경계가 있다.

경계 (측정 도구의 일부다)
-------------------------
* 시작 위치에서 ``--bound-deg`` 벗어나면 즉시 멈추고 위치 유지로 내려온다
* 온도 ``--temp-stop`` 도달 시 중단
* ``--seconds`` 초과 시 중단
* 모터가 응답을 멈추면 중단 — 안 보이는 채로 토크를 넣지 않는다
* 어떤 경로로 끝나든 **위치 유지 → 토크 0 → 토크 끄기** 순으로 내려온다

카메라
------
사진은 로봇이 **자기 자신에게서** 가져온다(``127.0.0.1:8099``). 밖에서 가져오면 터널을
건너느라 언제 찍힌 장인지 흐려지는데, 여기서 알고 싶은 것이 바로 "그 순간"이다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
  # 라즈베리파이: 파이썬 기본 기능만 쓰는 구동부. 아무것도 설치할 필요가 없다.
  from rs_direct_socket import Motor  # noqa: E402
except ImportError:                    # 로봇 가상머신: python-can 이 이미 깔려 있다
  from rs_direct import Motor  # noqa: E402

HARM = ((1.0, 1.0), (3.4, 0.6), (7.4, 0.3))
"""사전연구의 다중사인 그대로. 배수가 정수가 아닌 이유는 주기가 겹쳐 성분이 서로 지워지지
않게 하려는 것이다."""

CAM_URL = "http://127.0.0.1:8099/snapshot.jpg"


def snap(tag: str, outdir: str) -> str | None:
  try:
    data = urllib.request.urlopen(CAM_URL, timeout=2.0).read()
  except Exception:
    return None
  path = f"{outdir}/{tag}.jpg"
  with open(path, "wb") as f:
    f.write(data)
  return path


def multisine(t: float, amp: float, f0: float) -> float:
  return amp * sum(a * math.sin(2.0 * math.pi * (m * f0) * t) for m, a in HARM)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("mode", choices=["pd", "sweep"])
  ap.add_argument("--motor-id", type=int, default=4)
  ap.add_argument("--model", default="RS04")
  ap.add_argument("--channel", default="can0")
  # pd
  ap.add_argument("--move-deg", type=float, default=15.0)
  ap.add_argument("--kp", type=float, default=8.0)
  ap.add_argument("--kd", type=float, default=0.6)
  # sweep
  ap.add_argument("--amp", type=float, default=0.3, help="토크 진폭 Nm")
  ap.add_argument("--base-hz", type=float, default=4.0)
  # 공통
  ap.add_argument("--seconds", type=float, default=4.0)
  ap.add_argument("--rate-hz", type=float, default=1000.0)
  ap.add_argument("--bound-deg", type=float, default=25.0)
  ap.add_argument("--temp-stop", type=float, default=45.0)
  ap.add_argument("--outdir", default="/home/syaro/System_ID/data")
  ap.add_argument("--tag", default=None)
  a = ap.parse_args(argv)

  tag = a.tag or f"{a.mode}_{time.strftime('%H%M%S')}"
  rows, stop = [], None
  shots = {}

  with Motor(channel=a.channel, motor_id=a.motor_id, model=a.model) as m:
    st = m.read()
    if st is None:
      print("모터가 응답하지 않습니다", file=sys.stderr)
      return 1
    q0 = st.position_deg
    print(f"모터 id {a.motor_id} ({a.model}) · 시작 {q0:.3f}도 · 온도 {st.temp_c:.1f}도 · "
          f"모드 {st.mode} · 고장 {st.fault}")
    shots["before"] = snap(f"{tag}_before", a.outdir)

    if st.fault:
      m.clear_fault()
      time.sleep(0.05)
    m.enable()
    time.sleep(0.05)

    dt = 1.0 / a.rate_hz
    t_start = time.monotonic()
    next_t = t_start
    mid_shot_at = a.seconds * 0.5
    took_mid = False
    try:
      while True:
        t = time.monotonic() - t_start
        if t >= a.seconds:
          stop = "정해진 시간 끝"
          break
        if a.mode == "pd":
          # 0 -> +move -> 0 -> -move -> 0 한 바퀴. 목표가 있으므로 폭주가 불가능하다.
          phase = t / a.seconds
          target = q0 + a.move_deg * math.sin(2.0 * math.pi * phase)
          m.send(position_deg=target, kp=a.kp, kd=a.kd)
          cmd = target
        else:
          cmd = multisine(t, a.amp, a.base_hz)
          m.torque(cmd)
        st = m.recv(timeout_s=0.005)
        if st is None:
          stop = "모터가 응답하지 않음"
          break
        if abs(st.position_deg - q0) > a.bound_deg:
          stop = f"위치가 시작에서 {st.position_deg - q0:+.1f}도 벗어남 (한계 ±{a.bound_deg})"
          break
        if st.temp_c >= a.temp_stop:
          stop = f"온도 {st.temp_c:.1f}도 도달"
          break
        if st.fault:
          stop = "모터가 고장을 보고함"
          break
        rows.append((t, cmd, st.position_deg, st.velocity_deg_s, st.torque_nm, st.temp_c))
        if not took_mid and t >= mid_shot_at:
          shots["during"] = snap(f"{tag}_during", a.outdir)
          took_mid = True
        next_t += dt
        s = next_t - time.monotonic()
        if s > 0:
          time.sleep(s)
        else:
          next_t = time.monotonic()   # 뒤처지면 몰아 보내지 않는다
    except KeyboardInterrupt:
      stop = "사람이 중단"
    finally:
      try:
        last = m.recv(timeout_s=0.005)
        hold = last.position_deg if last else q0
        t_end = time.monotonic() + 0.7
        while time.monotonic() < t_end:
          m.hold(hold, kp=8.0, kd=0.6)
          m.recv(timeout_s=0.002)
        for _ in range(10):
          m.torque(0.0)
          m.recv(timeout_s=0.002)
      finally:
        pass  # Motor.close() 가 disable 을 부른다
    shots["after"] = snap(f"{tag}_after", a.outdir)

  print(f"\n끝난 이유: {stop}")
  if not rows:
    print("기록된 줄이 없습니다", file=sys.stderr)
    return 1
  qs = [r[2] for r in rows]
  vs = [abs(r[3]) for r in rows]
  achieved = len(rows) / rows[-1][0] if rows[-1][0] > 0 else 0
  print(f"줄 {len(rows)}개 · 실제 초당 {achieved:.0f}회")
  print(f"위치 {min(qs):.2f} ~ {max(qs):.2f}도 (폭 {max(qs)-min(qs):.2f}도) · "
        f"끝 {qs[-1]-q0:+.2f}도 표류")
  print(f"최대 속도 {max(vs):.1f}도/s · 온도 {rows[0][5]:.1f} → {rows[-1][5]:.1f}도")
  print("사진: " + ", ".join(f"{k}={v}" for k, v in shots.items() if v))

  out = f"{a.outdir}/{tag}.csv"
  with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_s", "cmd", "pos_deg", "vel_deg_s", "tau_est_nm", "temp_c"])
    w.writerows(rows)
  print(f"기록: {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
