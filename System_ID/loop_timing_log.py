#!/usr/bin/env python3
"""제어 주기가 왜 안 나오는지 재고, **매 주기를 파일에 남긴다.**

2026-09-12. 양다리 12관절을 영자세로 붙잡았더니 초당 82.8번밖에 못 돌았습니다
(목표 100, 사용자 요구 200). 밀림 548회, 최악 23.8밀리초, 그리고 전체 주기의
97.2퍼센트에서 모터 하나 이상이 그 주기 안에 답을 못 했습니다.

**원인을 추측하지 않고 재기 위한 도구입니다.** 평균만 보면 원인이 안 보입니다 --
한 주기 안에서 시간이 어디로 갔는지, 어느 모터가 늦는지, 늦는 것이 늘 같은
모터인지를 봐야 합니다.

토크가 절대 나가지 않습니다
---------------------------
보내는 것은 **세기 0 · 감쇠 0 · 토크 0** 프레임 하나뿐입니다
(`7F FF 7F F0 00 00 07 FF`). `can_status.py` 가 쓰는 것과 같은 프레임이고, 이
파일에는 켜기(0xFC)·끄기(0xFD)·영점(0xFE)이 **아예 들어 있지 않습니다.**

★ 주의: 어떤 모터가 **이미 힘이 켜진 채 자세를 붙잡고 있었다면** 세기 0 프레임이
그 힘을 놓게 합니다. 축을 돌리는 것이 아니라 붙잡던 것을 놓는 것입니다. 돌리기
전에 `can_status.py` 로 동작모드가 0(힘 꺼짐)인지 확인하십시오.

무엇을 재는가
-------------
주기마다 이만큼을 남깁니다. 시각은 전부 그 주기 시작으로부터의 **마이크로초**입니다.

    cycle          몇 번째 주기
    t_start_us     기준 시각(프로그램 시작)으로부터
    late_us        가려던 격자 시각보다 얼마나 늦게 시작했는가
    send_us        12개 명령을 socket 에 다 써 넣는 데 걸린 시간
    drain_us       응답을 기다린 시간 (창을 다 쓰면 창 크기와 같음)
    cycle_us       그 주기 전체
    got            받은 응답 개수
    rtt_<번호>_us  그 모터의 명령->응답 왕복 시간. 못 받았으면 빈칸

왜 이 칸들인가: 주기를 넘긴 시간이 **보내는 데** 갔는지 **기다리는 데** 갔는지가
갈라져야 합니다. 보내는 데 갔다면 통신선(slcan 은 글자로 바꿔 시리얼로 보냅니다)
이고, 기다리는 데 갔다면 모터나 창 크기입니다.

어떻게 가르는가 (권하는 순서)
-----------------------------
    # 1. 한 다리만. 짐이 절반이면 나오는가?
    loop_timing_log.py --channel can0 --ids 1 2 3 4 5 6 --hz 200 --seconds 20

    # 2. 양다리. 같은 조건에서 얼마나 나빠지는가?
    loop_timing_log.py --channel can0 --ids 1 2 3 4 5 6 \
                       --channel can1 --ids 7 8 9 10 11 12 --hz 200 --seconds 20

    # 3. 모터 수를 줄여 가며. 어디서 꺾이는가?
    loop_timing_log.py --channel can0 --ids 1 2 3 --hz 200 --seconds 20

    # 4. 기다리는 창을 바꿔 가며. 창이 범인인가?
    loop_timing_log.py ... --drain-ms 1 / 2 / 4

결과는 `--out` 에 준 이름으로 csv 두 개가 됩니다 -- 주기별 한 줄짜리와, 맨 끝에
붙는 요약. 요약만 보고 판단하지 말고 주기별 파일을 그려 볼 것.
"""

from __future__ import annotations

import argparse
import csv
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

CAN_FRAME = "=IB3x8s"

ZERO_FRAME = bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF])
"""세기·감쇠·토크가 전부 0. **이 파일이 만드는 유일한 내용입니다.**"""


class Bus:
  """한 채널. 보내는 것과 꺼내는 것을 나눠 둡니다 -- 두 다리가 **같이** 나가야
  하기 때문입니다. 한 다리를 보내고 그 답을 기다린 뒤 다음 다리를 보내면, 기다리는
  시간이 두 번 붙어 주기가 두 배로 듭니다."""

  def __init__(self, channel: str, ids):
    self.channel = channel
    self.ids = list(ids)
    self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    self.sock.bind((channel,))
    self.sock.setblocking(False)
    self.tx_fail = 0

  def flush(self) -> int:
    """주기 시작 전에 남은 것을 버립니다. 몇 개 버렸는지 돌려줍니다 -- 이 수가
    크면 지난 주기의 답이 늦게 온 것이라, 그 자체가 단서입니다."""
    n = 0
    while True:
      try:
        self.sock.recv(16)
        n += 1
      except (BlockingIOError, TimeoutError, socket.timeout):
        return n

  def send_all(self, sent_at: dict, now: float) -> None:
    for motor_id in self.ids:
      try:
        self.sock.send(struct.pack(CAN_FRAME, motor_id, 8, ZERO_FRAME))
        sent_at[motor_id] = time.perf_counter()
      except OSError:
        self.tx_fail += 1
    del now

  def take(self, rtt: dict, sent_at: dict) -> int:
    """지금 큐에 있는 것만 꺼냅니다. 기다리지 않습니다."""
    got = 0
    while True:
      try:
        raw = self.sock.recv(16)
      except (BlockingIOError, TimeoutError, socket.timeout):
        return got
      now = time.perf_counter()
      _, dlc, data = struct.unpack(CAN_FRAME, raw)
      if dlc < 8:
        continue
      motor_id = data[0]
      if motor_id in sent_at and motor_id not in rtt:
        rtt[motor_id] = (now - sent_at[motor_id]) * 1e6
        got += 1


def run(buses, *, hz: float, seconds: float, drain_s: float, out: Path,
        like_huphy: bool = False):
  period = 1.0 / hz
  every = [m for b in buses for m in b.ids]
  header = (["cycle", "t_start_us", "late_us", "send_us", "drain_us", "cycle_us",
             "got", "stale_dropped"] + [f"rtt_{m}_us" for m in every])

  t0 = time.perf_counter()
  rows, n = [], 0
  deadline = t0
  end = t0 + seconds

  while True:
    now = time.perf_counter()
    if now >= end:
      break
    # 격자에 맞춰 잔다. 이미 늦었으면 자지 않고 바로 간다.
    if now < deadline:
      time.sleep(deadline - now)
    start = time.perf_counter()
    late_us = (start - deadline) * 1e6

    # ★ `--like-huphy` 면 **버리지 않습니다.** HUPHY 는 큐에 남아 있던 답도
    # 이번 주기의 답으로 셉니다 (프레임에 일련번호가 없어 어느 명령에 대한
    # 답인지 모릅니다). 그래서 "답이 한 주기 늦는" 모터는 HUPHY 눈에는 **매 주기
    # 답하는** 것으로 보입니다. 두절 판정 문턱을 고르려면 그쪽 셈이 필요합니다.
    stale = 0 if like_huphy else sum(b.flush() for b in buses)

    sent_at, rtt = {}, {}
    for b in buses:
      b.send_all(sent_at, start)
    after_send = time.perf_counter()

    # 창이 다 차거나 전부 받을 때까지 꺼낸다.
    drain_end = after_send + drain_s
    while True:
      for b in buses:
        b.take(rtt, sent_at)
      if len(rtt) >= len(every) or time.perf_counter() >= drain_end:
        break
    done = time.perf_counter()

    rows.append([n, round((start - t0) * 1e6, 1), round(late_us, 1),
                 round((after_send - start) * 1e6, 1),
                 round((done - after_send) * 1e6, 1),
                 round((done - start) * 1e6, 1),
                 len(rtt), stale]
                + [round(rtt[m], 1) if m in rtt else "" for m in every])
    n += 1
    deadline += period

  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
  return rows, header, every


def summarise(rows, every, *, hz, drain_s, seconds, buses):
  def col(i):
    return [r[i] for r in rows]

  def pct(values, p):
    values = sorted(values)
    if not values:
      return float("nan")
    k = min(len(values) - 1, int(round(p / 100.0 * (len(values) - 1))))
    return values[k]

  n = len(rows)
  span_s = (rows[-1][1] - rows[0][1]) / 1e6 if n > 1 else 0.0
  actual_hz = (n - 1) / span_s if span_s > 0 else float("nan")
  late = col(2)
  cyc = col(5)
  got = col(6)
  full = sum(1 for g in got if g == len(every))

  lines = []
  add = lines.append
  add(f"\n  보낸 주기      {n}회 {seconds:.0f}초 — 실제 초당 {actual_hz:.1f}회 "
      f"(목표 {hz:.0f})")
  add(f"  기다린 창      {drain_s*1000:.1f}밀리초")
  add(f"  모터           {len(every)}개, 채널 {', '.join(b.channel for b in buses)}")
  add("")
  add(f"  한 주기        중앙 {statistics.median(cyc)/1000:.2f}ms · "
      f"95% {pct(cyc,95)/1000:.2f}ms · 최악 {max(cyc)/1000:.2f}ms")
  add(f"  격자 대비 늦음 중앙 {statistics.median(late)/1000:.2f}ms · "
      f"최악 {max(late)/1000:.2f}ms")
  add(f"  전부 받은 주기 {full}/{n} ({100.0*full/n:.1f}%)")
  add(f"  보내기 실패    {sum(b.tx_fail for b in buses)}회")
  add("")
  add("  모터별 왕복 시간 (밀리초) 과 빠진 비율")
  add(f"    {'번호':>4} {'중앙':>8} {'95%':>8} {'최악':>8} {'빠짐':>8}")
  base = 8
  for k, motor_id in enumerate(every):
    vals = [r[base + k] for r in rows if r[base + k] != ""]
    miss = 100.0 * (n - len(vals)) / n
    if vals:
      add(f"    {motor_id:>4} {statistics.median(vals)/1000:8.2f} "
          f"{pct(vals,95)/1000:8.2f} {max(vals)/1000:8.2f} {miss:7.1f}%")
    else:
      add(f"    {motor_id:>4} {'--':>8} {'--':>8} {'--':>8} {miss:7.1f}%")
  add("")
  add("  읽는 법: 왕복 시간이 창보다 길면 **창이 짧은 것**이고, 창 안에 드는데도")
  add("  빠진다면 **모터가 안 답한 것**입니다. 보내기가 오래 걸리면 통신선입니다.")
  return "\n".join(lines)


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", action="append", required=True,
                  help="채널. --ids 와 짝지어 되풀이해 줍니다")
  ap.add_argument("--ids", action="append", required=True,
                  help="그 채널의 모터 번호를 쉼표나 공백으로. 실수로 전체가 "
                       "나가지 않게 반드시 적어야 합니다")
  ap.add_argument("--hz", type=float, default=200.0)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--drain-ms", type=float, default=2.0,
                  help="응답을 기다리는 창. HUPHY 기본이 2.0 입니다")
  ap.add_argument("--like-huphy", action="store_true",
                  help="HUPHY 와 **같은 셈**으로 답을 셉니다 — 주기 시작에 큐를 "
                       "비우지 않아, 늦게 온 답도 이번 주기의 답으로 칩니다. "
                       "두절 판정 문턱을 고를 때 씁니다")
  ap.add_argument("--out", default="", help="주기별 csv 를 쓸 자리")
  ap.add_argument("--tag", default="", help="파일 이름에 붙일 말")
  a = ap.parse_args(argv)

  if len(a.channel) != len(a.ids):
    print("--channel 과 --ids 개수가 달라요. 짝지어 주세요.", file=sys.stderr)
    return 2

  buses = []
  for channel, ids_text in zip(a.channel, a.ids):
    ids = [int(x) for x in ids_text.replace(",", " ").split()]
    buses.append(Bus(channel, ids))

  stamp = time.strftime("%Y%m%d_%H%M%S")
  tag = f"_{a.tag}" if a.tag else ""
  out = Path(a.out or f"data/loop_timing/{stamp}{tag}.csv")

  print(f"\n세기 0 · 감쇠 0 · 토크 0 프레임만 보냅니다 ({ZERO_FRAME.hex(' ')}).")
  print("이 프레임으로는 축이 돌지 않습니다. 다만 **이미 힘이 켜져 붙잡고 있던**")
  print("관절은 놓습니다 — 먼저 can_status.py 로 동작모드가 0인지 확인하세요.\n")
  for b in buses:
    print(f"  {b.channel}: 모터 {b.ids}")
  print(f"  초당 {a.hz:.0f}번 · {a.seconds:.0f}초 · 창 {a.drain_ms:.1f}ms")
  print(f"  주기별 기록 -> {out}\n")

  rows, header, every = run(buses, hz=a.hz, seconds=a.seconds,
                            drain_s=a.drain_ms / 1000.0, out=out,
                            like_huphy=a.like_huphy)
  if not rows:
    print("한 주기도 못 돌았습니다.", file=sys.stderr)
    return 1

  text = summarise(rows, every, hz=a.hz, drain_s=a.drain_ms / 1000.0,
                   seconds=a.seconds, buses=buses)
  print(text)
  out.with_suffix(".txt").write_text(text, encoding="utf-8")
  print(f"\n  요약 -> {out.with_suffix('.txt')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
