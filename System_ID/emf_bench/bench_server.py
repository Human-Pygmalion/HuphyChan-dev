#!/usr/bin/env python3
"""Single-motor bench web UI for the back-EMF (regenerative over-voltage) absorber test.

Standard library only (Python 3.10).  One control thread talks to the motor, a
ThreadingHTTPServer serves the page, a Server-Sent-Events stream and a JSON command endpoint.

    bench_server.py --channel can0 --id 4 --model RS04            # real motor
    bench_server.py --fake                                         # simulated motor, no hardware
    bench_server.py --channel can0 --model RS04 --scan             # read-only id scan 1..127, then exit

What is and is NOT built in (the user forbade invented safety features):
  * there IS: an always-available Disable button and Esc key, a Clear-fault button, and a
    disable sent to the motor when this process exits normally (Ctrl+C / SIGTERM).
  * there is NOT: any timeout, auto-off, watchdog that drops torque, gain clamp or target slew
    limit.  Only the motor's own encoding range limits values (Kp 0..5000, Kd 0..100 for
    RS03/RS04, position +-12.57 rad), because the frame cannot carry more.

Enable never jumps: when enabling, the target position is set to the position just measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import emf_frames as fr  # noqa: E402
from bench_motor import BenchMotor, State  # noqa: E402
from fake_motor import FakeMotorTransport  # noqa: E402
from ov_monitor import OvMonitor  # noqa: E402
from rscodec import mit  # noqa: E402

MODE_NAMES = {0: "Reset", 1: "Cali", 2: "Motor"}
CSV_COLUMNS = ["t_s", "unix_s", "pos_deg", "vel_deg_s", "tau_nm", "temp_c", "mode", "fault", "warn",
               "target_deg", "kp", "kd", "enabled", "vbus_v", "iqf_a", "power_w", "fault_word"]
PARAM_CYCLE = (fr.PARAM_VBUS, fr.PARAM_IQF)
LOOP_ACTIONS = ("connect", "disconnect", "enable", "disable", "clear_fault", "fault_query")
SET_KEYS = ("kp", "kd", "target_deg", "sine", "sine_amp_deg", "sine_period_s", "vbus_priority")


class Cmd:
  def __init__(self, action: str, args: dict):
    self.action = action
    self.args = args
    self.done = threading.Event()
    self.result: dict = {"ok": True}


class Controller:
  def __init__(self, args):
    self.args = args
    self.period = 1.0 / args.rate
    self.lock = threading.Lock()
    self.cmdq: "queue.Queue[Cmd]" = queue.Queue()
    self.stop_evt = threading.Event()
    self.t_zero = time.monotonic()
    self.unix_zero = time.time()

    from rscodec import tables
    self.enc = tables.MIT_ENCODING[tables.Model(args.model)]
    self.pmax_deg = math.degrees(self.enc.pmax_rad)

    # settings (written by HTTP threads under lock, read by the loop thread)
    self.kp = float(args.kp)
    self.kd = float(args.kd)
    self.target_rel = 0.0          # degrees, relative to origin_deg
    self.target_event = 0          # +1 whenever the SERVER moves the target (enable snapshot)
    # sine position command (user request 2026-10-15): target = center + amp * sin(phase),
    # phase advances 2*pi per period (one full back-and-forth).  Computed in the loop thread
    # every cycle, so the waveform is as smooth as the control rate.
    self.sine_on = False
    self.sine_amp = 10.0           # degrees, +- around the center
    self.sine_period = 2.0         # seconds per full back-and-forth
    self.sine_center = 0.0         # degrees, relative to origin_deg
    self.sine_phase = 0.0          # radians
    self._sine_t: Optional[float] = None
    # VBUS-priority sampling (user request 2026-10-15): every cycle reads VBUS, iqf is not read.
    # Default alternation reads each at half the loop rate (100 Hz at 200 Hz) and misses short spikes.
    self.vbus_priority = bool(getattr(args, "vbus_priority", False))

    # loop-thread state
    self.motor: Optional[BenchMotor] = None
    self.enabled = False
    self.origin_deg: Optional[float] = None
    self.last_state: Optional[State] = None
    self.last_state_t = 0.0
    self.last_param_t = {p: 0.0 for p in PARAM_CYCLE}
    self.last_param_v: dict[int, Optional[float]] = {p: None for p in PARAM_CYCLE}
    self.pending_param: dict[int, float] = {}
    self.fault_word: Optional[int] = None
    self.prev_fault = False
    self.prev_warn = False
    self._first_state_pending = False     # first state after connect: a fault already latched is not a new event
    self.ov = OvMonitor()
    self.message: Optional[str] = None
    self.target_clamped = False
    self.cycle_n = 0
    self.lost_state = 0
    self.lost_param = {p: 0 for p in PARAM_CYCLE}
    self.send_fail = 0
    self.overruns = 0
    self.win = {"state": deque(), **{p: deque() for p in PARAM_CYCLE}}   # (t, ok) in last 2 s
    self.cycle_times: deque = deque()
    self.rates = dict(loop_hz=0.0, worst_gap_ms=0.0, state_pct=None, vbus_pct=None, iqf_pct=None, vbus_hz=0.0, iqf_hz=0.0)
    self._rates_t = 0.0

    # sample ring for SSE
    self.samples: deque = deque(maxlen=4000)
    self.seq = 0

    # recording
    self.rec_fh = None
    self.rec_writer = None
    self.rec_name: Optional[str] = None
    self.rec_rows = 0
    self.rec_last_flush = 0.0

    self.thread = threading.Thread(target=self._run, name="control", daemon=True)

  # -------------------------------------------------------------------- public (HTTP threads)
  def start(self):
    self.thread.start()

  def shutdown(self):
    self.stop_evt.set()
    self.thread.join(timeout=3.0)

  def submit(self, action: str, args: dict, wait_s: float = 3.0) -> dict:
    c = Cmd(action, args)
    self.cmdq.put(c)
    if not c.done.wait(wait_s):
      return {"ok": False, "error": "control loop did not answer in time"}
    return c.result

  def set_params(self, d: dict) -> dict:
    notes = []
    with self.lock:
      if "kp" in d:
        v = float(d["kp"])
        self.kp = min(max(v, 0.0), self.enc.kp_max)
        if self.kp != v:
          notes.append(f"Kp는 0..{self.enc.kp_max:g} 범위로 잘림")
      if "kd" in d:
        v = float(d["kd"])
        self.kd = min(max(v, 0.0), self.enc.kd_max)
        if self.kd != v:
          notes.append(f"Kd는 0..{self.enc.kd_max:g} 범위로 잘림")
      if "vbus_priority" in d:
        self.vbus_priority = d["vbus_priority"] in ("on", True, 1, "1")
      if "sine_amp_deg" in d:
        self.sine_amp = abs(float(d["sine_amp_deg"]))
      if "sine_period_s" in d:
        v = float(d["sine_period_s"])
        if v > 0:
          self.sine_period = v      # phase-continuous: only the speed of the phase changes
        else:
          notes.append("주기는 0보다 커야 함")
      if "target_deg" in d:
        if self.sine_on:
          self.sine_center = float(d["target_deg"])   # while the sine runs the slider moves its center
        else:
          self.target_rel = float(d["target_deg"])
      if "sine" in d:
        on = d["sine"] in ("on", True, 1, "1")
        if on and not self.sine_on:
          if not self.enabled:
            notes.append("사인파는 토크를 켠 뒤에 시작할 수 있음")
          else:
            self.sine_center = self.target_rel      # starts at sin(0) = 0 -> no jump
            self.sine_phase = 0.0
            self._sine_t = None
            self.sine_on = True
        elif not on and self.sine_on:
          self.sine_on = False                      # target stays where the sine was -> no jump
          self.target_event += 1
    return {"ok": True, "note": "; ".join(notes) or None}

  def ov_reset(self) -> dict:
    """Reset button: zero the counters and VBUS statistics (the CSV file keeps growing)."""
    self.ov.reset(time.monotonic() - self.t_zero)
    return {"ok": True}

  def status(self, events: bool = True) -> dict:
    now = time.monotonic()
    st = self.last_state
    alive = st is not None and now - self.last_state_t < 0.5
    names = fr.decode_fault_names(self.fault_word) if self.fault_word else []

    def ok(p):
      return now - self.last_param_t[p] < 0.5

    with self.lock:
      kp, kd, trel = self.kp, self.kd, self.target_rel
    ov = self.ov.summary(now - self.t_zero)
    extra = {"ov_events": self.ov.latest_events(30)} if events else {}
    return dict(
        ov=ov, vbus_priority=self.vbus_priority, **extra,
        connected=self.motor is not None, alive=alive, enabled=self.enabled,
        fake=self.args.fake, model=self.args.model, motor_id=self.args.id, channel=self.args.channel,
        origin_deg=self.origin_deg, kp=kp, kd=kd, target_rel_deg=trel, target_event=self.target_event,
        sine_on=self.sine_on, sine_amp_deg=self.sine_amp, sine_period_s=self.sine_period,
        sine_center_deg=self.sine_center,
        target_clamped=self.target_clamped,
        kp_max=self.enc.kp_max, kd_max=self.enc.kd_max, pmax_deg=self.pmax_deg,
        vmax_deg_s=math.degrees(self.enc.vmax_rad_s), tmax_nm=self.enc.tmax_nm,
        loop_hz=self.rates["loop_hz"], worst_gap_ms=self.rates["worst_gap_ms"], rate_hz=self.args.rate,
        lost_state=self.lost_state, lost_vbus=self.lost_param[fr.PARAM_VBUS],
        lost_iqf=self.lost_param[fr.PARAM_IQF], send_fail=self.send_fail, overruns=self.overruns,
        state_pct=self.rates["state_pct"], vbus_pct=self.rates["vbus_pct"], iqf_pct=self.rates["iqf_pct"],
        vbus_hz=self.rates["vbus_hz"], iqf_hz=self.rates["iqf_hz"],
        vbus_ok=ok(fr.PARAM_VBUS), iqf_ok=ok(fr.PARAM_IQF),
        param_fw_min=fr.PARAM_FIRMWARE_MIN.get(self.args.model),
        temp_c=st.temp_c if st else None,
        mode=st.mode if st else None, mode_name=MODE_NAMES.get(st.mode) if st else None,
        fault=bool(st.fault) if st else None, warn=bool(st.warn) if st else None,
        pos_deg=st.position_deg if st else None,
        fault_word=self.fault_word, fault_names=names,
        recording=self.rec_fh is not None, rec_file=self.rec_name, rec_rows=self.rec_rows,
        message=self.message, t=now - self.t_zero)

  def samples_since(self, seq: int):
    with self.lock:
      out = []
      for s, row in reversed(self.samples):
        if s <= seq:
          break
        out.append(row)
      out.reverse()
      return out, self.seq

  # -------------------------------------------------------------------- control thread
  def _run(self):
    next_t = time.monotonic()
    try:
      while not self.stop_evt.is_set():
        if self.motor is None:
          try:
            self._handle(self.cmdq.get(timeout=0.05))
          except queue.Empty:
            pass
          next_t = time.monotonic()
          continue
        while True:
          try:
            self._handle(self.cmdq.get_nowait())
          except queue.Empty:
            break
        if self.motor is None:
          continue
        self._cycle()
        next_t += self.period
        delay = next_t - time.monotonic()
        if delay > 0:
          time.sleep(delay)
        else:
          self.overruns += 1
          if delay < -5 * self.period:
            next_t = time.monotonic()
    finally:
      self._close_motor()
      self._stop_recording()
      self.ov.close_csv()

  def _close_motor(self):
    """Process exit path: disable first (Motor.close sends 5 disables), then close."""
    m, self.motor = self.motor, None
    self.enabled = False
    if m is not None:
      try:
        m.close()
      except Exception as exc:  # noqa: BLE001
        print(f"[bench] close failed: {exc}", flush=True)

  def _open_motor(self) -> BenchMotor:
    a = self.args
    if a.fake:
      kw = {}
      if getattr(a, "fake_ov_gain", None) is not None:
        kw["regen_v_per_w"] = a.fake_ov_gain
      tr = FakeMotorTransport(a.model, a.id, answer_params=not a.fake_no_param, **kw)
      return BenchMotor(transport=tr, channel="fake", motor_id=a.id, model=a.model)
    return BenchMotor(channel=a.channel, motor_id=a.id, model=a.model)

  def _handle(self, c: Cmd):
    fn = getattr(self, f"_do_{c.action}", None)
    try:
      if fn is None:
        c.result = {"ok": False, "error": f"unknown action {c.action}"}
      else:
        fn(c)
    except Exception as exc:  # noqa: BLE001
      c.result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    c.done.set()

  def _do_connect(self, c: Cmd):
    if self.motor is not None:
      return
    try:
      self.motor = self._open_motor()
    except OSError as exc:
      self.message = f"CAN 채널 {self.args.channel} 열기 실패: {exc} (README의 CAN 올리기 참고)"
      c.result = {"ok": False, "error": self.message}
      return
    self.origin_deg = None
    self.last_state = None
    self.fault_word = None
    self.prev_fault = False
    self.prev_warn = False
    self._first_state_pending = True
    self.message = None
    self.ov.reset(time.monotonic() - self.t_zero)
    try:
      self.ov.open_csv(self.args.data_dir, time.time())
    except OSError as exc:
      self.message = f"과전압 이벤트 파일을 만들 수 없음: {exc}"
    # zero-gain reads: state only, no torque
    m = self.motor
    for _ in range(30):
      m.send_passive()
      f = m.recv_frame(0.02)
      obj = m.classify(f) if f else None
      if isinstance(obj, State):
        self._on_state(obj)
        break
    if self.origin_deg is None:
      self.message = "모터 응답 없음 (id/모델/CAN 연결 확인). 응답이 오면 자동으로 원점이 잡힙니다."
      c.result = {"ok": True, "warning": self.message}

  def _do_disconnect(self, c: Cmd):
    self._close_motor()
    self.ov.stop(time.monotonic() - self.t_zero)
    self.ov.close_csv()
    self.last_state = None
    self.origin_deg = None

  def _do_enable(self, c: Cmd):
    m = self.motor
    if m is None or self.last_state is None or time.monotonic() - self.last_state_t > 0.5:
      c.result = {"ok": False, "error": "모터 상태를 아직 못 읽어서 켤 수 없음 (현재 위치를 모름)"}
      return
    with self.lock:
      self.target_rel = self.last_state.position_deg - self.origin_deg   # no jump on enable
      self.target_event += 1
    m.enable()
    m.enable()
    self.enabled = True

  def _do_disable(self, c: Cmd):
    self.enabled = False
    with self.lock:
      if self.sine_on:              # torque is off, so the sine has nothing to drive
        self.sine_on = False
        self.target_event += 1
    m = self.motor
    if m is not None:
      for _ in range(3):
        m.disable()

  def _do_clear_fault(self, c: Cmd):
    m = self.motor
    if m is None:
      c.result = {"ok": False, "error": "연결되지 않음"}
      return
    m.clear_fault()
    m.clear_fault()
    self.fault_word = None
    self.enabled = False     # after a fault the user must press Enable again (it re-snapshots the target)
    with self.lock:
      if self.sine_on:       # a running sine would override that snapshot and make Enable jump
        self.sine_on = False
        self.target_event += 1

  def _do_fault_query(self, c: Cmd):
    m = self.motor
    if m is None:
      c.result = {"ok": False, "error": "연결되지 않음"}
      return
    self.fault_word = m.query_fault_word()

  def _do_record(self, c: Cmd):
    if c.args.get("on"):
      self._start_recording()
    else:
      self._stop_recording()

  # -------------------------------------------------------------------- one cycle
  def _target_abs(self) -> float:
    with self.lock:
      if self.sine_on:
        now = time.monotonic()
        if self._sine_t is not None:
          self.sine_phase += 2.0 * math.pi * (now - self._sine_t) / self.sine_period
        self._sine_t = now
        self.target_rel = self.sine_center + self.sine_amp * math.sin(self.sine_phase)
      t = (self.origin_deg or 0.0) + self.target_rel
    lim = self.pmax_deg
    self.target_clamped = abs(t) > lim
    return max(-lim, min(lim, t))

  def _cycle(self):
    m = self.motor
    t0 = time.monotonic()
    with self.lock:
      kp, kd = self.kp, self.kd
    tgt = self._target_abs()
    if self.enabled:
      ok = m.send_frame(m.motor_id, mit.pack_command(
          position_deg=tgt, velocity_deg_s=0.0, kp=kp, kd=kd, torque_nm=0.0, enc=m.enc))
    else:
      ok = m.send_passive()
    idx = fr.PARAM_VBUS if self.vbus_priority else PARAM_CYCLE[self.cycle_n % 2]
    self.cycle_n += 1
    ok2 = m.send_param_read(idx)
    if not (ok and ok2):
      self.send_fail += 1

    need_state, need_param = ok, ok2
    state = None
    deadline = t0 + min(0.8 * self.period, 0.02)
    while need_state or need_param:
      left = deadline - time.monotonic()
      if left <= 0:
        break
      f = m.recv_frame(left)
      if f is None:
        break
      obj = m.classify(f)
      if isinstance(obj, State):
        state = obj
        need_state = False
      elif isinstance(obj, fr.ParamReply) and obj.index in PARAM_CYCLE:
        self.pending_param[obj.index] = obj.value
        self.last_param_t[obj.index] = time.monotonic()
        self.last_param_v[obj.index] = obj.value
        if obj.index == idx:
          need_param = False

    now = time.monotonic()
    if ok:
      self._note("state", state is not None, now)
      if state is None:
        self.lost_state += 1
    if ok2:
      got = idx in self.pending_param
      self._note(idx, got, now)
      if not got:
        self.lost_param[idx] += 1
    self.cycle_times.append(now)
    while self.cycle_times and now - self.cycle_times[0] > 1.0:
      self.cycle_times.popleft()

    if state is not None:
      self._on_state(state, tgt, kp, kd)
    if now - self._rates_t > 0.25:
      self._update_rates(now)

  def _on_state(self, st: State, tgt: Optional[float] = None, kp: float = 0.0, kd: float = 0.0):
    self.last_state = st
    self.last_state_t = st.stamp
    if self.origin_deg is None:
      self.origin_deg = st.position_deg
      with self.lock:
        self.target_rel = 0.0
        self.target_event += 1
    edge = bool(st.fault) and not self.prev_fault and self.motor is not None
    if edge:
      self.fault_word = self.motor.query_fault_word()      # documented fault query, read only
    elif not st.fault:
      self.fault_word = None
    if self._first_state_pending:
      # a fault that was already latched when we connected is not a new event of this session
      self._first_state_pending = False
      self.prev_warn = bool(st.warn)
      if edge:
        edge = False
        self.message = "연결했을 때 이미 고장 상태였음 (이벤트로 세지 않음). 고장 해제를 누르십시오."
    elif st.warn and not self.prev_warn:
      self.ov.on_warn_edge()
    self.prev_warn = bool(st.warn)
    self.prev_fault = bool(st.fault)
    if tgt is None:
      return
    v = self.pending_param.pop(fr.PARAM_VBUS, None)
    i = self.pending_param.pop(fr.PARAM_IQF, None)
    self.pending_param.clear()
    p_w = st.torque_nm * math.radians(st.velocity_deg_s)
    row = dict(t=round(st.stamp - self.t_zero, 5), pos=st.position_deg, vel=st.velocity_deg_s,
               tau=st.torque_nm, temp=st.temp_c, mode=st.mode, fault=int(st.fault), warn=int(st.warn),
               tgt=tgt, kp=kp, kd=kd, en=int(self.enabled), vbus=v, iqf=i, p=p_w)
    with self.lock:
      self.seq += 1
      self.samples.append((self.seq, row))
    self.ov.on_sample(row["t"], v, p_w, st.velocity_deg_s, st.torque_nm)
    if edge:
      self.ov.on_fault_edge(row["t"], self.unix_zero + row["t"], self.fault_word, kp, kd,
                            self.sine_on, self.sine_amp, self.sine_period)
    if self.rec_writer is not None:
      self.rec_writer.writerow([row["t"], round(self.unix_zero + row["t"], 4), row["pos"], row["vel"],
                                row["tau"], row["temp"], row["mode"], row["fault"], row["warn"], row["tgt"],
                                row["kp"], row["kd"], row["en"], "" if v is None else v,
                                "" if i is None else i, p_w,
                                "" if self.fault_word is None else f"0x{self.fault_word:08x}"])
      self.rec_rows += 1
      if st.stamp - self.rec_last_flush > 1.0:
        self.rec_fh.flush()
        self.rec_last_flush = st.stamp

  def _note(self, key, ok: bool, now: float):
    w = self.win[key]
    w.append((now, ok))
    while w and now - w[0][0] > 2.0:
      w.popleft()

  def _update_rates(self, now: float):
    self._rates_t = now
    ct = self.cycle_times
    hz = (len(ct) - 1) / (ct[-1] - ct[0]) if len(ct) > 2 and ct[-1] > ct[0] else 0.0
    gaps = [b - a for a, b in zip(ct, list(ct)[1:], strict=False)]
    worst = 1000.0 * max(gaps) if gaps else 0.0

    def pct(key):
      w = self.win[key]
      return None if not w else 100.0 * sum(1 for _, o in w if o) / len(w)

    def hzr(key):
      w = self.win[key]
      return sum(1 for _, o in w if o) / 2.0

    self.rates = dict(loop_hz=hz, worst_gap_ms=worst, state_pct=pct("state"), vbus_pct=pct(fr.PARAM_VBUS), iqf_pct=pct(fr.PARAM_IQF),
                      vbus_hz=hzr(fr.PARAM_VBUS), iqf_hz=hzr(fr.PARAM_IQF))

  # -------------------------------------------------------------------- recording
  def _start_recording(self):
    if self.rec_fh is not None:
      return
    os.makedirs(self.args.data_dir, exist_ok=True)
    name = datetime.now().strftime("emf_%Y%m%d_%H%M%S.csv")
    path = os.path.join(self.args.data_dir, name)
    self.rec_fh = open(path, "w", newline="")
    self.rec_writer = csv.writer(self.rec_fh)
    self.rec_writer.writerow(CSV_COLUMNS)
    self.rec_name = path
    self.rec_rows = 0

  def _stop_recording(self):
    if self.rec_fh is not None:
      self.rec_fh.flush()
      self.rec_fh.close()
    self.rec_fh = None
    self.rec_writer = None


# ========================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
  ctl: Controller = None  # set in main
  server_version = "emf_bench/1"

  def log_message(self, fmt, *a):   # keep the console quiet
    pass

  def _json(self, obj, code=200):
    body = json.dumps(obj).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    path = self.path.split("?")[0]
    if path in ("/", "/index.html"):
      with open(os.path.join(HERE, "index.html"), "rb") as fh:
        body = fh.read()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)
    elif path == "/status":
      self._json(self.ctl.status())
    elif path == "/stream":
      self._stream()
    else:
      self.send_error(404)

  def _stream(self):
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-store")
    self.send_header("Connection", "close")
    self.end_headers()
    ctl = self.ctl
    last = max(0, ctl.seq - 2000)      # replay up to 10 s of history after a page reload
    ev_rev, ev_sent = None, 0.0        # the event list is sent on change and once a second, not at 30 Hz
    try:
      while not ctl.stop_evt.is_set():
        rows, last = ctl.samples_since(last)
        now = time.monotonic()
        rev = ctl.ov.rev
        send_ev = rev != ev_rev or now - ev_sent > 1.0
        if send_ev:
          ev_rev, ev_sent = rev, now
        payload = json.dumps({"samples": rows, "status": ctl.status(events=send_ev)}, separators=(",", ":"))
        self.wfile.write(b"data: " + payload.encode() + b"\n\n")
        self.wfile.flush()
        time.sleep(1 / 30)
    except (BrokenPipeError, ConnectionResetError, OSError):
      pass

  def do_POST(self):
    if self.path != "/cmd":
      self.send_error(404)
      return
    try:
      n = int(self.headers.get("Content-Length", "0"))
      d = json.loads(self.rfile.read(n) or b"{}")
    except (ValueError, json.JSONDecodeError):
      self._json({"ok": False, "error": "bad JSON"}, 400)
      return
    ctl = self.ctl
    action = d.get("action", "set")
    out: dict = {"ok": True}
    if action in LOOP_ACTIONS:
      out = ctl.submit(action, d)
    elif action not in ("set", "record", "ov_reset"):
      out = {"ok": False, "error": f"unknown action {action}"}
    if action == "ov_reset":
      out = ctl.ov_reset()
    if any(k in d for k in SET_KEYS):
      out["note"] = ctl.set_params(d).get("note")
    if "record" in d:
      r = ctl.submit("record", {"on": d["record"] in ("on", True, 1, "1")})
      out["ok"] = out["ok"] and r.get("ok", True)
    out["status"] = ctl.status()
    self._json(out)


class Server(ThreadingHTTPServer):
  daemon_threads = True
  allow_reuse_address = True


# ========================================================================== scan (read-only)
def scan(args) -> int:
  """Zero-gain MIT frames to every id 1..127 (state only, no torque), then one Command 14
  VBUS read to each responder.  No enable/disable/mode/protocol frames are sent, and the
  socket is closed WITHOUT Motor.close() (which would send disable frames)."""
  if args.fake:
    tr = FakeMotorTransport(args.model, args.id, answer_params=not args.fake_no_param)
    m = BenchMotor(transport=tr, channel="fake", motor_id=1, model=args.model)
  else:
    try:
      m = BenchMotor(channel=args.channel, motor_id=1, model=args.model)
    except OSError as exc:
      print(f"CAN 채널 {args.channel} 을 열 수 없습니다: {exc}\nREADME.md 의 'CAN 올리기'를 먼저 하세요.")
      return 2
  print(f"scan {args.channel if not args.fake else 'FAKE'}  ids 1..127  (zero-gain MIT frames + Command 14 VBUS read; "
        f"decoded as {args.model})")
  found = []
  try:
    for mid in range(1, 128):
      m.motor_id = mid
      m.drain()
      m.send_passive()
      deadline = time.monotonic() + 0.03
      st = None
      while time.monotonic() < deadline:
        f = m.recv_frame(deadline - time.monotonic())
        if f is None:
          break
        obj = m.classify(f)
        if isinstance(obj, State):
          st = obj
          break
      if st is None:
        continue
      m.drain()
      m.send_param_read(fr.PARAM_VBUS)
      vb = None
      deadline = time.monotonic() + 0.03
      while time.monotonic() < deadline:
        f = m.recv_frame(deadline - time.monotonic())
        if f is None:
          break
        obj = m.classify(f)
        if isinstance(obj, fr.ParamReply) and obj.index == fr.PARAM_VBUS:
          vb = obj.value
          break
      found.append(mid)
      print(f"  id {mid:3d}: mode={MODE_NAMES.get(st.mode, st.mode)} pos={st.position_deg:8.2f} deg "
            f"temp={st.temp_c:5.1f} C fault={int(st.fault)} warn={int(st.warn)}  "
            + (f"VBUS={vb:.2f} V (Command 14 works)" if vb is not None
               else "Command 14: no reply (firmware too old? need RS04>=0.4.1.29 / RS03>=0.3.1.41)"))
  finally:
    m._s.close()
  print(f"found {len(found)} motor(s): {found}" if found else "no motor answered on ids 1..127")
  return 0 if found else 1


# ========================================================================== main
def lan_addresses() -> list[str]:
  try:
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
    return [a for a in out if "." in a]
  except Exception:  # noqa: BLE001
    return []


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--channel", default="can0")
  ap.add_argument("--id", type=int, default=4, help="motor CAN id (use --scan if unknown)")
  ap.add_argument("--model", default="RS04", choices=["RS03", "RS04"])
  ap.add_argument("--host", default="0.0.0.0")
  ap.add_argument("--port", type=int, default=8099, help="8095-8098 and 8081 are in use on the 3090 machine")
  ap.add_argument("--rate", type=float, default=200.0, help="control loop rate [Hz]")
  ap.add_argument("--kp", type=float, default=10.0, help="initial Kp shown in the UI [N m/rad]")
  ap.add_argument("--kd", type=float, default=1.0, help="initial Kd shown in the UI [N m s/rad]")
  ap.add_argument("--data-dir", default=os.path.expanduser("~/System_ID/data"))
  ap.add_argument("--vbus-priority", action="store_true",
                  help="read VBUS every cycle, iqf not at all (default: alternate, each at half the loop rate); "
                       "the page checkbox changes it at run time")
  ap.add_argument("--fake", action="store_true", help="simulated motor, no hardware")
  ap.add_argument("--fake-ov-gain", type=float, default=None,
                  help="fake motor: volts of bus rise per watt of regenerated power (default 0.08); raise it so a "
                       "fast sine trips the 60 V over-voltage fault")
  ap.add_argument("--fake-no-param", action="store_true", help="fake motor ignores Command 14 (old firmware)")
  ap.add_argument("--scan", action="store_true", help="read-only id scan 1..127, then exit")
  args = ap.parse_args(argv)

  if args.scan:
    return scan(args)

  ctl = Controller(args)
  Handler.ctl = ctl
  try:
    srv = Server((args.host, args.port), Handler)
  except OSError as exc:
    print(f"cannot bind {args.host}:{args.port}: {exc}  (try --port)")
    return 2
  ctl.start()

  def _term(_s, _f):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, _term)
  signal.signal(signal.SIGINT, _term)      # also when started in the background (shell sets SIGINT to ignore)
  tag = "FAKE" if args.fake else args.channel
  print(f"emf_bench  {args.model} id={args.id} {tag}  loop {args.rate:g} Hz  "
        f"{'VBUS every cycle' if args.vbus_priority else 'VBUS/iqf alternating'}  data -> {args.data_dir}")
  print(f"  http://127.0.0.1:{args.port}/")
  for a in lan_addresses():
    print(f"  http://{a}:{args.port}/")
  print("  Ctrl+C = disable the motor, then exit", flush=True)
  try:
    srv.serve_forever(poll_interval=0.2)
  except KeyboardInterrupt:
    print("\nstopping: sending disable ...", flush=True)
  finally:
    srv.server_close()
    ctl.shutdown()
  return 0


if __name__ == "__main__":
  sys.exit(main())
