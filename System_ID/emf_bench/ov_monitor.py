"""Over-voltage monitor for the back-EMF bench: counts motor faults by class, keeps VBUS
statistics, records one row per fault event.  Pure bookkeeping: no CAN, no motor commands, and
deliberately NO reaction to what it sees (no auto-clear, no auto-enable, no auto-off: the user
forbade unrequested protective automation).

Facts it is built on (RobStride RS04 manual, section 3.3.7 and the fault-query frame):
  * over-voltage fault when the bus voltage exceeds 60 V   -> fault word bit 3
  * under-voltage fault below 12 V                          -> fault word bit 2
  * the state frame's fault flag (Byte6 bit5) rises first; the fault word comes from the
    documented fault query that the controller sends on that rising edge.
  * the state frame's warning flag (Byte6 bit4) is a GENERIC "motor warning" (the manual defines
    only bit0 over-temperature in the warning word), so warnings are counted on their own and are
    never labelled over-voltage.

All times are the controller's loop time `t` (seconds, same base as the plotted samples) except
`unix`, which is wall-clock seconds.
"""

from __future__ import annotations

import csv
import os
import threading
from collections import deque
from datetime import datetime
from typing import Optional

from rscodec import tables

import emf_frames as fr

OV_LIMIT_V = 60.0             # manual: over-voltage protection threshold
UV_LIMIT_V = 12.0             # manual: under-voltage protection threshold
THRESHOLDS_V = (52.0, 55.0, 58.0)
WINDOW_S = 1.0                # "the 1 s before the event"
RATE_WINDOW_S = 300.0         # "events per minute over the last 5 min"
KEEP_EVENTS = 200             # events kept in memory (the CSV keeps all of them)
MAX_DT_S = 0.1                # a gap between two VBUS samples longer than this counts as 0.1 s

BIT_OV = tables.FAULT_BITS["overvoltage"]     # 3
BIT_UV = tables.FAULT_BITS["undervoltage"]    # 2

CLASS_OV, CLASS_UV, CLASS_OTHER, CLASS_UNREAD = "overvoltage", "undervoltage", "other", "unread"

EVENT_COLUMNS = ["wall_time", "unix_s", "t_s", "since_reset_s", "reset_n", "cls", "fault_word", "fault_names",
                 "vbus_max_1s_v", "vbus_min_1s_v", "vbus_at_event_v", "vel_deg_s", "tau_nm", "power_w",
                 "power_min_1s_w", "kp", "kd", "sine_on", "sine_amp_deg", "sine_period_s"]


def classify_fault(word: Optional[int]) -> str:
  """Fault class of a fault word.  None (the query got no answer) -> 'unread'.  Over-voltage wins
  if several bits are set (the word is kept in the event, so nothing is hidden)."""
  if word is None:
    return CLASS_UNREAD
  if (word >> BIT_OV) & 1:
    return CLASS_OV
  if (word >> BIT_UV) & 1:
    return CLASS_UV
  return CLASS_OTHER


class OvMonitor:
  def __init__(self):
    self.lock = threading.Lock()
    self.csv_fh = None
    self.csv_writer = None
    self.csv_path: Optional[str] = None
    self.csv_rows = 0
    self.csv_error: Optional[str] = None
    self.reset_n = 0
    self.rev = 0                       # bumps on every event / reset, lets the stream skip unchanged lists
    self.reset(0.0)

  # ------------------------------------------------------------------ lifecycle
  def reset(self, t: float) -> None:
    """Zero the counters and statistics (connect, or the Reset button).  The CSV file stays."""
    with self.lock:
      self.reset_n += 1
      self.rev += 1
      self.t_reset = t
      self.t_end: Optional[float] = None
      self.counts = {CLASS_OV: 0, CLASS_UV: 0, CLASS_OTHER: 0, CLASS_UNREAD: 0}
      self.warn_count = 0
      self.events: deque = deque(maxlen=KEEP_EVENTS)      # newest last
      self.times: deque = deque()                          # (t, cls) of every fault event, for the rates
      self.win: deque = deque()                            # (t, vbus|None, power, vel, tau)
      self.vbus_max: Optional[float] = None
      self.vbus_min: Optional[float] = None
      self.vbus_n = 0
      self.last_vbus_t: Optional[float] = None
      self.above = {thr: [0, 0.0] for thr in THRESHOLDS_V}    # samples, seconds
      self.last_ov_t: Optional[float] = None

  def stop(self, t: float) -> None:
    """Freeze the elapsed time (disconnect).  Counters stay visible until the next connect."""
    with self.lock:
      if self.t_end is None:
        self.t_end = t

  def open_csv(self, data_dir: str, unix: float) -> str:
    """Create ~/System_ID/data/emf_events_<YYYYmmdd_HHMMSS>.csv (header written and flushed now)."""
    self.close_csv()
    os.makedirs(data_dir, exist_ok=True)
    stamp = datetime.fromtimestamp(unix).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(data_dir, f"emf_events_{stamp}.csv")
    k = 1
    while os.path.exists(path):         # reconnect within the same second: do not append to the old file
      k += 1
      path = os.path.join(data_dir, f"emf_events_{stamp}_{k}.csv")
    fh = open(path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(EVENT_COLUMNS)
    fh.flush()
    with self.lock:
      self.csv_fh, self.csv_writer, self.csv_path, self.csv_rows, self.csv_error = fh, w, path, 0, None
    return path

  def close_csv(self) -> None:
    with self.lock:
      fh, self.csv_fh, self.csv_writer = self.csv_fh, None, None
    if fh is not None:
      try:
        fh.flush()
        fh.close()
      except OSError:
        pass

  # ------------------------------------------------------------------ feeding (control thread)
  def on_sample(self, t: float, vbus: Optional[float], power_w: float, vel_deg_s: float, tau_nm: float) -> None:
    with self.lock:
      w = self.win
      w.append((t, vbus, power_w, vel_deg_s, tau_nm))
      while len(w) > 2 and w[0][0] < t - WINDOW_S - 0.5:
        w.popleft()
      if vbus is None:
        return
      self.vbus_n += 1
      self.vbus_max = vbus if self.vbus_max is None else max(self.vbus_max, vbus)
      self.vbus_min = vbus if self.vbus_min is None else min(self.vbus_min, vbus)
      dt = 0.0 if self.last_vbus_t is None else min(max(t - self.last_vbus_t, 0.0), MAX_DT_S)
      self.last_vbus_t = t
      for thr, acc in self.above.items():
        if vbus > thr:
          acc[0] += 1
          acc[1] += dt

  def on_warn_edge(self) -> None:
    with self.lock:
      self.warn_count += 1

  def on_fault_edge(self, t: float, unix: float, word: Optional[int], kp: float, kd: float,
                    sine_on: bool, sine_amp: float, sine_period: float) -> dict:
    """Record one fault event.  Call AFTER on_sample() of the sample that first shows the flag.
    Velocity / torque / power are those of the last sample BEFORE that one (the flag sample itself
    already shows the motor in Reset)."""
    cls = classify_fault(word)
    with self.lock:
      in_win = [r for r in self.win if r[0] >= t - WINDOW_S]
      vb = [r[1] for r in in_win if r[1] is not None]
      pw = [r[2] for r in in_win]
      cur = self.win[-1] if self.win else None
      prev = self.win[-2] if len(self.win) >= 2 else cur
      ev = dict(
          n=sum(self.counts.values()) + 1, reset_n=self.reset_n, unix=unix,
          wall=datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
          t=t, since_reset_s=t - self.t_reset, cls=cls,
          fault_word=word, fault_hex=None if word is None else f"0x{word:08x}",
          fault_names=[] if word is None else fr.decode_fault_names(word),
          vbus_max_1s=max(vb) if vb else None, vbus_min_1s=min(vb) if vb else None,
          vbus_at_event=cur[1] if cur else None,
          vel_deg_s=prev[3] if prev else None, tau_nm=prev[4] if prev else None,
          power_w=prev[2] if prev else None, power_min_1s_w=min(pw) if pw else None,
          kp=kp, kd=kd, sine_on=bool(sine_on),
          sine_amp_deg=sine_amp if sine_on else None, sine_period_s=sine_period if sine_on else None)
      self.counts[cls] += 1
      self.times.append((t, cls))
      if cls == CLASS_OV:
        self.last_ov_t = t
      self.events.append(ev)
      self.rev += 1
      w, fh = self.csv_writer, self.csv_fh
      if w is not None:
        try:
          w.writerow([ev["wall"], round(unix, 4), round(t, 4), round(ev["since_reset_s"], 3), self.reset_n, cls,
                      "" if word is None else ev["fault_hex"], "|".join(ev["fault_names"]),
                      _n(ev["vbus_max_1s"]), _n(ev["vbus_min_1s"]), _n(ev["vbus_at_event"]),
                      _n(ev["vel_deg_s"]), _n(ev["tau_nm"]), _n(ev["power_w"]), _n(ev["power_min_1s_w"]),
                      kp, kd, int(bool(sine_on)), _n(ev["sine_amp_deg"]), _n(ev["sine_period_s"])])
          fh.flush()                         # immediately: the frequency must survive a crash / restart
          self.csv_rows += 1
        except (OSError, ValueError) as exc:
          self.csv_error = f"{type(exc).__name__}: {exc}"
    return ev

  # ------------------------------------------------------------------ reading (HTTP threads)
  def summary(self, now_t: float) -> dict:
    with self.lock:
      end = self.t_end if self.t_end is not None else now_t
      elapsed = max(0.0, end - self.t_reset)
      eff = min(RATE_WINDOW_S, elapsed)
      lo = end - RATE_WINDOW_S
      n5_ov = sum(1 for tt, c in self.times if tt >= lo and c == CLASS_OV)
      n5_all = sum(1 for tt, _ in self.times if tt >= lo)
      n_ov, n_all = self.counts[CLASS_OV], sum(self.counts.values())

      def per_min(n, span):
        return None if span < 1.0 else 60.0 * n / span

      return dict(
          counts=dict(self.counts), warn=self.warn_count, faults=n_all,
          elapsed_s=elapsed, reset_n=self.reset_n, rev=self.rev, t_reset=self.t_reset,
          since_last_ov_s=None if self.last_ov_t is None else max(0.0, end - self.last_ov_t),
          rate=dict(window_s=eff, ov_5m=n5_ov, faults_5m=n5_all,
                    ov_per_min_5m=per_min(n5_ov, eff), faults_per_min_5m=per_min(n5_all, eff),
                    ov_per_min_all=per_min(n_ov, elapsed), faults_per_min_all=per_min(n_all, elapsed)),
          vbus=dict(max=self.vbus_max, min=self.vbus_min, n=self.vbus_n,
                    margin_v=None if self.vbus_max is None else OV_LIMIT_V - self.vbus_max,
                    above=[dict(v=thr, n=a[0], s=a[1]) for thr, a in self.above.items()]),
          limits=dict(ov=OV_LIMIT_V, uv=UV_LIMIT_V),
          csv=self.csv_path, csv_rows=self.csv_rows, csv_error=self.csv_error)

  def latest_events(self, n: int = 30) -> list:
    """Newest first."""
    with self.lock:
      return list(reversed(list(self.events)[-n:]))


def _n(v):
  return "" if v is None else v
