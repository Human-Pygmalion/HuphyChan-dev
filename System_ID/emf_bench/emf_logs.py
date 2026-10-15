"""Log files of the back-EMF bench: column definitions, file naming, listing and safe download.

Pure helpers (no CAN, no motor, no HTTP) so they can be unit-tested.

Layout on the machine that runs the server (``data_dir`` defaults to ~/System_ID/data):
    <data_dir>/emf_logs/<name>.csv            one row per control cycle (this module's LOG_COLUMNS)
    <data_dir>/emf_logs/<name>.meta.json      sidecar: start/stop time, motor, rate, version, column notes
    <data_dir>/emf_logs/emf_events_*.csv      over-voltage event files (one row per fault event)
    <data_dir>/emf_*.csv                      "legacy" files written before the logging panel existed

Path safety: a download request names plain file names.  Anything containing a path separator,
"..", a NUL, or characters outside [A-Za-z0-9._-] is refused, and only .csv / .meta.json names
that really exist directly inside the two folders are served.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime
from typing import Optional

LOG_SUBDIR = "emf_logs"
NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_NAME = 100

# (column, description).  Units are part of the column name.
LOG_COLUMNS: list[tuple[str, str]] = [
    ("t_s", "seconds on the server's monotonic clock since the server started (cycle start)"),
    ("unix_s", "wall-clock time of the cycle start, Unix seconds"),
    ("wall_time", "wall-clock time of the cycle start, ISO 8601 local time with milliseconds"),
    ("cycle", "control-cycle counter since the server started (not reset by connect)"),
    ("cycle_dt_ms", "time since the previous cycle started [ms] (empty on the first cycle after connect)"),
    ("loop_period_ms", "configured control-loop period [ms] = 1000 / rate_hz"),
    ("rate_hz", "configured control-loop rate [Hz]"),
    ("connected", "1 if the server holds the CAN channel (always 1 in a cycle row)"),
    ("enabled_req", "1 if torque has been REQUESTED on (the Enable button); the motor's real state is `mode`"),
    ("state_ok", "1 if a motor state frame arrived in this cycle; 0 = state columns below are empty"),
    ("mode", "motor-reported mode: 0 Reset, 1 Cali, 2 Motor (torque on)"),
    ("mode_name", "Reset / Cali / Motor"),
    ("fault", "motor-reported fault flag (state frame)"),
    ("warn", "motor-reported warning flag (state frame; generic warning, not necessarily over-voltage)"),
    ("fault_word", "fault word in hex from the last fault query (empty if no fault or not read)"),
    ("fault_class", "overvoltage / undervoltage / other / unread, from fault_word (empty if no fault)"),
    ("fault_names", "fault bit names from fault_word, joined by |"),
    ("temp_c", "motor temperature [deg C]"),
    ("pos_deg", "absolute motor position [deg] as reported"),
    ("pos_rel_deg", "position relative to the connect origin [deg]"),
    ("origin_deg", "absolute position captured at connect (the 0 of the *_rel_deg columns) [deg]"),
    ("vel_deg_s", "motor-reported velocity [deg/s] (clipped at the motor's reporting limit)"),
    ("tau_nm", "motor-reported torque [N m]"),
    ("target_deg", "commanded absolute target position of this cycle [deg] (after range clipping)"),
    ("target_rel_deg", "commanded target relative to the connect origin [deg]"),
    ("target_clamped", "1 if the target was clipped to the motor's position range"),
    ("track_err_deg", "target_deg - pos_deg [deg]; only meaningful while mode == 2 (filter on the mode column)"),
    ("kp", "position gain sent [N m/rad]"),
    ("kd", "velocity gain sent [N m s/rad]"),
    ("sine_on", "1 while the sine position command runs"),
    ("sine_amp_deg", "sine amplitude setting [deg] (+-)"),
    ("sine_period_s", "sine period setting [s per full back-and-forth]"),
    ("sine_center_deg", "sine centre relative to the connect origin [deg]"),
    ("sine_phase_rad", "sine phase at this cycle [rad]"),
    ("vbus_v", "bus voltage VBUS [V] received in this cycle (empty if no reply)"),
    ("iqf_a", "filtered q-axis current iqf [A] received in this cycle (empty if no reply)"),
    ("param_req", "which parameter was requested this cycle: vbus or iqf"),
    ("power_w", "regenerated power = torque x velocity [W]; negative = power returned toward the supply"),
    ("vbus_priority", "1 = VBUS is read every cycle and iqf is not read"),
    ("ov_count", "over-voltage fault events counted by the monitor since its last reset"),
    ("uv_count", "under-voltage fault events counted since the last reset"),
    ("other_count", "other + unreadable fault events counted since the last reset"),
    ("warn_count", "warning-flag rising edges counted since the last reset"),
    ("lost_state", "state frames lost since connect"),
    ("lost_vbus", "VBUS replies lost since connect"),
    ("lost_iqf", "iqf replies lost since connect"),
    ("send_fail", "failed CAN sends since the server started"),
    ("overruns", "control cycles that finished late since the server started"),
]
COLUMN_NAMES = [c for c, _ in LOG_COLUMNS]


# ------------------------------------------------------------------ names
def default_stamp(now: Optional[datetime] = None) -> str:
  return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def sanitize_log_name(raw: Optional[str], stamp: Optional[str] = None) -> str:
  """User text -> a safe file name ending in .csv.  Keeps [A-Za-z0-9._-], turns everything else
  (spaces, '/', Korean, ...) into '_', strips leading dots (no hidden files, no '..'), cuts to 100
  characters, drops a typed '.csv' / '.CSV' (it is added back), and falls back to the timestamp when nothing is left."""
  s = (raw or "").strip()
  s = re.sub(r"(?i)\.csv$", "", s)
  s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
  s = re.sub(r"\.{2,}", ".", s)           # no ".." anywhere in a name
  s = s.lstrip(".")[:MAX_NAME].rstrip(".")
  if not s or set(s) <= {"_", "-", "."}:
    s = stamp or default_stamp()
  return s + ".csv"


def meta_name(csv_name: str) -> str:
  return re.sub(r"\.csv$", "", csv_name) + ".meta.json"


def open_unique(log_dir: str, csv_name: str, other_dirs: tuple[str, ...] = ()):
  """Create `<log_dir>/<csv_name>` exclusively (never overwrite).  When the name, its sidecar name
  or the same name in `other_dirs` already exists, try `<base>_1.csv`, `_2` ...  Returns (file object, path)."""
  os.makedirs(log_dir, exist_ok=True)
  base = re.sub(r"\.csv$", "", csv_name)
  k = 0
  while True:
    name = csv_name if k == 0 else f"{base}_{k}.csv"
    k += 1
    if any(os.path.exists(os.path.join(d, name)) for d in other_dirs):
      continue
    if os.path.exists(os.path.join(log_dir, meta_name(name))):
      continue
    try:
      fh = open(os.path.join(log_dir, name), "x", newline="")
    except FileExistsError:
      continue
    return fh, os.path.join(log_dir, name)


# ------------------------------------------------------------------ safe lookup
def is_safe_name(name: str) -> bool:
  return bool(name) and len(name) <= 160 and NAME_OK.match(name) is not None and ".." not in name \
      and not name.startswith(".") and (name.endswith(".csv") or name.endswith(".meta.json"))


def resolve_download(name: str, log_dir: str, legacy_dir: str) -> Optional[str]:
  """Absolute path of a downloadable file, or None.  `name` must be a plain, safe basename."""
  if not is_safe_name(name):
    return None
  p = os.path.join(log_dir, name)
  if os.path.isfile(p):
    return p
  if name.startswith("emf_") and name.endswith(".csv"):         # legacy samples / event files only
    p = os.path.join(legacy_dir, name)
    if os.path.isfile(p):
      return p
  return None


# ------------------------------------------------------------------ listing
_row_cache: dict = {}


def count_rows(path: str) -> int:
  """Data rows = newline count - 1 (header).  Cached on (size, mtime) because a long log is hundreds of MB."""
  try:
    st = os.stat(path)
  except OSError:
    return 0
  key = (path, st.st_size, st.st_mtime_ns)
  if key in _row_cache:
    return _row_cache[key]
  n = 0
  with open(path, "rb") as fh:
    while True:
      b = fh.read(8 << 20)
      if not b:
        break
      n += b.count(b"\n")
  n = max(0, n - 1)
  if len(_row_cache) > 400:
    _row_cache.clear()
  _row_cache[key] = n
  return n


def list_logs(log_dir: str, legacy_dir: str, open_rows: Optional[dict] = None) -> list[dict]:
  """Newest first.  `open_rows` maps the base names of files that are being written now to their live
  row count (None = unknown)."""
  open_rows = open_rows or {}
  out = []
  for d, legacy in ((log_dir, False), (legacy_dir, True)):
    try:
      names = os.listdir(d)
    except OSError:
      continue
    for n in names:
      if not n.endswith(".csv") or not is_safe_name(n):
        continue
      if legacy and not n.startswith("emf_"):
        continue
      p = os.path.join(d, n)
      try:
        st = os.stat(p)
      except OSError:
        continue
      if not os.path.isfile(p):
        continue
      is_open = n in open_rows
      rows = open_rows[n] if is_open and open_rows[n] is not None else count_rows(p)
      out.append(dict(
          name=n, size=st.st_size, mtime=st.st_mtime,
          mtime_iso=datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
          rows=rows, legacy=legacy, open=is_open,
          kind="events" if n.startswith("emf_events_") else "samples",
          meta=meta_name(n) if os.path.isfile(os.path.join(d, meta_name(n))) else None))
  out.sort(key=lambda r: r["mtime"], reverse=True)
  seen, uniq = set(), []
  for r in out:                      # a legacy file with the same name as a new one is shadowed (download serves the new)
    if r["name"] in seen:
      continue
    seen.add(r["name"])
    uniq.append(r)
  return uniq


# ------------------------------------------------------------------ download
def complete_length(path: str, size: int) -> int:
  """Length of the file prefix that ends on a line boundary (a file still being written may end
  inside a row; the download is cut at the last complete row)."""
  if size <= 0:
    return 0
  with open(path, "rb") as fh:
    pos = size
    while pos > 0:
      step = min(65536, pos)
      fh.seek(pos - step)
      b = fh.read(step)
      i = b.rfind(b"\n")
      if i >= 0:
        return pos - step + i + 1
      pos -= step
  return 0


def read_prefix(path: str, n: int):
  """Yield the first n bytes of path in chunks."""
  with open(path, "rb") as fh:
    left = n
    while left > 0:
      b = fh.read(min(1 << 20, left))
      if not b:
        break
      left -= len(b)
      yield b


def build_zip(paths: list[str]):
  """ZIP (stdlib, deflate) of the files -> (spooled temp file at position 0, byte length).  Each file is
  cut at its last complete row so a log that is still open comes out as a valid CSV."""
  tmp = tempfile.SpooledTemporaryFile(max_size=64 << 20)
  with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in paths:
      size = complete_length(p, os.path.getsize(p)) if p.endswith(".csv") else os.path.getsize(p)
      with z.open(os.path.basename(p), "w", force_zip64=True) as zf:
        for b in read_prefix(p, size):
          zf.write(b)
  n = tmp.tell()
  tmp.seek(0)
  return tmp, n


# ------------------------------------------------------------------ server version
def server_version(here: str) -> dict:
  """Git hash of the folder if git is available there, always the newest source-file mtime."""
  newest = 0.0
  for n in os.listdir(here):
    if n.endswith((".py", ".html")):
      newest = max(newest, os.path.getmtime(os.path.join(here, n)))
  info = dict(files_mtime=datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M:%S"), git=None)
  try:
    h = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       timeout=2)
    if h.returncode == 0 and h.stdout.strip():
      dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."], capture_output=True,
                             text=True, timeout=2).stdout.strip()
      info["git"] = h.stdout.strip() + ("+uncommitted" if dirty else "")
  except Exception:  # noqa: BLE001
    pass
  return info
