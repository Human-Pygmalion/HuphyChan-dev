"""Dummy transmitter: exercise the whole receive path without a robot.

From ONE underlying joint trajectory (sine / a script file / a recorded jsonl(.gz)) this can
produce either or both of:

  ws    canonical JointState (and, with ``--imu``, ImuState) sent over websocket to
        ``/ws/in`` - the direct path a future non-HUPHY host would also use.
  udp   the SAME trajectory re-expressed as HUPHY-format UDP packets (one per leg, the keys
        HUPHY's own ``telemetry/snapshot.py`` would produce) sent to the HUPHY UDP bridge's
        listening port - this is what proves the bridge's unit/sign/name conversion actually
        round-trips end to end, not just in a unit test against the bare function.

``--latency-ms``/``--jitter-ms``/``--drop-ratio`` inject network imperfections on the SENDING
side (a message is held ``latency_ms + uniform(-jitter,+jitter)`` before it goes out, and
dropped outright with probability ``drop-ratio``), so the receive-side staleness/seq-gap
logic in ``telemetry.RealState`` has something real to react to.

Only the AB variant has a HUPHY hardware analogue (the physical robot is the loop
mechanism); ``--target udp`` therefore requires an AB-family variant.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import json
import math
import random
import socket
import sys
import time

from .. import CACHE_DIR, VARIANTS
from ..contract import load_contract
from ..schema import ImuState, JointState, to_jsonl
from .huphy_udp import DEFAULT_PORT, WIRE_IS_ALREADY_MIRRORED, JointMap


# --------------------------------------------------------------------------- trajectories
class SineSource:
  def __init__(self, joints: list[str], default_q: dict[str, float], amplitude: float, freq_hz: float):
    self.joints = joints
    self.default_q = default_q
    self.amplitude = amplitude
    self.freq_hz = freq_hz

  def at(self, t: float) -> dict[str, float]:
    w = 2 * math.pi * self.freq_hz * t
    return {n: self.default_q[n] + self.amplitude * math.sin(w) for n in self.joints}


class ScriptSource:
  """``{"joint_names": [...], "rows": [[t_s, q...], ...]}``, held-value looped playback."""

  def __init__(self, path: str):
    d = json.loads(open(path).read())
    self.joint_names = d["joint_names"]
    self.rows = d["rows"]
    self.t0 = self.rows[0][0]
    self.span = max(self.rows[-1][0] - self.t0, 1e-6)

  def at(self, t: float) -> dict[str, float]:
    tm = self.t0 + (t % self.span)
    idx = 0
    for i, row in enumerate(self.rows):
      if row[0] <= tm:
        idx = i
      else:
        break
    return dict(zip(self.joint_names, self.rows[idx][1:]))


class JsonlSource:
  """Replays a pygviewer recording's ``JointState`` lines (the header line is skipped)."""

  def __init__(self, path: str):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
      lines = f.readlines()
    for i, line in enumerate(lines):
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      if i == 0 and obj.get("type") != "JointState":
        continue  # the recorder's plain-JSON header line
      if obj.get("type") == "JointState":
        rows.append(obj)
    if not rows:
      raise ValueError(f"{path}: no JointState rows found")
    self.rows = rows
    self.t0_ns = rows[0]["t_ns"]

  def at(self, t: float) -> dict[str, float]:
    target_ns = self.t0_ns + t * 1e9
    idx = 0
    for i, row in enumerate(self.rows):
      if row["t_ns"] <= target_ns:
        idx = i
      else:
        break
    row = self.rows[idx]
    return {n: q for n, q in zip(row["joint_names"], row["q"]) if q is not None}


# --------------------------------------------------------------------------- UDP encoding
def huphy_packet_for_limb(limb: str, side: str, q: dict[str, float], jmap: JointMap,
                           travel_sign: dict[str, float], t: float) -> dict:
  """Invert ``huphy_udp.huphy_deg_to_sim_rad`` to build one HUPHY-format FAST packet.

  Deliberately simple: only ``pos``/``tgt`` (== pos, no separate target injected) carry a
  value; ``vel``/``tau`` are sent as HUPHY's own -1 "no data" sentinel.  That is enough to
  exercise the adapter's name/unit/sign path end to end (task item 8's verification), which
  is the point of this transmitter - it is not trying to be a physically accurate robot.
  """
  out: dict[str, float] = {"t": round(t, 3), "loop_dt": 0.0}
  for (row_limb, motor), row in jmap.motors.items():
    if row_limb != limb:
      continue
    sim_joint = row["sim_joint"]
    if not sim_joint.startswith(side + "_"):
      continue
    qv = q.get(sim_joint)
    ts = travel_sign.get(sim_joint, 1.0)
    denom = ts * row["sign"]
    if qv is None or denom == 0:
      deg = -1.0
    else:
      deg = math.degrees((qv - row["offset_rad"]) / denom)
    out[f"{limb}/{motor}/pos"] = round(deg, 2)
    out[f"{limb}/{motor}/tgt"] = round(deg, 2)
    out[f"{limb}/{motor}/err"] = 0.0
    out[f"{limb}/{motor}/vel"] = -1.0
    out[f"{limb}/{motor}/tau"] = -1.0
  return out


# --------------------------------------------------------------------------- delivery
async def _drain_ws_replies(ws) -> None:
  """Read (and discard) whatever ``/ws/in`` acks back per accepted frame.

  Bug fixed 2026-09-04 (found via ``bridge/huphy_udp_forward.py``'s own identical mistake,
  same root cause here since this file never called ``ws.recv()`` either): a websocket
  client that only ever sends eventually stalls its own reader once the unread-message queue
  fills (default ``max_queue``, reached well under a second at 50 Hz), which then ALSO stops
  it from seeing PONG replies to its own keepalive pings, and the client self-disconnects
  with ``ConnectionClosedError: sent 1011 keepalive ping timeout`` after ~40-60s - this
  transmitter's default ``--seconds 10`` run is short enough to never have hit it, but a
  longer soak run (or ``--seconds`` raised for an endurance test) would have. Runs as a
  background task for the connection's lifetime; ``ws.__aiter__`` ends cleanly on close."""
  try:
    async for raw in ws:
      try:
        obj = json.loads(raw)
      except (json.JSONDecodeError, TypeError):
        continue
      if isinstance(obj, dict) and obj.get("error"):
        print(f"dummy_tx: server reported: {obj['error']}", file=sys.stderr, flush=True)
  except Exception:  # connection closing/closed while draining is expected, not an error here
    pass


async def _delayed_send(coro_factory, latency_s: float, jitter_s: float, drop_ratio: float):
  if drop_ratio > 0 and random.random() < drop_ratio:
    return
  delay = max(0.0, latency_s + random.uniform(-jitter_s, jitter_s))
  if delay > 0:
    await asyncio.sleep(delay)
  await coro_factory()


async def run(args) -> None:
  c = load_contract(args.cache, args.variant)
  default_q = dict(c.raw["default_q"])
  joints = args.joints.split(",") if args.joints else list(c.action_joint_names)
  unknown = [j for j in joints if j not in c.action_joint_names]
  if unknown:
    raise SystemExit(f"not actuated joints of {args.variant}: {unknown}")

  if args.pattern == "sine":
    src = SineSource(joints, default_q, args.amplitude, args.freq)
  elif args.pattern == "script":
    src = ScriptSource(args.file)
  elif args.pattern == "jsonl":
    src = JsonlSource(args.file)
  else:
    raise SystemExit(f"unknown --pattern {args.pattern!r}")

  jmap = JointMap() if "udp" in args.target else None
  # 2026-09-15: 같은 관문을 쓴다 (tx_map/huphy_udp 와 동일) - 전선과 모형 사이에
  # 곱할 것이 없다는 것이 로봇 교정으로 확인됐다.
  travel_sign = (
    {n: (1.0 if WIRE_IS_ALREADY_MIRRORED
         else float(c.raw["joint_contract"][n]["travel_sign"]))
     for n in c.action_joint_names}
    if jmap else {}
  )
  if "udp" in args.target and not c.is_loop:
    raise SystemExit("--target udp needs an AB (loop) variant - HUPHY's hardware has no RP analogue")

  ws = None
  ws_drain_task = None
  udp_sock = None
  if "ws" in args.target:
    import websockets

    ws = await websockets.connect(args.ws_url)
    ws_drain_task = asyncio.create_task(_drain_ws_replies(ws))
  if "udp" in args.target:
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  seq = 0
  t0 = time.monotonic()
  dt = 1.0 / args.hz
  print(f"dummy_tx: {args.pattern} -> {args.target} for {args.seconds:.0f}s @ {args.hz:.0f} Hz")
  try:
    n = int(args.seconds * args.hz)
    for i in range(n):
      t = i * dt
      q = src.at(t)
      seq += 1

      async def _send_ws(q=q, t=t, seq=seq):
        msg = JointState(
          t_ns=time.monotonic_ns(), seq=seq, src="dummy", contract_hash=c.contract_sha,
          joint_names=list(q), q=list(q.values()),
        )
        await ws.send(to_jsonl(msg).strip())
        if args.imu:
          tilt = 0.05 * math.sin(2 * math.pi * 0.1 * t)  # slow +-3 deg tilt around a static g
          gvec = [math.sin(tilt), 0.0, -math.cos(tilt)]
          imu = ImuState(
            t_ns=time.monotonic_ns(), seq=seq, src="dummy", contract_hash=c.contract_sha,
            gravity_b=gvec, gyro_rad_s=[0.0, 0.0, 0.0], acc_m_s2=[g * 9.81 for g in gvec],
          )
          await ws.send(to_jsonl(imu).strip())

      async def _send_udp(q=q, t=t):
        for limb, side in (("left", "L"), ("right", "R")):
          pkt = huphy_packet_for_limb(limb, side, q, jmap, travel_sign, t)
          udp_sock.sendto(json.dumps(pkt).encode(), (args.udp_host, args.udp_port))

      if ws is not None:
        await _delayed_send(_send_ws, args.latency_ms / 1e3, args.jitter_ms / 1e3, args.drop_ratio)
      if udp_sock is not None:
        await _delayed_send(_send_udp, args.latency_ms / 1e3, args.jitter_ms / 1e3, args.drop_ratio)
      await asyncio.sleep(max(0.0, t0 + t + dt - time.monotonic()))
  finally:
    if ws is not None:
      await ws.close()
    if ws_drain_task is not None:
      ws_drain_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await ws_drain_task
    if udp_sock is not None:
      udp_sock.close()
  print(f"dummy_tx: done, {seq} ticks sent")


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="pygviewer bridge dummy_tx")
  ap.add_argument("--variant", default="LegOnly-AB", choices=list(VARIANTS))
  ap.add_argument("--cache", default=CACHE_DIR)
  ap.add_argument("--pattern", choices=("sine", "script", "jsonl"), default="sine")
  ap.add_argument("--joints", default=None, help="comma-separated; default = all actuated")
  ap.add_argument("--amplitude", type=float, default=0.2, help="sine amplitude [rad]")
  ap.add_argument("--freq", type=float, default=0.5, help="sine frequency [Hz]")
  ap.add_argument("--file", default=None, help="script (.json) or recording (.jsonl[.gz])")
  ap.add_argument("--target", default="ws", help="comma-separated subset of ws,udp")
  ap.add_argument("--ws-url", default="ws://127.0.0.1:8095/ws/in")
  ap.add_argument("--udp-host", default="127.0.0.1")
  ap.add_argument("--udp-port", type=int, default=DEFAULT_PORT)
  ap.add_argument("--hz", type=float, default=50.0)
  ap.add_argument("--seconds", type=float, default=10.0)
  ap.add_argument("--latency-ms", type=float, default=0.0)
  ap.add_argument("--jitter-ms", type=float, default=0.0)
  ap.add_argument("--drop-ratio", type=float, default=0.0)
  ap.add_argument("--imu", action="store_true", help="also send ImuState (static gravity + slow tilt)")
  args = ap.parse_args(argv)
  args.target = set(args.target.split(","))
  asyncio.run(run(args))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
