"""RobStride 버스 — 런타임 조작.

세 조각을 잇는 곳임.

    base.py       MotorsBus 계약, MotorState
    codec/mit.py  숫자 <-> 8바이트
    canbus.py     8바이트 <-> CAN 선

각도는 전부 **raw 공간**임. 관절 이름도 모름 — 모터 id 로만 말함. cal 변환과 관절
이름은 `robots/` 가 함.


## 상태는 명령의 응답으로 옴

MIT 모드에는 "상태 읽기" 명령이 따로 없음. 모터는 **동작 명령을 받으면 응답으로
현재 상태를 돌려줌.**

그래서 움직이지 않고 상태만 보려면 게인과 토크가 0인 명령을 보냄.

    tau = kp*(목표각 - 현재각) + kd*(0 - 현재속도) + 토크_FF
        = 0*(...) + 0*(...) + 0
        = 0

토크가 0이니 아무 일도 일어나지 않고 응답만 옴. `refresh_states()` 가 이걸 씀.


## 전송과 수거를 나눠 둠

    send_mit()   보내기만
    collect()    수거만
    refresh_states()   둘 다 (`MotorsBus` 계약)

버스가 둘일 때 "왼다리 보내기 -> 오른다리 보내기 -> 왼다리 수거 -> 오른다리 수거"
순서를 짜려면 나뉘어 있어야 함 (이슈 #10).


## can_link 개선판에서 바뀐 것 (원본 a1b67b1 대비) -- `collect` 의 끝 판정

원본은 `drain(expect=N)` 이 **프레임 개수**로 끝남. 직전 주기의 늦은 응답이 큐에
남아 있으면 그것도 한 개로 세어, 이번 주기 응답 하나가 다음 주기로 밀리고 그 밀림이
이어짐 (밀림이 안 풀림: `can_link/tests/test_collect_slip.py` 가 재현함).

개선판은 (`completion="per_motor"`, 기본) **모터마다** 끝남. `completion="count"` 는 원본의 개수 판정. 모터는 "이번 송신 시각 이후에 도착한 응답(커널 도착 시각)"을
받았을 때 끝난 것이고, 송신 이전에 도착한 프레임은 상태에 반영하되 `late` 로만 기록함.
반환값(응답이 없던 모터 id)은 그대로이고, 상세는 `bus.last_collect` 에 있음.

**동작이 바뀌는 곳 (한 군데)**: 모터 하나에 응답 없는 명령이 `max_outstanding`(켰을 때 권장 2)개
쌓이면 그 모터에는 이번 주기 `send_mit` 명령을 **보내지 않음**. 모터는 마지막으로 받은
MIT 명령(목표각·게인·토크)을 계속 유지하므로 토크가 갑자기 0 이 되지는 않음 -- 대신
그 주기의 새 목표값이 그 모터에는 전달되지 않음. 모터 수신 칸이 3개라 4번째부터 버려지는
것(2026-09-09 실측)을 호스트에서 먼저 막는 장치임. 쌓인 기록은 `outstanding_ttl_s`
(기본 50 ms)가 지나면 버려져서, 죽은 모터에도 주기적으로 다시 시도함.
**기본은 꺼짐**(`max_outstanding=None`); 켜려면 `max_outstanding=2`. `enable/disable/clear_fault` 는 이 제한을 받지 않음.

## 여기 없는 것

CAN id 변경, 프로토콜 전환, 기계영점, 플래시 저장은 `commissioning.py` 로 감.
되돌리기 어렵고 한 번만 하는 조작이라, 제어 루프에서 부를 수 있는 자리에 두지 않음.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from ..base import Motor, MotorFault, MotorState, MotorsBus, resolve_motor_list
from ..canbus import DEFAULT_DRAIN_S, DEFAULT_POLL_S, CanBus, CanFrame
from . import tables
from .codec import mit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MitCommand:
    """MIT 동작 명령 하나. 모터 펌웨어가 이 다섯 값으로 토크를 계산함.

        tau = kp*(position_deg - 현재각) + kd*(velocity_deg_s - 현재속도) + torque_nm

    `position_deg` 는 **raw 공간**임.

    다섯 개를 튜플로 넘기지 않는 이유: 순서를 틀려도 타입이 같아 조용히 통과함.
    kp 자리에 위치가 들어가면 모터가 전력으로 튐.
    """

    position_deg: float
    velocity_deg_s: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque_nm: float = 0.0


DEFAULT_HOST_ID = 0xFD
"""공장값 호스트 ID. 모든 모터가 이 번호로 응답함 (파라미터 0x200A CAN_MASTER)."""

PASSIVE = MitCommand(position_deg=0.0)
"""토크가 나가지 않는 명령. 응답만 받아 오는 데 씀."""


def _command_frame(motor_id: int, command: int, f_cmd: int = tables.F_CMD_DEFAULT) -> CanFrame:
    """제어 명령 프레임을 만듦.

        data[0:6] = 0xFF    data[6] = F_CMD    data[7] = 명령 바이트

    같은 명령 바이트가 `F_CMD` 에 따라 두 가지 뜻을 가짐 (`tables.py` 참조).
    """
    return CanFrame(
        can_id=int(motor_id),
        data=bytes([0xFF] * 6 + [int(f_cmd) & 0xFF, int(command) & 0xFF]),
    )


@dataclass
class CollectStats:
    """직전 `collect()` 한 번의 결과. `RobStrideBus.last_collect` 에 있음."""

    awaited: Tuple[int, ...] = ()
    """직전 `send_mit` 에서 명령을 **실제로 보낸** 모터 (미응답 한도로 건너뛴 것 제외)."""

    fresh: Tuple[int, ...] = ()
    """이번 송신 이후에 도착한 응답을 받은 모터."""

    missing: Tuple[int, ...] = ()
    """`awaited` 중 `fresh` 가 아닌 모터."""

    late: Tuple[int, ...] = ()
    """송신 **이전에** 도착한 프레임(직전 명령의 늦은 응답)이 나온 모터."""

    late_age_s: Dict[int, float] = field(default_factory=dict)
    """`late` 프레임의 나이 = 수거를 마친 시각 - 그 프레임의 도착 시각. 이만큼 묵은 값."""

    fresh_age_s: Dict[int, float] = field(default_factory=dict)
    """`fresh` 응답의 나이. 정상이면 수거 대기 안쪽의 작은 값."""

    skipped: Tuple[int, ...] = ()
    """미응답 한도 때문에 이번 주기 명령을 보내지 않은 모터."""

    waited_s: float = 0.0
    timed_out: bool = False
    """기다린 모터가 다 오기 전에 마감에 걸렸는지."""


@dataclass
class LinkCounters:
    """버스 수명 동안 누적. `as_fields()` 는 텔레메트리용 (0 이어도 모든 키)."""

    cycles: int = 0
    late_replies: int = 0
    skipped_commands: int = 0
    timeouts: int = 0
    id_mismatch: int = 0
    undecodable: int = 0
    unknown_motor: int = 0

    def as_fields(self) -> Dict[str, int]:
        return {f"bus.{k}": v for k, v in self.__dict__.items()}


class RobStrideBus(MotorsBus):
    """CAN 채널 하나에 물린 RobStride 모터들.

    모터마다 모델이 다를 수 있음 — 다리는 RS02 4개와 RS00 2개임. 모델이 다르면
    인코딩 범위가 다르므로 모터별로 표를 들고 있음.
    """

    def __init__(
        self,
        bus: CanBus,
        motors: Mapping[int, Motor],
        *,
        protocol: tables.Protocol = tables.Protocol.MIT,
        drain_s: float = DEFAULT_DRAIN_S,
        reply_ids: Optional[Mapping[int, int]] = None,
        max_outstanding: Optional[int] = None,
        completion: str = "per_motor",
        outstanding_ttl_s: float = 0.05,
        kernel_filter: bool = False,
    ) -> None:
        if drain_s < 0:
            raise ValueError(f"drain_s 는 0 이상이어야 함 (받은 값 {drain_s})")
        if completion not in ("per_motor", "count"):
            raise ValueError(f"completion 은 'per_motor' 또는 'count' (받은 값 {completion!r})")
        self.completion = completion
        """`collect` 끝 판정. `per_motor`(기본): 모터별 "송신 이후 도착". `count`: 원본처럼 프레임 개수."""
        max_outstanding = 0 if max_outstanding is None else max_outstanding
        if not 0 <= int(max_outstanding) <= 3:
            raise ValueError(
                f"max_outstanding 은 0(끔)~3 이어야 함 -- 모터 수신 칸이 3개임 (받은 값 {max_outstanding})"
            )
        self.bus = bus
        self.motors = dict(motors)
        self.max_outstanding = int(max_outstanding)
        """모터 하나에 응답 없는 명령을 이 개수까지만 둠. `None`/0 이면 제한 없음 (원본 동작, **기본**)."""
        self.outstanding_ttl_s = float(outstanding_ttl_s)
        """응답 없는 명령 기록을 이 시간이 지나면 버림 (모터가 버렸거나 응답을 잃은 것으로 봄)."""

        self.reply_ids: Dict[int, int] = {int(m): int(h) for m, h in (reply_ids or {}).items()}
        """모터 id -> 그 모터가 응답에 쓰는 11비트 번호 (모터별 호스트 ID). 비우면 전부
        공장값 0xFD 를 공유하고 `data[0]` 으로만 구분함."""
        for mid, host in self.reply_ids.items():
            if mid not in self.motors:
                raise ValueError(f"reply_ids: 이 버스에 없는 모터 {mid}")
            if not 0 <= host <= 0x7FF:
                raise ValueError(f"reply_ids: m{mid} 의 번호 0x{host:X} 가 11비트를 넘음")
            if host in self.motors:
                # 응답 번호가 다른 모터의 명령 번호와 같으면 그 모터가 응답 프레임을 명령으로
                # 읽을 수 있음 -- 상태 바이트가 MIT 게인·토크로 해석됨.
                raise ValueError(
                    f"reply_ids: m{mid} 의 응답 번호 0x{host:X} 가 모터 id {host} 와 같음 (거부)"
                )
        owners: Dict[int, List[int]] = {}
        for mid, host in self.reply_ids.items():
            owners.setdefault(host, []).append(mid)
        self._mid_by_reply_id: Dict[int, int] = {h: m[0] for h, m in owners.items() if len(m) == 1}
        """번호 하나를 모터 하나만 쓰는 경우의 역방향 표. 여럿이 나눠 쓰는 번호는 빠지고
        그 프레임은 `data[0]` 으로 구분함."""
        self.kernel_filter = bool(kernel_filter)
        self.protocol = protocol
        self.drain_s = float(drain_s)
        """응답을 기다리는 최대 시간 (초). `collect` 와 `refresh_states` 가 씀.

        기다린 개수가 다 오면 즉시 빠져나가므로 **정상 주기에는 이만큼 쓰지 않음.**
        응답이 늦거나 빠질 때만 끝까지 기다림.

        크게 잡으면 늦게 온 응답을 이번 주기에 받아 무응답이 줄지만, 모터 하나가
        죽으면 매 주기 이만큼을 통째로 씀 -- 주기 예산을 넘겨 밀림이 됨. 작게 잡으면
        그 반대임. 기본 2ms 는 100Hz 예산 10ms 의 20% 임.
        """

        # 모델 문자열을 벤더 enum 으로 옮기는 유일한 지점임. 여기서 걸러 두면
        # 제어 중에 오타가 드러나는 일이 없음.
        self._encoding: Dict[int, tables.EncodingRange] = {}
        for motor_id, motor in self.motors.items():
            try:
                model = tables.Model(motor.model)
            except ValueError as e:
                raise ValueError(
                    f"m{motor_id}: 모르는 모델 {motor.model!r}. "
                    f"가용: {[m.value for m in tables.Model]}"
                ) from e
            self._encoding[motor_id] = tables.encoding_for(model, protocol)

        self._states: Dict[int, MotorState] = {mid: MotorState() for mid in self.motors}
        self._torque_on: Dict[int, bool] = {mid: False for mid in self.motors}

        self.latency = None
        """지연 기록기. 붙이면 `sent(mid)` 와 `received(mid, 초, late)` 를 부름.

        기본은 없음 -- 없으면 시각을 재지도 않음. `telemetry.latency_log` 가 붙임.
        """
        self._sent_at: Dict[int, float] = {}
        self._prev_sent_at: Dict[int, float] = {}
        self._tx_at: Dict[int, float] = {}
        """모터별 마지막 송신 시각. `_sent_at` 과 같은 값이지만 **지연 기록기가 없어도** 남김 --
        `collect` 가 "송신 이후 도착"을 가리는 데 씀. (`_sent_at` 은 원본처럼 기록기가 있을 때만.)"""

        self._outstanding: Dict[int, Deque[float]] = {mid: deque() for mid in self.motors}
        """모터별로 응답을 아직 못 받은 명령의 송신 시각 (오래된 것이 앞)."""
        self._cycle_sent: Tuple[int, ...] = ()
        self._cycle_skipped: Tuple[int, ...] = ()
        self._cycle_open = False
        """`send_mit` 를 부른 뒤 아직 `collect` 를 안 불렀는지."""
        self.last_collect = CollectStats()
        self.link = LinkCounters()

    def __repr__(self) -> str:
        return f"RobStrideBus({self.bus.channel}, 모터 {len(self.motors)}개)"

    # ---- 구성 -------------------------------------------------------------
    @property
    def motor_ids(self) -> Tuple[int, ...]:
        return tuple(self.motors)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def encoding(self, motor_id: int) -> tables.EncodingRange:
        """모터의 인코딩 범위. RS02 와 RS00 은 토크 범위가 다름."""
        return self._encoding[motor_id]

    # ---- 수명 -------------------------------------------------------------
    def connect(self) -> None:
        self.bus.connect()
        if self.kernel_filter and hasattr(self.bus, "set_reply_filter"):
            # 호스트 ID 를 아직 안 바꾼 모터는 0xFD 로 계속 응답하므로 그 번호도 열어 둠.
            ids = set(self.reply_ids.values())
            if any(mid not in self.reply_ids for mid in self.motors):
                ids.add(DEFAULT_HOST_ID)
            self.bus.set_reply_filter(ids)

    def disconnect(self) -> None:
        """**토크를 먼저 끊고** 채널을 닫음. 여러 번 불려도 안전함.

        순서가 반대면 채널이 닫힌 뒤라 정지 명령을 보낼 방법이 없어짐. 모터는
        마지막 명령을 계속 유지하므로 사람이 전원을 뽑을 때까지 힘을 씀.
        """
        if self.bus.is_connected:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning("%s 토크 차단 실패 (계속 진행함): %s", self.bus.channel, e)
        self.bus.disconnect()

    # ---- 토크 -------------------------------------------------------------
    def enable_torque(self, motors: Optional[Iterable[int]] = None) -> None:
        """모터를 활성화함. 이 뒤부터 MIT 명령이 실제로 힘을 씀."""
        ids = resolve_motor_list(motors, self.motor_ids)
        self.bus.send_many([_command_frame(mid, tables.CMD_ENABLE) for mid in ids])
        for mid in ids:
            self._torque_on[mid] = True

    def disable_torque(self, motors: Optional[Iterable[int]] = None) -> None:
        """모터를 정지시킴. 힘이 빠져 관절이 중력에 떨어짐."""
        ids = resolve_motor_list(motors, self.motor_ids)
        self.bus.send_many([_command_frame(mid, tables.CMD_STOP) for mid in ids])
        for mid in ids:
            self._torque_on[mid] = False

    def is_torque_on(self, motor_id: int) -> bool:
        """마지막으로 보낸 명령 기준임. 모터에 물어본 값이 아님."""
        return self._torque_on[motor_id]

    def clear_fault(self, motors: Optional[Iterable[int]] = None) -> None:
        """고장 상태를 지움. 원인이 남아 있으면 다시 뜸."""
        ids = resolve_motor_list(motors, self.motor_ids)
        self.bus.send_many([_command_frame(mid, tables.CMD_FAULT) for mid in ids])

    # ---- 명령 -------------------------------------------------------------
    def send_mit(self, commands: Mapping[int, MitCommand]) -> int:
        """MIT 명령들을 **한 번에** 보냄. 보낸 개수를 반환함.

        수거하지 않음 — 응답은 `collect()` 로 받음. 버스가 둘일 때 전송을 먼저
        몰아야 두 다리의 명령 시각이 붙음 (이슈 #10).

        여기서 한계를 검사하지 않음. 그건 `safety.guards` 가 cal 공간에서 이미 함.
        같은 검사를 두 군데 두면 한쪽만 고쳐졌을 때 어느 쪽이 맞는지 알 수 없음.
        """
        unknown = [mid for mid in commands if mid not in self.motors]
        if unknown:
            raise ValueError(
                f"이 버스에 없는 모터 id: {unknown} (가용: {list(self.motor_ids)})"
            )

        now = time.monotonic()
        sendable: Dict[int, MitCommand] = {}
        skipped: List[int] = []
        for mid, c in commands.items():
            if self._outstanding_full(mid, now):
                skipped.append(mid)
            else:
                sendable[mid] = c

        frames = [
            CanFrame(
                can_id=mid,
                data=mit.pack_command(
                    position_deg=c.position_deg,
                    velocity_deg_s=c.velocity_deg_s,
                    kp=c.kp,
                    kd=c.kd,
                    torque_nm=c.torque_nm,
                    enc=self._encoding[mid],
                ),
            )
            for mid, c in sendable.items()
        ]
        self._cycle_sent = tuple(sendable)
        self._cycle_skipped = tuple(skipped)
        self._cycle_open = True
        self.link.skipped_commands += len(skipped)
        if skipped:
            logger.debug("%s 미응답 한도로 명령 건너뜀: %s", self.bus.channel, skipped)
        self._note_sent(sendable)
        return self.bus.send_many(frames)

    def _outstanding_full(self, mid: int, now: float) -> bool:
        """이 모터에 응답 없는 명령이 한도에 닿았는지. 오래된 기록은 먼저 버림."""
        if self.max_outstanding <= 0:
            return False
        q = self._outstanding[mid]
        while q and now - q[0] > self.outstanding_ttl_s:
            q.popleft()
        return len(q) >= self.max_outstanding

    def _note_sent(self, commands: Mapping[int, MitCommand]) -> None:
        """보낸 시각을 모터별로 남김. **묶음을 보내기 직전 한 시각**을 씀.

        프레임별 시각이 아닌 이유: 제어 루프가 겪는 지연은 "명령을 내기로 한 때부터
        응답이 손에 들어온 때까지" 임. slcan 에서 앞 프레임이 직렬로 나가는 동안 뒤
        프레임이 기다리는 시간도 그 안에 들어가야 함.

        직전 시각도 들고 있음 -- 다음 주기에 늦게 들어온 응답을 그 명령에 맞추려고.
        지연 기록기용 `_sent_at` 은 원본처럼 기록기가 있을 때만 남기고, `collect` 가 쓰는
        `_tx_at` 은 항상 남김.
        """
        now = time.monotonic()
        for mid in commands:
            self._tx_at[mid] = now
            self._outstanding[mid].append(now)
            if self.latency is not None:
                if mid in self._sent_at:
                    self._prev_sent_at[mid] = self._sent_at[mid]
                self._sent_at[mid] = now
                self.latency.sent(mid)

    def _note_received(self, mid: int, stamp: float) -> None:
        """응답 하나의 지연을 남김.

        MIT 응답에는 순번이 없음. 그래서 **도착 시각이 이번 송신보다 앞이면** 직전
        명령의 응답이 늦게 온 것으로 봄 (`late`). 도착 시각은 커널이 찍은 값임.
        """
        sent = self._sent_at.get(mid)
        if sent is None or not stamp:
            return
        if stamp >= sent:
            self.latency.received(mid, stamp - sent, False)
            return
        prev = self._prev_sent_at.get(mid)
        if prev is not None and stamp >= prev:
            self.latency.received(mid, stamp - prev, True)

    # ---- 상태 -------------------------------------------------------------
    def collect(
        self, *, expect: Optional[int] = None, timeout_s: Optional[float] = None
    ) -> List[int]:
        """도착한 응답을 해석해 캐시에 넣음. **응답이 없었던 모터 id** 를 반환함.

        끝 판정은 **모터별**임: 직전 `send_mit` 에서 명령을 보낸 모터 각각이 "그 송신
        시각 이후에 도착한 응답"을 받으면 끝. 전부 받으면 즉시 빠져나가고, 아니면
        `timeout_s`(기본 `self.drain_s`)에서 끊음. 송신 이전에 도착한 프레임(직전
        주기의 늦은 응답)은 상태에 반영하되 끝 판정에는 안 쓰고 `last_collect.late` 에
        나이와 함께 남김. 한 모터의 응답이 여럿이면 가장 늦게 도착한 것이 상태가 됨.

        `expect` 는 **호환용**임. `send_mit` 없이 부르는 경우(예: 활성화 응답 수거)에만
        예전처럼 프레임 개수로 끝남.

        `timeout_s` 를 안 주면 `self.drain_s` 를 씀.
        """
        wait = self.drain_s if timeout_s is None else float(timeout_s)
        awaited = self._cycle_sent
        skipped = self._cycle_skipped
        sent_this_cycle = self._cycle_open
        self._cycle_sent = ()
        self._cycle_skipped = ()
        self._cycle_open = False
        t0 = time.monotonic()
        stats = CollectStats(awaited=awaited, skipped=skipped)
        fresh: Dict[int, float] = {}
        late: Dict[int, float] = {}
        late_stamp: Dict[int, float] = {}
        seen_any = set()

        if self.completion == "count":
            # 원본 동작: 프레임 개수로 끝남 (비교·되돌리기용)
            frames = self.bus.drain(expect=expect, timeout_s=wait)
            self._ingest(frames, fresh, late, late_stamp, seen_any, awaited=())
        elif not awaited and sent_this_cycle:
            # 이번 주기에 보낸 명령이 하나도 없음(전부 건너뜀·빈 명령): 기다릴 응답이 없으므로 있는 것만 받음.
            frames = self.bus.drain(expect=0, timeout_s=0.0) if getattr(self.bus, "poll", None) is None \
                else self.bus.poll(0.0)
            self._ingest(frames, fresh, late, late_stamp, seen_any, awaited=())
        elif not awaited:
            # 호환 경로: `send_mit` 없이 부르면 개수로 기다림 (원본 동작).
            frames = self.bus.drain(expect=expect, timeout_s=wait)
            self._ingest(frames, fresh, late, late_stamp, seen_any, awaited=())
        else:
            deadline = t0 + wait
            pending = set(awaited)
            poll = getattr(self.bus, "poll", None)
            idle = 0.0                       # poll 이 없는 버스에서 잔 시간 (시계가 안 가는 시험 대역 대비)
            while True:
                remaining = deadline - time.monotonic()
                if poll is not None:
                    frames = poll(max(0.0, remaining) if pending else 0.0)
                else:
                    # poll 이 없는 버스(시험 대역 등): drain 으로 있는 것만 꺼내고 짧게 잠
                    frames = self.bus.drain(expect=None, timeout_s=0.0)
                    if not frames and pending:
                        time.sleep(min(max(0.0, remaining), DEFAULT_POLL_S))
                        idle += DEFAULT_POLL_S
                if frames:
                    self._ingest(frames, fresh, late, late_stamp, seen_any, awaited=awaited)
                    pending = {m for m in awaited if m not in fresh}
                if not pending:
                    break
                if time.monotonic() >= deadline or idle >= wait:
                    stats.timed_out = True
                    self.link.timeouts += 1
                    break

        end = time.monotonic()
        stats.waited_s = end - t0
        stats.fresh = tuple(m for m in self.motor_ids if m in fresh)
        stats.missing = tuple(m for m in awaited if m not in fresh)
        stats.late = tuple(m for m in self.motor_ids if m in late)
        stats.late_age_s = {m: end - late_stamp[m] for m in stats.late}
        stats.fresh_age_s = {m: end - fresh[m] for m in stats.fresh}
        self.last_collect = stats
        self.link.cycles += 1
        self.link.late_replies += len(stats.late)

        # 보내지 않은 모터(명령 안 한 것·건너뛴 것)는 이번 수거에서 아무 프레임이나 왔으면 응답한 것으로 봄.
        ok = set(fresh)
        ok.update(m for m in seen_any if m not in awaited)
        return [mid for mid in self.motor_ids if mid not in ok]

    def _mid_of(self, frame: CanFrame) -> Optional[int]:
        """응답 프레임이 어느 모터 것인지. 번호가 한 모터 전용이면 번호로, 아니면 `data[0]`.

        전용 번호로 왔는데 `data[0]` 이 다르면 어느 쪽도 믿을 수 없으므로 버림 (세어 둠).
        """
        owner = self._mid_by_reply_id.get(frame.can_id)
        if owner is not None:
            if frame.data and frame.data[0] != owner:
                self.link.id_mismatch += 1
                return None
            return owner
        return int(frame.data[0]) if frame.data else None

    def _ingest(
        self,
        frames: List[CanFrame],
        fresh: Dict[int, float],
        late: Dict[int, float],
        late_stamp: Dict[int, float],
        seen_any: set,
        *,
        awaited: Tuple[int, ...],
    ) -> None:
        """프레임들을 해석해 상태를 갱신하고 fresh/late 로 가름. 도착 순서대로 처리함."""
        now = time.monotonic()
        for f in frames:
            mid = self._mid_of(f)
            if mid is None:
                continue
            if mid not in self.motors:
                self.link.unknown_motor += 1
                logger.debug("%s 모르는 모터 응답 id=%d", self.bus.channel, mid)
                continue
            try:
                _, pos, vel, tau, temp = mit.decode_state(f.data, enc=self._encoding[mid])
            except (ValueError, KeyError) as e:
                self.link.undecodable += 1
                logger.debug("%s 해석 실패 %s: %s", self.bus.channel, f.data.hex(), e)
                continue

            stamp = f.stamp or now
            seen_any.add(mid)
            q = self._outstanding[mid]
            if q:
                q.popleft()                     # 응답 하나가 명령 하나를 갚음 (오래된 것부터)

            # 상태는 시간을 거슬러 가지 않음: 이미 더 늦게 도착한 값이 있으면 덮지 않음.
            if stamp >= self._states[mid].stamp:
                self._states[mid] = MotorState(
                    position_deg=pos,
                    velocity_deg_s=vel,
                    torque_nm=tau,
                    temp_c=temp,
                    stamp=stamp,
                )
            if self.latency is not None:
                self._note_received(mid, f.stamp)

            sent = self._tx_at.get(mid)
            if mid in awaited:
                if sent is not None and stamp >= sent:
                    fresh[mid] = stamp
                else:
                    late[mid] = late.get(mid, 0.0)
                    late_stamp[mid] = min(late_stamp.get(mid, stamp), stamp)
            else:
                if sent is not None and stamp < sent:
                    late[mid] = 0.0
                    late_stamp[mid] = min(late_stamp.get(mid, stamp), stamp)
                else:
                    fresh[mid] = stamp

    def refresh_states(self, motors: Optional[Iterable[int]] = None) -> List[int]:
        """상태를 요청하고 응답을 수거함. 응답이 없었던 모터 id 를 반환함.

        게인과 토크가 0인 명령을 보냄 — MIT 모드에는 읽기 전용 명령이 없고, 모터는
        동작 명령의 응답으로 상태를 돌려줌. 토크가 0이라 아무 일도 일어나지 않음.

        제어 루프에서는 이걸 쓰지 않음. 이미 보낸 명령의 응답을 `collect()` 로
        받으면 되므로, 프레임을 두 배로 보낼 이유가 없음.

        **요청 없이 받는 길도 있음** — Command 13(`tables.CMD_ACTIVE_REPORT`)으로
        능동 보고를 켜면 모터가 스스로 상태 프레임을 보냄(기본 10ms 주기,
        `EPScan_time` 으로 조정). 미구현이고, 제어 루프에는 쓸 것이 못 됨 —
        모터 타이머 기준이라 상태 시각이 명령과 안 붙음(이슈 #10). 명령 없이 상태만
        보는 자리(브링업·모니터링)에 맞음.
        """
        ids = resolve_motor_list(motors, self.motor_ids)
        self.bus.flush_rx()
        for mid in ids:
            self._outstanding[mid].clear()      # 비운 큐의 응답은 더 안 옴
        self.send_mit({mid: PASSIVE for mid in ids})
        missing = self.collect(expect=len(ids))
        return [mid for mid in ids if mid in set(missing)]

    def state(self, motor_id: int) -> MotorState:
        """마지막으로 수거된 상태. 새로 읽지 않음."""
        return self._states[motor_id]

    def states(self) -> Dict[int, MotorState]:
        return dict(self._states)

    # ---- 고장 -------------------------------------------------------------
    def read_fault(self, motor_id: int, *, timeout_s: float = 0.01) -> Optional[MotorFault]:
        """고장값을 조회함. 응답이 없으면 `None`.

        조회 전에 큐를 비움 — 고장 응답은 일반 상태 프레임과 CAN ID 가 같아
        구분되지 않으므로, 묵은 프레임이 남아 있으면 그걸 고장값으로 읽음.
        """
        self.bus.flush_rx()
        self.bus.send(
            _command_frame(motor_id, tables.CMD_FAULT, f_cmd=tables.F_CMD_FAULT_QUERY)
        )
        for f in self.bus.drain(expect=1, timeout_s=timeout_s):
            try:
                mid, word = mit.decode_fault(f.data)
            except ValueError:
                continue
            if mid != motor_id:
                continue
            return MotorFault(
                raw=word,
                bits={name: bool(word & (1 << bit)) for name, bit in tables.FAULT_BITS.items()},
            )
        return None
