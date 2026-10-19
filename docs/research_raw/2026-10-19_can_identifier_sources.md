# CAN 식별자(ID) 동작 — 원문 발췌 (2026-10-19)

> 질문: 0xFD 를 분리하려면 호스트(PC)를 바꿔야 하나? 모터는 이미 모터 ID 로 분리되어 있나?
> 정리 문서: [[178_can_identifier_and_host_id]]

## S1. Bosch CAN Specification 2.0 (1991)
- 링크: http://esd.cs.ucr.edu/webres/can20.pdf (72 쪽, Bosch 원문 사본. 같은 파일이 can-cia.org, esd-electronics.com 에도 있음). 받은 날 2026-10-19.
- Part A p.6 "Information Routing" — 노드는 주소를 쓰지 않는다:
  > "In CAN systems a CAN node does not make use of any information about the system configuration (e.g. station addresses)."
  > "The IDENTIFIER does not indicate the destination of the message, but describes the meaning of the data, so that all nodes in the network are able to decide by MESSAGE FILTERING whether the data is to be acted upon by them or not."
- Part A p.7 "Priorities", "Arbitration":
  > "The IDENTIFIER defines a static message priority during bus access."
  > "If 2 or more units start transmitting messages at the same time, the bus access conflict is resolved by bitwise arbitration using the IDENTIFIER."
  > "When a 'recessive' level is sent and a 'dominant' level is monitored (see Bus Values), the unit has lost arbitration and must withdraw without sending one more bit."
- Part A p.23 "BIT ERROR" — 중재 구간 밖에서 값이 다르면 오류:
  > "A BIT ERROR has to be detected at that bit time, when the bit value that is monitored is different from the bit value that is sent. An exception is the sending of a 'recessive' bit during the stuffed bit stream of the ARBITRATION FIELD or during the ACK SLOT."
- 함의 (필자 해석): 같은 ID 로 동시에 시작한 두 송신기는 중재 구간에서 갈리지 않는다. 데이터 구간에서 처음 다른 비트에서 recessive 를 보낸 쪽이 BIT ERROR 를 낸다. 오류 프레임 후 둘 다 다시 보낸다.

## S2. Linux kernel `Documentation/networking/can.rst` (master, 2026-10-19)
- 링크: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/networking/can.rst (105 ~ 117 행, "SocketCAN Concept")
  > "the CAN bus is a broadcast-only(!) medium that has no MAC-layer addressing like ethernet. The CAN-identifier (can_id) is used for arbitration on the CAN-bus. Therefore the CAN-IDs have to be chosen uniquely on the bus."
  > "For this reason a CAN-ID can be treated best as a kind of source address."

## S3. RobStride RS04 User Manual 260713 (`Motor_Spec/manuals/RS04_User_Manual_260713.pdf`)
- 6 장 머리: 11비트 ID = bit10~8 "Mode type" + bit7~0 "ID".
- 6.1 Response Command 1 (상태 응답): 11-bit ID `0FD`, 설명 "Host ID". Byte0 = "Motor CAN ID".
- 6.3 Command 1 등 명령: 11-bit ID = "Target motor CAN ID".
- 6.12 Command 9 "Modify Host CAN ID": ID = 대상 모터 ID, 데이터 `FF FF FF FF FF FF F_CMD 01`, "F_CMD: Host CAN ID". 응답은 Response Command 2.
- 매뉴얼에는 Command 9 가 저장까지 하는지 쓰여 있지 않다.

## S4. ISO 11898-1
- 유료 규격이라 열어 보지 못했다. 이 노트의 "ID 는 유일해야 한다" 근거는 S2 와 S1 의 중재·비트 오류 규칙이다.

## S5. 우리 실측
- HUPHY `2fdc139` 개발자 실측 (2026-10-16, 양다리, 토크 0, 100 Hz): 몰아 보내기에서 무응답 주기 39 ~ 43 %, 버스 오류 약 1,300. 하나씩 보내기에서 무응답 0 %, 버스 오류 4. ([[174_findings_digest_bench_can_2026-10-17]] §3-3)
- 벤치 RS04 (2026-10-19): 응답은 모두 11비트 `0x0FD`, Byte0 = 4 ([[174_findings_digest_bench_can_2026-10-17]] §2-2a).
