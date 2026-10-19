# slcan 대 gs_usb: "gs_usb 는 slcan 이 못 하는 병렬성을 준다" 주장 사실 확인 (원문 발췌 노트)

작성일: 2026-10-19 · 작성: researcher (Sonnet) · 성격: 조사 기록 (설계 판단 없음)

용어 정의 (처음 나올 때)

- CAN (Controller Area Network): 한 번에 한 프레임만 흐르는 직렬 방송 버스.
- SocketCAN: 리눅스 커널의 CAN 네트워크 계층. `can0` 같은 네트워크 장치로 보임.
- slcan (serial line CAN): CAN 프레임을 ASCII 글자열(Lawicel 방식)로 바꿔 직렬 포트(여기서는 USB CDC-ACM)로 주고받는 방식. 리눅스에서는 `slcand` 가 직렬 포트에 줄 규율(line discipline)을 붙여 `slcan0` 장치를 만든다.
- gs_usb: Geschwister Schneider 사의 USB-CAN 프로토콜을 쓰는 커널 드라이버 (`drivers/net/can/usb/gs_usb.c`). candleLight 펌웨어가 이 프로토콜을 구현한다. 직렬 포트 계층 없이 USB 벌크 전송으로 CAN 프레임(이진 구조체)을 직접 주고받는다.
- USB CDC-ACM (Communications Device Class, Abstract Control Model): USB 가상 직렬 포트 클래스.
- URB (USB Request Block): 리눅스 USB 코어에서 전송 한 건을 나타내는 객체.
- 벌크 전송 (bulk transfer): USB 의 한 전송 종류. 대역폭·지연 보장이 없고, 남는 대역폭을 쓴다.
- SOF (Start of Frame): USB 풀스피드에서 1 ms 마다 호스트가 보내는 프레임 시작 토큰.
- FS (full-speed): USB 12 Mbit/s 속도 등급. STM32G431 의 USB 장치 블록은 FS 이다.
- TX / RX: 송신 / 수신. echo: 보낸 프레임을 송신 완료 시점에 호스트로 되돌려 주는 것(SocketCAN 의 자체 수신 확인).
- berr (bus error): CAN 버스 오류(비트 오류, 스터프 오류 등) 보고.
- ID: CAN 식별자. 중재(arbitration) 우선순위를 결정한다.
- FD (flexible data-rate): CAN FD 규격.
- RTT (round-trip time): 요청을 보내고 응답을 받을 때까지의 왕복 시간.
- ISR (interrupt service routine): 인터럽트 처리 루틴.
- 표기: "스냅샷" = 2026-10-19 에 git.kernel.org / raw.githubusercontent.com 에서 받은 최신 본.

---

## 1. 질문

검증 대상 주장: "USB-CAN 어댑터에서 slcan 대신 gs_usb(candleLight 펌웨어)를 쓰면, slcan 이 허용하지 않는 병렬성이 생긴다."

맥락: RobStride 준직접구동(QDD) 모터 6개, MIT 프로토콜, 11비트 프레임, 1 Mbit/s CAN, 리눅스 SocketCAN. 어댑터는 CANable 2.0 (STM32G431, USB FS), 현재 slcan 펌웨어(normaldotcom canable2-fw), `slcand -o -c -s8`. 라즈베리파이에서 slcan 으로 측정: 모터 6개를 하나씩(송신 후 응답 대기) 하면 한 주기 4.07 ms, 응답은 약 1 ms 단위 USB 묶음으로 도착, 버스 점유 약 18 %. 모터당 응답 안 한 프레임을 약 3개까지만 허용. 모든 모터 응답이 같은 ID(호스트 ID 0xFD)를 쓴다.

답해야 할 5가지: (1) 단일 CAN 버스에서 "병렬"의 뜻, (2) slcan 의 TX 경로, (3) gs_usb 의 TX 경로, (4) USB FS 1 ms 지연의 정체와 slcan 대 gs_usb 왕복 지연 실측 자료, (5) 판정.

## 2. 방법 (실제로 한 검색·취득)

- 커널 소스 직접 취득 (curl, git.kernel.org): `drivers/net/can/usb/gs_usb.c` (master 스냅샷, v6.1, v6.6 태그), `drivers/net/can/slcan/slcan-core.c` (master, v6.1), `drivers/usb/class/cdc-acm.c`/`.h`, `include/linux/can/length.h`, `Documentation/networking/can.rst`. 아래 코드 발췌는 이 파일에서 grep/sed 로 뽑은 글자 그대로이다.
- 펌웨어 소스 직접 취득 (curl, raw.githubusercontent.com): `candle-usb/candleLight_fw` (README.md, include/config.h, include/usbd_gs_can.h, src/main.c, src/usbd_gs_can.c, src/can_common.c, src/can/bxcan.c, src/can/m_can.c), `normaldotcom/canable2-fw` (README, src/main.c, src/can.c, src/slcan.c, src/usbd_cdc_if.c, inc/*.h).
- 웹 문서 취득: canable.io getting-started (curl 후 HTML 글자 추출), Elmue 의 CANable 2.5 펌웨어 문서 (curl 후 글자 추출), Hackaday CANine 로그 (curl 후 글자 추출), Infineon AN57294 PDF (pdftotext), Sojka/Píša RTLWS 논문 PDF (pdftotext).
- 웹 검색 (WebSearch) 질의 약 14회: 위 주제의 영어 질의 (timing analysis Linux CAN drivers, slcan vs candlelight latency, USB FS bulk 19 transactions per frame, CAN frame 111/135 bits, FTDI latency timer, SocketCAN echo, RobStride + gs_usb, linux-can 메일링 리스트 지연 등).
- 한계: WebFetch 요약 도구가 돌려준 문장은 요약기를 거친 것이라, 아래 각 출처에 "원문 직접 확인(직접)" 또는 "도구 요약 경유(경유)" 를 표시했다.

---

## 3. 출처별 기록

### 3.1 질문 1: 단일 CAN 버스에서 "병렬"의 뜻

**S1. Linux Kernel docs, SocketCAN** — `Documentation/networking/can.rst` (직접, git.kernel.org 스냅샷)
URL: https://docs.kernel.org/networking/can.html (원문 파일: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/Documentation/networking/can.rst)

> the CAN bus is a broadcast-only(!) medium that has no MAC-layer addressing like ethernet. The CAN-identifier (can_id) is used for arbitration on the CAN-bus. Therefore the CAN-IDs have to be chosen uniquely on the bus.

뒷받침: CAN 은 한 번에 프레임 하나만 흐르는 방송 매체이고, ID 는 버스에서 유일해야 한다. 같은 ID 를 여러 노드가 쓰는 것은 설계 규칙 위반이다. (질문 5의 응답 ID 충돌 근거.)

**S2. NXP Community, "CAN bus arbitration issue"** (경유, 검색 결과 요약. 페이지 자체는 로봇 차단으로 직접 열람 못 함)
URL: https://community.nxp.com/t5/LPC-Microcontrollers/CAN-bus-arbitration-issue/m-p/514254

검색 도구가 전한 문장: "Bit errors are triggered and both transmissions are repeated." / "bit errors are no indication that the same ID is used". 뒷받침: 같은 ID·다른 데이터가 동시에 송신되면 비트 오류가 나고 양쪽이 재전송한다. **원문 직접 확인 실패.** 그러므로 이 항목은 "미확인(2차)" 으로 둔다. 같은 내용의 1차 근거는 Bosch CAN 2.0 규격이나 확인하지 못했다.

**S3. 프레임 비트 수**: `include/linux/can/length.h` (직접) + 패치 설명 (경유)
URL: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/include/linux/can/length.h
패치: https://lkml.iu.edu/hypermail/linux/kernel/2306.1/08769.html ("[PATCH v5 3/3] can: length: refactor frame lengths definition to add size in bits")

length.h 발췌 (직접):
> * First part of the Inter Frame Space (a.k.a. IMF - intermission field)
> #define CAN_INTERMISSION_BITS 3
> * @intermission: if and only if true, include the inter frame space assuming no bus idle (i.e. only the intermission). Strictly speaking, the inter frame space is not part of the frame. However, it is needed when calculating the delay between the Start Of Frame of two consecutive frames.

패치 설명의 표 (경유, 요약 도구가 표를 그대로 옮김):
> Classic CAN SFF no bitstuffing 111 14 / Classic CAN SFF bitstuffing 135 17 (bits, bytes; maximum data length, including intermission)

뒷받침: 표준(11비트) 프레임, 데이터 8바이트의 선상 길이는 스터프 비트 없이 111 비트(= 108 + 휴지 3), 최악 스터프 시 135 비트. 1 Mbit/s 에서 1비트 = 1 µs 이므로 프레임당 111–135 µs. 사용자가 적은 "111–135 µs" 와 일치한다. 함께 Vector 교육자료(검색 결과 경유)도 최악 스터프 24비트, 132비트 프레임(휴지 제외)이라고 한다: https://elearning.vector.com/mod/page/view.php?id=349 (직접 열람은 안 함).

**"병렬" 네 갈래 정리 (출처 아님, 위 근거로 필자가 분류)**

| 갈래 | 뜻 | 전선 위에서 병렬인가 |
|---|---|---|
| (a) 비행 중 프레임 수 | 어댑터·드라이버가 송신 요청을 몇 개 동시에 붙들고 있는가 (TX 큐 깊이) | 아니오. 전선은 직렬. 호스트와 어댑터 사이만 겹침 |
| (b) 한 장치 다채널 | 한 USB 장치에 CAN 컨트롤러가 여러 개 | 예. 서로 다른 버스이므로 진짜 병렬 |
| (c) 어댑터 여러 개 | 어댑터마다 별도 버스 | 예. 단 우리는 모터가 한 버스에 있음 |
| (d) 호스트 파이프라인 | 응답을 기다리지 않고 다음 요청을 보내고 나중에 모아서 읽음 | 아니오. 호스트 소프트웨어 방식이라 슬캔이든 gs_usb든 가능 |

---

### 3.2 질문 2: slcan 경로

**S4. `drivers/net/can/slcan/slcan-core.c`** (직접, master 스냅샷. v6.1 태그에도 `netif_stop_queue(sl->dev)` 와 `can_put_echo_skb` 0건 확인)
URL: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/net/can/slcan/slcan-core.c

송신 진입점 `slcan_netdev_xmit` (글자 그대로):
```c
	netif_stop_queue(sl->dev);
	slcan_encaps(sl, (struct can_frame *)skb->data); /* encaps & send */
	spin_unlock(&sl->lock);

	skb_tx_timestamp(skb);

out:
	kfree_skb(skb);
	return NETDEV_TX_OK;
```
`slcan_encaps` 끝부분:
```c
	set_bit(TTY_DO_WRITE_WAKEUP, &sl->tty->flags);
	actual = sl->tty->ops->write(sl->tty, sl->xbuff, pos - sl->xbuff);
	sl->xleft = (pos - sl->xbuff) - actual;
	sl->xhead = sl->xbuff + actual;
```
`slcan_transmit` (작업 큐에서 실행, 직렬 포트에 쓸 공간이 생기면):
```c
		/* Now serial buffer is almost free & we can start
		 * transmission of another packet
		 */
		sl->dev->stats.tx_packets++;
		clear_bit(TTY_DO_WRITE_WAKEUP, &sl->tty->flags);
		spin_unlock_bh(&sl->lock);
		netif_wake_queue(sl->dev);
```
버퍼와 등록:
```c
#define SLCAN_MTU (sizeof("T1111222281122334455667788EA5F\r") + 1)
	dev = alloc_candev(sizeof(*sl), 1);
	sl->can.ctrlmode_supported = CAN_CTRLMODE_LISTENONLY;
```
ASCII 부호화 (글자 그대로): `*pos |= 0x20; /* convert R/T to lower case for SFF */`, ID 를 `hex_asc_upper` 3자리, `*pos++ = cf->len + '0';`, 데이터 `hex_byte_pack_upper`, 끝 `*pos++ = '\r';`.

뒷받침 (필자 해석을 구분해서 적음):
- 사실: 프레임 하나를 보낼 때마다 `netif_stop_queue` 를 호출하고, 직렬 쓰기 완료 후 작업 큐(`slcan_transmit`)가 `netif_wake_queue` 를 호출한다. 드라이버가 한 시점에 붙들 수 있는 프레임은 `xbuff` 하나뿐이다 (`xbuff[SLCAN_MTU]` 단일 버퍼).
- 사실: `can_put_echo_skb` / `can_get_echo_skb` 가 파일에 없다. 송신 `skb` 는 직렬 쓰기 직후 `kfree_skb` 로 버려진다. 따라서 버스에서의 송신 성공(echo)은 드라이버가 알 수 없고, `tx_packets` 는 "직렬 포트 쓰기 완료" 시점에 올라간다.
- 사실: 오류 프레임 `'e'` 와 상태 프레임 `'s'` 구문 해석 함수(`slcan_bump_err`, `slcan_bump_state`)가 있다. 그러나 이는 펌웨어가 해당 글자를 보내야 동작한다. 아래 canable2-fw 소스에는 오류·상태 프레임을 내보내는 코드가 없다 (S6).
- 사실: `do_get_berr_counter` 콜백이 없다 (파일 grep 0건). 하드웨어 타임스탬프 없음: 쓰는 것은 `skb_tx_timestamp(skb)` (소프트웨어 송신 타임스탬프) 뿐.
- 프레임당 ASCII 길이 (필자 계산): 표준 프레임, 데이터 8바이트 = `t` 1 + ID 3 + 길이 1 + 데이터 16 + `\r` 1 = **21 바이트**. gs_usb 는 20 바이트(S7). 크기 차이는 작다. 이 비교에서 크기는 결정 요인이 아니다.

**S5. `drivers/usb/class/cdc-acm.c` / `.h`** (직접, master 스냅샷)
URL: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/usb/class/cdc-acm.c

```c
#define ACM_NW  16
#define ACM_NR  16
...
	readsize = usb_endpoint_maxp(epread) *
				(quirks == SINGLE_RX_URB ? 1 : 2);
	acm->writesize = usb_endpoint_maxp(epwrite) * 20;
...
static void acm_write_bulk(struct urb *urb)
...
	acm_write_done(acm, wb);
	spin_unlock_irqrestore(&acm->write_lock, flags);
	set_bit(EVENT_TTY_WAKEUP, &acm->flags);
	schedule_delayed_work(&acm->dwork, 0);
...
	if (test_and_clear_bit(EVENT_TTY_WAKEUP, &acm->flags))
		tty_port_tty_wakeup(&acm->port);
```
뒷받침: 직렬 쓰기 완료 알림(`write_wakeup`)은 USB OUT URB 가 끝난 뒤 작업 큐 두 단계(`acm_softint` 와 `slcan_transmit`)를 거쳐서야 올라온다. 곧 slcan 에서는 "프레임 하나의 USB OUT 전송 완료 → 작업 큐 → 다음 프레임 허용" 이 프레임마다 반복된다. 읽기 쪽은 URB 16개, 한 개 크기 = 엔드포인트 최대 패킷 64 × 2 = 128 바이트. 필자 해석이다: 송신 큐는 Qdisc(`qlen 10` 기본값)에 쌓일 수 있지만 장치로는 한 프레임씩만 나간다.

**S6 (펌웨어 쪽). normaldotcom/canable2-fw** (직접, main 브랜치 스냅샷)
URL: https://github.com/normaldotcom/canable2-fw (파일: src/main.c, src/usbd_cdc_if.c, src/can.c, inc/usbd_cdc_if.h, inc/can.h)

README:
> This repository contains sources for the slcan CANable 2.0 firmware.
> This firmware currently does not provide any ACK/NACK feedback for serial commands.

(두 문장은 WebFetch 요약 도구를 경유해 받음. 경유. 소스 파일은 직접.)

`inc/usbd_cdc_if.h` (직접): `#define NUM_RX_BUFS 8`, `#define TX_LINBUF_SIZE 64 // Set to 64 for max single packet size`, `#define USBTXQUEUE_LEN 2048`. `inc/can.h`: `#define TXQUEUE_LEN 64`.
`src/usbd_cdc_if.c` `cdc_process` (직접): USB 송신이 쉬는 상태(`hcdc->TxState == 0`)일 때 대기 바이트를 최대 64바이트 모아 한 번에 보낸다.
```c
    if(hcdc->TxState == 0)
    {
    	uint16_t linbuf_ctr = 0;
    	while(txbuf.tail != txbuf.head)
    	{
    		tx_linbuf[linbuf_ctr++] = txbuf.data[txbuf.tail];
    		...
    		if(linbuf_ctr >= TX_LINBUF_SIZE)
    			break;
```
`src/main.c` 주 반복문: `led_process(); can_process(); cdc_process();` 를 돌고, 수신 프레임이 있으면 `slcan_parse_frame` 후 `cdc_transmit`.
`src/can.c`: `can_tx` 에 주석 `// Send a message on the CAN bus. Called from USB ISR.`, `can_process` 는 하드웨어 송신 FIFO 에 빈 칸이 있을 때만 소프트웨어 큐(64칸)에서 `HAL_FDCAN_AddMessageToTxFifoQ`. `src/slcan.c` 에서 `.TxEventFifoControl = FDCAN_NO_TX_EVENTS, // don't record tx events`. 타임스탬프·오류 프레임 출력 코드는 `src/slcan.c` 에서 찾지 못함 (grep: timestamp 0건, 오류 레지스터 문자열 `E` 명령만 있음).

뒷받침: 펌웨어는 응답 프레임 여러 개를 한 USB 패킷(최대 64바이트, 21바이트 프레임이면 3개)에 묶어 보낼 수 있다. 이는 사용자가 본 "묶음 도착" 과 모순되지 않는다. 그러나 USB 패킷이 언제 호스트로 넘어가는지는 호스트 컨트롤러가 정한다 (질문 4). 이 펌웨어는 송신 완료 이벤트를 켜지 않는다 (`FDCAN_NO_TX_EVENTS`).

주의: 사용자가 실제로 올린 CANable 2.0 펌웨어 버전은 위 main 브랜치와 다를 수 있다. 미확인.

---

### 3.3 질문 3: gs_usb 경로

**S7. `drivers/net/can/usb/gs_usb.c`** (직접, master 스냅샷. 상수는 v6.1 과 v6.6 태그에서도 동일 확인: `GS_MAX_TX_URBS 10`, `GS_MAX_RX_URBS 30`)
URL: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/net/can/usb/gs_usb.c

```c
/* Only send a max of GS_MAX_TX_URBS frames per channel at a time. */
#define GS_MAX_TX_URBS 10
/* Only launch a max of GS_MAX_RX_URBS usb requests at a time. */
#define GS_MAX_RX_URBS 30
```
송신 `gs_can_start_xmit` (글자 그대로):
```c
	txc = gs_alloc_tx_context(dev);
	if (!txc)
		return NETDEV_TX_BUSY;
...
	usb_fill_bulk_urb(urb, dev->udev,
			  dev->parent->pipe_out,
			  hf, dev->hf_size_tx,
			  gs_usb_xmit_callback, txc);
...
	can_put_echo_skb(skb, netdev, idx, 0);

	atomic_inc(&dev->active_tx_urbs);

	rc = usb_submit_urb(urb, GFP_ATOMIC);
...
		/* Slow down tx path */
		if (atomic_read(&dev->active_tx_urbs) >= GS_MAX_TX_URBS)
			netif_stop_queue(netdev);
```
수신 완료(에코) 처리:
```c
		skb = dev->can.echo_skb[hf->echo_id];
		stats->tx_packets++;
		stats->tx_bytes += gs_usb_get_echo_skb(dev, skb, hf);
		gs_free_tx_context(txc);

		atomic_dec(&dev->active_tx_urbs);

		netif_wake_queue(netdev);
```
RX URB: `for (i = 0; i < GS_MAX_RX_URBS; i++) { ... usb_fill_bulk_urb(urb, dev->udev, dev->parent->pipe_in, buf, dev->parent->hf_size_rx, gs_usb_receive_bulk_callback, parent);`
다채널: `icount = dconf.icount + 1; dev_info(&intf->dev, "Configuring for %u interfaces\n", icount);` 후 채널마다 `gs_make_candev`.
타임스탬프: `/* Timestamp 32 bit timer runs at 1 MHz (1 µs tick). Worker accounts for timer overflow (will be after ~71 minutes) */`, `#define GS_CAN_FEATURE_HW_TIMESTAMP BIT(4)`, `can_rx_offload_queue_timestamp(offload, skb, ts)`.
오류·버스 오류: `#define GS_CAN_FEATURE_BERR_REPORTING BIT(12)`, `dev->can.ctrlmode_supported |= CAN_CTRLMODE_BERR_REPORTING;`, `dev->can.do_get_berr_counter = gs_usb_can_get_berr_counter;` (`GS_CAN_FEATURE_GET_STATE` 지원 시), 수신 경로에서 `if (le32_to_cpu(hf->can_id) & CAN_ERR_FLAG) gs_update_state(dev, cf);`.
프레임 구조체 `struct gs_host_frame`: 머리 12바이트(`echo_id`, `can_id`, `can_dlc`, `channel`, `flags`, `reserved`) + 데이터 8바이트 = 20바이트 (타임스탬프 사용 시 +4 = 24).

뒷받침 (사실):
- 채널마다 최대 **10개** 송신 URB 가 동시에 비행 중일 수 있다 (`GS_MAX_TX_URBS`). 각 프레임에 `echo_id` 를 붙이고, 장치가 되돌려 주는 에코로 송신 완료를 인식한다. 꽉 차야 `netif_stop_queue`.
- 수신 URB 30개가 미리 걸려 있다.
- 채널 수는 장치가 알려 준다 (`icount`). 한 장치에서 `can0`, `can1` 이 생긴다.
- 하드웨어 타임스탬프(1 µs), `berr` 보고, 상태·오류 카운터가 **기능 비트로** 지원된다. 펌웨어가 해당 기능을 광고해야 켜진다.
- 수신 콜백은 URB 하나에서 `gs_host_frame` 하나만 해석한다 (`gs_usb_receive_bulk_callback` 에 프레임 반복문 없음). 곧 **메인라인 gs_usb 는 USB 전송 하나에 CAN 프레임 하나**이다. (Elmue 의 "한 덩어리에 250 프레임" 은 메인라인 드라이버 밖의 확장이다. 3.5 참조.)

**S8. candle-usb/candleLight_fw** (직접, master 스냅샷)
URL: https://github.com/candle-usb/candleLight_fw (README.md, include/config.h, include/usbd_gs_can.h, src/main.c, src/usbd_gs_can.c, src/can_common.c, src/can/bxcan.c, src/can/m_can.c)

README (직접):
> This implements the interface of the mainline linux gs_usb kernel module and works out-of-the-box with linux distros packaging this module, e.g. Ubuntu.
> STM32G431-based devices (e.g. CANable-MKS 2.0) are not supported by this project yet.
> Currently, the firmware sends back an echo frame to the host when the frame is written to the CAN peripheral, and not when the frame is actually sent successfully on the bus. This affects timestamps, one-shot mode, and other edge cases.

소스 (직접):
- `include/config.h`: `#define CAN_QUEUE_SIZE (64 * NUM_CAN_CHANNEL)`: 프레임 풀 64칸 × 채널 수.
- `include/usbd_gs_can.h`: `#define USBD_GS_CAN_RX_BUFFER_COUNT 2` (이중 버퍼 USB 블록), 그 외 1. `struct gs_host_frame_object msgbuf[CAN_QUEUE_SIZE];`
- `src/main.c` 주 반복문: `CAN_SendFrame` → `USBD_GS_CAN_ReceiveFromHost` → `USBD_GS_CAN_SendToHost` → `CAN_ReceiveFrame` 순 반복.
- `src/can_common.c` `CAN_SendFrame`: 호스트에서 온 프레임 목록(`list_from_host`)의 맨 앞 프레임을 `can_send` 로 하드웨어에 넣고, 성공하면 곧바로 `timer_get()` 시각을 단 에코를 `list_to_host` 로 넣는다. 실패하면(빈 칸 없음) 맨 앞에 되돌려 놓는다.
- `src/can/bxcan.c` `can_send`: `can_find_free_mailbox` 로 빈 송신 우편함이 있을 때만 쓴다 (bxCAN 은 우편함 3개). `src/can/m_can.c` (FDCAN, G0B1 등): `HAL_FDCAN_AddMessageToTxFifoQ`, `.TxEventFifoControl = FDCAN_NO_TX_EVENTS`.
- `src/usbd_gs_can.c` `USBD_GS_CAN_SendToHost`/`SendFrame`: `to_host_buf` 가 있으면 즉시 반환하고, 없으면 목록 맨 앞 프레임 하나를 `USBD_LL_Transmit` 한다. 그 전송이 끝나는 `DataIn` 콜백이 `to_host_buf = NULL` 로 비운 뒤에야 다음 프레임을 보낸다. 즉 **장치→호스트는 USB 전송 하나에 프레임 하나**를 순서대로 보낸다 (프레임 개수는 USB 전송 개수와 같음). 엔드포인트는 벌크, 최대 패킷 `CAN_DATA_MAX_PACKET_SIZE` = 64 (FS 장치) 또는 32.

뒷받침 (사실):
- 장치 쪽 송신 대기열: 프레임 풀 64칸을 호스트 방향(`list_from_host`)과 호스트로 보낼 방향(`list_to_host`)이 나눠 쓴다. 하드웨어 송신 우편함이 가득 차면 풀에서 기다린다.
- 에코는 "CAN 컨트롤러에 써 넣은 시점"에 온다 (README 가 명시). 버스에서 실제 송신이 끝난 시점이 **아니다**. 곧 gs_usb 에코는 "프레임이 버스에서 끝났다" 의 증거가 아니다.
- **중요 주의**: 메인라인 candleLight_fw 는 STM32G431 (CANable 2.0, CANable-MKS 2.0 포함)을 **지원하지 않는다고 README 에 쓰여 있다**. 사용자가 "candleLight 로 플래시했다" 고 한 어댑터가 (i) 다른 보드(F042/F072/G0B1)인지, (ii) G431 용 별도 포크(예: Elmue 2.5)인지, 확인이 필요하다. 이 노트의 gs_usb 결론은 메인라인 프로토콜 기준이다. 포크의 TX 큐 깊이·USB 묶음은 다를 수 있다.

---

### 3.4 질문 4: USB 풀스피드 타이밍

**S9. Infineon (Cypress) AN57294 "USB 101: An Introduction to Universal Serial Bus 2.0"** (직접, PDF 를 pdftotext 로 변환)
URL: https://www.infineon.com/dgdl/Infineon-AN57294_USB_101_An_Introduction_to_Universal_Serial_Bus_2.0-ApplicationNotes-v09_00-EN.pdf

> SOF packets are only seen on full- and high-speed devices and are sent every millisecond ... With a High-Speed device, an SOF is sent out every 125 us and frame count is only incremented every 1 ms.
> Full-Speed capable devices support a maximum packet size of 64-bytes.
> Delivery time with a bulk transfer is variable because there is no set aside bandwidth for the transfer. The delivery time varies depending on how much bandwidth on the bus is available, which makes the actual delivery time unpredictable.
> Table 4: Guaranteed Latency — Bulk: No ... Maximum Transfer Size 64 bytes ... Maximum Transfer Speed 1.216 MB/s (Bulk)

뒷받침: FS 프레임은 1 ms. 벌크는 대역폭·지연 보장이 없다. 벌크 최대 전송 속도 1.216 MB/s = 64 B × 19 전송 × 1000 프레임/s (필자 산술, 아래 S10 로 보강). 따라서 **한 프레임(1 ms) 안에 벌크 전송이 여러 번 일어날 수 있다.**

**S10. 보강 (2차): "한 프레임 19회"** — NXP 커뮤니티 글(경유, 검색 요약. 열람 403): https://community.nxp.com/thread/78160 ; ARM Keil 포럼(경유): https://community.arm.com/forums/f/keil-forum/39475/usb-bulk-driver . 검색 도구가 전한 문장: "For FS bulk transfer, 19 full-size packets (transactions) / frame is the theoretical max." 본 규격 USB 2.0 문서(5.8.4절)는 직접 열람하지 못했다. **1차 규격 확인 못 함 (2차 근거만).** 또한 mikrocontroller.net 글은 버스 전체 한도로 다른 수치(8바이트 72회)를 인용한다: https://www.mikrocontroller.net/topic/176736 (경유, 요지만).

**S11. Beyond Logic, USB in a NutShell, 4장** (경유)
URL: https://www.beyondlogic.org/usbnutshell/usb4.shtml
요약기 전달: "Bulk transfers will use spare un-allocated bandwidth on the bus after all other transactions have been allocated." / "For full speed endpoints, the maximum bulk packet size is either 8, 16, 32 or 64 bytes long." 뒷받침: 벌크는 남는 대역폭을 쓴다. (9번과 같은 내용.)

**S12. FTDI 지연 타이머 (유추 근거, 직접 대응 아님)**: 리눅스 커널 커밋 "USB: serial: ftdi_sio: fix extreme low-latency setting" (검색 결과 경유) https://lkml.iu.edu/hypermail/linux/kernel/1703.0/03248.html 와 FTDI 지식 베이스 "1ms is not recommended as this is the same as the USB frame length"(경유). 뒷받침: USB 직렬 장치는 장치 쪽 타이머로 묶음 송신 시기를 정하는 경우가 있다. **CANable2 의 STM32 CDC 는 FTDI 와 구조가 다르다** (`cdc_process` 가 주 반복문에서 USB 송신 쉬는 상태이면 바로 보냄, S6). 유추로만 사용하고, 판정에는 쓰지 않는다.

**S13. Linux dwc2 호스트 NAK 재시도 패치 (Douglas Anderson, 2017-10-25)** (경유, 요약기)
URL: https://lkml.iu.edu/hypermail/linux/kernel/1710.3/01836.html
요약기 전달: "there is no requirement to retry super quickly after a NAK" — 호스트 컨트롤러가 NAK 받은 벌크 IN 을 언제 재시도하는지는 구현에 달렸다. 이 글은 분할 전송(고속 허브 뒤 FS 장치) 사례이고, **라즈베리파이 직결 FS 장치의 NAK 재시도 간격은 이 자료로 확인되지 않는다.** 미확인.

**S14. Hackaday.io, CANine 프로젝트 로그 (Tinymovr), 2021-08-08** (직접, curl 후 글자 추출)
URL: https://hackaday.io/project/168650/log/196365
> Initial tests indicate that CANine-fw pushes performance compared to slcan. It is now possible to do a round-trip message in exactly 1ms, which is the polling time for USB full-speed. With slcan, this was around 1.5 - 2.0 ms for Linux ans Macos, and around 20ms for Windows!
> The new firmware is currently in alpha

뒷받침: **slcan 왕복 1.5–2.0 ms (리눅스·맥) 대 직접 USB(libusb) 펌웨어 1 ms** 라는 단일 보고. 주의: 그 대안은 gs_usb 가 아니라 자체 libusb 펌웨어, 측정 방법·표본 수 미공개, 알파 단계. **gs_usb 에 그대로 일반화할 수 없다.** 그래도 "FS 직접 USB 방식의 왕복 하한이 1 ms 부근" 이라는 방향성의 유일한 수치 근거이다.

**S15. Kvaser 개발자 블로그, "Comparing CAN interface transfer delays"** (경유)
URL: https://www.kvaser.com/developer-blog/comparing-can-interface-transfer-delays/
요약기 전달: "Sending and receiving with a Kvaser PCIEcan 4xHS, I measure a total delay of 225µs." / USB 는 PCIe 에 비해 "it adds an extra delay of 100µs" (Leaf Light HS v2). 변동폭 ±50 % 도 흔하다고 함. 뒷받침: 상용 USB-CAN 어댑터도 호스트 지연이 100 µs 수준 추가라는 한 제조사 측정. Kvaser 는 자체 USB 프로토콜이며 candleLight 와 무관, **사실상 참고용.**

**S16. Sojka, Píša, "Timing Analysis of Linux CAN Drivers" (RTLWS 2011)** (직접, PDF)
URL: https://static.lwn.net/images/conf/rtlws11/papers/proc/p30.pdf
> There exists many CAN drivers for Linux and one of them – Socketcan – is being merged to the mainline kernel. Another often used driver is called LinCAN. ... we provide the results of several experiments that compare latencies introduced by this two drivers.
뒷받침: **이 논문은 SocketCAN 과 LinCAN 의 비교이며 USB 어댑터, slcan, gs_usb 를 다루지 않는다** (pdftotext 전체에서 "usb", "slcan", "gs_usb" 0건). 후속 연구 (Píša 등 ewc24 "Continuous CAN Bus Subsystem Latency Evaluation", https://canbus.pages.fel.cvut.cz:/files/ewc24-can-latester.pdf, 검색 결과로만 확인)도 자체 CAN 컨트롤러 기준이다. **요청하신 자료 유형(slcan 대 gs_usb 왕복 지연의 학술 측정)은 찾지 못했다.**

**S17. 그 밖에 찾지 못한 것 (부정 결과 기록)**: linux-can 메일링 리스트, ODrive·mjbots·Open Dynamic Robot Initiative·RobStride 포럼에서 "slcan 대 gs_usb 왕복 지연"을 숫자로 비교한 글을 검색으로 찾지 못했다. 가장 가까운 것은 tuna-f1sh 의 gist (gs_usb 하드웨어·소프트웨어 타임스탬프 간격 비교) https://gist.github.com/tuna-f1sh/2abe421c2547d41e638253e04e509331 이나 결과 수치가 페이지에 없다 (직접 열람). PCAN-USB 의 1–2 ms 프레임 간 지연 글 https://industrialmonitordirect.com/blogs/knowledgebase/reducing-pcan-usb-latency-for-fast-can-bootloading 은 출처 없는 보고 값이라 근거로 쓰지 않는다.

---

### 3.5 어댑터 제조사·펌웨어 저자의 비교 주장

**S18. canable.io, Getting Started** (직접, curl 후 글자 추출)
URL: https://canable.io/getting-started.html
> This firmware does not use slcan, so it is not interchangable with the stock firmware. However, the CANable appears as a CAN interface natively in Linux, works with the Cangaroo app (below) on Windows, and performs very well under high bus load conditions.
뒷받침: gs_usb 는 `slcand` 없이 네이티브 `can0` 이 된다. 수치 없음. (검색 요약에 있던 "slcand is bypassed entirely" 문구는 이 페이지 본문에서 같은 글자로는 확인하지 못했고, 위 문장이 확인된 내용이다.)

**S19. Elmue, CANable 2.5 firmware (Slcan and Candlelight)** (직접, curl 후 글자 추출, 저자 본인의 주장·제3자 검증 없음)
URL: https://netcult.ch/elmue/CANable%20Firmware%20Update/ · 소스 저장소: https://github.com/Elmue/CANable-2.5-firmware-Slcan-and-Candlelight
> Candlelight uses the slow and misdesigned Geschwister Schneider protocol that sends always 80 bytes over USB for any CAN FD packet no matter how many data bytes it has. ... ElmüSoft Candlelight is 3 times faster than gs_usb.
> The new Candlelight protocol can pack up to 250 frames with up to 2048 byte into one USB blob. Blobs are only sent at very high CAN speeds if there are multiple frames waiting in the USB FIFO.
> The Full Speed USB transfer with 12 Mbit/s is the bottleneck of the CANable adapter.
> Therefore timestamps are not implemented for Slcan , by purpose.
> The maximum count of packets that can be stored in the adapter is 64 + 3 = 67.

뒷받침: (1) 레거시 gs_usb 프로토콜은 FD 프레임이면 항상 80바이트를 보낸다. 클래식 CAN 프레임은 20바이트라서 이 문제는 우리 경우(클래식 8바이트)에는 해당하지 않는다 (S7 구조체). (2) "3배 빠름" 은 CAN FD·고부하 처리량 기준 저자 주장이다. 우리처럼 부하가 낮고 왕복 지연이 문제인 사용에는 직접 적용되지 않는다. (3) 슬캔 펌웨어는 타임스탬프를 일부러 넣지 않았다고 한다. (4) 이 저장소는 G431 용 CANable 에서 slcan 과 Candlelight 두 펌웨어를 지원하는 포크이다. 메인라인 gs_usb 드라이버와의 호환 범위("legacy" 대 "ElmüSoft" 프로토콜)는 직접 확인하지 않았다 (미확인).

---

## 4. 판정표

| # | 주장 (또는 하위 주장) | 판정 | 근거 |
|---|---|---|---|
| 0 | "gs_usb 는 slcan 이 못 하는 병렬성을 준다" (전체) | **부분적으로 참.** 호스트↔어댑터 구간의 비행 중 프레임 수, 에코, 다채널은 맞다. 단일 버스 전선 위의 병렬성은 어떤 방식으로도 불가능하고, 우리 문제(응답 ID 충돌, 모터당 3프레임 한도)는 어댑터로 바뀌지 않는다 | S1, S4, S7 |
| 1 | 단일 CAN 버스는 한 번에 한 프레임 (전선 위 병렬 없음) | 참 | S1 (broadcast-only), S3 (프레임당 111–135 비트) |
| 2 | gs_usb: 채널당 최대 10개 TX URB 동시 비행, 프레임별 echo_id | 참 (v6.1, v6.6, master 동일) | S7 `GS_MAX_TX_URBS 10`, `can_put_echo_skb` |
| 3 | slcan: 드라이버가 프레임 하나씩만 장치로 내보냄 (프레임마다 큐 정지 → USB OUT 완료 → 작업 큐 → 큐 재개) | 참 | S4 `netif_stop_queue`/`netif_wake_queue`, S5 `acm_write_bulk` |
| 4 | slcan: TX echo 없음 | 참 | S4 (`can_put_echo_skb` 0건, `kfree_skb`) |
| 5 | slcan: 하드웨어 타임스탬프 없음, berr 카운터 콜백 없음 | 참 (드라이버 기준). 펌웨어도 타임스탬프 코드를 못 찾음 | S4, S6 |
| 6 | slcan 오류 프레임 지원 | 드라이버는 해석하나, canable2-fw 소스에서 오류·상태 프레임 송출 코드는 못 찾음. 즉 실제 사용에서는 **사실상 없음 (소스 기준)** | S4, S6 |
| 7 | gs_usb: 하드웨어 타임스탬프 1 µs, berr 보고, 다채널 | 참 (펌웨어가 기능 비트를 광고할 때) | S7 |
| 8 | gs_usb 에코 = 버스 송신 완료 | **거짓.** candleLight 는 CAN 컨트롤러에 쓴 시점에 에코한다 | S8 README |
| 9 | 메인라인 candleLight_fw 가 CANable 2.0 (STM32G431) 에서 동작 | **거짓/미지원** (README). 사용자 어댑터의 정확한 펌웨어 확인 필요 | S8 |
| 10 | FS USB 프레임은 1 ms, 벌크는 지연 보장 없음 | 참 | S9 |
| 11 | 한 1 ms 프레임에 벌크 전송이 여러 번 가능 (최대 64B × 19) | 참 (산술·2차 근거). 1차 규격 열람은 못 함 | S9 (1.216 MB/s), S10 (2차) |
| 12 | slcan·gs_usb 모두 모든 전송에 정확히 1 ms 를 더한다 | **근거 없음/미확인.** 둘 다 같은 FS 벌크 위라서 하한 구조는 비슷하나, 사용자가 본 "1 ms 묶음"의 원인(호스트 컨트롤러의 IN 재시도 간격, 펌웨어 송신 시점, 읽기 URB 크기)은 어떤 자료로도 가려지지 않았다 | S5, S6, S13 (모두 간접) |
| 13 | gs_usb 가 slcan 보다 왕복 지연이 짧다 | **미확인.** 직접 비교 측정 자료를 찾지 못함. 방향성 단서: CANine 의 slcan 1.5–2.0 ms 대 자체 libusb 펌웨어 1 ms (gs_usb 아님, 알파, 방법 미공개) | S14, S17 |
| 14 | 학술 논문에 slcan 대 gs_usb 지연 비교가 있다 | **찾지 못함.** Sojka/Píša 는 SocketCAN 대 LinCAN, USB 미포함 | S16 |
| 15 | gs_usb 로 바꾸면 같은 ID 응답 충돌이 없어진다 | **거짓.** 충돌은 CAN 중재 계층 문제이며 어댑터와 무관 (ID 는 유일해야 함) | S1, (S2 2차) |
| 16 | gs_usb 로 모터당 3프레임 한도가 풀린다 | **거짓 (모터 쪽 한도).** 어댑터 사양과 무관. 이 한도 자체는 사용자가 준 값이며 이 조사에서 검증하지 않음 | 사용자 제공, 미검증 |
| 17 | 프레임 선상 시간 111–135 µs (표준, 8바이트, 1 Mbit/s) | 참 | S3 |
| 18 | gs_usb 는 USB 전송 하나에 CAN 프레임 하나 (메인라인) | 참 (호스트 파싱 코드, 장치 펌웨어 모두) | S7 수신 콜백, S8 `SendToHost`. 이는 "쓰루풋이 커지는" 이점이 아니라는 뜻이기도 함 |

---

## 5. 요약 (필자 해석, 판정은 위 표가 우선)

1. gs_usb 가 실제로 개선하는 것: 호스트가 프레임 여러 개를 장치에 연달아 넘길 수 있음 (최대 10), 프레임별 송신 에코(컨트롤러 기록 시점), 1 µs 하드웨어 타임스탬프로 지연을 정확히 측정할 수 있음, 오류·상태 보고, 다채널, `slcand`·직렬 포트 계층 제거.
2. 바꾸지 못하는 것: 버스는 직렬이라 요청 6개와 응답 6개는 어차피 한 줄로 흐른다 (프레임당 111–135 µs). 같은 ID 응답 충돌, 모터당 미응답 3프레임 한도, USB FS 1 ms 프레임 구조.
3. 가장 값싼 검증은 어댑터를 바꾸기 전에 **usbmon**(리눅스 USB 패킷 캡처)이나 gs_usb 하드웨어 타임스탬프(`candump -H`)로 1 ms 묶음의 원인을 직접 보는 것이다. 이 노트는 이를 실행하지 않았다.

## 6. 주의 사항 (최대 5개)

1. 사용자가 적은 "버스 점유 약 18 %" 는 재확인이 필요하다. 요청 프레임 6개 + 응답 프레임 6개 = 12프레임이 모두 8바이트 표준 프레임이면 12 × 111–135 µs = 1.33–1.62 ms 이고, 4.07 ms 주기의 약 33–40 % 이다. 18 % 는 프레임 6개(0.67–0.81 ms)에 해당한다. 프레임 길이(데이터 바이트 수) 또는 센 방식을 확인할 것. (필자 산술)
2. 메인라인 candleLight_fw 는 STM32G431 미지원이다 (S8). 사용자 어댑터의 정확한 보드와 펌웨어(`lsusb -v`, `dmesg`, `ethtool -i can0`, 펌웨어 빌드 이름)를 먼저 확인할 것. 포크마다 TX 큐 깊이와 USB 묶음이 다르다.
3. "1 ms 묶음" 의 원인은 이 조사로 확정되지 않았다. 호스트 컨트롤러의 NAK 재시도 간격(라즈베리파이 직결 FS 장치)에 대한 1차 자료를 찾지 못했다 (S13 은 분할 전송 사례).
4. slcan 대 gs_usb 의 왕복 지연 실측 자료는 공개 자료에서 찾지 못했다. 유일한 수치(S14)는 gs_usb 가 아닌 자체 펌웨어이고 방법이 비공개이다. Elmue 의 "3배" (S19) 는 FD·고부하 처리량에 대한 저자 본인의 주장이다.
5. USB 2.0 규격 원문(19회/프레임, NAK 재시도 규칙)은 열람하지 못했고, 2차 자료와 산술로 대체했다. NXP 포럼 글(S2, S10)은 접속 차단으로 검색 요약만 확인했다.

## 7. 원본 파일 위치

- 소스 사본(휘발성 스크래치패드): `/tmp/claude-1000/-home-syaro-MikuchanRemote-Human-Pygmalion/d73f94c0-8344-44b5-a1d2-d7761a614c28/scratchpad/src/` (gs_usb.c, slcan-core.c, cdc-acm.c, length.h, can.rst, candle_*.c, c2_*.c, an57294.txt, sojka.txt). 세션이 끝나면 사라질 수 있으므로 필요하면 위 URL 에서 다시 받을 것.
