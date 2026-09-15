# 147. IMU — 지금 어디까지 돼 있고, 꽂으면 무엇을 하면 되나

2026-09-15. 사용자 지시: *"huphy - biped branch 확인해봐. imu 는 ttyacm0? 로 잡힐거고
그것도 다 구현되어있다. 변경사항, imu관련 구현체 전부 확인해"* 이어서
*"imu 빼뒀어. 기록해. 다음에 imu 꽂고 다시말할테니 차분확인해."*

## 0. 한 줄

**로봇 쪽(HUPHY)은 다 돼 있습니다. 브리지(뷰어 쪽)도 다 돼 있습니다.
비어 있는 곳은 우리 송신기 `huphy_live.py` 한 군데뿐이고, 장치는 지금 빠져 있습니다.**

```
[IMU 센서] --(안 꽂힘)--> [HUPHY: 다 됨] --(★ 여기가 빔)--> [전선] --> [브리지: 다 됨] --> [뷰어]
```

## 1. 로봇에 올라간 코드가 biped 최신과 같은가 — 같습니다

IMU 관련 파일 10개를 `origin/biped` 와 해시로 대조했습니다 (로봇의 `~/HUPHY` 에는
`.git` 이 없어서 파일 내용을 직접 비교했습니다).

| 결과 | 파일 |
|---|---|
| 일치 9개 | `sensors/{base,group,registry,__init__}.py` · `sensors/ebimu/{imu,protocol,commands}.py` · `robots/biped.py` · `scripts/imu.py` |
| 다름 1개 | `telemetry/snapshot.py` |

**다른 그 하나도 IMU 와 무관합니다.** 차이는 `lost` 필드 추가 두 군데뿐입니다
(`DIAG_MOTOR_FIELDS` 에 한 개, 모터 상태에 한 줄 — 끊김 판정용이고 `link-health` 쪽
작업입니다). IMU 를 만드는 `build_imu` 는 **글자까지 같습니다.**

biped 의 IMU 계보 (오래된 것부터):

```
175ad76  feat(sensors): 센서 계층과 Xsens MTi IMU
365b219  refactor(sensors): ImuState 를 중력방향과 센서별 extra 로 나눔
6493bb3  feat(sensors): E2BOX EBIMU-9DOF 드라이버
7686e8c  feat(scripts): huphy-imu 진입점
3004db0  feat(sensors): IMU 를 팔다리의 부속에서 로봇의 단위로 분리
```

## 2. 구현체가 실제로 무엇을 하는가

* **`EbimuImu.connect()` 가 자기 읽기 스레드를 띄웁니다** (데몬). 그래서 우리가 주기마다
  IMU 를 따로 읽어 줄 필요가 **없습니다** — 한 번 연결하면 알아서 최신값을 들고 있습니다.
* `Biped.connect()` 가 `self.imus.connect()` 를 부릅니다. `Biped.imu_states()` 는 몸통에
  달린 것과 다리에 달린 것을 **합쳐서** 냅니다.
* 센서 하나를 못 만들어도 **나머지로 그냥 진행합니다** (`registry.make_imus`). 즉 IMU 가
  없다고 로봇이 안 뜨지는 않습니다 — 조용히 빠질 뿐이라, 없는 것을 **눈치채기 어렵습니다.**
* `build_imu(robot, t=...)` 가 내는 열쇠 모양:

```
imu/<이름>/gx  gy  gz          각속도 [도/초]
imu/<이름>/ax  ay  az          가속도 [m/s^2, 중력 포함]
imu/<이름>/grav_x grav_y grav_z  몸체에서 본 중력 단위벡터 (수평이면 0,0,-1)
imu/<이름>/age                 마지막 패킷 이후 [ms]. 한 번도 없었으면 -1
```

설정(`config/robot.yaml`)의 이름이 `main` 이므로 **`imu/main/*`** 로 옵니다.

```yaml
imus:
  main:
    model: ebimu
    port: /dev/ebimu          # ★ 이 링크가 지금 없습니다
    mount: right_leg
    output: [quat, gyro, accel, dist, temp, time]
    accel_mode: gravity
    rate_hz: 100
```

## 3. 브리지(뷰어 쪽)도 이미 돼 있습니다

`tools/pygviewer/pygviewer/bridge/huphy_udp.py` 의 `parse_imu` 가 위 열쇠 모양을
**그대로** 읽습니다 — `gx/gy/gz`(도→라디안 변환 포함), `ax/ay/az`, `grav_*`,
쿼터니언 `qw/qx/qy/qz`, `age`. 손댈 것이 없습니다.

거기 이미 적혀 있는 주의 하나를 옮겨 둡니다: **`-1` 을 "없음" 으로 보는 것은 `age` 와
`sensor_dt` 뿐입니다.** `grav_z` 는 똑바로 선 로봇이 내는 값이 정확히 `-1.0` 이라,
거기까지 `-1 = 없음` 으로 처리했다가 **가장 흔한 IMU 값을 통째로 지운** 적이 있습니다.

## 4. ★ 비어 있는 곳 — `huphy_live.py` 의 `flat()`

`System_ID/huphy_live.py` 의 `flat()` 은 **관절 열쇠만** 냅니다
(`<팔다리>/<모터>/<항목>`). IMU 가 잡혀도 **전선에 안 실립니다.** 실제로 확인했습니다 —
20초 동안 60개 패킷을 받아 보니 열쇠 앞머리가 `left_leg`/`right_leg` 뿐이고
`imu/` 는 하나도 없었습니다.

고칠 방향은 **새로 짜지 않는 것**입니다. HUPHY 자신의 `build_imu(robot, t=...)` 를 불러
그 결과를 `flat()` 이 내는 꾸러미에 합치면 됩니다. 그러면 브리지가 이미 아는 모양이
그대로 나갑니다.

관찰 모드(`observe()`)에서도 IMU 가 연결돼 있어야 하므로, 거기서 `connect()` 가
불렸는지 같이 확인해야 합니다 (모터를 켜는 것과는 **무관**합니다 — IMU 는 읽기 전용
시리얼이고, 관찰 모드의 안전 성질을 바꾸지 않습니다).

## 5. 지금 장치가 빠져 있습니다 — 기준선을 찍어 뒀습니다

사용자가 IMU 를 뽑아 두었습니다. 다음에 꽂고 나서 **차분**을 보면 새로 생긴 자리가
무엇인지 추측 없이 나옵니다.

```
usb_snapshot.py --save 2026-09-15_imu_unplugged     (찍어 둠)
usb_snapshot.py --save <꽂은뒤이름>
usb_snapshot.py --diff 2026-09-15_imu_unplugged <꽂은뒤이름>
```

기준선 (2026-09-15 15:37, IMU 뺀 상태):

```
[tty]    /dev/ttyACM2   /dev/ttyACM3          <- 둘 다 CAN 어댑터
[by_id]  ...CANable..._003E00315550530B20303731-if00   -> can0
         ...CANable..._004100375034570520323530-if00   -> can1
[lsusb]  허브들 + 4K USB Camera + CANable x2   <- IMU 로 보이는 것 없음
```

**`ttyACM` 번호를 믿지 마십시오.** 꽂는 순서로 바뀝니다 — 오늘도 어댑터가 재부팅 뒤
`ttyACM0/1` 에서 `ttyACM2/3` 으로 옮겨 갔습니다. `by_id` 의 일련번호 이름을 쓰거나,
그 이름으로 `/dev/ebimu` udev 링크를 만드는 것이 맞습니다 (`can_up.sh` 가 이미 같은
이유로 `by-id` 를 씁니다).

## 5b. 꽂았습니다 — 실제로 어디로 잡혔나 (2026-09-15 15:39)

차분 결과입니다. **`ttyACM0` 이 아니라 `/dev/ttyUSB0` 였습니다** — 예상과 달랐고,
그래서 기준선을 찍어 둔 것이 값을 했습니다.

```
[tty]    + /dev/ttyUSB0
[by_id]  + usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0
[lsusb]  + Bus 001 Device 013: ID 10c4:ea60 Silicon Labs CP210x UART Bridge
```

udev 속성(실제로 읽은 값, 추측 아님):

```
ATTRS{idVendor}=="10c4"   ATTRS{idProduct}=="ea60"   ATTRS{serial}=="0001"
ATTRS{manufacturer}=="Silicon Labs"
ATTRS{product}=="CP2102 USB to UART Bridge Controller"
```

**센서는 정상입니다.** HUPHY 자신의 파서(`ebimu/protocol.decode`)로 생중계 한 줄을
풀어 확인했습니다:

```
원본  *-0.9844,0.0079,0.0165,0.1750, 0.00,0.00,0.00, 0.037,-0.010,1.001, ... 29.5, 46705

중력방향 = (0.0353, 0.0098, -0.9993)    크기 1.000, 거의 (0,0,-1) -> 거의 수평
각속도   = (0.0, 0.0, 0.0) deg/s        가만히 있음
가속도   = (0.363, -0.098, 9.816) m/s^2 크기 9.82 = g, g->m/s^2 변환도 맞음
```

초당 **100줄**(설정 `rate_hz: 100` 과 일치), 15개 항목이 `output` 순서와 맞습니다.

### 포트를 맞추는 법 — udev 링크 (사용자 선택, 2026-09-15)

`ttyUSB0` 번호는 꽂는 순서로 변하므로 번호를 설정에 적지 않습니다. 대신 링크를 만듭니다.
**루트 권한이 필요해 사람이 실행합니다.**

```
/etc/udev/rules.d/60-ebimu.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", ATTRS{serial}=="0001", SYMLINK+="ebimu"
```

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
ls -l /dev/ebimu          # -> ttyUSB0 을 가리켜야 함
```

★ **주의**: 이 CP2102 의 일련번호가 `0001` 입니다 — 제조사 기본값이라 **같은 칩을 쓰는
장치를 하나 더 꽂으면 규칙이 둘 다 잡습니다.** 그때는 `KERNELS=="1-1.1.3"` 처럼 꽂은
자리로 좁히거나, 센서 일련번호를 바꿔야 합니다. CAN 어댑터들은 `ad50:60c4` 라 안 겹칩니다.

## 6. 남은 순서


1. ~~새로 생긴 자리 찾기~~ ✅ `/dev/ttyUSB0` (§5b)
2. **udev 링크 만들기** — 위 규칙. 루트 권한이라 사람이 실행합니다. ← **여기서 막혀 있음**
3. `python -m huphy.scripts.imu check` 로 센서 설정과 붙은 방향을 본다.
   안 맞으면 `huphy-imu apply --yes` — **센서 비휘발성 메모리에 씁니다.**
4. ~~`flat()` 에 `build_imu` 합치기~~ ✅ 2026-09-15. `StateSender.imu_fields()` 가
   HUPHY 의 `build_imu` 를 그대로 부릅니다. IMU 가 하나도 안 붙으면 **한 번 알리고**
   관절만 보냅니다 — HUPHY 는 조용히 빼고 진행하므로 우리가 말해 줘야 합니다.
5. `huphy_live` 를 다시 띄우고, 전선에서 `imu/main/*` 가 실제로 오는지 본다.
   **이때 돌고 있는 기록이 있으면 끊깁니다** — 기록 중에는 재시작하지 않습니다.
6. 뷰어에서 자세가 실물과 같이 기우는지 본다.
