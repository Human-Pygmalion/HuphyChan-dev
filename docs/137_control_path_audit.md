# 제어 경로 전수 점검 — 무엇을 중복해 만들었고, 무엇에 얹을 것인가

2026-09-13 · 사용자 지시로 처음부터 다시 봅니다.

> *"개발 시작하기 전에, HUPHY 내에 이미 관련된 구현체 있는지 확인하고, 있으면 그
> 구현체를 써. 만약 문제가 있다면 뭐가 문제인지 확인하고 보고하고 계획해서 거기에
> 래핑해서 개발하고."*

---

## 1. 중복해서 만든 것 — 전수

| 우리가 만든 것 | 이미 있던 것 | 판정 |
|---|---|---|
| `System_ID/huphy_live.py` 의 `StateSender` (UDP JSON 상태 송신) | **HUPHY `telemetry/udp.py`** (`UdpSink`) + `telemetry/snapshot.py` (필드 이름을 정하는 유일한 곳) | **중복.** 설정에 `telemetry.host/port` 가 이미 있고 파이 설정에도 `10.8.0.16:9870` 이 적혀 있었습니다 |
| `System_ID/huphy_log.py` (상태를 CSV 로) | **HUPHY `telemetry/csv_log.py`** (`CsvSink`). 설정의 `telemetry.csv_path` 한 줄이면 켜집니다 | **중복.** 이것 때문에 사용자가 22분 조작한 기록이 안 남았습니다 — 켜 둘 수 있는 기능을 안 쓰고 따로 만들다 놓쳤습니다 |
| `System_ID/huphy_live.py` 전체 (UDP 목표 -> Motion -> HUPHY 가드 -> CAN) | **`tools/pygviewer/pygviewer/bridge/huphy_remote_motion.py`** (1,456줄, 단위검사 있음) — 같은 일을 하고, HUPHY 의 `bringup.build_leg`·`ControlLoop`·**자체 텔레메트리**를 그대로 씁니다 | **중복.** 우리 저장소 안에, 우리가 만든 것이 이미 있었습니다 |
| `System_ID/joint_web.py` 의 원격 조작 화면 | **pygviewer** 의 조작 경로 (`tx.py`, `bridge/tx_client.py`, `bridge/tx_map.py`, `bridge/remote_target.py`) — 지금 **떠 있습니다**(프로세스 4개) | **중복** |
| `System_ID/can_status.py` (상태 읽기) | `huphy-commission --limb X state`, `huphy-bringup` | 부분 중복 (진단 편의는 있음) |

**중복이 아닌 것** (HUPHY 에 없어서 우리가 만든 것):

* 발목 A·B **쌍** 클리핑 — HUPHY 는 모터 공간(AB)에서 두 각을 따로만 자릅니다
* 과열 차단 — HUPHY 는 온도로 아무것도 막지 않습니다
* 통신 건강 판정 고치기 — **HUPHY 안에서** 고쳤습니다 (가지 `link-health-outstanding`)
* 주기별 왕복시간 기록(`loop_timing_log.py`) — 진단용, HUPHY 에 없음

## 2. 그 구현체들의 **문제** (래핑하려면 알아야 하는 것)

### 2a. HUPHY 텔레메트리에 없는 값

`snapshot.py` 가 내는 것: 모터마다 `pos · tgt · err · vel · tau` 와
`temp · age · ack · miss · lost`, 그리고 `missing · since_clip · since_reject`.

**없는 것**: `kp`, `kd`, **통신선에 실제로 실리는 kp**, 관절의 **허용 구간(lo/hi)**,
모터 모델 이름. 화면이 슬라이더를 그리려면 구간이 필요하고, 세기를 숫자로 넣게
하려면 지금 걸린 게인이 필요합니다.

→ **래핑 방법**: 구간과 게인은 **설정에서 오는 값**이라 매 주기 실어 보낼 필요가
없습니다. 화면이 시작할 때 한 번 읽어 오는 자리(`/limits` 같은 것)를 만들고,
초당 스무 번 흐르는 것은 HUPHY 텔레메트리 그대로 씁니다.

### 2b. `huphy_remote_motion.py` 는 **한 다리만** 명령합니다

`--limb` 이 필수이고 한 개만 받습니다. 우리는 12관절을 같이 잡아야 합니다.
(안에서 `Biped` 를 만들기는 합니다 — `ControlLoop` 이 `Robot` 을 요구하기 때문.)

→ **래핑 방법**: `--limb` 을 여러 개 받게 넓히는 것이 맞습니다. CAN 선 두 개를
한 프로세스가 잡아야 하므로 프로세스를 둘로 띄우는 것은 **안 됩니다**(한 선에 두
프로그램 금지).

### 2c. 우리 쪽에만 있는 기능을 옮겨야 합니다

발목 쌍 클리핑 · 과열 차단 · 관절별 게인 변경 · 통신 탐침. 이것들은
`huphy_remote_motion.py` 에 `Motion` 을 감싸는 방식으로 넣을 수 있습니다 —
지금 `huphy_live.py` 에서도 그 모양(`AnklePairGuard(ThermalCutoff(LiveTargets))`)
입니다.

## 3. 프로세스·루프 구조가 왜 불안정한가

오늘 하루의 실제 증상입니다.

| 증상 | 원인 |
|---|---|
| 제어 루프가 계속 죽음 | ① **통신 두절 판정**이 실제로 걸림(오른 발목·오른 엉덩이좌우, 10주기) ② 내가 ssh 래퍼로 띄워서, 래퍼가 제한 시간에 죽을 때 같이 죽음 |
| 로그가 어디 있는지 모름 | `live.log · live2~6.log · rewire.log · rewire2.log · drain3.log · knee.log …` 로 흩어짐. 어느 것이 지금 것인지 파일 이름으로 알 수 없음 |
| 무엇이 떠 있는지 모름 | 확인 방법이 `pgrep` 뿐이고, 그마저 내 셸 래퍼와 겹쳐 잘못 셈 |
| 상태 경로가 둘 | HUPHY 텔레메트리(9870, 안 씀) + 우리 StateSender(9873/9874, 씀) |
| 화면이 둘 | `joint_web`(8098, 씀) + pygviewer(떠 있는데 하드웨어에 안 붙음) |

## 4. 계획 — 순서대로, 각 단계에서 모터가 언제 도는지 명시

**원칙**: 모터에 힘이 들어가는 단계는 **전부 사람 승인 뒤에** 합니다. 아래 (a)(b)는
힘이 안 들어갑니다.

| 순서 | 무엇 | 모터 | 끝났다고 할 수 있는 근거 |
|---|---|---|---|
| **(a)** | **프로세스 한 자리로**: 시작·중지·상태를 명령 하나로. 로그도 한 자리(`~/System_ID/logs/<시각>.log`), "지금 뜬 것" 을 묻는 `status` | 안 돎 | `status` 가 루프·화면·기록기·카메라를 한 화면에 보여 줌 |
| **(b)** | **기록을 HUPHY 것으로**: 설정의 `telemetry.csv_path` 를 채우고 `host/port` 를 쓰기. 우리 `huphy_log.py` 는 폐기 예정으로 표시 | 안 돎 | 파일이 생기고, 열 이름이 `snapshot.field_names()` 와 같음 |
| **(c)** | **명령 경로를 `huphy_remote_motion` 으로**: `--limb` 여러 개 받기 + 우리 기능 네 개 얹기 | **돎** — 승인 뒤 | 같은 입력에 대해 지금 경로와 **같은 패킷**이 나오는지 대조(이미 있는 검사 E15~E18 방식) |
| **(d)** | **화면을 pygviewer 로**: 관절 조작을 거기서. `joint_web` 은 카메라·표만 남기거나 폐기 | **돎** — 승인 뒤 | 화면에서 눌러 관절이 움직이는 것을 영상으로 |
| **(e)** | 통신 두절 문턱 결정 (20 / 끄기 / 배선 수리) | — | 사용자 결정 필요 |

**(c) 를 (d) 보다 먼저** 하는 이유: 화면을 바꾸기 전에 로봇 쪽 경로가 하나여야
합니다. 둘을 같이 바꾸면 어느 쪽이 문제인지 못 가립니다.

## 5. 지금 당장 쓰지 말아야 할 것

* `System_ID/huphy_log.py` — HUPHY `csv_log` 로 대체 예정. **새로 쓰지 않습니다**
* `System_ID/huphy_live.py` 의 `StateSender` — HUPHY 텔레메트리로 대체 예정
* 새 기능을 `huphy_live.py` 에 더 붙이지 않습니다. (c) 에서 옮길 것이 늘어납니다
