# 중복 구현 정리 — 파일별 수정안

2026-09-13 · `docs/137` 의 점검 결과를 **실제로 고칠 파일 단위**로 옮긴 것입니다.
모터는 한 번도 돌지 않는 단계까지 먼저 하고, 도는 단계는 승인 뒤에 합니다.

---

## 0. 한 장 요약

| 파일 | 줄 | 판정 | 한 줄 |
|---|---|---|---|
| `System_ID/huphy_live.py` | 1,202 | **폐기(단계적)** | 고유 기능 네 개를 옮긴 뒤 지움 |
| `System_ID/huphy_log.py` | 95 | **폐기** | HUPHY 설정 `telemetry.csv_path` 한 줄로 대체 |
| `System_ID/huphy_send.py` | 141 | **폐기** | 뷰어 `/target`·`/script/run` 로 대체 |
| `System_ID/joint_web.py` | 1,353 | **축소** | 제어 경로 떼고 카메라·표만 (읽기 전용) |
| `System_ID/cmd_guard.py` | 605 | **강등·유지** | 제어 아님. 패킷 동일성 검사의 **대조군** |
| `tools/huphy_audit/e2e_live.py` | 1,802 | **재배선** | 62개 검사를 브리지 쪽으로 겨냥 |
| `bridge/huphy_remote_motion.py` | 1,456 | **본체로 승격** | 여기에 얹는다 |
| `bridge/guards.py` | (새로) | **신설** | 발목 쌍·과열·게인, `huphy_live` 에서 이사 |

---

## 1. `System_ID/huphy_live.py` — 폐기, 단 네 조각만 살린다

같은 일을 하는 `bridge/huphy_remote_motion.py` 가 이미 있고, 그쪽이 **HUPHY 자체
텔레메트리를 쓰고 단위검사도 있습니다.** 게다가 오늘 밤 제가 헤맨 "모터가 꺼져
있으면 명령이 아무 일도 안 한다" 를 그 파일은 **2026-09-05 에 이미 측정해서
주석으로 적어** 두었습니다 (`huphy_remote_motion.py:760-782`).

| `huphy_live` 의 조각 | 어디로 | 왜 |
|---|---|---|
| `AnklePairGuard` + `ANKLE_AXIS_DEG` · `ANKLE_ROM_PITCH` · `ANKLE_ROM_ROLL` | **`bridge/guards.py` (신설)** | HUPHY 에 없음. 모터 공간(AB)에서 두 각을 따로만 자르므로 조합을 볼 곳이 필요 |
| `ThermalCutoff` | 〃 | HUPHY 는 온도로 아무것도 막지 않음 |
| `GainControl` (눈금·상한·0 함정) | 〃 + 뷰어 `POST /gains` 에 연결 | 브리지에는 `--kp-max/--kd-max` 상한만 있고 **관절별 변경**이 없음 |
| `LinkProbe` | **`tools/huphy_audit/link_probe.py`** | 진단 도구. 제어 경로에 있을 이유가 없음 |
| `trim_limits` | 브리지 옵션 `--trim-deg` 로 | 한계를 양쪽으로 좁히는 기능은 브리지에 없음 |
| `LiveTargets` · `TargetServer` · `StateSender` | **버림** | 각각 브리지의 목표 보유·`--listen`·HUPHY 텔레메트리와 중복 |
| `pick_joints` · `limbs_of` · `joint_prefix` | **버림** | 브리지의 `--enable` 과 관절 대응표가 같은 일을 함 |
| 켜기 3회 반복 (오늘 추가) | 브리지의 `biped.enable()` 자리에 | 브리지도 한 번만 보내고 확인 안 함 — 오늘 실측 근거와 함께 옮김 |

## 2. `System_ID/huphy_log.py` — 폐기

HUPHY `telemetry/csv_log.py` 가 같은 일을 하고, 열 이름은 `snapshot.py` 가 단독으로
정합니다. 켜는 법은 설정 한 줄입니다.

```yaml
telemetry:
  host: 10.8.0.16
  port: 9870
  csv_path: /home/maido/System_ID/logs/run.csv   # <- 이 줄
  csv_flush_every: 50
```

**이걸 안 써서 사용자가 22분 조작한 기록이 안 남았습니다.**

## 3. `System_ID/huphy_send.py` — 폐기

목표를 천천히 보내는 손도구인데, 뷰어에 `POST /target` 과 `POST /script/run`
(`scripts/*.json` 시퀀스 재생)이 이미 있습니다. 파이에서 CLI 로 보내야 할 때는
`tools/pygviewer/scripts/` 의 기존 스크립트를 씁니다.

## 4. `System_ID/joint_web.py` — 카메라·표만 남긴다

| 뗄 것 | 남길 것 |
|---|---|
| `--command`/`--live` 경로 전체 (`RemoteGuard`, `/cmd`, `/release`, `/gains`, 슬라이더·숫자칸·잠금) | 벤치 카메라(`/cam.mjpg`), 12행 상태표, 통신 상태, 온도 경고 |
| `cmd_guard` import | |

조작은 pygviewer 조종판(`/tx/config → /tx/enable → /tx/arm → /target`)에서 합니다.
**같이 고쳐야 하는 파일**: `tools/pygviewer/pygviewer/scenario.py` 가 프로그램 셋
(`joint_web(read-only)` · `joint_web(dry-run)` · `joint_web(live)`)을 들고 있으므로,
뒤의 둘을 지우고 읽기 전용 하나만 남깁니다. 대응 검사 `tests/test_scenario.py` 도.

## 5. `System_ID/cmd_guard.py` — 지우지 않고 **강등**

패킷 동일성 검사(E15~E18)가 "우리 구현과 HUPHY 가 같은 바이트를 내는가" 를 보는데,
그 **대조군이 이 파일**입니다. 지우면 그 검사가 의미를 잃습니다.

* 머리말에 **"제어 경로 아님. 검사 대조군"** 을 박습니다
* `joint_web.py` 의 import 를 뗍니다 (4번과 함께)
* `System_ID/test_cmd_guard_homing.py` 는 그대로 둡니다

## 6. `tools/huphy_audit/e2e_live.py` — 62개 검사를 다시 겨냥

지금 `huphy_live` 의 이름 43곳을 씁니다. 옮긴 뒤에는:

| 지금 | 바꾼 뒤 |
|---|---|
| `import huphy_live as HL` | `from pygviewer.bridge import guards` |
| `HL.AnklePairGuard` · `HL.ThermalCutoff` · `HL.GainControl` | `guards.*` |
| `HL.LiveTargets` · `HL.TargetServer` · `HL.StateSender` | 브리지의 목표 보유·수신·HUPHY 텔레메트리 |
| `HL.pick_joints` · `HL.trim_limits` | 브리지의 `--enable` · `--trim-deg` |
| E43~E47 (화면 배선) | pygviewer 조종판 쪽으로 다시 겨냥 |

**검사 내용은 하나도 안 줄입니다.** 62개가 그대로 통과해야 다음 단계로 갑니다.

## 7. 손대지 않는 것

`can_status.py` · `can_off.py` · `record_cam.py` · `loop_timing_log.py` ·
`test_cmd_guard_homing.py` · HUPHY 가지 `link-health-outstanding`.

오늘 `/tmp` 에 급하게 만든 탐침 둘(켜기 확인·토크 확인)은 **`System_ID/motor_probe.py`
하나로** 정리해 리포에 넣습니다 — 같은 것을 또 만들지 않도록.

---

## 8. 순서와 확인 방법

| | 단계 | 모터 | 통과 기준 |
|---|---|---|---|
| 1 | `bridge/guards.py` 신설 + 네 조각 이사 + 단위검사 | 안 돎 | 새 단위검사 통과 |
| 2 | 브리지에 `--limb` 여러 개 · `--trim-deg` · guards 연결 · 켜기 3회 | 안 돎 | `--dry-run` 으로 12관절 명령이 나오는 것 확인 |
| 3 | `e2e_live.py` 재배선 | 안 돎 | **62개 지킴 59 이상 · 구멍 0** |
| 4 | HUPHY 설정: 텔레메트리 9870 + `csv_path` | 안 돎 | 파일이 생기고 viser 화면에 값이 들어옴 |
| 5 | `joint_web` 축소 · `scenario.py` 정리 | 안 돎 | pygviewer 검사 51개 통과 |
| 6 | `huphy_live.py` · `huphy_log.py` · `huphy_send.py` 삭제 | 안 돎 | 남은 참조 0 (`grep`) |
| 7 | 실물에서 브리지로 한 관절 움직이기 | **돎** | **사용자 승인 뒤**, 녹화하며 |

1~6 은 모터가 한 번도 돌지 않습니다. 7 만 승인이 필요합니다.

## 9. 위험

* **검사 62개가 잠시 깨집니다** (3단계). 그 사이에는 실물을 만지지 않습니다.
* 문서 여러 곳이 `huphy_live.py` 를 가리킵니다 (`docs/128·132·133·137`,
  실험 노트 4개). **문서는 그때의 기록이므로 고치지 않고**, 6단계에서 "이 파일은
  `bridge/huphy_remote_motion.py` 로 옮겨졌다" 는 줄만 `docs/137` 에 남깁니다.
* pygviewer 는 다른 세션이 만든 부분이 있습니다. `bridge/` 는 우리 몫이지만
  `scenario.py` 를 건드리므로 `session_bus` 에 알립니다.


---

## 10. 정정 (2026-09-13, 1단계 직후)

**1단계에서 제가 또 같은 실수를 했습니다** — 옮기기 전에 목적지를 안 봤습니다.

| 조각 | 계획 | 실제 | 조치 |
|---|---|---|---|
| `ThermalCutoff` | `guards.py` 로 이사 | **브리지에 이미 있음** (`bridge/motor_fault.py`). 그쪽이 더 강함: 50도 차단·45도 복귀 이력이 있고, 목표를 묶는 데 그치지 않고 **게인을 0 으로** 내림. `huphy_remote_motion._update_thermal_cutoff` 가 매 주기 돌리고 `{관절}/temp_valid`·`{관절}/cutoff` 를 관측으로 냄 | **되물렸습니다.** `guards.py` 에서 지우고 왜 지웠는지 주석으로 남김 |
| `GainControl` | 이사 | 브리지에 `plan_gains()` 가 있으나 **메시지마다 들어오는 kp/kd** 를 계획하는 것이고, "운용 중에 사람이 관절별로 바꿔 두는" 것은 없음 | **유지**. 다만 둘이 같은 `leg.config.motors` 를 건드리므로 2단계에서 **한쪽으로 몰아야** 함 |
| `AnklePairGuard` | 이사 | 브리지는 **AB → RP 로 바꿔서** HUPHY 에 넣음 (`solve_fk` 로 pitch/roll 을 만들어 `action["ankle_pitch"]`). 우리 것은 **AB 공간 그대로** 자름 | **설계 결정 필요** — 아래 |
| `LinkProbe` | 이사 | 중복 아님 | 유지 |

### 발목을 어느 공간으로 줄 것인가 — 물어봐야 하는 것

* **브리지 방식 (AB → RP 변환)**: HUPHY 의 발목 역기구학·도달성 검사를 그대로 탐.
  대신 크랭크 각을 한 번 변환했다 되돌리므로 정책이 낸 AB 값과 **정확히 같은 크랭크
  각이 나간다는 보장이 없음**.
* **우리 방식 (AB 그대로)**: 정책이 낸 값이 그대로 나감. 대신 HUPHY 가 A·B 를
  따로만 자르므로 **조합을 우리가 봐야 함** (`AnklePairGuard` 가 그 일).

사용자가 2026-09-12 에 **AB** 로 정했습니다(*"AB. Huphy 지금 구현체도, 실제 Policy 도
그 공간으로 돌잖아"*). 그러면 브리지의 변환 경로를 AB 직통으로 바꾸고 그 자리에
`AnklePairGuard` 를 넣는 것이 맞습니다. **2단계에서 이것부터 확인하고 시작합니다.**
