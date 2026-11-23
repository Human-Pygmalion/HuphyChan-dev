# huphy_rough — 험지 보행 정책 배포 꾸러미

HUPHY LegOnly 이족(모터 12 개, 2-RSU "AB" 발목)이 **속도 명령을 따라 걷는** 정책입니다.
발 주변의 높이 지도를 보고 계단·박스·경사·기울어진 판·징검돌을 넘습니다. 앞뒤·옆·회전 명령을 모두 학습했고, 명령 (0, 0, 0) 이 서기입니다.
꾸러미 자체는 `numpy` 와 `onnxruntime` 만 있으면 돕니다.

- 학습 방법: Terrain-Grounded Imagination (TGI). 시뮬레이터 transition 으로 월드 모델을 학습하고, 상상 rollout 의 return 을 월드 모델을 통해 backprop 해서 정책을 학습합니다.
- 학습 코드: `terrain-grounded-imagination-huphy` 저장소 (브랜치 `patch9-limits`). 학습 설정은 [11 절](#11-학습-설정) 에 있습니다.

## 0. 로봇 쪽에 있어야 하는 것

이 정책은 **odometry estimator 와 mapper 가 붙은 로봇** 을 전제로 합니다. 둘 중 하나라도 없으면 쓰지 않습니다.

| 출처 | 정책에 주는 값 | 절 |
|---|---|---|
| IMU | 자이로, 중력 방향. **IMU 축이 골반 축과 같아야 함** (+x 앞, +y 왼쪽, +z 위). 다르게 붙었으면 회전해서 넣을 것 | 3 |
| 모터 엔코더 | 모터 12 개 각도 (크랭크는 크랭크 모터 각) | 3, 7 |
| **odometry estimator** | 골반 선속도 (골반 좌표계), 월드 좌표 골반 위치·자세 | [5](#5-odometry-estimator) |
| **mapper** | 지면 높이 (월드 지도, 로봇 중심 격자 등 네 형태 중 하나) | [6](#6-mapper) |
| 조작 쪽 (조이스틱, 상위 계획기) | 속도 명령 | 4 |

- **선속도는 대신할 수 없습니다.** 0 이나 속도 명령으로 대신하면 걷지 못합니다 (5 절).
- 높이 지도는 계단·턱을 넘는 데 쓰입니다. 정책은 높이 없이 돌리지 않도록 만들었습니다 (`height=None` 은 오류).

> [!WARNING]
> **기본자세가 huphy_squat 과 다릅니다. 기존 Huphy 꾸러미(`pyg_*`)의 자세와 같습니다.**
>
> 정책은 켜지는 순간 "지금 기본자세로 서 있다" 고 보고 움직이기 시작합니다. 행동 0 이 곧 기본자세이기 때문입니다.
> huphy_squat 의 기본자세로 고정해 두고 켜면 첫 주기의 목표각이 한 번에 튑니다.
>
> | 관절 (R, L 은 부호 반대) | huphy_squat 기본자세 | 이 꾸러미 기본자세 | 차이 |
> |---|---|---|---|
> | hip roll | +0.05 | 0 | 0.05 rad (2.9°) |
> | hip yaw | +0.0088 | 0 | 0.009 rad |
> | crank A | +0.1070 | +0.1440 | 0.037 rad |
> | crank B | −0.1847 | −0.1471 | 0.038 rad |
>
> 켜는 순서와 확인 방법은 [7 절](#7-기본자세-행동-오프셋)과 [8 절 6 번](#8-켜기-전-점검)을 보십시오.

## 1. 들어 있는 것

| 파일 | 내용 |
|---|---|
| `policy_40000.onnx` | 정책 (관측 214 → 행동 12). 관측 정규화가 ONNX 안에 들어 있음 |
| `foot_model_40000.onnx` | FPM (착지점 예측, 입력 36 → 두 발의 다음 착지점 4). 정규화 포함 |
| `deploy_contract.json` | 입출력 명세: 관측 214 칸·행동 12 칸 정의, 기본자세, 높이 격자, 명령 범위, 참조 입출력, 해시 |
| `policy_runner.py` | `RoughPolicy`: IMU + 엔코더 + odometry + mapper + 명령 → 모터 목표각. 직전 행동, 엔코더 이력, FPM, 높이 패치를 안에서 처리 |
| `obs_adapter.py` | 관측 앞부분 48 칸과 FPM 입력 조립 / 행동 → 목표각 변환 |
| `height_scan.py` | mapper 출력 → 높이 패치 166 칸. mapper 출력의 네 가지 형태 ([6 절](#6-mapper)) |
| `example_loop.py` | 가짜 로봇·odometry·mapper 로 도는 50 Hz 루프 예제. 실제 로봇은 이 세 클래스만 바꾸면 됨 |
| `motor_params.json` | 모터별 Kp/Kd, 토크 한계, T-N 곡선, 관절 범위, 관성·마찰 |
| `train_env_manifest.json` | 학습 조건 요약 (지형, 명령, 랜덤화 범위, 노이즈, 지연, 보상 가중치) |

체크포인트는 하나(40,000 iteration)입니다. 학습 체크포인트 원본은 학습 저장소에 있고 (`logs/main_patch9_limits_s0/ckpt_040000.pt`), 해시는 `deploy_contract.json` 에 있습니다.

## 2. 바로 확인

```bash
pip install -r requirements.txt
python3 policy_runner.py --self-test                     # 참조 입출력 대조
python3 policy_runner.py --self-test --dry-run 50        # + 가짜 입력으로 배선 확인 (목표각이 기본자세에서 수 ° 벗어나는 것은 정상)
python3 example_loop.py --seconds 20                     # 가짜 로봇·odometry·mapper 로 배선·주기 확인 (월드 지도)
python3 example_loop.py --seconds 20 --map robot_grid    # mapper 가 로봇 중심 격자를 낼 때
```

`--self-test` 는 세 가지를 봅니다. 허용치는 1e-4 이고, 만든 기계에서의 차이는 괄호 안 값입니다.
1. 정책 ONNX 가 학습 체크포인트(torch)와 같은 행동을 내는가 (2.9e-6)
2. FPM ONNX 가 같은 착지점을 내는가 (6.8e-7)
3. **runner 전체**(관측 조립, FPM, 높이 패치, 정책, 목표각 변환)가 기록된 16 주기의 센서값·높이 격자에서 같은 목표각을 내는가 (8.3e-7)

**여기서 어긋나면 모터를 켜지 마십시오.**

## 3. 코드에 붙이기

```python
from policy_runner import RoughPolicy

pol = RoughPolicy.from_package()
pol.self_test()
pol.reset()                                    # 로봇을 세울 때마다

while True:                                    # 50 Hz (20 ms)
    targets = pol.inference(
        gyro=imu.gyro,                         # [rad/s] IMU 좌표계 (+x 앞, +y 왼쪽, +z 위)
        projected_gravity=imu.gravity_dir,     # 중력 방향 단위벡터, 똑바로 서면 (0, 0, -1)
        joint_pos=encoders,                    # {"R_hip_pitch_joint": rad, ...} 모터 12 개
        base_lin_vel=odom.v_body,              # [m/s] 골반 선속도, 골반 좌표계 (5 절)
        base_pos=odom.pos,                     # [m] 월드 좌표 골반 원점, mapper 지도와 같은 좌표계 (5 절)
        base_quat=odom.quat,                   # (w, x, y, z), 골반 → 월드
        height=mapper.output,                  # mapper 출력 (6 절): WorldGrid 등
        command=(vx, vy, wz),                  # 속도 명령 (4 절)
    )
    for name, q in targets.items():
        driver[name].set_position(q)           # PD 는 드라이버가 (8 절)
```

기록·표시용으로 `pol.last_footholds` (FPM 이 낸 다음 착지점, 골반 yaw 좌표, (R, L)), `pol.last_scan_info` (높이 패치 점과 값), `pol.last_raw_action` 을 볼 수 있습니다.

## 4. 속도 명령 주는 법

**속도 명령은 센서로 재는 값이 아닙니다.** 로봇 쪽에서 "이 속도로 걸어라" 하고 정해 주는 값입니다 (조이스틱, UI, 상위 계획기).

| 항목 | 값 |
|---|---|
| 순서·단위 | (vx [m/s], vy [m/s], wz [rad/s]) |
| 좌표계 | 골반 좌표계. vx 앞, vy 왼쪽, wz 반시계 (위에서 볼 때) |
| 학습 범위 | vx −1.0 ~ 1.5, vy −0.5 ~ 0.5, wz −1.0 ~ 1.0. 밖의 값은 잘라서 넣고 한 번 경고함 |
| 서기 | (0, 0, 0). 학습 환경의 10 % 가 서기 명령이었음 |
| 넣는 주기 | 매 주기(50 Hz). 바뀌지 않아도 같은 값을 계속 넣으면 됨 |
| 바꾸는 법 | 학습 때는 10 초마다 계단식으로 바뀌었으므로 계단 입력도 견딤. 실기에서는 천천히 바꾸기를 권장 (`example_loop.py` 의 `CMD_RATE` 0.5 m/s²) |

학습 때 명령 방식: 환경마다 10 초마다 (vx, vy) 와 목표 방향(heading)을 새로 뽑고, wz 는 heading 오차에서 만들었습니다 (heading 제어). 정책이 보는 것은 wz 뿐이므로, 로봇에서는 wz 를 직접 줘도 되고 heading 을 유지하도록 wz 를 계산해 줘도 됩니다.

## 5. odometry estimator

| 값 | 정의 | 학습 때 품질 |
|---|---|---|
| `base_lin_vel` | 골반(base_link) **원점** 의 선속도 [m/s], **골반 좌표계**. IMU 위치 (원점보다 6.95 cm 아래) 의 속도가 아님. 월드 좌표면 `obs_adapter.world_to_body(v_world, base_quat)` | 참값 + 매 주기 균등 노이즈 ±0.15 m/s (축마다). 지연 없음 |
| `base_pos` | 월드 좌표 골반 원점 [m]. **mapper 지도와 같은 좌표계, 같은 높이 기준** | 지도 어긋남으로 들어감 (6 절) |
| `base_quat` | 골반 → 월드 회전 (w, x, y, z). 높이 패치에는 yaw 만 쓰임 | 위와 같음 |

`base_pos`·`base_quat` 는 월드 좌표 지도 (`WorldGrid`, 월드 좌표 함수) 에만 필요합니다. mapper 가 로봇 중심 격자를 바로 내면 없어도 됩니다.

정책은 이 값을 실제 몸의 움직임으로 믿고 균형과 걸음을 조절합니다.
조금 늦더라도 실제 움직임을 따라가는 추정값이 필요하고, 추정값의 치우침은 그대로 속도 오차가 됩니다.

## 6. mapper

### 6-1. 정책이 받는 값 (최종 입력)

정책 관측의 마지막 166 칸입니다. mapper 가 어떤 형태로 지도를 내든, **결국 이 값이 만들어져야 합니다.**

1. **격자**: 골반 원점을 중심으로, **골반 yaw 만 따라 도는 수평 격자** 를 놓습니다 (roll·pitch 는 따라가지 않음).
   - x(앞) −0.80 ~ +0.80 m, y(왼쪽) −0.50 ~ +0.50 m, 0.05 m 간격 → **33 × 21 점**.
   - 점 (i, j) 의 위치: x = −0.80 + 0.05 i, y = −0.50 + 0.05 j.
2. **착지점**: FPM 이 두 발의 다음 착지점 (fx, fy) 를 같은 격자 좌표로 냅니다 (R, L 순). runner 가 안에서 합니다.
3. **패치**: 발마다 착지점에서 가장 가까운 격자점 (ax, ay) = (round((fx + 0.8) / 0.05), round((fy + 0.5) / 0.05)) 둘레의 **9 × 9 점** 을 봅니다.
   - 점 k = 9a + b (a, b = 0 … 8): i = clip(ax + a − 4, 0, 32), j = clip(ay + b − 4, 0, 20).
4. **값**: h_k = clip(z_pelvis − z_terrain(점 k) − 0.5, −1, 1) [m].
   - z_pelvis 는 골반(base_link) 원점의 높이, z_terrain 은 그 점의 지면 높이입니다.
   - 평지에서 기본자세로 서면 z_pelvis − z_terrain ≈ 0.91 m, 그래서 h ≈ 0.41 입니다.
5. **배치**: 발마다 [fx, fy, h_0 … h_80] (83 칸, fx·fy 도 ±1 로 자름), R 다음 L → 166 칸.

즉 mapper 와 odometry 가 함께 만들어야 하는 것은 **각 격자점에서 "골반이 지면보다 몇 m 위에 있나"** 하나입니다. 9 × 9 패치를 고르는 것은 runner 가 합니다.

| 학습 때 높이 정보 품질 | 값 |
|---|---|
| 지도 | 4 cm 격자, 가장 가까운 칸 값 |
| 칸마다 노이즈 | 매 주기 균등 ±0.10 m (착지점 fx, fy 에도 ±0.10 m) |
| 지도 어긋남 (drift) | 에피소드마다 xy ±0.05 m, z ±0.05 m |
| 지연 | 없음 |

### 6-2. mapper 출력을 넣는 방법

`inference(height=...)` 에 아래 중 하나를 넣습니다. 모두 6-1 의 값을 만들고, 어떤 형태든 결과는 같습니다.

| mapper 가 내는 것 | 클래스 (`height_scan.py`) | odometry 의 `base_pos`·`base_quat` |
|---|---|---|
| 월드 좌표 elevation map (2.5D 지도). `z[i, j]` 는 점 (origin_x + i·res, origin_y + j·res) 의 지면 높이. 가장 가까운 칸 값을 씀 | `WorldGrid(z, origin_xy, resolution)` | **필요** |
| 월드 좌표 질의 함수: 점 (N, 2) → 지면 높이 z (N,). 그냥 callable 을 넣으면 이것 | `WorldHeightFunction(fn)` | **필요** |
| 로봇 중심 격자 33 × 21 (6-1 의 격자), 값 = z_pelvis − z_terrain [m]. 그 모양의 배열을 넣으면 이것 | `RobotGrid(grid)` | 필요 없음 |
| 로봇 중심 질의 함수: 골반 yaw 좌표 점 (N, 2) → z_pelvis − z_terrain (N,) | `RobotHeightFunction(fn)` | 필요 없음 |

```python
from height_scan import RobotGrid, RobotHeightFunction, WorldGrid

height = WorldGrid(elev_map, (x0, y0), 0.04)          # 월드 지도. 새 지도가 오면 height.update(z, origin, res)
height = lambda xy: elevation.query(xy)               # 월드 좌표 함수
height = RobotGrid(rel_33x21)                         # 로봇 중심 격자 (z_pelvis − z_terrain)
height = RobotHeightFunction(lambda xy: ...)          # 로봇 중심 함수
```

- **월드 좌표 형태** 는 odometry 의 `base_pos`·`base_quat` 가 지도와 **같은 월드 좌표계** 여야 합니다. 높이(z)도 같은 기준이어야 합니다. 정책은 z_pelvis − z_terrain 만 보므로, 골반 높이와 지면 높이의 기준이 어긋나면 그대로 오차가 됩니다.
- **로봇 중심 형태** 는 격자가 골반 yaw 를 따라 돈다는 것과, 값이 골반 원점 기준이라는 것을 지켜야 합니다.
- **지도 범위**: `WorldGrid` 는 지도 밖의 점에 가장자리 칸 값을 씁니다. 로봇 둘레 최소 ±0.9 m 를 덮어야 합니다.
- **켤 때 높이 점검**: 리셋 뒤 처음 5 주기 동안 z_pelvis − z_terrain 의 중앙값이 0.5 ~ 1.3 m 밖이면 (서 있으면 약 0.9 m) 모터 목표를 내기 전에 예외를 냅니다. 로봇 중심 형태에 지면 높이 z 자체를 넣었거나, 월드 좌표 형태에서 `base_pos` 의 z 와 지도의 높이 기준이 다를 때 걸립니다.
- **빈 칸 (NaN/Inf, 아직 못 본 곳)**: 그 칸은 같은 발 패치의 유효한 칸들의 중앙값으로 채웁니다. 패치 전체가 비면 그 발의 직전 주기 값을 씁니다. 처음 한 번만 경고하고 계속 돕니다. 리셋 직후에 패치 전체가 비어 있으면 예외를 냅니다 — 지도가 준비된 뒤 켜십시오. 채운 칸 수는 `pol.scan_nan_count` 입니다.

## 7. 기본자세 (행동 오프셋)

**기본자세** 는 `deploy_contract.json` 의 `default_pose_rad` 입니다. 이 값 하나가 세 곳에 그대로 쓰입니다.

1. **목표각의 기준점**: 목표각 = 기본자세 + 0.25 · 행동. 정책이 0 을 내면 목표각이 정확히 기본자세입니다.
2. **엔코더 관측의 기준점**: 관측에는 엔코더각 − 기본자세 가 들어갑니다.
3. **시작 자세**: 정책을 켜기 전에 로봇을 이 자세로 보내 둡니다.

| 관절 [rad] | R | L |
|---|---|---|
| hip pitch | −0.175 | +0.175 |
| hip roll | 0 | 0 |
| hip yaw | 0 | 0 |
| knee | +0.35 | −0.35 |
| crank A | +0.143955 | −0.143885 |
| crank B | −0.147063 | +0.146870 |
| (수동, 참고) ankle pitch / roll | +0.1752 / −0.0001 | +0.1750 / 0.0000 |

- 무릎을 살짝 굽혀 골반은 똑바르고 발바닥은 수평인 자세입니다. 발 간격은 24.7 cm, 골반 높이는 약 0.91 m 입니다 (학습 리셋 높이 0.9115 m).
- 크랭크 값은 발목 pitch 0.175 rad, roll 0 을 만드는 값입니다. 좌우 값이 1e-4 rad 수준으로 다른 것은 모델의 좌우 링크 치수 차이 때문입니다.
- 기존 Huphy 꾸러미 (`pyg_stand_*`, `pyg_rough_main_20261012`) 의 기본자세와 같습니다 (그 꾸러미들의 크랭크 값 +0.1439 / −0.1469 는 2e-4 rad 이내로 같음).
- 발목은 수동 관절입니다. 정책은 발목 각을 내지 않고 크랭크 모터 각만 냅니다.

## 8. 켜기 전 점검

1. **제어 주기 50 Hz.** 20 ms 마다 `inference` 를 한 번씩 부릅니다. 주기가 20 % 넘게 어긋나면 runner 가 한 번 경고합니다.
2. **드라이버 PD 게인**: `motor_params.json` 값을 그대로 설정합니다. huphy_squat 과 같은 게인입니다.
   - 학습에서는 모터마다 ×0.8 ~ 1.2 로 흔들었습니다.
   - 위치 목표만 보냅니다. 속도 목표는 0, 피드포워드 토크도 0 입니다.

   | 모터 | Kp [N·m/rad] | Kd [N·m·s/rad] |
   |---|---|---|
   | hip pitch / hip roll (RS04) | 135 | 6 |
   | knee (RS04) | 130 | 6 |
   | hip yaw (RS03) | 75 | 4 |
   | crank A / B (RS03) | 60 | 2.7 |
3. **크랭크는 모터 각입니다.** 발목 피치·롤 각도가 아닙니다.
4. **관절 0 점과 부호**는 학습한 MJCF 모델 기준입니다. L 관절은 R 의 거울이라 부호가 반대입니다. 축 방향은 `motor_params.json` 의 `joint_axis_in_parent_link` 를 보십시오.
5. **중력 방향 부호**: 똑바로 서면 (0, 0, −1) 입니다. 위 방향 벡터나 가속도 [m/s²] 를 그대로 넣으면 runner 가 리셋 직후 멈추고 알려 줍니다.
6. **시작 자세.** 정책을 켜는 순서는 다음과 같습니다.
   1. 드라이버 PD 로 모터를 **이 꾸러미의 기본자세** (`pol.default_pose`, 7 절 표) 로 천천히 보냅니다.
   2. 양발이 바닥에 닿고, 몸통이 똑바르고 (중력 방향 ≈ (0, 0, −1)), 로봇이 멈춰 있는지 봅니다.
   3. **엔코더각과 기본자세의 차이를 모터마다 확인합니다.** 모두 0.05 rad 이내여야 합니다.
      - 학습에서 리셋 자세는 기본자세 그대로였고 (관절 노이즈 없음), 엔코더 오프셋은 ±0.015 rad 였습니다.
   4. `pol.reset()` 을 부르고, 명령 (0, 0, 0) 으로 정책을 켭니다. 첫 주기부터 정책 목표각을 그대로 보냅니다. 고정 자세 목표와 섞거나 보간하지 마십시오.
   5. 몇 초 동안 제자리에서 안정적으로 서는지 본 뒤에 명령을 천천히 올립니다.

   ```python
   q = read_encoders()
   err = {j: abs(q[j] - pol.default_pose[j]) for j in pol.joint_names}
   worst = max(err, key=err.get)
   assert err[worst] < 0.05, f"{worst} 가 기본자세에서 {err[worst]:.3f} rad 벗어남 — 정책을 켜지 마십시오"
   ```

## 9. 이 코드가 하지 않는 것

- **안전 정지.** 넘어짐 감지, 토크·온도 감시, 비상 정지는 로봇 쪽에서 하십시오. 학습에서는 몸통이 70° 넘게 기울거나 골반이 지면 위 0.45 m 아래로 내려가면 에피소드를 끝냈습니다.
- **PD 게인 설정과 토크 제한.** 드라이버가 합니다.
- **odometry 추정과 지도 만들기.** estimator 와 mapper 쪽 일입니다.
- **관측 노이즈·지연·지도 어긋남을 흉내 내는 것.** 이것들은 학습 때만 넣었습니다 (11-5).

## 10. 알려진 한계

- **실기에서 아직 돌려 보지 않았습니다.** 모든 수치는 학습 환경과 시뮬레이션 기준입니다. 시뮬레이션 결과는 이 꾸러미에 넣지 않았고, 학습 저장소의 `LOCAL_SETUP.md` 에 있습니다.
- 첫 시험은 8 절 6 번의 순서로, 명령 (0, 0, 0) 에서 시작해 천천히 올리십시오.
- 학습에 넣지 않은 실기 차이는 11-5 끝에 적었습니다 (모터 세기, 응답 유실, 주기 흔들림 등).

## 11. 학습 설정

### 11-1. 과제

| | |
|---|---|
| 명령 | (vx, vy, wz), vx −1.0 ~ 1.5 m/s, vy ±0.5 m/s, wz ±1.0 rad/s. 10 초마다 새로 뽑음, heading 제어, 10 % 서기 |
| 지형 | 평지, 피라미드 계단 오르막·내리막 (디딤판 0.33 m, 높이 3 ~ 18 cm), 랜덤 박스 (2 ~ 15 cm), 기울어진 판 (15° 까지), 징검돌 (0.50 m 돌·간격 5 cm 부터 0.33 m 돌·간격 약 22 cm 까지). 발 크기 (0.26 × 0.116 m) 에 맞춤 |
| 커리큘럼 | 지형 종류마다 난이도 10 단계, success-guided sampling 이 난이도를 올림 |
| 주기 | 정책 50 Hz, 물리 5 ms |
| 에피소드 | 20 초 |
| 종료 | 기울기 > 70°, 골반이 지면 위 0.45 m 아래, 구멍에 빠짐, NaN |
| 학습 | 8192 env, 40,000 iteration, seed 0, H200 한 장에서 7.9 시간 |
| 시뮬레이터 | mjlab (MuJoCo-Warp, upstream 20f10e96) |

### 11-2. 로봇과 액추에이터

- **로봇 모델**: jbae49 의 최신 모델 `pygmalion_locomotion/assets/pygmalion_v2/LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts2_capsule7plus5_loop.xml` (26.9 kg, 캡슐 발바닥). 학습에는 시각 메쉬만 뺀 사본을 썼습니다.
- **모터 12 개** (다리마다 hip pitch / roll / yaw, knee, crank A / B). 발목은 2-RSU 병렬 구조이고, 닫힌 링크는 MJCF 의 equality 제약으로 풉니다.
- **명시적 PD + T-N 곡선**: τ = Kp (q\* − q) − Kd q̇ 를 5 ms 마다 계산하고, 측정한 48 V 토크-속도 곡선으로 자릅니다 (구동 방향만, 제동은 최대 토크까지). 게인은 huphy_squat 과 같은 실제 드라이버 값입니다 (8 절 표).
- **모터 동특성**: 벤치에서 잰 armature·마찰·점성 (`motor_params.json`).
- **행동**: 목표각 = 기본자세 + 0.25 · a. hip·knee 는 관절 범위의 90 %, 크랭크는 ±0.95 rad 로 자릅니다.

### 11-3. 관측

| 정책 관측 (214) | 칸 | 학습 노이즈 (균등, 매 주기) |
|---|---|---|
| 골반 선속도 (골반 좌표계) | 3 | ±0.15 m/s |
| 0.2 × IMU 자이로 | 3 | ±0.2 rad/s (배율 전) |
| IMU 중력 방향 | 3 | ±0.05 |
| 속도 명령 | 3 | – |
| 모터 엔코더각 − 기본자세, 2 프레임 (이전, 지금) | 24 | ±0.01 rad |
| 직전 행동 (±10 으로 자름) | 12 | – |
| 높이 패치: 발마다 착지점 2 + 9 × 9 높이 (6-1) | 166 | ±0.10 m |

- **모터 속도는 관측에 없습니다.** 엔코더 두 프레임만 봅니다.
- 자이로·중력·엔코더는 하나의 센서 묶음으로 0 · 20 · 40 ms 지연을 받습니다. 선속도·높이·명령은 지연 없이 넣었습니다.
- **FPM (착지점 예측)**: 입력 36 = 선속도, 자이로, 중력, 명령, 엔코더 두 프레임 (노이즈 없음). 출력 = 두 발의 다음 착지점. 학습 중 실제 착지점으로 계속 학습했고, 정책은 이 착지점 둘레의 높이만 봅니다.

### 11-4. 보상

모두 상태와 행동의 미분 가능한 함수입니다 (월드 모델을 통해 backprop 하기 때문).

| 묶음 | 항 | 가중치 |
|---|---|---|
| 추종 | 선속도 추종 exp(−‖v_xy − v\*‖² / 0.25) / 회전 추종 exp(−(ω_z − ω\*)² / 0.5) | +2.0 / +2.0 |
| 자세 | upright / 서기 명령일 때 기본자세 / 서기 명령일 때 발 들기 | +0.25 / +2.5 / −1.0 |
| | hip roll·yaw 의 기본자세 편차 (L1) | −1.0 |
| 걸음 | gait (발 교대) / swing 발 높이 0.1 m 미만 / 미끄러짐 | +0.25 / −2.0 / −0.1 |
| | 착지점 지지 (발바닥이 지면 밖으로 나간 비율) / 두 발 간격 0.17 m 미만 | −3.0 / −5.0 |
| 안정 | 골반 수직 속도² / 롤·피치 각속도² / 몸 각속도 | −0.5 / −0.05 / −0.05 |
| 제약 | 관절 한계 (hip·knee·크랭크 90 % 범위 밖, 목표각 clip 밖) / 수동 발목 한계 (90 % 범위 밖) | −5 / −10 |
| 하드웨어 | 토크 여유 (PD 토크가 T-N 한계의 0.9 배를 넘은 양) / 발열 Σ(τ / 정격)² | −2 / −0.02 |
| | 회생 전력 relu(−τ·q̇) [W] / 착지 충격 (하강 속도²) | −0.002 / −0.5 |
| 부드러움 | action rate | −0.1 |

### 11-5. 도메인 랜덤화

huphy_squat 과 같은 범위입니다. 모델 파라미터는 **환경마다 시작할 때 한 번** 뽑습니다 (8192 env 가 범위를 덮음). 지연은 에피소드마다 다시 뽑습니다.

| 대상 | 범위 |
|---|---|
| 링크 관성 (pseudo-inertia, CAD 불확실성), 링크 묶음 7 개 | huphy_squat 과 같음 (`train_env_manifest.json`) |
| 짐 (골반 CoM 점질량) | −1 … +8 kg |
| PD 게인 Kp, Kd | 모터마다 × U(0.8, 1.2) |
| 모터 frictionloss / damping / armature | 모터마다 × U(0.8, 1.2) |
| 로드·발목 수동 관절 frictionloss / damping | U(0, 0.5) N·m / U(0, 0.05) N·m·s/rad |
| 발 마찰 | U(0.3, 1.2) |
| 엔코더 오프셋 (관측과 명령에 같이 들어감) | 모터마다 ±0.015 rad |
| IMU 장착 각 (roll, pitch, yaw) | 각 ±2° |
| 명령 지연 / 관측 지연 (IMU + 엔코더) | 0 ~ 20 ms / 0 · 20 · 40 ms |
| 관측 노이즈 | 11-3 표 |
| 지도 어긋남 | 에피소드마다 xy ±0.05 m, z ±0.05 m |
| 골반 밀기 (속도 점프) | 2 ~ 5 초마다, 수평 ±0.45 m/s, 수직 ±0.1 m/s, 회전 ±0.3 ~ 0.4 rad/s |
| 리셋 | 기본자세, 위치 ±0.5 m, 방향 무작위, 몸 속도 ±0.2 |

학습에 넣지 않은 것: 모터 세기 (토크 한계) 랜덤화, 모터 응답 유실, 제어 주기 흔들림, 정지 마찰, IMU 값 멈춤, 발목 링크 유격, 로봇 쪽 목표각 가드.

### 11-6. 네트워크 / 학습

| 항목 | 값 |
|---|---|
| 정책 | MLP 512-256-128 ELU, 관측 정규화 (ONNX 에 포함), 결정적 출력 |
| FPM | MLP 128-128 (LayerNorm, LeakyReLU), 출력 4 |
| 월드 모델 | 결정적 MLP 3 × 1024. 8 스텝 자기회귀 loss, 발 접촉 head |
| 정책 학습 | 32 스텝 상상 rollout 의 return 을 월드 모델을 통해 backprop + critic bootstrap, 좌우 대칭 loss |

## 12. 검증

- `--self-test` (2 절): 정책·FPM ONNX 가 학습 체크포인트와 같은 출력을 내고, runner 전체가 기록된 16 주기에서 같은 목표각을 내는지 봅니다.
- 학습 저장소의 컨트롤러 (학습 파이프라인과 관측이 2e-4 안에서 같음을 단위 테스트로 확인) 와 이 꾸러미에 같은 센서값을 넣으면, 500 주기 동안 목표각 차이가 0 이었습니다 (월드 좌표 함수, 로봇 중심 격자).

## 13. 꾸러미 다시 만들기 (학습 컴퓨터에서)

학습 저장소 `terrain-grounded-imagination-huphy` (브랜치 `patch9-limits`) 에서:

```bash
tools/python.sh tgi/scripts/export.py --ckpt logs/main_patch9_limits_s0/ckpt_040000.pt
tools/python.sh tools/build_huphy_rough.py --run logs/main_patch9_limits_s0 --it 40000 --out ../HuphyChan-dev/deploy_asset/huphy_rough
tools/python.sh tools/verify_huphy_rough.py --package ../HuphyChan-dev/deploy_asset/huphy_rough
```

- `build_huphy_rough.py` 는 ONNX 를 복사하고 `deploy_contract.json` (참조 입출력 포함), `motor_params.json`, `train_env_manifest.json` 을 만듭니다. runner 코드와 이 README 는 꾸러미에 직접 있습니다.
- `verify_huphy_rough.py` 는 12 절의 비교를 다시 합니다.
