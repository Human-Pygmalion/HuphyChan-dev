# 상체 자유도 학습 전 보상 배선 감사 (2026-09-01)

> 한 줄 결론: GPU는 비어 있지만 `v2u1`을 그대로 시작하면 안 된다. 팔·허리 관절이
> 자세 보상의 허용범위 표와 모터 정격 표에서 빠져 있어, 오류가 나거나 상체를 사실상
> 못 쓰게 만드는 과벌점이 생긴다.

## ELI5

기존 학습에서는 로봇의 상체 질량·형상은 그대로 두고 허리·어깨 5자유도를 **임시로
고정**했다. 이제 그 고정을 풀려는데, 시험지에는 상체 항목이 생겨도 채점표에는 상체를
어떻게 매길지 빠져 있었다. 게다가 실제로는 20~40 N·m급 모터인데 1 N·m짜리로 채점하고
있었다. 이 상태로 학습하면 로봇이 상체를 안 쓰는 것이 유리하다. 먼저 채점표를 완성한 뒤
짧은 예행연습을 통과시켜야 한다.

## 발견한 결함

1. `variable_posture`는 선택된 모든 관절에 대해 `std_standing`, `std_walking`,
   `std_running` 벡터를 만든다. 상체 자유 모드에서는 pose 대상이 12개에서 17개로
   늘지만 Pygmalion의 walking/running 표에는 하체 12개 패턴만 있다.
2. `thermal_effort`는 정격 표에서 매칭되지 않은 관절을 1 N·m로 둔다. 현재 표에는
   waist/shoulder가 없다. 따라서 RS03 어깨(연속 20 N·m)와 RS04 허리(연속 40 N·m)가
   각각 20배·40배 작은 모터로 채점되고, 제곱 비용은 최대 400배·1,600배 커진다.
3. 고정 상체 기준정책 `v2s1`은 팔을 15도로 벌린 채 용접하지만, 자유 상체의 초기
   shoulder roll은 0도다. 그러면 A/B가 “자유도 유무”뿐 아니라 시작 자세까지 달라진다.

## 변경 원칙

- 새 보상을 발명하지 않는다. 같은 코드베이스의 Unitree G1 상체 값(waist yaw
  walking/running 0.2/0.3 rad, shoulder pitch 0.15/0.5 rad, shoulder roll
  0.15/0.2 rad)을 검증된 기본값으로 재사용한다.
- thermal 정격은 모델에 이미 선언된 모터 계열과 맞춘다: waist RS04=40 N·m,
  shoulders RS03=20 N·m.
- `PYG_ARM_ABD_DEG=15`를 상체 자유 모드의 초기 shoulder roll에도 적용해 두 정책의
  시작 형상을 맞춘다.
- 이 수정 후 1환경 CPU 구성검사와 짧은 GPU 스모크를 먼저 수행한다. 스모크에서
  action=17, actor observation=68, NaN=0, 보상 계산 성공, host swap=0을 확인하기 전에는
  본학습을 시작하지 않는다.

## 간과하기 쉬운 변수

- action/observation 차원 증가 자체, 상체 관절의 action-rate 비용, self-collision 제외,
  초기 팔 자세, upper motor T-N 정격, seed, 명령 커리큘럼, domain randomization 시작점,
  ROM, 센서 관절 순서, 영상·측정 시 같은 `PYG_UPPER_DOF` 토글 사용 여부.
- 단일 seed의 우열은 탐색 결과일 뿐 설계 결론이 아니다. seed 42는 선별 실험이고,
  효과가 보이면 최소 3개 seed로 재현한다.

## 근거

- 코드: `mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py`
- 보상 구현: `mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/rewards.py`
- 모터 정의·초기자세: `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py`
- 비교 기준: `mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/g1/env_cfgs.py`
- 실행 계획: [[103_v2_training_plan]], [[106_session_backlog]], [[107_joint_rom_table]]
