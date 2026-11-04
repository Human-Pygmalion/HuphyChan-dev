# 로봇 모델 파일(URDF · MJCF) 폴더

**버전 대장(무엇이 언제 왜 바뀌었나)은 같은 폴더의 `MODEL_VERSION_LEDGER.md` 입니다.** 이 폴더와 mjlab `xmls/` 의 모든 `.urdf`/`.xml` 이름이 그 문서 §5 표에 있고, 빠지면 `python3 tools/robot_model/check_model_ledger.py` 가 알려 줍니다.

- **지금 쓸 최신 모델**: `LegOnly_prototype-tempmass-motormeasured-armfix_v30_proxyfix_footv5_d455_reinf_alparts_loop.{urdf,xml}` (2026-11-04, 25.51 kg).
- MJCF 의 mjlab 용 사본은 `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/` 에 있습니다(그 폴더는 mjlab 저장소 안이라 이 저장소의 git 에 없음).
- 파일은 손으로 고치지 않고 `tools/robot_model/` 의 빌더가 만듭니다. 원본은 지우지 않고 새 이름의 새 파일로 만듭니다.
- 새 모델을 만들면 대장 §6 의 칸을 모두 채워 항목을 더하세요.
- 좌우 이름을 바꾼 일(2026-09-12)로 옮긴 파일은 `legacy/README.md` 에 있습니다.
