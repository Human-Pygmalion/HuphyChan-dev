# legacy/ — 좌우 이름 바꾸기(2026-09-12) 이전 것들

2026-09-12 커밋 `a98d0d9`("name the legs after the side they are actually on")에서 `L_`/`R_`
접두어가 실제 다리 쪽과 맞도록 바뀌었다. **+y 가 왼쪽**이고, 그전까지 `L_` 는 −y = 오른다리에 있었다.

모델 파일(xml·urdf)은 **전부 제자리에서 고쳐졌으므로 여기에 없다.** 확인함(2026-09-13):
자산 폴더 19+19개, 에셋주 폴더 32개 모두 `L_` 엉덩이가 +0.1237 = 왼쪽. 옛 규약 파일 0개.

여기 있는 것은 이름을 따라가지 못해 **틀린 다리에 붙게 된 파일**들이다.

| 파일 | 무엇이었나 | 왜 옮겼나 |
|---|---|---|
| `LegOnly_..._loop_bent.prerename.json` | 웅크림 시작 자세의 발목 폐루프 각도 풀이값(2026-09-05 풀이) | 관절 **이름**으로 저장돼 있어 이름 바꾸기 뒤 크랭크 값이 반대 다리에 걸렸다. 리셋마다 폐루프가 **37.04 mm** 찢긴 채 시작했다(원래 0.00 mm) — v3_keyfix가 없앴던 바로 그 결함(당시 37.23 mm). 2026-09-13에 `tools/robot_model/loop_bent_keyframe.py --tag <tag>_loop` 로 다시 풀어 0.00000 mm 복구 |

## 같이 옮긴 것 (다른 폴더)

* `mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/xmls/legacy/_scene_tmp.xml`
  — 도구가 만든 임시 장면 파일. 이름 바꾸기 뒤에도 옛 규약(`L_` 가 −0.0709)으로 남아 있었고
  이름으로 참조하는 코드가 없다.
* `mujoco-sim/mjlab/analysis/out/legacy_prerename/legonly_ab_v4_18099_{clean,push}.{npz,mjb}`
  — 이름 바꾸기 시각(22:04)을 **가로질러** 측정된 짝이라 못 쓴다. 앞 절반은 옛 이름,
  뒤 절반은 새 이름 + 폐루프 37 mm 찢김. 2026-09-13에 다시 잰다.

## 옮기지 않은 것

* `../pygmalion_v3_printed_loop_bent.json` — `PYG_BENT_KEYFRAME_LEGACY=1` 이 읽는 v3 빌드용
  풀이값. 이 파일은 좌우 크랭크가 **둘 다 같은 값**(≈ −0.299)이라 이름을 맞바꿔도 물리적으로
  0.0003 rad 차이뿐이다. 2026-09-05 이전 정책을 재생하는 경로라 그대로 둔다.

## 읽는 사람에게

이 폴더의 파일을 학습이나 측정에 쓰지 말 것. 과거에 무슨 값이었는지 확인할 때만 본다.
좌우 이름의 내력은 `docs/136_left_right_naming_history.md` 에 있다.
