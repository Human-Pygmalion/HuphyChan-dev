# 원 학습 코드 (mjlab fork) 자료

배포 policy 는 upstream mjlab `20f10e96` 에서 갈라진 fork 로 학습했다. Fork 의 commit 은 공개 remote 에 없다.
그래서 audit 문서가 인용하는 `MJ/...` 경로 (fork 코드) 를 읽을 수 있도록 아래를 넣었다. **학습 재현에는 필요 없다**
(이 폴더의 `pygmalion_mjlab` 패키지는 순정 mjlab 만 쓴다).

| 파일 | 내용 |
|---|---|
| `mjlab_fork_src_20f10e96_to_383a02ac.patch` | Fork 의 `src/` 전체 변경 (upstream `20f10e96` → fork `383a02ac`, 98 파일, +15,775/−80). Run 이후 `src/` 변경 없음 (audit §1.11) |
| `analysis/launch_rough_main.py` | 배포 policy (rough 단계) launcher. sha256 이 run 의 launch manifest 와 같다 |
| `analysis/train_actor_warmstart.py` | rough 단계 train script (actor-only load, step reset, critic warm-up) |
| `analysis/train_wandb_video.py` | parent flat 단계 (P1, P2) train script. Fork 에서도 git 에 없던 파일 |

Fork 코드를 audit 의 줄 번호대로 보려면:

```bash
git clone https://github.com/mujocolab/mjlab.git mjlab_fork && cd mjlab_fork
git checkout -b pygmalion-fork 20f10e96
git apply /path/to/vibecoded-audit-20261012/fork/mjlab_fork_src_20f10e96_to_383a02ac.patch
cp /path/to/vibecoded-audit-20261012/fork/analysis/*.py analysis/   # analysis/ 는 src 밖
```

각 run 이 실제로 쓴 source 의 snapshot 은 `../reference/<run>/repro_snapshot/` 에 있다 (launch 때 복사된 파일).
