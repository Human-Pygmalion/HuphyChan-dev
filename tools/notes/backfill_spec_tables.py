"""Backfill §1b (rewards + Kp/Kd) and §1b-2/3/4 (actuator limits · ROM/action window ·
PYG_* stack flags) into EVERY training-run note under docs/experiments/.

User instruction 2026-09-03: "이건 모든 Docs에 다 넣으라고" — the four tables that
2026-09-03_legonly_ab_v2.md §1b~§1b-4 carries must exist in every run note, and must
be generated from that run's own saved config, never retyped.

  note -> run dir   : the timestamped token the note itself cites, else the run whose
                      directory name ends with the note's run name.
  run dir -> tables : mujoco-sim/mjlab/analysis/run_spec_tables.py (single source of
                      the arithmetic; also used by the launcher for NEW runs).
  run dir -> model  : repro/ snapshot when the run kept one; otherwise MODEL_RULES
                      below, each entry carrying the evidence it rests on.  Anything
                      the evidence does not settle stays "모델 미해석" — never guessed.

Usage:
  python3 tools/notes/backfill_spec_tables.py [--dry-run] [--force] [--only <substr>]
  python3 tools/notes/backfill_spec_tables.py --inventory     # table only, no writes
  python3 tools/notes/backfill_spec_tables.py --only-1b [--dry-run]
      # refresh §1b only (e.g. after a reward_gains_table fix); §1b-2/3/4 untouched
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "experiments"
MJLAB = ROOT / "mujoco-sim" / "mjlab"
PY = MJLAB / ".venv" / "bin" / "python3"
TOOL = MJLAB / "analysis" / "run_spec_tables.py"
LOG_ROOTS = [
    MJLAB / "logs/rsl_rl/pygmalion_velocity",
    ROOT / "pygmalion_locomotion/logs/rsl_rl/pygmalion_flat",
    ROOT / "pygmalion_locomotion/logs/rsl_rl/pygmalion_rough",
]

# Notes that summarise several runs (or none): they get a pointer, not a table.
AGGREGATE = {
    "INDEX.md": "학습 실험 대장(전체 목록)",
    "sweep_gear_ratio.md": "감속비 스윕 4런 종합",
    "2026-06-30to07-07_pre-flat25_backfill.md": "14개 런 일괄 소급 노트",
    "2026-07-09to10_superseded_runs.md": "폐기된 6개 런 정리",
    "2026-07-11_bentinit_ab_plan.md": "A/B 실행 계획(런 없음)",
    "2026-07-12_bentinit_ab_result.md": "bentinit A/B 2런 비교",
    "2026-08-26_ankleAB_vs_RP_comparison.md": "AB↔RP 계열 비교",
    "2026-08-28_ab_ankle_usage_audit.md": "다수 정책 발목 사용 감사",
    # 2026-09-09~10 벤치 모터 측정 노트. **학습 런이 아니다** — 실물 모터 한 대를
    # 라즈베리파이에 직결해 잰 기록이라 대응하는 logs/rsl_rl 런 디렉터리가 없고,
    # §1b-2(액추에이터 한계)·§1b-3(ROM/행동창)·§1b-4(PYG_* 플래그)는 학습 설정에서
    # 나오는 표라 여기 붙일 근거가 없다. 대신 각 노트가 자기 실험 조건(모터 번호·이득·
    # 진폭·온도·경계·표본수)을 머리에 적는다.
    "2026-09-09_motor_rotor_inertia.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_motor_friction_retest.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_motor_report_and_timing.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_motor_sysid_attempt.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_sim_vs_real_sine_sweep.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_pd_step_chirp_sim_vs_real.md": "벤치 모터 측정(학습 런 아님)",
    "2026-09-09_pd_ablation_five_commands.md": "벤치 모터 측정(학습 런 아님)",
}

# note stem -> (p1 run dir name, p2 run dir name or None). Only where the automatic
# token/name match needs help; everything else is resolved by find_runs().
MANUAL_RUNS = {
    # docs/62 + memory: the note's "05-14-13" is a typo for the 05-16-24 P2-final run.
    "2026-07-07_P2_final_flat": ("2026-07-07_05-16-24", None),
    "2026-07-07_P2_final_analysis": ("2026-07-07_05-16-24", None),
    "2026-07-09_rough_p2_final": ("2026-07-09_16-21-00_rough_p1_nodr",
                                  "2026-07-09_21-09-15_rough_p2_dr"),
    "2026-09-03_legonly_gait_kinematics": ("2026-09-03_02-47-35_legonly_ab_v1_p1",
                                           "2026-09-03_08-44-28_legonly_ab_v1_p2"),
}

V3 = "pygmalion_v3_printed_loop.xml"
V3S = "pygmalion_v3_printed.xml"
V4 = "pygmalion_v4_printed_loop.xml"
# v4_printed_loop.xml was created 2026-08-26 21:22; a run started before that could
# not have loaded it. That is the objective boundary the v3 attributions rest on.
V4_BORN = "2026-08-26_21-22"

SERIAL_SRC = (
    "이 시기 `pygmalion_constants._XML_NAME` 기본 분기 — `PYG_V2`/`PYG_HIP_CANT*`/"
    "`PYG_ROLLOFF30` 미설정 시 `pygmalion.xml`. 노트의 hip_roll 하드스톱 진술"
    "(외전 −45° / 내전 +25°)과 이 파일의 range가 일치"
)
V3_SRC = (
    "v3 printed 계열 — 노트 [[2026-08-23_ankleAB_c2]]/[[2026-08-23_ankleRP_c2]]가 "
    f"선언한 모델이며, v4 XML은 {V4_BORN}에야 생성되어 이 런이 로드할 수 없었다(객관 상한)"
)
V34_SRC = (
    "v3/v4 printed 폐루프 **계열** — 두 XML의 관절 range는 완전히 동일하므로 아래 ROM 표는 "
    "확정이다. 다만 이 런이 v3(35.35 kg)인지 v4(31.316 kg)인지는 런에 기록이 없어 "
    "**질량은 미해석**"
)


def model_rule(run: str):
    """-> (xml or None, source label or None). None = let the tool auto-resolve."""
    d = run[:16]  # 2026-08-26_21-22
    if run < "2026-08-01":
        if "cant30" in run:
            return "pygmalion_cant30.xml", "노트 선언 `PYG_HIP_CANT30=1`; " + SERIAL_SRC
        if "cant20" in run:
            return "pygmalion_cant20.xml", "노트 선언 `PYG_HIP_CANT20=1`; " + SERIAL_SRC
        if "rolloff30" in run:
            return "pygmalion_rolloff30.xml", "노트 선언 `PYG_ROLLOFF30=1`; " + SERIAL_SRC
        if run.startswith(("2026-06-30", "2026-07")):
            return "pygmalion.xml", SERIAL_SRC
        return None, None  # IsaacLab era: usd_path / repro snapshot
    if "bundleV4" in run or "v2s1" in run:
        return V4, ("노트 [[2026-08-27_bundleV4_AB]]/[[2026-08-28_v2s1_AB]] 선언 "
                    "(`PYG_MODEL_V4=1`, 31.316 kg)")
    if run.startswith("2026-09"):
        return None, None  # repro snapshot / launch manifest
    rp = "RP" in run
    if d < V4_BORN:
        return (V3S if rp else V3), V3_SRC
    return (V3S if rp else V3), V34_SRC


def find_runs(stem: str, text: str):
    runs = {}
    for r in LOG_ROOTS:
        if r.is_dir():
            for d in os.listdir(r):
                runs.setdefault(d, r / d)
    if stem in MANUAL_RUNS:
        p1, p2 = MANUAL_RUNS[stem]
        return runs.get(p1), runs.get(p2)
    toks = [t for t in sorted(set(re.findall(
        r"2026-\d\d-\d\d_\d\d-\d\d-\d\d(?:_[A-Za-z0-9_]+)?", text))) if t in runs]
    pick = [t for t in toks if t == stem or stem.startswith(t)]
    if not pick:
        date = stem[:10]
        base = re.sub(r"^2026-\d\d-\d\d_", "", stem)
        tails = [base, re.sub(r"_aborted$", "", base),
                 re.sub(r"_(AB|RP)$", "", base)]
        for tail in dict.fromkeys(tails):
            hits = [k for k in runs
                    if k.endswith("_" + tail) or k.endswith("_" + tail + "_p1")]
            # the note's own date disambiguates re-launches of the same run name
            same_day = [k for k in hits if k.startswith(date)]
            pick = same_day or hits
            if pick:
                break
    if not pick:
        return None, None
    p1 = sorted(pick)[0]
    p2 = None
    if p1.endswith("_p1"):
        # 2026-09-03_02-47-35_legonly_ab_v1_p1 -> any run ending legonly_ab_v1_p2
        name = re.sub(r"^2026-\d\d-\d\d_\d\d-\d\d-\d\d_", "", p1)[:-3]
        cand = sorted(k for k in runs if k.endswith(name + "_p2") and k > p1)
        if cand:
            p2 = cand[0]
    else:
        cand = sorted(t for t in toks if t != p1 and t.endswith("_p2"))
        if cand:
            p2 = cand[-1]
    return runs.get(p1), (runs.get(p2) if p2 else None)


MARK_BEGIN = "<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->"
MARK_END = "<!-- SPEC-TABLES:END -->"

POINTER = (
    "<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->\n\n"
    "**§1b-2 / §1b-3 / §1b-4 (설정 명세 표)** — 이 노트는 {why}이라 단일 런의 config가 "
    "없다. 리워드 가중치·모터 게인·토크 한계·ROM/액션 창·`PYG_*` 플래그는 **각 런 노트의 "
    "§1b~§1b-4**에 있다(모두 그 런의 `params/env.yaml`에서 기계 생성).\n\n"
    "<!-- SPEC-TABLES:END -->"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--only-1b", action="store_true",
                    help="rewrite only §1b (rewards + gains) inside existing blocks; "
                         "writes nothing else (no pointer or placeholder blocks)")
    a = ap.parse_args()

    rows = []
    for note in sorted(DOCS.glob("*.md")):
        if a.only and a.only not in note.name:
            continue
        stem = note.stem
        text = note.read_text()
        if note.name in AGGREGATE:
            status = "aggregate"
            if not (a.inventory or a.dry_run or a.only_1b):
                if "SPEC-TABLES:BEGIN" not in text:
                    note.write_text(text.rstrip() + "\n\n"
                                    + POINTER.format(why=AGGREGATE[note.name]) + "\n")
                    status = "aggregate/pointer 삽입"
            rows.append((note.name, "—", "종합노트", status))
            continue
        p1, p2 = find_runs(stem, text)
        if p1 is None:
            rows.append((note.name, "—", "런 디렉토리 없음", "원본 설정 소실"))
            if (not (a.inventory or a.dry_run or a.only_1b)
                    and "SPEC-TABLES:BEGIN" not in text):
                note.write_text(
                    text.rstrip() + "\n\n<!-- SPEC-TABLES:BEGIN (analysis/run_spec_tables.py) -->"
                    "\n\n**§1b-2 / §1b-3 / §1b-4 (설정 명세 표)** — 이 노트가 가리키는 학습 런의 "
                    "디렉토리(`params/env.yaml`)가 남아 있지 않다: **원본 설정 소실**. "
                    "액추에이터 한계·ROM·플래그를 추측으로 채우지 않는다.\n\n"
                    "<!-- SPEC-TABLES:END -->\n")
            continue
        xml, src = model_rule(p1.name)
        cmd = [str(PY), str(TOOL), str(p1), "--insert", str(note)]
        # The tool rewrites only what sits between the markers. Keep §1b in the
        # block whenever it is there now — checking the whole note made --force drop
        # a generated §1b table — and add it when the note has no reward table.
        # A §1b HEADING is not a §1b TABLE: several notes only point at a sibling
        # note's tables.
        m = re.search(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END), text, re.S)
        block = m.group(0) if m else ""
        rest = text.replace(block, "", 1) if block else text
        table_outside = "| reward | weight |" in rest
        with_1b = "| reward | weight |" in block or not table_outside
        b1_sub = bool(re.search(r"^## (?:§)?1b\b", rest, re.M))
        if a.only_1b and "| reward | weight |" in block:
            b1_sub = "**§1b-1." in block  # keep the heading style the block has
        if a.only_1b:
            # §1b only, inside the existing block; §1b-2/3/4 stay as they are.
            if not with_1b:
                rows.append((note.name, p1.name, xml or "(자동)",
                             "skip (§1b 표가 표식 밖 — 수동/구식 삽입, 손대지 않음)"))
                continue
            cmd.append("--only-1b")
        elif with_1b:
            cmd.append("--with-1b")
        if with_1b and b1_sub:
            cmd.append("--b1-sub")
        if p2 is not None and not a.only_1b:
            cmd += ["--p2", str(p2)]
        if xml and not a.only_1b:
            cmd += ["--model-xml", xml, "--model-source", src]
        if a.force:
            cmd.append("--force")
        if a.dry_run or a.inventory:
            cmd.append("--dry-run")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(cmd, cwd=MJLAB, capture_output=True, text=True, env=env)
        out = (r.stdout.splitlines() or [""])[0]
        if r.returncode:
            out = "ERROR: " + (r.stderr.strip().splitlines() or [""])[-1]
        if table_outside:
            out += " · §1b 표가 표식 밖(수동/구식 삽입) — 이 도구로 갱신 안 됨"
        rows.append((note.name, p1.name + (f" + {p2.name}" if p2 else ""),
                     xml or "(자동)", out.split(": ", 1)[-1]))

    w = max(len(r[0]) for r in rows)
    print(f"| {'노트'.ljust(w)} | 런 디렉토리 | 모델 | 결과 |")
    print(f"|{'-' * (w + 2)}|---|---|---|")
    for n, run, m, st in rows:
        print(f"| {n.ljust(w)} | {run} | {m} | {st} |")
    print(f"\n총 {len(rows)}개 노트")
    return 0


if __name__ == "__main__":
    sys.exit(main())
