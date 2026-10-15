#!/usr/bin/env python3
"""Tables and figures for docs/169 s8 from rough_load_probe.py / quasi_static_terrain_torque.py results.

  python3 rough_load_report.py --data results/rough_load --out /tmp/sec8 --fig docs/img/pd_gain

Reads  <data>/{box,slope,stairs}_{main,trial}.json  and  results/quasi_static_terrain.json.
Writes one markdown file per table into --out (the doc section is assembled from them) and two PNG figures.
Criterion (2) of docs/169 s7: Kp >= tau_P99 / 0.5236 (30 deg error), and Kp >= tau_P99.9 / half action window.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GROUPS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "crank"]
KO = {"hip_pitch": "엉덩이 앞뒤", "hip_roll": "엉덩이 좌우", "hip_yaw": "엉덩이 비틀기", "knee": "무릎", "crank": "발목 크랭크"}
RATED = {"hip_pitch": 40.0, "hip_roll": 40.0, "knee": 40.0, "hip_yaw": 20.0, "crank": 20.0}  # continuous, N*m
LIMIT = {"hip_pitch": 120.0, "hip_roll": 120.0, "knee": 120.0, "hip_yaw": 60.0, "crank": 60.0}
# env.yaml actions.clip half-width [rad] = (hi - lo) / 2; crank +-1.08 rad = +-62 deg
HALF_WINDOW = {"hip_pitch": (1.9678634 + 0.3097954) / 2, "hip_roll": (1.3875369 + 0.3403389) / 2,
               "hip_yaw": 0.7068582, "knee": (1.98968 - 0.10472) / 2, "crank": 1.08}
ERR30 = math.radians(30.0)
POL = {"main": "기준 정책 (배포, 크랭크 22.3)", "trial": "시험 정책 (크랭크 55)"}
DOC7_FLAT_P99 = {"hip_pitch": 59.1, "hip_roll": 59.1, "hip_yaw": 38.1, "knee": 41.1, "crank": 23.0}


def load(d: Path):
    out = {}
    for suite in ("box", "slope", "stairs"):
        for pol in ("main", "trial"):
            p = d / f"{suite}_{pol}.json"
            if p.exists():
                out[(suite, pol)] = {c["cond"]: c for c in json.load(open(p))["conditions"]}
    return out


def tau(c, g, k="p99", scope="all", sub=None):
    base = c["all"] if sub is None else c["per_type"][sub]["stats"]
    v = base[g]["tau" if scope == "all" else f"tau_{scope}"]
    return v.get(k, float("nan"))


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def md(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def req_kp(c, g, sub=None):
    p99 = tau(c, g, "p99", sub=sub)
    p999 = tau(c, g, "p999", sub=sub)
    return p99 / ERR30, p999 / HALF_WINDOW[g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", default="")
    a = ap.parse_args()
    D = load(Path(a.data))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = {}

    # ---- box: geometry per level
    rows = []
    for L in (0, 3, 6, 9):
        d = L / 9
        rows.append([L, f"{d:.2f}", f"{100 * 0.1 * d:.1f}", f"{100 * 0.12 * d:.1f}", f"{100 * 0.24 * d:.1f}",
                     f"{15 * d:.1f}", f"{100 * 0.1 * d:.1f}"])
    T["box_geometry"] = md(rows, ["단계", "난이도 d", "계단 한 단 높이 [cm]", "무작위 상자 높이 ±[cm]",
                                  "이웃 상자 최대 높이차 [cm]", "기울어진 판 최대 기울기 [도]", "기울어진 판 높이 ±[cm]"])

    # ---- falls
    for suite, key, vxs, lab in (("box", "box_L{}_vx{}", (0.3, 0.6, 1.0), "box"),):
        rows = []
        for pol in ("main", "trial"):
            if (suite, pol) not in D:
                continue
            for L in (0, 3, 6, 9):
                r = [POL[pol], L]
                for vx in vxs:
                    c = D[(suite, pol)].get(key.format(L, vx))
                    r.append(fmt(100 * c["falls"]) if c else "-")
                rows.append(r)
        T["box_falls"] = md(rows, ["정책", "단계", "넘어짐 [%] 0.3 m/s", "0.6 m/s", "1.0 m/s"])
    # falls per terrain type at vx 0.6 and 1.0
    rows = []
    for pol in ("main", "trial"):
        if ("box", pol) not in D:
            continue
        for vx in (0.6, 1.0):
            for L in (0, 3, 6, 9):
                c = D[("box", pol)].get(f"box_L{L}_vx{vx}")
                if not c:
                    continue
                pt = c["per_type"]
                rows.append([POL[pol], vx, L] + [fmt(100 * pt[t]["falls"]) for t in
                                                 ("flat", "pyramid_stairs", "pyramid_stairs_inv", "box_random_grid", "tilted_grid")])
    T["box_falls_type"] = md(rows, ["정책", "속도 [m/s]", "단계", "평지", "계단(올라감 쪽 중앙 평지에서 내려감)", "역계단", "무작위 상자", "기울어진 판"])

    # ---- box torque P99: flat tile vs rough pooled, per policy / vx
    for pol in ("main", "trial"):
        if ("box", pol) not in D:
            continue
        for vx in (0.3, 0.6, 1.0):
            rows = []
            for g in GROUPS:
                r = [KO[g], fmt(DOC7_FLAT_P99[g])]
                c0 = D[("box", pol)].get(f"box_L0_vx{vx}")
                r.append(fmt(tau(c0, g, "p99", sub="flat")) if c0 else "-")
                for L in (0, 3, 6, 9):
                    c = D[("box", pol)].get(f"box_L{L}_vx{vx}")
                    r.append(fmt(tau(c, g, "p99")) if c else "-")
                rows.append(r)
            T[f"box_tau_p99_{pol}_vx{vx}"] = md(rows, ["관절", "평지 걷기 (문서 §7)", "평지 타일 (이번 측정)", "단계 0", "단계 3", "단계 6", "단계 9"])

    # ---- full detail for the hardest level, per policy: rms p50 p99 p999 max err sat clip
    for pol in ("main", "trial"):
        if ("box", pol) not in D:
            continue
        for vx in (0.6, 1.0):
            c = D[("box", pol)].get(f"box_L9_vx{vx}")
            if not c:
                continue
            rows = []
            for g in GROUPS:
                s = c["all"][g]
                t = s["tau"]
                rows.append([KO[g], fmt(t["rms"]), fmt(t["p50"]), fmt(t["p99"]), fmt(t["p999"]), fmt(t["max"]),
                             fmt(s["err_deg"]["p50"]), fmt(s["err_deg"]["p99"]), fmt(s["err_deg"]["p999"]),
                             f"{100 * s['sat_frac']:.3f}", f"{100 * s['clip_frac']:.2f}"])
            T[f"box_detail_L9_{pol}_vx{vx}"] = md(rows, ["관절", "토크 실효값", "중앙값", "99 백분위수", "99.9 백분위수", "최대", "오차 중앙값 [도]", "오차 99 백분위수", "오차 99.9 백분위수",
                                                         "토크 한계 도달 [%]", "목표가 동작 한계에 걸린 [%]"])

    # ---- landing transient
    rows = []
    for pol in ("main", "trial"):
        if ("box", pol) not in D:
            continue
        for L in (0, 9):
            c = D[("box", pol)].get(f"box_L{L}_vx0.6")
            if not c:
                continue
            for g in GROUPS:
                rows.append([POL[pol], L, KO[g], fmt(tau(c, g, "p99", "landing")), fmt(tau(c, g, "p99", "other")),
                             fmt(tau(c, g, "p999", "landing")), fmt(tau(c, g, "p999", "other")),
                             f"{100 * c['all'][g]['landing_frac']:.0f}"])
    T["landing"] = md(rows, ["정책", "단계", "관절", "착지 직후 0.1초 99 백분위수", "그 밖 99 백분위수", "착지 직후 99.9 백분위수", "그 밖 99.9 백분위수", "착지 구간 비율 [%]"])

    # ---- GRF
    rows = []
    for pol in ("main", "trial"):
        if ("box", pol) not in D:
            continue
        for L in (0, 3, 6, 9):
            for vx in (0.6, 1.0):
                c = D[("box", pol)].get(f"box_L{L}_vx{vx}")
                if not c:
                    continue
                gr = c["grf"]
                p99 = max(v["p99_bw"] for v in gr.values())
                mx = max(v["max_bw"] for v in gr.values())
                pk = sum(v["env_peak_p50_bw"] for v in gr.values()) / 2
                rows.append([POL[pol], L, vx, fmt(p99, 2), fmt(pk, 2), fmt(mx, 1)])
    T["grf_box"] = md(rows, ["정책", "단계", "속도", "발 지면반력 99 백분위수 [체중]", "환경별 최대값의 중앙값 [체중]", "전체 최대 [체중]"])

    # ---- slopes / stairs per condition tables
    def cond_table(suite, names, pols=("main", "trial")):
        rows = []
        for pol in pols:
            if (suite, pol) not in D:
                continue
            for nm in names:
                c = D[(suite, pol)].get(nm)
                if not c:
                    continue
                rows.append([POL[pol], nm.replace("_vx", " / ").replace("slope_", "").replace("stairs_", ""),
                             fmt(100 * c["falls"]), fmt(100 * c.get("reached_feature", float("nan")), 0),
                             fmt(100 * c.get("fell_after_feature", float("nan"))), fmt(c.get("end_x_mean_m", float("nan")), 1), c["valid_steps"]]
                            + [fmt(tau(c, g, "p99")) for g in GROUPS])
        return md(rows, ["정책", "조건", "넘어짐 [%]", "경사·계단에 닿은 비율 [%]", "닿은 뒤 넘어짐 [%]", "끝 위치 [m] (시작 3.0)", "분석 걸음 수"]
                  + [f"{KO[g]} 99 백분위수" for g in GROUPS])

    sl = [f"slope_{d}_{ang}_vx{vx}" for ang in (10, 20, 30) for d in ("up", "down") for vx in (0.3, 0.6)]
    st = [f"stairs_{d}_{cm}cm_vx0.3" for cm in (10, 15, 20, 25) for d in ("up", "down")]
    def limit_table(suite, names):
        rows = []
        for pol in ("main", "trial"):
            for nm in names:
                c = D.get((suite, pol), {}).get(nm)
                if not c:
                    continue
                al = c["all"]
                rows.append([POL[pol], nm.replace("_vx", " / ").replace("slope_", "").replace("stairs_", "")]
                            + [f"{100 * al[g]['sat_frac']:.2f}" for g in ("hip_roll", "hip_yaw", "knee")]
                            + [f"{100 * al['crank']['clip_frac']:.1f}", fmt(al["crank"]["err_deg"]["p99"], 0), fmt(al["knee"]["err_deg"]["p99"], 0),
                               fmt(max(v["p99_bw"] for v in c["grf"].values()), 1)])
        return md(rows, ["정책", "조건", "엉덩이 좌우 토크 한계 도달 [%]", "엉덩이 비틀기 [%]", "무릎 [%]", "크랭크 목표가 동작 한계에 걸린 [%]",
                         "크랭크 오차 99 백분위수 [도]", "무릎 오차 99 백분위수 [도]", "발 지면반력 99 백분위수 [체중]"])

    T["limit_slope"] = limit_table("slope", ["slope_down_20_vx0.6", "slope_up_30_vx0.6", "slope_down_30_vx0.3"])
    T["limit_stairs"] = limit_table("stairs", ["stairs_up_15cm_vx0.3", "stairs_down_15cm_vx0.3", "stairs_down_25cm_vx0.3"])
    T["slope"] = cond_table("slope", sl)
    T["stairs"] = cond_table("stairs", st)

    # ---- required Kp, worst over conditions in each family (only conditions with >= 2000 valid steps)
    fam = {
        "평지 걷기 (문서 §7)": None,
        "평지 타일 (이번 측정, 속도 0.3 ~ 1.0)": [("box", f"box_L{L}_vx{vx}", "flat") for L in (0,) for vx in (0.3, 0.6, 1.0)],
        "박스 지형 단계 0 ~ 9 전체 (훈련 범위)": [("box", f"box_L{L}_vx{vx}", None) for L in (0, 3, 6, 9) for vx in (0.3, 0.6, 1.0)],
        "박스 지형 단계 9 (가장 어려움)": [("box", f"box_L9_vx{vx}", None) for vx in (0.3, 0.6, 1.0)],
        "오르막 10 ~ 30도 (훈련 밖)": [("slope", n, None) for n in sl if "_up_" in n],
        "내리막 10 ~ 30도 (훈련 밖)": [("slope", n, None) for n in sl if "_down_" in n],
        "계단 오름 10 ~ 25 cm (훈련 밖)": [("stairs", n, None) for n in st if "_up_" in n],
        "계단 내림 10 ~ 25 cm (훈련 밖)": [("stairs", n, None) for n in st if "_down_" in n],
    }
    for pol in ("main", "trial"):
        rows, rows_w = [], []
        for name, items in fam.items():
            if items is None:
                rows.append([name] + [fmt(DOC7_FLAT_P99[g]) + " / " + fmt(DOC7_FLAT_P99[g] / ERR30, 0) for g in GROUPS])
                continue
            r, rw = [name], [name]
            for g in GROUPS:
                best99, best999, bn = 0.0, 0.0, ""
                for suite, nm, sub in items:
                    c = D.get((suite, pol), {}).get(nm)
                    if not c or c["valid_steps"] < 2000:
                        continue
                    t99, t999 = tau(c, g, "p99", sub=sub), tau(c, g, "p999", sub=sub)
                    if t99 > best99:
                        best99, bn = t99, nm
                    best999 = max(best999, t999)
                r.append(f"{best99:.1f} / {best99 / ERR30:.0f}")
                rw.append(f"{best999:.1f} / {best999 / HALF_WINDOW[g]:.0f}")
            rows.append(r)
            rows_w.append(rw)
        T[f"reqkp_{pol}"] = md(rows, ["지형 묶음"] + [f"{KO[g]} 토크 99 백분위수 [N·m] / 필요 Kp" for g in GROUPS])
        T[f"reqkp_window_{pol}"] = md(rows_w, ["지형 묶음"] + [f"{KO[g]} 토크 99.9 백분위수 [N·m] / 필요 Kp" for g in GROUPS])

    # ---- thermal: RMS vs rated, per family (worst condition) and a per-condition table for the baseline
    for pol in ("main", "trial"):
        rows = []
        for name, items in fam.items():
            if items is None:
                continue
            r = [name]
            for g in GROUPS:
                best, bn = 0.0, ""
                for suite, nm, sub in items:
                    c = D.get((suite, pol), {}).get(nm)
                    if not c or c["valid_steps"] < 2000:
                        continue
                    v = (c["all"] if sub is None else c["per_type"][sub]["stats"])[g]["tau"]["rms"]
                    if v > best:
                        best, bn = v, nm
                r.append(f"{best:.1f} ({100 * best / RATED[g]:.0f} %)")
            rows.append(r)
        T[f"thermal_{pol}"] = md(rows, ["지형 묶음 (그중 가장 큰 조건)"] + [f"{KO[g]} 토크 실효값 [N·m] (정격 {RATED[g]:.0f} 대비)" for g in GROUPS])
    # per-condition RMS for slopes+stairs, baseline
    rows = []
    for suite, names in (("slope", [n for n in sl if n.endswith("vx0.6")]), ("stairs", st)):
        for nm in names:
            c = D.get((suite, "main"), {}).get(nm)
            if c:
                rows.append([nm.replace("_vx", " / ")] + [f"{c['all'][g]['tau']['rms']:.1f} ({100 * c['all'][g]['tau']['rms'] / RATED[g]:.0f} %)" for g in GROUPS]
                            + [f"{c['all']['crank']['rms_env_p95']:.1f}"])
    T["thermal_cond_main"] = md(rows, ["조건"] + [KO[g] for g in GROUPS] + ["크랭크 환경별 실효값 상위 5 %"])

    # ---- quasi-static
    qs = Path(a.data).parent / "quasi_static_terrain.json"
    if qs.exists():
        Q = json.load(open(qs))
        rows = []
        for k, v in Q["scenarios"].items():
            t = v["tau"]
            p = v["params"]
            rows.append([k, f"{p.get('slope_deg', 0):g}", f"{p.get('foot_h', 0) * 100:g}", f"{p.get('g_scale', 1):g}",
                         fmt(abs(t["hip_pitch"])), fmt(abs(t["hip_roll"])), fmt(abs(t["knee"])), fmt(abs(t["crank_A"])), fmt(abs(t["crank_B"])),
                         fmt(v["knee_deg"], 0), fmt(abs(v["abs_joint_deg"].get("ap", float("nan"))), 0),
                         fmt(abs(v["abs_joint_deg"].get("hr", float("nan"))), 0) if "hr" in v["abs_joint_deg"] else "-",
                         fmt(p.get("extra_pelvis_lowering_m", 0) * 100, 0),
                         "예" if v["cop_in_sole"] else "아니오", "관절 한계 밖" if p.get("joint_limits_relaxed") else "한계 안"])
        T["quasi"] = md(rows, ["상황", "경사 [도]", "단 높이 [cm]", "하중 [체중]", "엉덩이 앞뒤", "엉덩이 좌우", "무릎", "크랭크 A", "크랭크 B",
                               "무릎 굽힘 [도]", "발목 앞뒤 각도 [도]", "엉덩이 좌우 각도 [도]", "골반 추가 낮춤 [cm]", "압력중심이 발바닥 안", "자세"])

    for k, v in T.items():
        (out / f"{k}.md").write_text(v)
    print("tables:", ", ".join(sorted(T)))

    if a.fig:
        make_figs(D, Path(a.fig))


def make_figs(D, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#2a78d6", "#eb6834", "#52514e"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: P99 torque, baseline policy, per joint group across terrain
    conds = [("flat tile\n0.6 m/s", "box", "box_L0_vx0.6", "flat"), ("box level 9\n0.6 m/s", "box", "box_L9_vx0.6", None),
             ("box level 9\n1.0 m/s", "box", "box_L9_vx1.0", None), ("slope up 20°\n0.6 m/s", "slope", "slope_up_20_vx0.6", None),
             ("slope down 20°\n0.6 m/s", "slope", "slope_down_20_vx0.6", None), ("stairs up\n15 cm, 0.3 m/s", "stairs", "stairs_up_15cm_vx0.3", None),
             ("stairs down\n15 cm, 0.3 m/s", "stairs", "stairs_down_15cm_vx0.3", None)]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    names = {"hip_pitch": "hip pitch", "hip_roll": "hip roll", "hip_yaw": "hip yaw", "knee": "knee", "crank": "ankle crank"}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    n = len(GROUPS)
    w = 0.16
    for gi, g in enumerate(GROUPS):
        ys = []
        for lab, suite, nm, sub in conds:
            c = D.get((suite, "main"), {}).get(nm)
            ys.append(tau(c, g, "p99", sub=sub) if c else float("nan"))
        ax.bar([i + (gi - 2) * (w + 0.015) for i in range(len(conds))], ys, w, color=pal[gi], label=names[g])
    ax.axhline(40, color=grey, lw=1, ls="--")
    ax.set_xlim(-0.6, len(conds) + 0.5)
    ax.text(len(conds) + 0.45, 41, "RS04 rated\n40 N·m", ha="right", va="bottom", color=grey, fontsize=8)
    ax.axhline(20, color=grey, lw=1, ls=":")
    ax.text(len(conds) + 0.45, 21, "RS03 rated\n20 N·m", ha="right", va="bottom", color=grey, fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds])
    ax.set_ylabel("|torque| 99th percentile [N·m]")
    ax.set_title("Deployed rough policy: per-joint torque (99th percentile) by terrain (slopes and stairs are outside the training range)", fontsize=9.5)
    ax.legend(ncol=5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "2026-10-15_rough_torque_p99.png", dpi=150)
    plt.close(fig)

    # fig 2: falls vs box level, two policies, three speeds
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for pol, col in (("main", blue), ("trial", orange)):
        if ("box", pol) not in D:
            continue
        for vx, ls in ((0.3, ":"), (0.6, "-"), (1.0, "--")):
            ys = [100 * D[("box", pol)][f"box_L{L}_vx{vx}"]["falls"] for L in (0, 3, 6, 9)]
            ax.plot([0, 3, 6, 9], ys, color=col, ls=ls, marker="o", ms=4, lw=1.6,
                    label=f"{'baseline' if pol == 'main' else 'trial'} {vx} m/s")
    ax.set_xlabel("box terrain level (0 easiest, 9 hardest)")
    ax.set_ylabel("episodes with a fall within 20 s [%]")
    ax.set_title("Falls per box-terrain level", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "2026-10-15_rough_falls_by_level.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
