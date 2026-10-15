#!/usr/bin/env python3
"""Figures for docs/169 section 9: our current / flat-recommended / mountain-recommended PD gains
against the published gains of other robots, with the evidence behind each recommendation.

  python3 tools/robot_gains/plot_recommendation_vs_robots.py [--table]

Reads tools/robot_gains/results/{g1_a,g1_b,h1,t1,berkeley,pyg_v7b,pyg_rec_flat,pyg_rec_mountain}.json
(compare_robot_gains.py), results/rough_load/*_main.json (rough_load_probe.py) and
results/quasi_static_terrain.json.  Needs only numpy + matplotlib (no MuJoCo).  English labels only.
Writes docs/img/pd_gain/2026-10-15_rec_F1..F7_*.png.   --table prints the markdown comparison table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
OUT = REPO / "docs" / "img" / "pd_gain"
sys.path.insert(0, str(HERE))
from rough_load_report import DOC7_FLAT_P99, ERR30, load as load_rough, tau as rough_tau  # noqa: E402

OTHERS = ["g1_a", "g1_b", "h1", "t1", "berkeley"]
OURS = ["pyg_v7b", "pyg_rec_flat", "pyg_rec_mountain"]
ALL = OTHERS + OURS
NAME = {"g1_a": "G1-a (rl_gym)", "g1_b": "G1-b (mjlab)", "h1": "H1", "t1": "T1", "berkeley": "Berkeley",
        "pyg_v7b": "Pyg current", "pyg_rec_flat": "Pyg flat-rec", "pyg_rec_mountain": "Pyg mountain-rec"}
SHORT = {"g1_a": "G1-a", "g1_b": "G1-b", "h1": "H1", "t1": "T1", "berkeley": "Berkeley",
         "pyg_v7b": "Pyg\ncurrent", "pyg_rec_flat": "Pyg\nflat-rec", "pyg_rec_mountain": "Pyg\nmountain-rec"}
# other robots: cool / neutral hues; ours: red family with a black outline
COL = {"g1_a": "#1f77b4", "g1_b": "#8fc1e3", "h1": "#2ca02c", "t1": "#9467bd", "berkeley": "#7f7f7f",
       "pyg_v7b": "#7f0000", "pyg_rec_flat": "#e34a33", "pyg_rec_mountain": "#fdae6b"}
HATCH = {"pyg_v7b": "", "pyg_rec_flat": "", "pyg_rec_mountain": ""}
GROUPS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle"]
GLAB = {"hip_pitch": "hip pitch", "hip_roll": "hip roll", "hip_yaw": "hip yaw", "knee": "knee",
        "ankle": "ankle\n(Pyg: crank)"}
GLAB1 = {"hip_pitch": "hip pitch", "hip_roll": "hip roll", "hip_yaw": "hip yaw", "knee": "knee", "ankle": "ankle / crank"}
BAND = (17.0, 28.0)           # rule 1: action = 1 is 17-28 % of the effort limit
GENERIC_INERTIA = {"h1", "t1"}   # Menagerie class default rotor inertia, not a hardware value
PYG_GAINS = {"pyg_v7b": dict(hip_pitch=(150, 6), hip_roll=(150, 6), hip_yaw=(150, 6), knee=(220, 6), crank=(22.3, 1.41)),
             "pyg_rec_flat": dict(hip_pitch=(120, 6), hip_roll=(120, 6), hip_yaw=(75, 4), knee=(130, 6), crank=(55, 2.5)),
             "pyg_rec_mountain": dict(hip_pitch=(135, 6), hip_roll=(135, 6), hip_yaw=(75, 4), knee=(130, 6), crank=(60, 2.7))}
EFFORT = {"hip_pitch": 120.0, "hip_roll": 120.0, "hip_yaw": 60.0, "knee": 120.0, "crank": 60.0}


def group_of(n: str):
    s = n.lower()
    if "crank" in s:
        return "ankle"
    if "hip_yaw" in s or s.endswith("_hr"):
        return "hip_yaw"
    if "hip_roll" in s or s.endswith("_haa"):
        return "hip_roll"
    if "hip_pitch" in s or s.endswith("_hfe"):
        return "hip_pitch"
    if "knee" in s or s.endswith("_kfe"):
        return "knee"
    if "ankle_roll" in s or s.endswith("_faa"):
        return None
    if "ankle" in s or s.endswith("_ffe"):
        return "ankle"
    return None


def gather():
    D = {}
    for k in ALL:
        r = json.loads((RES / f"{k}.json").read_text())
        d = {"mass": r["mass_kg"], "ratio": r["A3"]["ratio"],
             "stand": r["A2"]["no_push"].get("min_refined"),
             "stand_ankle": r["A2"]["ankle_only_no_push"].get("min_refined"), "g": {}}
        acc = {}
        for n, v in r["A5"].items():
            g = group_of(n)
            if g is None:
                continue
            a = r["A4"][n]
            acc.setdefault(g, []).append(dict(pct=v["pct_effort_at_action1"], kp=v["kp"], kd=v["kd"],
                                              effort=v["effort"], kpkg=v["kp_per_kg"], zeta=a["zeta_rotor"],
                                              wn=a["wn_rotor_hz"], arm=a["armature"]))
        for g, L in acc.items():
            d["g"][g] = {f: float(np.mean([x[f] for x in L])) for f in L[0]}
        # ankle: per-ankle equivalent Kp (pitch) per kg, as in docs/170 table (Pyg crank pair -> ankle pitch)
        eq = r["A3"]["ankle_equivalent"]["L"]
        d["g"]["ankle"]["kpkg"] = eq["Kp_pitch"] / d["mass"]
        d["g"]["ankle"]["kp_eq"] = eq["Kp_pitch"]
        D[k] = d
    return D


def required_kp():
    """Required Kp (30 deg error, criterion 2) = torque P99 / 0.5236, baseline (main) policy only."""
    R = load_rough(RES / "rough_load")
    gm = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "crank"]
    lab = ["flat walk (docs/169 s7)", "box terrain level 9", "uphill 10-30 deg", "stairs up 10-25 cm",
           "25 cm landing x2 (quasi-static)"]
    fam = {
        1: [("box", f"box_L9_vx{vx}") for vx in (0.3, 0.6, 1.0)],
        2: [("slope", n) for n in R[("slope", "main")] if "_up_" in n],
        3: [("stairs", n) for n in R[("stairs", "main")] if "_up_" in n],
    }
    out = {g: [None] * 5 for g in gm}
    for g in gm:
        out[g][0] = DOC7_FLAT_P99[g] / ERR30
        for i, items in fam.items():
            best = 0.0
            for suite, nm in items:
                c = R[(suite, "main")].get(nm)
                if c and c["valid_steps"] >= 2000:
                    best = max(best, rough_tau(c, g, "p99"))
            out[g][i] = best / ERR30
    qs = json.loads((RES / "quasi_static_terrain.json").read_text())["scenarios"]["landing_25cm__x2"]["tau"]
    for g in gm:
        t = max(abs(qs["crank_A"]), abs(qs["crank_B"])) if g == "crank" else abs(qs[g])
        out[g][4] = t / ERR30
    return lab, out


# ------------------------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def draw_grouped(ax, D, field, ylab, band=None, ymax=None, fs=7, compact=False, fmt="{:.0f}"):
    n = len(ALL)
    w = 0.8 / n
    x = np.arange(len(GROUPS))
    if band:
        ax.axhspan(band[0], band[1], color="#2a9d4b", alpha=0.15, zorder=0)
    for i, k in enumerate(ALL):
        vals = [D[k]["g"].get(g, {}).get(field, np.nan) for g in GROUPS]
        xs = x + (i - (n - 1) / 2) * w
        ours = k in OURS
        for xx, v in zip(xs, vals):
            if np.isnan(v):
                continue
            h = min(v, ymax) if ymax else v
            ax.bar(xx, h, w * 0.92, color=COL[k], edgecolor="k" if ours else "none", linewidth=1.1 if ours else 0,
                   zorder=3, label=NAME[k] if xx == xs[0] else None)
            if ymax and v > ymax:
                ax.text(xx, ymax * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=fs - 1, rotation=90)
            elif not compact and ours:
                ax.text(xx, v + (ymax or max(vals)) * 0.012, fmt.format(v), ha="center", va="bottom", fontsize=fs - 1.5,
                        rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([GLAB[g] if not compact else GLAB1[g] for g in GROUPS], fontsize=8)
    ax.set_ylabel(ylab, fontsize=9)
    if ymax:
        ax.set_ylim(0, ymax * 1.16)
    _style(ax)


def fig_f1(ax=None, compact=False):
    D = gather()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(12.5, 5.6))
    draw_grouped(ax, D, "pct", "torque at action = 1  [% of effort limit]", band=BAND, ymax=70, compact=compact)
    ax.text(0.01, 0.97, "green band = 17-28 % (rule 1: G1, H1, Berkeley, mjlab)\nbars above 70 % are clipped, value printed on top",
            transform=ax.transAxes, fontsize=7.5, va="top")
    if own:
        ax.set_title("F1  Rule 1: torque at action = 1 as % of effort limit (hip/knee/ankle, all robots)", fontsize=11)
        ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.0))
        ax.set_ylim(0, 70 * 1.16)
        fig.tight_layout()
        fig.savefig(OUT / "2026-10-15_rec_F1_rule1_action1_pct.png", dpi=150)
        plt.close(fig)
    else:
        ax.set_title("Rule 1: action = 1 torque, % of limit", fontsize=9)


def fig_f2():
    D = gather()
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    draw_grouped(ax, D, "kpkg", "Kp per kg body mass  [N m/rad/kg]", ymax=10, fmt="{:.1f}")
    ax.set_title("F2  Kp per body mass (ankle = per-ankle pitch equivalent; Pyg crank pair converted)", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="upper right", frameon=False)
    ax.set_ylim(0, 10 * 1.3)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_rec_F2_kp_per_kg.png", dpi=150)
    plt.close(fig)


def ax_max(gk, vals):
    return max(max(vals) * 1.15, PYG_GAINS["pyg_v7b"][gk][0] * 1.08, BAND[1] / 100 * EFFORT[gk] / 0.25 * 1.05)


def f3_panel(ax, g, lab, req, compact=False):
    gk = "crank" if g == "crank" else g
    vals = req[gk]
    y = np.arange(len(lab))[::-1]
    lo, hi = (BAND[0] / 100 * EFFORT[gk] / 0.25, BAND[1] / 100 * EFFORT[gk] / 0.25)
    ax.axvspan(lo, hi, color="#2a9d4b", alpha=0.15, zorder=0)
    ax.barh(y, vals, 0.6, color="#5b7fa6", zorder=2)
    for yy, v in zip(y, vals):
        if v > 0.14 * ax_max(gk, vals):
            ax.text(v - 2, yy, f"{v:.0f}", va="center", ha="right", fontsize=8, color="white", weight="bold", zorder=6)
        else:
            ax.text(v + 2, yy, f"{v:.0f}", va="center", fontsize=8, zorder=6)
    for k, ls in zip(OURS, ["-", "-", "-"]):
        kp = PYG_GAINS[k][gk][0]
        ax.axvline(kp, color=COL[k], lw=2.2, ls=ls, zorder=4, label=f"{NAME[k]} {kp:g}")
        ax.plot([kp], [len(lab) - 0.35], marker="v", color=COL[k], ms=6, mec="k", zorder=5, clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels(lab if not compact else [""] * len(lab), fontsize=7.5)
    ax.set_xlim(0, ax_max(gk, vals))
    ax.set_title({"hip_pitch": "hip pitch", "hip_roll": "hip roll", "hip_yaw": "hip yaw", "knee": "knee",
                  "crank": "ankle crank"}[gk] + f"  (limit {EFFORT[gk]:.0f} N m)", fontsize=9)
    ax.set_xlabel("required Kp  [N m/rad]", fontsize=8)
    _style(ax)


def fig_f3():
    lab, req = required_kp()
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.2))
    gs = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "crank"]
    for ax, g in zip(axs.flat, gs):
        f3_panel(ax, g, lab, req)
    ax = axs.flat[5]
    ax.axis("off")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[k], lw=3, label=f"{NAME[k]} Kp") for k in OURS]
    ax.legend(handles=hs, loc="upper left", fontsize=9, frameon=False, bbox_to_anchor=(0, 1.0))
    ax.text(0, 0.68, "Bar = Kp needed so the P99 torque stays\nwithin 30 deg of position error\n(torque / 0.5236 rad).\n"
            "Baseline (deployed) policy only; downhill\nand stairs-down rows excluded (policy fell).\n\n"
            "Green band = rule 1 (action 1 = 17-28 %\nof the effort limit).\n\n"
            "Landing row: quasi-static whole-leg model,\n2x body weight on one leg after a 25 cm\n"
            "drop; not a policy measurement.\ndocs/169 s8-7 uses it for the knee only.\n\n"
            "A line left of a bar's end = that terrain\nexceeds the 30 deg error budget.", fontsize=8.5, va="top")
    fig.suptitle("F3  Criterion 2 evidence: required Kp by terrain vs our three gain sets", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "2026-10-15_rec_F3_required_kp_by_terrain.png", dpi=150)
    plt.close(fig)


def draw_f4(ax, D, compact=False):
    x = np.arange(len(ALL))
    vals = [D[k]["ratio"] for k in ALL]
    for xi, k, v in zip(x, ALL, vals):
        ax.bar(xi, v, 0.7, color=COL[k], edgecolor="k" if k in OURS else "none", linewidth=1.1, zorder=3)
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5 if not compact else 6.5)
    ax.axhline(1.0, color="k", lw=1.4, zorder=4)
    ax.text(-0.45, 1.02, "1.0 = falls over under gravity alone", ha="left", fontsize=7.5)
    for yv, t in ((0.64, "human passive 0.64"), (0.91, "human passive 0.91"), (1.33, "human total 1.33")):
        ax.axhline(yv, color="#444", lw=1, ls="--", zorder=1)
        ax.text(-0.45, yv + 0.015, t, fontsize=7, color="#222", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] if not compact else SHORT[k].replace("mountain-rec", "mtn-rec") for k in ALL],
                       fontsize=7.5 if not compact else 6.5)
    ax.set_ylabel("ankle stiffness / (m g h)", fontsize=9)
    ax.set_ylim(0, 1.55)
    _style(ax)


def fig_f4():
    D = gather()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    draw_f4(ax, D)
    ax.set_title("F4  Criterion 3: sum of both ankles' pitch stiffness / (mass x g x CoM height above ankle)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_rec_F4_ankle_stiffness_over_mgh.png", dpi=150)
    plt.close(fig)


def fig_f5():
    D = gather()
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.axhspan(0.6, 1.0, color="#2a9d4b", alpha=0.15, zorder=0)
    mk = {"hip_pitch": "o", "hip_roll": "s", "hip_yaw": "D", "knee": "^", "ankle": "P"}
    for k in ALL:
        for g in GROUPS:
            v = D[k]["g"].get(g)
            if not v:
                continue
            ours = k in OURS
            gen = k in GENERIC_INERTIA
            ax.scatter(v["wn"], v["zeta"], marker=mk[g], s=95 if ours else 60,
                       facecolor="white" if gen else COL[k], edgecolor="k" if ours else COL[k],
                       linewidth=1.6 if gen else (1.1 if ours else 0.6), zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.8, 40)
    ax.set_ylim(0.22, 4.2)
    ax.set_xticks([2, 3, 5, 7, 10, 15, 20, 30])
    ax.set_xticklabels(["2", "3", "5", "7", "10", "15", "20", "30"])
    ax.set_yticks([0.25, 0.5, 0.6, 1.0, 2.0, 3.0])
    ax.set_yticklabels(["0.25", "0.5", "0.6", "1.0", "2", "3"])
    ax.text(1.9, 0.98, "target band 0.6-1.0", fontsize=8, color="#1b6e34", va="top")
    ax.set_xlabel("natural frequency of the rotor alone  sqrt(Kp / I_rotor) / 2 pi  [Hz]", fontsize=9)
    ax.set_ylabel("damping ratio  Kd / (2 sqrt(Kp I_rotor)), rotor only", fontsize=9)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], marker="o", ls="", mfc=COL[k], mec="k" if k in OURS else COL[k], ms=8, label=NAME[k]) for k in ALL]
    h2 = [Line2D([], [], marker=mk[g], ls="", mfc="#999", mec="k", ms=7, label=GLAB1[g]) for g in GROUPS]
    h3 = [Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=8, mew=1.6,
                 label="hollow = rotor inertia is a generic\nvalue, UNVERIFIED (H1 0.1, T1 0.005)")]
    l1 = ax.legend(handles=h1, fontsize=7.5, loc="upper left", title="robot", title_fontsize=8, frameon=True)
    ax.add_artist(l1)
    ax.legend(handles=h2 + h3, fontsize=7.5, loc="lower right", title="joint", title_fontsize=8, frameon=True)
    ax.set_title("F5  Criterion 4: rotor-based damping ratio vs natural frequency, every joint group\n(Pyg hip pitch and hip roll share one point per set; Pyg yaw/knee/crank are separate)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_rec_F5_damping_vs_frequency.png", dpi=150)
    plt.close(fig)


def draw_f6(ax, D, compact=False):
    cap = 40.0
    x = np.arange(len(ALL))
    for xi, k in zip(x, ALL):
        v = D[k]["stand"]
        none = v is None
        h = cap if none else v
        ax.bar(xi, h, 0.7, color=COL[k], edgecolor="k" if k in OURS else "none", linewidth=1.1, zorder=3,
               hatch="///" if none else None, alpha=0.5 if none else 1.0)
        ax.text(xi, h * 1.08, "none" if none else f"{v:.1f}", ha="center", fontsize=7.5 if not compact else 6.5)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(0.01, 0.97, "dashed line = x1: published gains stand unaided", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5 if not compact else 6.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 80)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] if not compact else SHORT[k].replace("mountain-rec", "mtn-rec") for k in ALL],
                       fontsize=7.5 if not compact else 6.5)
    ax.set_ylabel("smallest gain multiplier that stands 5 s", fontsize=9)
    _style(ax)


def fig_f6():
    D = gather()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    draw_f6(ax, D)
    ax.set_title("F6  PD-only stand multiplier (all leg joints scaled together, rigid contact, no policy); log scale", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_rec_F6_stand_multiplier.png", dpi=150)
    plt.close(fig)


def fig_f7():
    D = gather()
    lab, req = required_kp()
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.26, left=0.05, right=0.985, top=0.9, bottom=0.07)
    # (a) rule 1
    ax = fig.add_subplot(gs[0, 0])
    draw_grouped(ax, D, "pct", "action = 1 torque [% of limit]", band=BAND, ymax=70, compact=True, fs=7)
    ax.set_ylim(0, 70 * 1.16)
    ax.set_title("A  Rule 1: action = 1 torque, % of limit (green = 17-28 %)", fontsize=10, loc="left")
    # (b) criterion 2 compact: required Kp (dots per terrain) and our Kp (bars)
    ax = fig.add_subplot(gs[0, 1])
    gk = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "crank"]
    mkr = ["o", "s", "^", "D", "v"]
    xs = np.arange(len(gk))
    wd = 0.2
    for j, k in enumerate(OURS):
        for i, g in enumerate(gk):
            kp = PYG_GAINS[k][g][0]
            ax.bar(i + (j - 1) * wd, kp, wd * 0.92, color=COL[k], edgecolor="k", linewidth=0.8, zorder=2,
                   label=NAME[k] if i == 0 else None)
    for t in range(5):
        ax.scatter([i for i in range(len(gk))], [req[g][t] for g in gk], marker=mkr[t], s=34, c="#1f3b5c",
                   zorder=5, label=lab[t].replace(" (docs/169 s7)", "").replace(" (quasi-static)", ""))
    ax.set_xticks(xs)
    ax.set_xticklabels(["hip pitch", "hip roll", "hip yaw", "knee", "crank"], fontsize=8)
    ax.set_ylabel("Kp  [N m/rad]", fontsize=9)
    ax.set_ylim(0, 330)
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", frameon=True)
    ax.set_title("B  Criterion 2: Kp needed (dots) vs our Kp (bars)", fontsize=10, loc="left")
    _style(ax)
    # (c) ankle / mgh
    ax = fig.add_subplot(gs[0, 2])
    draw_f4(ax, D, compact=True)
    ax.set_title("C  Criterion 3: ankle stiffness / (m g h)", fontsize=10, loc="left")
    # (d) stand multiplier
    ax = fig.add_subplot(gs[1, 0])
    draw_f6(ax, D, compact=True)
    ax.set_title("D  PD-only stand multiplier (log; hatched = none up to 32x)", fontsize=10, loc="left")
    # (e) table of our three sets
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    rows = [["joint", "current\nKp / Kd", "flat-rec\nKp / Kd", "mountain-rec\nKp / Kd"]]
    for g, nm in (("hip_pitch", "hip pitch"), ("hip_roll", "hip roll"), ("hip_yaw", "hip yaw"), ("knee", "knee"),
                  ("crank", "ankle crank")):
        rows.append([nm] + [f"{PYG_GAINS[k][g][0]:g} / {PYG_GAINS[k][g][1]:g}" for k in OURS])
    rows.append(["action 1 (% limit)"] + [
        " / ".join(f"{D[k]['g'][g]['pct']:.0f}" for g in ("hip_pitch", "hip_yaw", "knee", "ankle")) for k in OURS])
    rows.append(["ankle / (m g h)"] + [f"{D[k]['ratio']:.2f}" for k in OURS])
    rows.append(["PD-only stand x"] + [f"{D[k]['stand']:.2f}" for k in OURS])
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center", colWidths=[0.28, 0.23, 0.23, 0.26])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dddddd")
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("E  Our three gain sets (action 1 row: hip pitch / hip yaw / knee / crank)", fontsize=10, loc="left")
    # (f) key
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=COL[k], edgecolor="k" if k in OURS else "none", label=NAME[k]) for k in ALL]
    ax.legend(handles=hs, loc="upper left", fontsize=10, title="robot colours (same in F1-F6)", title_fontsize=10, frameon=False)
    ax.text(0.0, 0.42, "Reading guide\n"
            "A  green band = where G1, H1, Berkeley and mjlab sit\n"
            "B  a dot to the right of a bar = that terrain needs\n    more Kp than the set provides (30 deg budget)\n"
            "C  line at 1.0: below it, gravity alone topples the\n    robot over its ankles\n"
            "D  below x1 would mean the gains stand unaided; every\n    robot is above it", fontsize=9, va="top")
    fig.suptitle("F7  Our recommended joint gains vs published gains of other robots - summary "
                 "(details: F1-F6)", fontsize=13)
    fig.savefig(OUT / "2026-10-15_rec_F7_summary_dashboard.png", dpi=150)
    plt.close(fig)


def print_table():
    D = gather()
    print("| robot | mass kg | hip pitch Kp/Kd | knee Kp/Kd | ankle Kp/Kd | action 1 % hip pitch / knee / ankle | Kp per kg hip pitch / knee / ankle | ankle/mgh | stand x (all) | stand x (ankle only) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    f = lambda v: "none" if v is None else f"{v:.2f}"
    for k in ALL:
        d = D[k]
        g = d["g"]
        kk = lambda n: f"{g[n]['kp']:g} / {g[n]['kd']:g}"
        print(f"| {NAME[k]} | {d['mass']:.1f} | {kk('hip_pitch')} | {kk('knee')} | {kk('ankle')} | "
              f"{g['hip_pitch']['pct']:.0f} / {g['knee']['pct']:.0f} / {g['ankle']['pct']:.0f} | "
              f"{g['hip_pitch']['kpkg']:.2f} / {g['knee']['kpkg']:.2f} / {g['ankle']['kpkg']:.2f} | "
              f"{d['ratio']:.2f} | {f(d['stand'])} | {f(d['stand_ankle'])} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        print_table()
        sys.exit(0)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    OUT.mkdir(parents=True, exist_ok=True)
    fig_f1()
    fig_f2()
    fig_f3()
    fig_f4()
    fig_f5()
    fig_f6()
    fig_f7()
    print("wrote", sorted(p.name for p in OUT.glob("2026-10-15_rec_F*.png")))
