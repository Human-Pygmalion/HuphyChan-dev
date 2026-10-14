#!/usr/bin/env python3
"""How reward and curriculum were given over training iterations, for stand-only runs.

Reads, for each run (= a chain of training phases: phase 1, optional restart, phase 2),
  <phase>/events.out.tfevents.*      TensorBoard scalars (ground truth for what happened)
  <phase>/params/env.yaml            reward weights, curriculum, event ranges as built
  <phase>/repro/launch_manifest.json PYG_* flags the phase was launched with
and writes
  <out>/standonly_schedule.png                randomization / limits / stage / LR / std / entropy
  <out>/standonly_reward_terms.png            per-term reward the policy collected, runs overlaid
  <out>/standonly_reward_stack_<label>.png    stacked positive terms above zero, penalties below
and, with --tables, prints the reward-weight table, the flag differences and key numbers.

TB step convention (verified, see --tables output): the TB step IS the absolute iteration.
A resumed phase restarts its step at the loaded checkpoint's iteration (e.g. v9 phase 2 starts
at 1999, not 0). Overlapping steps between phases are cut: a phase contributes only the steps
before the next phase's first step.

Run with the mjlab venv (needs tensorboard + matplotlib):
  cd mujoco-sim/mjlab && CUDA_VISIBLE_DEVICES="" .venv/bin/python3 \
      ../../tools/notes/standonly_schedule_plots.py --tables
Other runs: --run "label=phase1_dir,phase2_dir" (repeat; dirs under --logs or absolute).
"""
import argparse
import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOGS = os.path.join(REPO, "mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity")
DEFAULT_RUNS = [
    ("v7 stand-only", ["2026-09-16_14-25-25_legonly_ab_v7_standonly_p1",
                       "2026-09-16_18-21-14_legonly_ab_v7_standonly_p2"]),
    ("v7b robust stand", ["2026-09-16_22-47-17_legonly_ab_v7b_robust_stand_p1",
                          "2026-09-16_22-57-26_legonly_ab_v7b_robust_stand_p1r",
                          "2026-09-16_23-49-54_legonly_ab_v7b_robust_stand_p2"]),
    ("v9 caps", ["2026-09-20_18-09-37_legonly_ab_v9_caps_p1",
                 "2026-09-20_19-17-25_legonly_ab_v9_caps_p2"]),
]
AUDIT_CKPTS = [100, 300, 1000, 2000, 3000, 5000, 7000, 9000]
RUN_COLORS = ["#2b6cb0", "#e07b24", "#2f9e5b", "#8e44ad", "#c0392b"]
SMOOTH = 50          # rolling-mean window (iterations) for the per-term plots
RESUME_MASK_CAP = 300  # max iterations dropped after a resume (see Phase.warm_end)
STEPS_PER_ITER = 24


# ----------------------------------------------------------------------------- loading
class _Loader(yaml.SafeLoader):
    pass


def _py_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return str(loader.construct_scalar(node))


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _py_tag)


class Phase:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.env = yaml.load(open(os.path.join(path, "params/env.yaml")), _Loader)
        mf = os.path.join(path, "repro/launch_manifest.json")
        self.flags = json.load(open(mf))["environment"] if os.path.exists(mf) else {}
        ev = sorted(glob.glob(os.path.join(path, "events.out.tfevents.*")))
        if not ev:
            raise FileNotFoundError(f"no events file in {path}")
        ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
        ea.Reload()
        self.tb = {}
        for tag in ea.Tags()["scalars"]:
            a = np.array([(e.step, e.value) for e in ea.Scalars(tag)], dtype=float)
            self.tb[tag] = a
        self.first_step = int(self.tb["Train/mean_reward"][0, 0])
        self.last_step = int(self.tb["Train/mean_reward"][-1, 0])
        # After a resume every environment starts a fresh episode at the same moment, so for
        # the first iterations only early-terminated episodes are logged and the per-episode
        # reward is a short-episode artifact. End of that warm-up = first iteration whose
        # logged mean episode length reaches 990 of 1000 steps (capped).
        el = self.tb["Train/mean_episode_length"]
        ok = el[el[:, 1] >= 990, 0]
        self.warm_end = int(min(ok[0] if len(ok) else self.first_step, self.first_step + RESUME_MASK_CAP))

    def rewards(self):
        return {k: v for k, v in self.env["rewards"].items()}


class Run:
    def __init__(self, label, dirs, color):
        self.label, self.color = label, color
        self.phases = [Phase(d) for d in dirs]
        self.bounds = []   # (iteration, kind) of every later phase start
        for i, p in enumerate(self.phases[1:], 1):
            self.bounds.append((p.first_step, "restart" if i < len(self.phases) - 1 else "phase2"))

    def _cut(self, i):
        return self.phases[i + 1].first_step if i + 1 < len(self.phases) else 10 ** 9

    def stitched(self, tag, mask_resume=False, fn=None):
        """Absolute-iteration series. `fn(phase, steps, values)` may transform per phase."""
        xs, ys = [], []
        for i, p in enumerate(self.phases):
            if tag not in p.tb:
                continue
            a = p.tb[tag]
            lo = p.warm_end if (mask_resume and i > 0) else p.first_step
            m = (a[:, 0] >= lo) & (a[:, 0] < self._cut(i))
            s, v = a[m, 0], a[m, 1]
            if fn is not None:
                v = fn(p, s, v)
            xs.append(s)
            ys.append(v)
        if not xs:
            return np.array([]), np.array([])
        return np.concatenate(xs), np.concatenate(ys)

    def computed(self, fn):
        """Series computed from a phase's config over its own step range (e.g. entropy coef)."""
        xs, ys = [], []
        for i, p in enumerate(self.phases):
            s = np.arange(p.first_step, min(p.last_step + 1, self._cut(i)), dtype=float)
            xs.append(s)
            ys.append(fn(p, s))
        return np.concatenate(xs), np.concatenate(ys)


def smooth(y, w):
    if len(y) == 0:
        return y
    k = np.ones(w)
    num = np.convolve(np.nan_to_num(y), k, "same")
    den = np.convolve(np.isfinite(y).astype(float), k, "same")
    return num / np.maximum(den, 1)


def window_mean(run, tag, it, w=100, mask=True):
    x, y = run.stitched(tag, mask_resume=mask)
    m = (x > it - w / 2) & (x <= it + w / 2)
    return float(np.nanmean(y[m])) if m.any() else float("nan")


# ----------------------------------------------------------------------------- helpers
def entropy_coef(p, s):
    f = p.flags
    if not f.get("PYG_ENTROPY_ANNEAL"):
        return np.full_like(s, 0.01)           # rl_cfg entropy_coef
    a, b = float(f["PYG_ENTROPY_ANNEAL_START"]), float(f["PYG_ENTROPY_ANNEAL_END"])
    c0, c1 = float(f.get("PYG_ENTROPY_START_COEF", 0.01)), float(f.get("PYG_ENTROPY_END_COEF", 0.002))
    frac = np.clip((s - a) / max(1.0, b - a), 0, 1)
    return c0 + (c1 - c0) * frac


def push_limit(p, s, v):
    pm = p.env["curriculum"]["dr_levels"]["params"]["push_max"]["x"][1]
    return v * pm


def add_ckpt_ticks(ax, top=False):
    for c in AUDIT_CKPTS:
        ax.axvline(c, color="0.6", lw=0.7, alpha=0.35, zorder=0)
    if top:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(AUDIT_CKPTS)
        sec.set_xticklabels([str(c) for c in AUDIT_CKPTS], rotation=90, fontsize=8, color="0.35")
        sec.tick_params(length=3, color="0.5")
        sec.set_xlabel("stand-pose audit checkpoints", fontsize=8, color="0.35")


def add_bounds(ax, run, label=False):
    for it, kind in run.bounds:
        ax.axvline(it, color="k", lw=1.0, ls="--" if kind == "phase2" else ":", alpha=0.8)
        if label:
            ax.text(it, 1.0, " phase 2" if kind == "phase2" else " restart", rotation=90,
                    va="top", ha="left", fontsize=8, transform=ax.get_xaxis_transform())


def style(ax, ylabel=None, title=None):
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


# ----------------------------------------------------------------------------- figure 1
def fig_schedule(runs, out):
    rows = [
        ("DR factor (0 = off, 1 = full)", "Curriculum/dr_levels/dr_factor"),
        ("Push speed limit x (m/s)", None),
        ("Command range limit (nominal)", None),
        ("Velocity-command stage", "Curriculum/command_vel/stage"),
        ("Learning rate (log)", "Loss/learning_rate"),
        ("Action std (mean)", "Policy/mean_std"),
        ("Entropy coefficient (from flags)", None),
        ("CAPS term inside loss", "Loss/smooth"),
    ]
    n = len(runs)
    fig, axes = plt.subplots(len(rows), n, figsize=(6.0 * n, 2.5 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    xmax = max(p.last_step for r in runs for p in r.phases)
    for j, r in enumerate(runs):
        for i, (lab, tag) in enumerate(rows):
            ax = axes[i, j]
            style(ax, lab if j == 0 else None)
            add_ckpt_ticks(ax, top=(i == 0))
            add_bounds(ax, r, label=(i == 0))
            if i == 0:
                x, y = r.stitched(tag)
                ax.plot(x, y, color=r.color, lw=1.6)
                ax.set_ylim(-0.03, 1.08)
            elif i == 1:
                x, y = r.stitched("Curriculum/dr_levels/dr_factor", fn=push_limit)
                ax.plot(x, y, color=r.color, lw=1.6)
                ax.set_ylim(-0.05, 1.7)
            elif i == 2:
                for t, c, ls in (("lin_vel_x_max", "#1f4e8c", "-"), ("lin_vel_y_max", "#c0392b", "--"),
                                 ("ang_vel_z_max", "#2f9e5b", ":")):
                    x, y = r.stitched("Curriculum/command_vel/" + t)
                    ax.plot(x, y, color=c, lw=1.6, ls=ls, label=t.replace("_max", "") + " upper limit")
                ax.set_ylim(0, 2.9)
                if j == 0:
                    ax.legend(fontsize=8, loc="lower right")
                ax.text(0.68, 0.55, "every environment is commanded zero\n(rel_standing_envs = 1.0):\nthese limits never change a command",
                        transform=ax.transAxes, fontsize=8, ha="center", color="0.3")
            elif i == 3:
                x, y = r.stitched(tag)
                ax.plot(x, y, color=r.color, lw=1.6)
                ax.set_ylim(-0.2, 4.4)
            elif i in (4, 5):
                x, y = r.stitched(tag)
                ax.plot(x, y, color=r.color, lw=0.5, alpha=0.3)
                ax.plot(x, smooth(y, 25), color=r.color, lw=1.5)
                if i == 4:
                    ax.set_yscale("log")
            elif i == 6:
                x, y = r.computed(entropy_coef)
                ax.plot(x, y, color=r.color, lw=1.6)
                ax.set_ylim(0, 0.0115)
            elif i == 7:
                x, y = r.stitched(tag)
                if len(x):
                    ax.plot(x, y, color=r.color, lw=0.5, alpha=0.3)
                    ax.plot(x, smooth(y, 25), color=r.color, lw=1.5)
                else:
                    ax.text(0.5, 0.5, "no CAPS term (plain PPO loss)", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="0.4")
            if i == len(rows) - 1:
                ax.set_xlabel("absolute training iteration", fontsize=10)
        f0, f1 = r.phases[0].flags, r.phases[-1].flags
        rn = lambda f: (f"joints +-{f['PYG_RESET_JOINT_NOISE']} rad, tilt +-{f['PYG_RESET_TILT']} rad"
                        if f.get("PYG_RESET_JOINT_NOISE") else "none (exact keyframe, upright)")
        sub = (f"reset noise: {rn(f1)}\n"
               f"DR window {f1.get('PYG_DR_START_ITER')}-{f1.get('PYG_DR_END_ITER')} (log ends {r.phases[-1].last_step}), "
               f"strong DR {'on' if f1.get('PYG_DR_STRONG') else 'off'}, CAPS lambda {f1.get('PYG_CAPS_S', 'off')}")
        axes[0, j].annotate(f"{r.label}\n{sub}", xy=(0.5, 1.30), xycoords="axes fraction", ha="center",
                            va="bottom", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlim(-100, xmax + 150)
    fig.suptitle("Randomization and curriculum schedule over training iterations "
                 "(dashed line = phase 2 starts, dotted = restart; grey ticks = audit checkpoints)",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 2 / 3
def reward_meta(runs):
    """Ordered term list: (name, weight) from the first phase of the first run; verifies identical."""
    base = {k: v["weight"] for k, v in runs[0].phases[0].rewards().items()}
    for r in runs:
        for p in r.phases:
            w = {k: v["weight"] for k, v in p.rewards().items()}
            if w != base:
                print(f"WARNING reward weights differ in {p.name}")
    return base


def nonzero_terms(runs, weights):
    nz, zero = [], []
    for t in weights:
        tag = "Episode_Reward/" + t
        mx = 0.0
        for r in runs:
            for p in r.phases:
                if tag in p.tb:
                    mx = max(mx, float(np.abs(p.tb[tag][:, 1]).max()))
        (nz if mx > 0 else zero).append(t)
    pos = sorted([t for t in nz if weights[t] > 0], key=lambda t: -weights[t])
    neg = [t for t in nz if weights[t] < 0]
    return pos, neg, zero


def fig_terms(runs, out):
    weights = reward_meta(runs)
    pos, neg, zero = nonzero_terms(runs, weights)
    panels = [("Episode_Reward/" + t, f"{t}  (weight {weights[t]:+g})") for t in pos + neg]
    panels.append(("Train/mean_reward", "TOTAL mean episode return (20 s)"))
    panels.append(("Loss/smooth", "CAPS term in loss (v9 only)"))
    ncol = 4
    nrow = int(np.ceil((len(panels) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.1 * nrow), squeeze=False)
    flat = axes.ravel()
    for k, (tag, title) in enumerate(panels):
        ax = flat[k]
        style(ax, None, title)
        add_ckpt_ticks(ax, top=(k < ncol))
        lo, hi = np.inf, -np.inf
        for r in runs:
            x, y = r.stitched(tag, mask_resume=True)
            if len(x) == 0:
                continue
            ys = smooth(y, SMOOTH)
            ax.plot(x, y, color=r.color, lw=0.4, alpha=0.18)
            ax.plot(x, ys, color=r.color, lw=1.5, label=r.label)
            lo, hi = min(lo, ys.min()), max(hi, ys.max())
            for it, kind in r.bounds:
                if kind == "phase2":
                    ax.axvline(it, color=r.color, lw=0.9, ls="--", alpha=0.7)
        if np.isfinite(lo):
            pad = 0.08 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
        if k >= len(panels) - ncol:
            ax.set_xlabel("absolute iteration", fontsize=9)
    leg = flat[len(panels)]
    leg.axis("off")
    for r in runs:
        leg.plot([], [], color=r.color, lw=2, label=r.label)
    leg.plot([], [], color="k", lw=0.9, ls="--", label="phase 2 starts (per run colour)")
    leg.plot([], [], color="0.6", lw=0.7, label="audit checkpoints")
    leg.legend(loc="upper left", fontsize=10, frameon=False)
    leg.text(0.0, 0.40,
             f"Smoothed: rolling mean of {SMOOTH} iterations; faint line = raw.\n"
             "Warm-up after each resume dropped (until mean episode\nlength first reaches 990/1000 steps; 44-130 iterations).\n"
             "Episode_Reward = per-episode sum of weight x value x dt,\ndivided by 20 s (already weighted).\n"
             "Always exactly zero (command-gated, all commands are zero):\n" + textwrap.fill(", ".join(zero), 62),
             transform=leg.transAxes, fontsize=8.5, va="top")
    for k in range(len(panels) + 1, len(flat)):
        flat[k].axis("off")
    fig.suptitle("Reward collected per term over training (same y-range for every run in a panel)",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=100)
    plt.close(fig)


def fig_stack(run, weights, out):
    pos, neg, zero = nonzero_terms([run], weights)
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(pos + neg)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for ax, which in ((a1, "all"), (a2, "pen")):
        style(ax)
        add_ckpt_ticks(ax, top=(which == "all"))
        add_bounds(ax, run, label=(which == "all"))
    # common x grid with masked resume intervals removed
    xs = None
    series = {}
    for t in pos + neg:
        x, y = run.stitched("Episode_Reward/" + t, mask_resume=True)
        series[t] = smooth(y, SMOOTH)
        xs = x
    p_stack = [series[t] for t in pos]
    n_stack = [series[t] for t in neg]
    a1.stackplot(xs, *p_stack, colors=[colors[t] for t in pos],
                 labels=[f"{t} ({weights[t]:+g})" for t in pos], lw=0)
    a1.stackplot(xs, *n_stack, colors=[colors[t] for t in neg],
                 labels=[f"{t} ({weights[t]:+g})" for t in neg], lw=0)
    a2.stackplot(xs, *n_stack, colors=[colors[t] for t in neg], lw=0)
    tot = sum(series[t] for t in pos + neg)
    xr, yr = run.stitched("Train/mean_reward", mask_resume=True)
    a1.plot(xs, tot, color="k", lw=1.2, label="sum of all terms")
    a1.plot(xr, smooth(yr / 20.0, SMOOTH), color="k", lw=1.0, ls=":", label="Train/mean_reward / 20 s")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_ylabel("reward per second (weighted)", fontsize=10)
    a2.set_ylabel("penalty terms only (zoom)", fontsize=10)
    a2.set_xlabel("absolute training iteration", fontsize=10)
    a1.legend(fontsize=8.5, ncol=2, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    a1.set_title(f"{run.label}: positive terms stacked above zero, penalties below "
                 f"(rolling mean {SMOOTH} it.)", fontsize=11, loc="left", pad=60)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def print_tables(runs):
    weights = reward_meta(runs)
    print("\n## reward weights per phase (env.yaml) -- identical across phases:",
          all({k: v["weight"] for k, v in p.rewards().items()} == weights
              for r in runs for p in r.phases))
    for t, w in weights.items():
        prm = runs[0].phases[0].rewards()[t]["params"]
        short = {k: v for k, v in prm.items() if k not in ("asset_cfg", "func", "rated", "std_walking", "std_running")}
        print(f"{t:26s} {w:+g}  {json.dumps(short)[:150]}")
    print("\n## flag differences (launch_manifest environment)")
    allp = [(r.label, p) for r in runs for p in r.phases]
    keys = sorted(set().union(*[set(p.flags) for _, p in allp]))
    for k in keys:
        vals = [p.flags.get(k, "-") for _, p in allp]
        if len(set(vals)) > 1:
            print(f"{k:28s}", " | ".join(f"{p.name.split('_', 2)[2][:22]}={v[:12]}" for (_, p), v in zip(allp, vals)))
    print("\n## step convention: first/last TB step per phase, and overlap check")
    for r in runs:
        for i, p in enumerate(r.phases):
            print(f"{r.label:18s} {p.name[-30:]:32s} steps {p.first_step}..{p.last_step}  "
                  f"max_iterations(agent.yaml) {yaml.load(open(os.path.join(p.path, 'params/agent.yaml')), _Loader)['max_iterations']}")
    print("\n## key schedule facts (TB)")
    for r in runs:
        x, y = r.stitched("Curriculum/dr_levels/dr_factor")
        nz = x[y > 0]
        full = x[y >= 0.9999]
        print(f"{r.label:18s} dr_factor>0 from {int(nz[0]) if len(nz) else None}, reaches 1.0 at "
              f"{int(full[0]) if len(full) else None}, max {y.max():.3f} at last step {int(x[-1])}")
        x, y = r.stitched("Curriculum/command_vel/stage")
        ch = [(int(x[i]), round(float(y[i]), 2)) for i in range(1, len(y)) if abs(y[i] - y[i - 1]) >= 0.9]
        print("    stage jumps (iteration, new stage):", ch[:8])
        for t in ("lin_vel_x_max", "lin_vel_y_max", "ang_vel_z_max"):
            x, y = r.stitched("Curriculum/command_vel/" + t)
            print(f"    {t} final {y[-1]:.2f}")
        x, y = r.stitched("Loss/learning_rate")
        print(f"    LR at 2000/5000/last: {np.interp([2000, 5000, x[-1]], x, y)}")
        x, y = r.stitched("Policy/mean_std")
        print(f"    action std at 100/1000/2000/3000/5000/7000/last: {np.round(np.interp([100, 1000, 2000, 3000, 5000, 7000, x[-1]], x, y), 4)}")
    print("\n## per-term reward, mean over a 100-iteration window around each iteration")
    pos, neg, zero = nonzero_terms(runs, weights)
    its = [1000, 3000, 5000, 7000, 8998]
    print("term".ljust(26), *[f"{r.label[:9]}@{it}".rjust(15) for it in its for r in runs])
    for t in pos + neg + ["__total__"]:
        tag = "Episode_Reward/" + t if t != "__total__" else "Train/mean_reward"
        vals = [window_mean(r, tag, it) for it in its for r in runs]
        print(t.ljust(26), *[f"{v:15.4f}" for v in vals])
    print("\nalways zero:", zero)
    for r in runs:
        x, y = r.stitched("Loss/smooth")
        if len(x):
            print(f"{r.label} Loss/smooth (CAPS value) at 2000/3000/5000/7000/last:",
                  [round(window_mean(r, 'Loss/smooth', i), 4) for i in (2000, 3000, 5000, 7000, 8998)],
                  " Loss/surrogate:", [round(window_mean(r, 'Loss/surrogate', i), 4) for i in (3000, 8998)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", help='"label=dir1,dir2,..." (repeatable)')
    ap.add_argument("--logs", default=DEFAULT_LOGS)
    ap.add_argument("--out", default=os.path.join(REPO, "docs/mujoco/assets"))
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    spec = DEFAULT_RUNS if not a.run else [(s.split("=", 1)[0], s.split("=", 1)[1].split(",")) for s in a.run]
    runs = []
    for i, (label, dirs) in enumerate(spec):
        full = [d if os.path.isabs(d) else os.path.join(a.logs, d) for d in dirs]
        runs.append(Run(label, full, RUN_COLORS[i % len(RUN_COLORS)]))
    os.makedirs(a.out, exist_ok=True)
    if a.tables:
        print_tables(runs)
    fig_schedule(runs, os.path.join(a.out, "standonly_schedule.png"))
    fig_terms(runs, os.path.join(a.out, "standonly_reward_terms.png"))
    weights = reward_meta(runs)
    for r in runs:
        slug = r.label.lower().replace(" ", "_").replace("-", "")
        fig_stack(r, weights, os.path.join(a.out, f"standonly_reward_stack_{slug}.png"))
    print("wrote figures to", a.out)


if __name__ == "__main__":
    main()
