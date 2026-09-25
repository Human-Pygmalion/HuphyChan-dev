#!/usr/bin/env python3
"""Figures for docs/155_training_pipeline_guide.md (the training walkthrough).

Every number drawn here is read from a real run, not typed in:
  - gains, action scale, effort limits, reward std values: v10a params/env.yaml
  - curriculum stage, DR factor, learning rate, reward, episode length,
    per-term reward contributions: v7 (legonly_ab_v7_heading_revert) tensorboard, P1 + P2

Run with the mjlab venv (tensorboard + yaml live there):
  mujoco-sim/mjlab/.venv/bin/python3 tools/notes/training_guide_figures.py
Labels are English only (matplotlib has no Korean glyphs here).
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJLAB = os.path.join(REPO, "mujoco-sim", "mjlab")
LOGS = os.path.join(MJLAB, "logs", "rsl_rl", "pygmalion_velocity")
OUT = os.path.join(REPO, "docs", "mujoco", "assets")
V10A = "2026-09-25_16-56-05_legonly_ab_v10a_stand01_p1"
V7 = "legonly_ab_v7_heading_revert"


class _AnyLoader(yaml.SafeLoader):
  pass


def _any(loader, suffix, node):
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node, deep=True)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  return loader.construct_scalar(node)


_AnyLoader.add_multi_constructor("", _any)


def env_cfg() -> dict:
  with open(os.path.join(LOGS, V10A, "params", "env.yaml")) as fh:
    return yaml.load(fh, Loader=_AnyLoader)


def scalars(phase: str) -> EventAccumulator:
  f = sorted(glob.glob(os.path.join(LOGS, f"*_{V7}_{phase}", "events*")))[-1]
  e = EventAccumulator(f, size_guidance={"scalars": 0})
  e.Reload()
  return e


def series(e: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
  s = e.Scalars(tag)
  return np.array([x.step for x in s]), np.array([x.value for x in s])


# ---------------------------------------------------------------------------------------
def fig_loop() -> None:
  """One training iteration, drawn as the data actually flows."""
  fig, ax = plt.subplots(figsize=(13, 7.2))
  ax.set_xlim(0, 13)
  ax.set_ylim(0, 7.2)
  ax.axis("off")

  def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color,
                                ec="0.25", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

  def arrow(x0, y0, x1, y1, text="", dy=0.15, color="0.2"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.4))
    if text:
      ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=8.5,
              color=color)

  # the environment side
  ax.add_patch(FancyBboxPatch((0.2, 3.3), 6.3, 3.7, boxstyle="round,pad=0.05",
                              fc="#f3f7fb", ec="#6b8fb3", lw=1.0, ls="--"))
  ax.text(0.35, 6.75, "16,384 simulated robots in parallel (GPU, MuJoCo-Warp)",
          fontsize=10, weight="bold", color="#35597f")
  box(0.5, 5.3, 2.6, 1.1, "Physics step\n200 Hz (dt 0.005 s)", "#dbe8f5")
  box(3.6, 5.3, 2.6, 1.1, "PD controller per motor\ntau = Kp(q* - q) - Kd qdot\nclamped by T-N curve", "#dbe8f5")
  box(3.6, 3.6, 2.6, 1.1, "Observation (actor 45)\n+ noise; critic 82\n(no noise, privileged)", "#e8f1e0")
  box(0.5, 3.6, 2.6, 1.1, "Reward: 22 terms\nx weight x dt, summed\n+ termination / reset", "#fbeede")
  arrow(3.1, 5.85, 3.6, 5.85)
  arrow(3.6, 5.6, 3.1, 5.6)
  ax.text(3.35, 5.05, "x4 physics steps\nper policy step\n(decimation 4 -> 50 Hz)",
          ha="center", fontsize=8, color="0.3")

  # the policy / learner side
  box(7.3, 5.3, 2.5, 1.1, "Actor network\n45 -> 512-256-128 -> 12\n(Gaussian mean + std)", "#efe3f5")
  box(10.3, 5.3, 2.5, 1.1, "Action a (12)\nq* = default + 0.25 a\nclipped to safe window", "#efe3f5")
  box(7.3, 3.6, 2.5, 1.1, "Critic network\n82 -> 512-256-128 -> 1\n(value estimate)", "#efe3f5")
  box(10.3, 3.6, 2.5, 1.1, "Rollout buffer\n24 steps x 16,384 envs\n= 393,216 samples", "#f5f0d8")
  arrow(6.2, 4.4, 7.3, 5.6, "obs", dy=0.05)
  arrow(9.8, 5.85, 10.3, 5.85)
  arrow(11.55, 6.4, 5.0, 6.62, "targets q* back to the motors", dy=0.12, color="#7a3e9d")
  arrow(6.2, 4.15, 7.3, 4.15, "obs", dy=0.1)
  ax.plot([1.8, 1.8, 11.2], [3.6, 3.15, 3.15], color="#b36b00", lw=1.4)
  arrow(11.2, 3.15, 11.2, 3.6, "", color="#b36b00")
  ax.text(8.8, 2.95, "reward, done (per robot, per step)", fontsize=8.5, color="#b36b00",
          ha="center")
  arrow(9.8, 4.15, 10.3, 4.15)

  # the update
  box(5.4, 0.4, 7.4, 1.9,
      "PPO update (once per iteration, after the 24 steps)\n"
      "advantages by GAE (gamma 0.99, lambda 0.95)  ->  5 epochs x 4 mini-batches\n"
      "clipped surrogate (0.2) + value loss (x1.0) - entropy bonus (0.01)\n"
      "Adam, learning rate adapted to keep KL near 0.01, grad norm clipped at 1.0",
      "#fde2e2")
  arrow(12.2, 3.6, 12.2, 2.5, "")
  arrow(8.55, 2.5, 8.55, 3.6, "", color="#a33")
  ax.text(8.65, 2.62, "new weights to critic and actor", fontsize=8.5, color="#a33")

  box(0.3, 0.4, 4.6, 1.9,
      "Around the loop (launcher)\n"
      "P1: gated command curriculum, no randomisation\n"
      "P2: randomisation ramp 0 -> 1 over 10,000 iter\n"
      "     + 4,000 iter at full strength; entropy 0.01 -> 0.002",
      "#eeeeee")
  ax.set_title("One training iteration of the Pygmalion velocity policy "
               "(values from run v10a)", fontsize=12)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "training_guide_loop.png"), dpi=130)
  plt.close(fig)


# ---------------------------------------------------------------------------------------
def fig_pd(cfg: dict) -> None:
  """What one unit of action means in torque, per joint group."""
  groups = [("hip pitch / roll / yaw", 150.0, 120.0, "#1f77b4", 60.0),
            ("knee", 220.0, 120.0, "#d62728", None),
            ("ankle crank A/B", 22.266, 60.0, "#2ca02c", None)]
  scale = 0.25
  err = np.linspace(0, 1.2, 400)
  fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
  for name, kp, eff, col, eff2 in groups:
    tau = np.minimum(kp * err, eff)
    ax[0].plot(np.degrees(err), tau, color=col, lw=2, label=f"{name}: Kp {kp:g}, limit {eff:g} N m")
    if eff2:
      ax[0].plot(np.degrees(err), np.minimum(kp * err, eff2), color=col, lw=1.2, ls="--",
                 label="hip yaw (RS03): limit 60 N m")
    if eff / kp <= err[-1]:
      ax[0].plot(np.degrees(eff / kp), eff, "o", color=col)
    else:
      ax[0].text(np.degrees(err[-1]), kp * err[-1] + 3,
                 f"saturates only at {np.degrees(eff / kp):.0f} deg:\nnever inside its window",
                 color=col, fontsize=8, ha="right")
  for k in (1, 2, 3, 4):
    ax[0].axvline(np.degrees(k * scale), color="0.6", lw=0.8, ls=":")
    ax[0].text(np.degrees(k * scale), 2, f" a={k}", fontsize=8, color="0.4")
  ax[0].set_xlabel("position error q* - q  [deg]  (static, joint at rest)")
  ax[0].set_ylabel("commanded torque [N m]")
  ax[0].set_title("Stiffness Kp turns a position error into torque\n"
                  "(dots: error where the motor saturates)")
  ax[0].legend(fontsize=8, loc="lower right")
  ax[0].grid(alpha=0.3)

  # damping: the same error while the joint is already moving
  v = np.linspace(-10, 10, 200)
  for name, kp, eff, col, _ in groups:
    kd = 1.414 if "crank" in name else 6.0
    ax[1].plot(v, np.clip(kp * np.radians(10) - kd * v, -eff, eff), color=col, lw=2,
               label=f"{name}: Kd {kd:g}")
  ax[1].axhline(0, color="0.5", lw=0.8)
  ax[1].set_xlabel("joint velocity qdot [rad/s]  (error fixed at 10 deg)")
  ax[1].set_ylabel("commanded torque [N m]")
  ax[1].set_title("Damping Kd subtracts torque in proportion to speed\n"
                  "(effective natural frequency = sqrt(Kp / inertia))")
  ax[1].legend(fontsize=8)
  ax[1].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "training_guide_pd.png"), dpi=130)
  plt.close(fig)


# ---------------------------------------------------------------------------------------
def fig_shapes(cfg: dict) -> None:
  r = cfg["rewards"]
  fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
  e = np.linspace(0, 2.0, 400)
  for name, key, col in [("linear velocity (x, y)", "track_linear_velocity", "#1f77b4"),
                         ("turn rate (yaw)", "track_angular_velocity", "#ff7f0e"),
                         ("upright (tilt)", "upright", "#2ca02c")]:
    std = float(r[key]["params"]["std"])
    ax[0].plot(e, np.exp(-(e**2) / std**2), color=col, lw=2, label=f"{name}: std {std:.3f}")
  ax[0].set_xlabel("tracking error (m/s, rad/s, or tilt measure)")
  ax[0].set_ylabel("reward per step (before weight)")
  ax[0].set_title("Tracking kernels exp(-error^2 / std^2)\n"
                  "never negative; gradient vanishes when far off")
  ax[0].legend(fontsize=8)
  ax[0].grid(alpha=0.3)

  d = np.linspace(-0.5, 0.5, 400)
  ax[1].plot(d, np.abs(d), lw=2, label="L1 |x|: same pull at any distance")
  ax[1].plot(d, d**2 / 0.5, lw=2, label="L2 x^2 (scaled): weak near 0, strong far")
  ax[1].set_xlabel("deviation from default pose [rad]")
  ax[1].set_ylabel("cost")
  ax[1].set_title("Why v10b's standing term is L1\n"
                  "L2 barely pushes small drifts back")
  ax[1].legend(fontsize=8)
  ax[1].grid(alpha=0.3)

  k = np.arange(0, 400)
  t = k * 0.02
  for g, col in [(0.99, "#d62728"), (0.995, "0.5"), (0.95, "0.7")]:
    lab = f"gamma {g}: half weight after {np.log(0.5) / np.log(g) * 0.02:.2f} s"
    ax[2].plot(t, g**k, lw=2 if g == 0.99 else 1.2, color=col, label=lab)
  ax[2].set_xlabel("time into the future [s] (policy step 0.02 s)")
  ax[2].set_ylabel("weight of a future reward")
  ax[2].set_title("Discount gamma sets how far ahead the policy plans\n"
                  "(ours: 0.99 -> about 2 s effective horizon)")
  ax[2].legend(fontsize=8)
  ax[2].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "training_guide_reward_shapes.png"), dpi=130)
  plt.close(fig)


# ---------------------------------------------------------------------------------------
def fig_timeline() -> None:
  p1, p2 = scalars("p1"), scalars("p2")
  fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

  def both(tag):
    a = series(p1, tag)
    b = series(p2, tag)
    return a, b

  (s1, v1), (s2, v2) = both("Curriculum/command_vel/stage")
  p1_end = s1.max()
  ax[0].step(s1, v1, where="post", color="#1f77b4", label="P1 command stage")
  ax[0].step(s2, v2, where="post", color="#1f77b4", alpha=0.5)
  (f1, fv1), _ = both("Curriculum/command_vel/forced_promotions")
  jumps = f1[1:][np.diff(fv1) > 0]
  for j in jumps:
    ax[0].axvline(j, color="#d62728", ls="--", lw=1)
    ax[0].text(j, 0.3, " forced\n promotion", color="#d62728", fontsize=8)
  ax[0].set_ylabel("stage (0..4)")
  ax[0].set_title(f"Run v7 as it actually trained: P1 ended at iteration {p1_end:,}, "
                  "then P2 resumed from its last checkpoint")
  ax[0].legend(fontsize=8, loc="lower right")
  ax[0].grid(alpha=0.3)

  (d1, dv1), (d2, dv2) = both("Curriculum/dr_levels/dr_factor")
  ax[1].plot(d1, dv1, color="#2ca02c")
  ax[1].plot(d2, dv2, color="#2ca02c", label="randomisation strength (0 = off, 1 = full)")
  ax[1].set_ylabel("DR factor")
  ax[1].legend(fontsize=8)
  ax[1].grid(alpha=0.3)

  for tag, col, lab in [("Train/mean_reward", "#9467bd", "mean episode reward"),
                        ("Train/mean_episode_length", "#8c564b", "mean episode length [steps]")]:
    (a1, av1), (a2, av2) = both(tag)
    tgt = ax[2] if "reward" in tag else ax[2].twinx() if not hasattr(fig, "_tw") else fig._tw
    if "length" in tag:
      fig._tw = tgt
    tgt.plot(a1, av1, color=col, lw=1, label=lab)
    tgt.plot(a2, av2, color=col, lw=1)
    tgt.set_ylabel(lab, color=col)
  ax[2].grid(alpha=0.3)

  (l1, lv1), (l2, lv2) = both("Loss/learning_rate")
  ax[3].semilogy(l1, lv1, color="#17becf", lw=0.8)
  ax[3].semilogy(l2, lv2, color="#17becf", lw=0.8, label="learning rate (adaptive, KL target 0.01)")
  ax[3].set_ylabel("learning rate")
  ax[3].set_xlabel("iteration (one iteration = 24 policy steps x 16,384 robots)")
  ax[3].legend(fontsize=8)
  ax[3].grid(alpha=0.3)
  for a in ax:
    a.axvline(p1_end, color="0.3", lw=1)
  ax[1].text(p1_end, 0.5, "  P1 | P2", fontsize=9)
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "training_guide_v7_timeline.png"), dpi=130)
  plt.close(fig)


# ---------------------------------------------------------------------------------------
def fig_contrib() -> None:
  e = scalars("p2")
  tags = [t for t in e.Tags()["scalars"] if t.startswith("Episode_Reward/")]
  vals = {}
  for t in tags:
    s = e.Scalars(t)
    vals[t.split("/", 1)[1]] = float(np.mean([x.value for x in s[-100:]]))
  items = sorted(vals.items(), key=lambda kv: kv[1])
  fig, ax = plt.subplots(figsize=(10, 7))
  ax.barh([k for k, _ in items], [v for _, v in items],
          color=["#d62728" if v < 0 else "#1f77b4" for _, v in items])
  ax.axvline(0, color="0.3", lw=0.8)
  ax.set_xlabel("episode sum of (weight x term x dt) / 20 s episode, mean of last 100 iterations\n"
                "for an episode that lasts 20 s this equals weight x (time-average of the term)")
  ax.set_title("Where the reward of a finished policy comes from (run v7, end of P2)\n"
               "blue = earned, red = paid")
  ax.grid(alpha=0.3, axis="x")
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "training_guide_reward_contrib.png"), dpi=130)
  plt.close(fig)
  print("contributions:", {k: round(v, 3) for k, v in items})


if __name__ == "__main__":
  os.makedirs(OUT, exist_ok=True)
  cfg = env_cfg()
  fig_loop()
  fig_pd(cfg)
  fig_shapes(cfg)
  fig_timeline()
  fig_contrib()
  print("wrote training_guide_*.png to", OUT)
