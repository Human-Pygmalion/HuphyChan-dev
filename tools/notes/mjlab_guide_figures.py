#!/usr/bin/env python3
"""Diagrams for docs/156_mjlab_and_robot_modeling_guide.md.

  mjlab_guide_layers.png    mjlab's two layers + where our code sits (upstream vs ours)
  mjlab_guide_modeling.png  real robot -> CAD -> mass/geometry/motors -> MJCF -> checks -> training

Box contents name real files; line counts are measured at run time from the repository, so
the figure cannot drift from the code. English labels only.
  python3 tools/notes/mjlab_guide_figures.py
"""

from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MJ = os.path.join(REPO, "mujoco-sim", "mjlab")
SRC = os.path.join(MJ, "src", "mjlab")
OUT = os.path.join(REPO, "docs", "mujoco", "assets")


def nlines(rel: str) -> int:
  p = os.path.join(REPO, rel)
  with open(p, encoding="utf-8", errors="ignore") as fh:
    return sum(1 for _ in fh)


def box(ax, x, y, w, h, text, fc, ec="0.25", fs=9, lw=1.2, ls="-"):
  ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=lw,
                              ls=ls))
  ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color="0.25", text="", fs=8):
  ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                               color=color, lw=1.3))
  if text:
    ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, text, fontsize=fs, color=color)


UP = "#e3ecf6"   # upstream mjlab
OURS = "#fbe3cf"  # written or heavily changed by us
EXT = "#e6f2e1"  # outside mjlab (our repo tools)


def fig_layers() -> None:
  fig, ax = plt.subplots(figsize=(14, 9.2))
  ax.set_xlim(0, 14)
  ax.set_ylim(0, 9.2)
  ax.axis("off")
  pc = "mujoco-sim/mjlab/src/mjlab/asset_zoo/robots/pygmalion/pygmalion_constants.py"
  ec = "mujoco-sim/mjlab/src/mjlab/tasks/velocity/config/pygmalion/env_cfgs.py"
  rw = "mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/rewards.py"
  cu = "mujoco-sim/mjlab/src/mjlab/tasks/velocity/mdp/curriculums.py"
  rl = "mujoco-sim/mjlab/analysis/run_v2_scratch.py"

  ax.text(0.2, 8.85, "mjlab = upstream framework (blue)  +  our robot, task and tools (orange)  "
          "+  repository tools outside mjlab (green)", fontsize=11, weight="bold")

  # training entry
  box(ax, 0.3, 7.6, 4.3, 0.9, f"analysis/run_v2_scratch.py  ({nlines(rl):,} lines)\n"
      "two-phase launcher: P1 curriculum -> P2 DR ramp", OURS)
  box(ax, 5.0, 7.6, 3.9, 0.9, "scripts/train.py  /  analysis/train_wandb_video.py\n"
      "parse task id + CLI overrides", UP)
  box(ax, 9.3, 7.6, 4.4, 0.9, "rl/: RSL-RL OnPolicyRunner, PPO, ONNX export\n"
      "(+ ours: smooth_ppo.py, connectome_policy.py)", UP)
  arrow(ax, 4.6, 8.05, 5.0, 8.05)
  arrow(ax, 8.9, 8.05, 9.3, 8.05)

  # task layer
  ax.add_patch(FancyBboxPatch((0.2, 4.1), 13.6, 3.2, boxstyle="round,pad=0.04", fc="#fafafa",
                              ec="0.5", ls="--"))
  ax.text(0.35, 7.0, "MANAGER LAYER  (the RL problem)   ManagerBasedRlEnv + 8 managers",
          fontsize=10, weight="bold", color="0.3")
  box(ax, 0.4, 5.6, 4.4, 1.2, "tasks/velocity/config/pygmalion/\n"
      f"env_cfgs.py ({nlines(ec):,})  rl_cfg.py  __init__.py\n"
      "task id 'Mjlab-Velocity-Flat-Pygmalion'\nall PYG_* switches are read here", OURS)
  box(ax, 5.1, 5.6, 4.1, 1.2, "tasks/velocity/mdp/  (terms)\n"
      f"rewards.py ({nlines(rw):,})  curriculums.py ({nlines(cu):,})\n"
      "observations.py  velocity_command.py", OURS)
  box(ax, 9.5, 5.6, 4.1, 1.2, "managers/  envs/  (upstream)\n"
      "observation, action, reward, termination,\nevent, command, curriculum, metrics", UP)
  box(ax, 0.4, 4.3, 6.4, 1.0, "tasks/velocity/velocity_env_cfg.py  (upstream base task, small edits)\n"
      "height scan moved to critic only; PYG_UNEVEN terrain", UP)
  box(ax, 7.1, 4.3, 6.5, 1.0, "envs/mdp/events.py (+62: our DR events)   terrains/config.py\n"
      "(+ UNEVEN_TERRAINS_CFG: no stairs for a blind policy)", UP)

  # sim layer
  ax.add_patch(FancyBboxPatch((0.2, 0.3), 13.6, 3.5, boxstyle="round,pad=0.04", fc="#fafafa",
                              ec="0.5", ls="--"))
  ax.text(0.35, 3.5, "SIMULATION LAYER  (the robot and world)   MjSpec -> MjModel -> MuJoCo Warp",
          fontsize=10, weight="bold", color="0.3")
  box(ax, 0.4, 1.8, 4.6, 1.4, "asset_zoo/robots/pygmalion/\n"
      f"pygmalion_constants.py ({nlines(pc):,})\nmotors, gains, keyframes, joint names,\n"
      "action scale, safe clip, collisions", OURS)
  box(ax, 5.3, 1.8, 3.9, 1.4, "xmls/<Variant>_<tag>{,_loop}.xml\n"
      "MJCF written by build_robot.py\n(never edited by hand)", OURS)
  box(ax, 9.5, 1.8, 4.1, 1.4, "tn_actuator.py  ankle_rp_actuator.py\n"
      "PD clamped by measured T-N curve\n(extends actuator/pd_actuator.py)", OURS)
  box(ax, 0.4, 0.5, 4.2, 1.0, "entity/  scene/  sensor/\nentity = MJCF + Python cfg", UP)
  box(ax, 4.9, 0.5, 4.2, 1.0, "actuator/  (upstream)\nIdealPd, DcMotor, Builtin, Learned", UP)
  box(ax, 9.4, 0.5, 4.2, 1.0, "sim/  (upstream)\nMuJoCo Warp, CUDA graph, N worlds", UP)

  arrow(ax, 11.5, 7.6, 11.5, 6.8)
  arrow(ax, 5.3, 2.5, 5.0, 2.5)
  ax.text(7.0, 3.85, "the manager layer's terms read the robot through Entity (joint/body regex)",
          fontsize=8.5, color="0.35", ha="center")

  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "mjlab_guide_layers.png"), dpi=130)
  plt.close(fig)


def fig_modeling() -> None:
  fig, ax = plt.subplots(figsize=(14, 10.5))
  ax.set_xlim(0, 14)
  ax.set_ylim(0, 10.5)
  ax.axis("off")
  ax.set_title("From the real robot to a trainable model: what is measured, where, and in which order",
               fontsize=12)
  t = "tools/robot_model/"

  # column 1: sources of truth
  ax.text(0.3, 10.0, "1  SOURCES OF TRUTH", fontsize=10, weight="bold")
  box(ax, 0.2, 8.5, 3.4, 1.2, "Fusion 360 CAD (one leg)\nwork on an EXPORT COPY only\n"
      "(tools/fusion/mcp_client.py)", EXT)
  box(ax, 0.2, 7.0, 3.4, 1.2, "Scale: printed parts weighed\nalu_parts_measured.json\n"
      "-> density per part", EXT)
  box(ax, 0.2, 5.5, 3.4, 1.2, "Motor bench (System_ID/)\ncoast-down, inertia, stiction\n"
      "-> MOTOR_MEAS", EXT)
  box(ax, 0.2, 4.0, 3.4, 1.2, "Motor_Spec/*_TN_curve_48V.csv\nvendor torque-speed curve\n"
      "+ measured no-load 19.9 rad/s", EXT)
  box(ax, 0.2, 2.5, 3.4, 1.2, "Sign convention (user)\nrotor face, clockwise = +\n"
      "motor_sign_check.py", EXT)

  # column 2: build pipeline (make_printed_robot.sh)
  ax.text(4.2, 10.0, "2  BUILD  (tools/robot_model/make_printed_robot.sh)", fontsize=10,
          weight="bold")
  steps = [
    ("1 dump_bodies.py", "every CAD body: mass, COM,\ninertia, bbox, visibility"),
    ("2 massprops_fusion.py", "bodies -> rigid LINKS\ncatalogue motor masses asserted"),
    ("3 motor_proxies_fusion.py", "motor cylinders: centre, axis,\nradius from the SAME dump"),
    ("4 build_robot.py  (+ rom_check.py)", "URDF + MJCF, mirror to 2 legs,\n"
     "CAD->sim frame, ROM, 3 variants"),
    ("5 validate_robot.py", "mass, L/R, joint sweeps,\ninertia read-back, figures"),
    ("6 mass_dr.py", "per-body uncertainty ->\nper-link DR ranges (json)"),
  ]
  y = 8.6
  for i, (a, b) in enumerate(steps):
    box(ax, 4.1, y, 4.4, 1.05, f"{a}\n{b}", OURS if i != 4 else "#fff3c4", fs=8.6)
    if i:
      arrow(ax, 6.3, y + 1.05 + 0.3, 6.3, y + 1.05)
    y -= 1.35
  arrow(ax, 3.6, 9.1, 4.1, 9.1)
  arrow(ax, 3.6, 7.6, 4.1, 7.8)
  arrow(ax, 3.6, 3.1, 4.1, 4.7, text="signs")

  # column 3: into mjlab + checks
  ax.text(9.1, 10.0, "3  INTO mjlab  +  CHECKS BEFORE TRAINING", fontsize=10, weight="bold")
  box(ax, 9.0, 8.5, 4.8, 1.2, "xmls/LegOnly_<tag>_loop.xml\n2-RSU ankle as equality/connect\n"
      "selected by PYG_MODEL_TAG + PYG_ANKLE_MODE=AB", OURS)
  box(ax, 9.0, 7.0, 4.8, 1.2, "pygmalion_constants.py\nMOTOR_MEAS -> armature/friction/viscous\n"
      "gains (crank = ankle x r^2 / 2), T-N clamp", OURS)
  box(ax, 9.0, 5.5, 4.8, 1.2, "loop_ankle_verify.py\nclosure ~0 mm, lever ratio 1.25,\n"
      "reach design ROM, sole settles flat", "#fff3c4")
  box(ax, 9.0, 4.0, 4.8, 1.2, "check_motor_params.py  pd_sim_vs_real.py\n"
      "same command to real motor and 1-joint sim\nparam_provenance.py (file:line:value)", "#fff3c4")
  box(ax, 9.0, 2.5, 4.8, 1.2, "loop_bent_keyframe.py  (reset pose)\nankle = knee - hip so the sole is flat\n"
      "render it and LOOK", "#fff3c4")
  box(ax, 9.0, 0.8, 4.8, 1.2, "TRAIN  (docs/155)\nrun_v2_scratch.py with the same PYG_* \n"
      "in training, measurement and play", "#f0e0f5")
  arrow(ax, 8.5, 5.07, 9.0, 9.0)
  ax.text(8.55, 6.2, "xml", fontsize=8, color="0.25")
  for (x0, y0, lab, yl) in [(3.6, 6.1, "MOTOR_MEAS", 7.75), (3.6, 4.6, "T-N csv", 7.35)]:
    ax.add_patch(FancyArrowPatch((x0, y0), (9.0, yl - 0.25), arrowstyle="-|>",
                                 mutation_scale=12, color="#1f6f8b", lw=1.0, ls="--",
                                 alpha=0.55))
    ax.text(8.95, yl - 0.1, lab, fontsize=7.5, color="#1f6f8b", ha="right")
  arrow(ax, 8.5, 2.37, 9.0, 1.4)
  ax.text(6.3, 0.55, "mass_dr json -> PYG_MASS_DR_JSON (used in P2 with PYG_INERTIAL_DR=1)",
          fontsize=8, color="0.25", ha="center")
  for yy in (8.5, 7.0, 5.5, 4.0):
    arrow(ax, 11.4, yy, 11.4, yy - 0.3)
  arrow(ax, 11.4, 2.5, 11.4, 2.0)

  ax.text(0.2, 1.25, "yellow = a check that has caught a real, silent error in this project\n"
          "(sign flips x9, mirrored ranges x8, motor moved 75 mm, ankle constant from the\n"
          "wrong crouch, 4.2x motor mismatch inside one model)", fontsize=8.5, color="0.3")
  fig.tight_layout()
  fig.savefig(os.path.join(OUT, "mjlab_guide_modeling.png"), dpi=130)
  plt.close(fig)


if __name__ == "__main__":
  os.makedirs(OUT, exist_ok=True)
  fig_layers()
  fig_modeling()
  src = os.path.join(REPO, "tools", "robot_model", "fusion_snapshots", "v30_inspection",
                     "renders_armfix", "Pygmalion_v30_variants_all_views.png")
  shutil.copy2(src, os.path.join(OUT, "mjlab_guide_variants.png"))
  print("wrote mjlab_guide_*.png to", OUT)
