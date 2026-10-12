#!/usr/bin/env python3
"""V2 scratch-training launcher: phase 1 (gated curriculum, DR off) -> phase 2 (DR ramp).

docs/103_v2_training_plan.md §4 step 6 asks for exactly this: "런처 수정: P1 실측 길이 ->
DR start/end 자동 계산". The gated curriculum (V3) made it mandatory rather than nice to
have -- with promotion by report card, P1's length is an OUTCOME, not a schedule, so the
DR ramp window CANNOT be a constant baked into the command line. The failure this prevents
is on record: 2026-07-15, uneven2 P1 finished at 12k while PYG_DR_START_ITER still said
20000, common_step_counter never reached start_step, and dr_factor sat at 0.33 for the
whole of phase 2 (env_cfgs.py, dr_levels comment).

WHAT IT DOES
  P1  from scratch (no --agent.resume), flat, DR OFF (PYG_DR_START_ITER pushed far past
      the horizon so dr_levels holds factor at 0), PYG_GATED_CURRICULUM=1, entropy anneal
      OFF, PYG_STUDENT_TEACHER=1, PYG_CRITIC_DR_OBS=1 *from the start* (planner decision:
      the actor is the deployable 45-D Student and the critic obs width is
      then identical in both phases, so P2 can FULL-resume actor+critic+optimizer; a critic
      that grew at the phase boundary would force an actor-only warm start and throw the
      value function away).
      Watched until the command curriculum SETTLES at its top stage, or a hard iter cap.
  P2  FULL resume from P1's last checkpoint. THEN, and only then, the DR window is computed
      from the measured P1 length:  PYG_DR_START_ITER = P1_END,
      PYG_DR_END_ITER = P1_END + ramp (default 10000), plus a digest tail (default 4000).
      Entropy anneal ON across the whole P2 window (V4), DR events ramp 0 -> 1 (push,
      friction, encoder bias) exactly as docs/103 §3b describes.

P1 END DETECTION -- why the console log and not the tfevents
  Both carry the same numbers (rsl_rl's logger writes every curriculum return value to BOTH
  the tfevents scalar store and the printed block, keyed by the same absolute iteration).
  The console log wins on three counts:
    1. It is the only place the PROMOTION LINES appear ("[gated-curriculum] stage 2 -> 3 at
       iter 194: gate passed after dwell 49 iters ..."), which is what makes a graduation
       decision auditable after the fact.
    2. It exists regardless of the logger backend; the tfevents file is a side effect of
       whichever writer --agent.logger selects.
    3. No flush lag. tf SummaryWriter buffers on a 120 s timer; the print block is ~5 KB,
       larger than stdio's buffer, so it lands essentially per iteration.
  tfevents is kept as an automatic fallback (--source auto) for the case where a log file
  was lost or rotated: same keys, same iteration index, so the two merge cleanly.

  "Settled" mirrors the gate's OWN arithmetic (mdp.commands_vel_gated), because the top
  stage never promotes and therefore never prints a line of its own:
      stage == n_stages-1  AND  stage_dwell_iters >= PYG_GATE_MIN_DWELL
      AND gate_window_episodes >= PYG_GATE_MIN_EPISODES
      AND gate_fell_over <= PYG_GATE_FELL_MAX
      AND (gate_baseline is nan  OR  gate_err_steady <= gate_baseline * PYG_GATE_ERR_RATIO)
  held for --settle-hold consecutive logged iterations (the trailing window is 100 iters, so
  consecutive samples are correlated; the hold is cheap insurance against one lucky window).

IDEMPOTENT
  State lives in analysis/out/v2_scratch_<run>.json. Re-invoking the launcher CONTINUES:
  a live phase is re-attached and watched, a dead-but-unfinished phase is resumed from its
  latest checkpoint (watchdog-style 'r' suffix chain), a finished P1 goes straight to P2.
  Nothing is ever restarted from scratch once a checkpoint exists.

BOTH PHASES REGISTER IN analysis/out/watchdog_runs.json (enabled, full env) so watchdog.sh
  auto-resumes a crash. The P1 entry is DISABLED at the instant of graduation -- otherwise
  the watchdog would see "trainer gone at iter 9000 < total 16000" and resume the phase we
  just deliberately ended.

USAGE
  bash analysis/run_v2_scratch.sh --dry-run
  bash analysis/run_v2_scratch.sh --smoke
  nohup bash analysis/run_v2_scratch.sh --run v2s1_AB --ankle AB > analysis/out/v2s1_AB.out 2>&1 &

CONTINUE MODE (2026-09-05) -- "+N more iterations from an existing checkpoint"
  bash analysis/run_v2_scratch.sh --continue-from logs/rsl_rl/pygmalion_velocity/<P1-or-P2 dir> \
       --continue-ckpt model_18099.pt --extra-iters 800 --run <new_run_name> \
       [--env PYG_SOME_FLAG=value ...]

  WHY THIS EXISTS, AND WHY IT IS NOT JUST `train_wandb_video.py --agent.resume True`
    A plain resume command is a perfectly good way to add iterations, but it records
    nothing: no run note gate, no reproducibility snapshot, no watchdog entry, no hourly
    review loop, no accumulate video, no GPU lock. Every one of those is mandatory for a
    training attempt on this project (see the module-level rules this file already
    enforces for P1/P2). CONTINUE mode is the standard resume command wrapped in exactly
    the same recording stack P1/P2 use, so an "add 800 iterations and see if the fix
    helped" A/B run is not a second-class, unrecorded experiment.

  WHAT GETS INHERITED, AND WHY THAT IS THE WHOLE POINT
    The parent run's OWN `repro/launch_manifest.json` (written by snapshot_repro() at its
    launch) is the source of truth for the new run's environment: every PYG_* key it
    carried is copied verbatim, and only keys named on this invocation's --env are
    changed. This is deliberate, not a shortcut. It means:
      - PYG_GATED_CURRICULUM, PYG_DR_START_ITER / PYG_DR_END_ITER, PYG_ENTROPY_ANNEAL* all
        come from the PARENT's values. Since the parent already finished its curriculum
        and DR ramp, those windows are iteration ranges the new run's absolute iteration
        counter has already passed -- there is no re-gating and no re-ramping to configure
        here, because the parent's own numbers already describe "fully promoted, DR at
        1.0" once the checkpoint's iteration is past them.
      - Reading the manifest (rather than recomputing p1_env()/p2_env() for a "similar"
        config) is what makes --env the ONLY deliberate variable between the parent and
        the continued run -- the guarantee an A/B comparison needs. Recomputing from
        scratch risks silently drifting on a flag this launcher's defaults changed since
        the parent was launched.
    A run without a repro/launch_manifest.json (pre-dates this snapshot mechanism, or lost
    it) cannot be continued this way -- there is nothing to inherit from.

  RESUME SEMANTICS -- full resume, single untracked phase
    `--agent.resume True --agent.load-run <parent dir basename> --agent.load-checkpoint
    <ckpt>` loads actor + critic + optimizer (same task, same observation layout as P1/P2
    guarantee already, so nothing forces an actor-only warm start here). There is no
    curriculum gate to watch and no DR window to compute -- the run is driven to
    completion the same way P2's tail is (poll for the checkpoint, resume across a crash
    via the same watchdog-style 'r'-suffix chain), just without P2's settle/gate logic.
    `--agent.max-iterations` passed to the trainer is --extra-iters (iterations REMAINING
    from the loaded checkpoint, exactly like every other resume in this file); the
    absolute iteration the run stops at is (checkpoint's own iteration) + --extra-iters,
    and that sum is what gets registered as the watchdog's total_iters so an unfinished
    continue run is never mistaken for a finished one.

  STATE FILE (analysis/out/v2_scratch_<run>.json) -- new keys, P1/P2 schema untouched
    "mode": "continue" at the top level, plus a "continue" phase dict carrying
    parent_run_dir, parent_ckpt, parent_iter (parsed from the checkpoint filename),
    extra_iters, total_iters_abs, env_overrides (ONLY the --env keys that changed
    something, so the state file itself documents the A/B's single variable), env (the
    full inherited+overridden environment actually launched), status/end_iter/ckpt_dir/
    ckpt/finished_at (same names P1/P2 use). A state file that already has "p1"/"p2" keys
    is untouched by this mode and vice versa -- the two schemas never mix in one file
    because a run name is either continue-mode or scratch-mode for its whole lifetime.

  NAMING -- no forced suffix
    Unlike P1/P2 (which the launcher appends itself), --run in CONTINUE mode is used
    VERBATIM as the trainer's --agent.run-name and the state-file/watchdog key. Name it
    so it cannot collide with a scratch run's "<run>_p1"/"<run>_p2" (a trailing "_c" is
    the convention used in this project's run notes, e.g. "legonly_ab_v2_c"), and so the
    run note glob (`docs/experiments/*<run>*.md`) matches only the note meant for it.

  WHAT IS DELIBERATELY NOT REUSED FROM P1/P2
    The gated-curriculum settle detection (settle_check(), LogReader.gate_points()) does
    not apply -- there is no stage transition to detect, the run either reaches its
    target absolute iteration or it does not. Log lines use the "[C]" prefix (see the
    module's "[P1]"/"[P2]" convention) so a shared log stream stays greppable by mode.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

MJLAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(MJLAB))
OUT = os.path.join(MJLAB, "analysis", "out")
EXPERIMENT_NOTES = os.path.join(REPO, "docs", "experiments")
DOC_ASSETS = os.path.join(REPO, "docs", "mujoco", "assets")
WATCHDOG_SPEC = os.path.join(OUT, "watchdog_runs.json")
GPU_LOCK = "/home/syaro/pyg_fea/locks/gpu.lock"
TASK = "Mjlab-Velocity-Flat-Pygmalion"
EXPERIMENT = "pygmalion_velocity"
SAVE_INTERVAL = 100  # rl_cfg.py: pygmalion_ppo_runner_cfg().save_interval
STEPS_PER_ITER = 24  # rl_cfg.py: num_steps_per_env

# DR off in P1: dr_levels holds factor at 0.0 while common_step_counter <= start_step.
# NOT PYG_NO_DR=1 -- that POPS push/friction/encoder/base_com entirely, which would (a) drop
# the dr_levels curriculum so dr_factor is never logged, (b) disable PYG_INERTIAL_DR (env_cfgs
# warns and skips it), and (c) change the critic's dr_* observation semantics between phases.
DR_OFF_START = 100_000_000
DR_OFF_END = 100_000_001

ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
KV_RE = re.compile(r"^\s*([A-Za-z_][\w/]*)\s*:\s*(-?(?:\d+\.?\d*|\.\d+|nan|inf))\s*$")
PROMO_RE = re.compile(
  r"\[gated-curriculum\]\s+(WARNING\s+)?stage (\d+) -> (\d+) at iter (\d+): (.*)"
)

GATE_KEYS = (
  "Curriculum/command_vel/stage",
  "Curriculum/command_vel/stage_dwell_iters",
  "Curriculum/command_vel/gate_err_steady",
  "Curriculum/command_vel/gate_fell_over",
  "Curriculum/command_vel/gate_baseline",
  "Curriculum/command_vel/gate_window_episodes",
  "Curriculum/command_vel/forced_promotions",
  "Curriculum/command_vel/lin_vel_x_max",
  "Curriculum/dr_levels/dr_factor",
)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
_LOGF = None
_NOTE = None


def say(msg: str) -> None:
  line = f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}"
  print(line, flush=True)
  if _LOGF is not None:
    _LOGF.write(line + "\n")
    _LOGF.flush()


def _sha256(path: str) -> str | None:
  if not os.path.isfile(path):
    return None
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for block in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def require_experiment_note(run: str) -> str:
  """Memory gate: no training attempt, including a smoke, starts without its run note."""
  matches = sorted(glob.glob(os.path.join(EXPERIMENT_NOTES, f"*{run}*.md")))
  if len(matches) != 1:
    raise SystemExit(
      f"RECORDING PRE-FLIGHT BLOCKED: expected exactly one docs/experiments/*{run}*.md, "
      f"found {len(matches)}. Create/register the run note before allocating the GPU."
    )
  return matches[0]


def start_review_loop(run: str, note: str) -> None:
  """Start the run-specific <=1 h numeric review recorder, once."""
  probe = subprocess.run(["pgrep", "-af", "review_loop.sh"], capture_output=True, text=True)
  if any(run in line for line in probe.stdout.splitlines()):
    say(f"[record] review_loop already running for {run}")
    return
  logp = os.path.join(OUT, f"review_loop_{run}.log")
  fh = open(logp, "a")
  subprocess.Popen(
    ["bash", "analysis/review_loop.sh", run, note], cwd=MJLAB,
    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
  )
  say(f"[record] hourly review loop -> {os.path.relpath(logp, MJLAB)}")


def snapshot_repro(name: str, env: dict, cmd: list[str]) -> None:
  """Freeze every input that can drift after a trainer has created its run directory."""
  started = time.time()
  # 300 s, not 90: on 2026-09-26 the v10a P2 trainer created its run dir 58 s after launch
  # and the launcher, listing the NFS parent through its attribute cache, still had not
  # seen it at 90 s -> RuntimeError killed the launcher (the trainer kept running unwatched).
  deadline = started + 300
  dirs = []
  while time.time() < deadline:
    # Only a directory the new trainer is writing counts; an older one of the same name
    # would otherwise be snapshotted before the new one appears.
    dirs = [d for d in run_dirs(name) if os.path.getmtime(d) >= started - 5]
    if dirs:
      break
    time.sleep(2)
  if not dirs:
    raise RuntimeError(f"run directory for {name} did not appear; cannot snapshot reproducibility")
  run_dir = dirs[-1]
  repro = os.path.join(run_dir, "repro")
  os.makedirs(repro, exist_ok=True)
  sources = [
    os.path.join(MJLAB, "analysis", "run_v2_scratch.py"),
    os.path.join(MJLAB, "analysis", "run_v2_scratch.sh"),
    os.path.join(MJLAB, "src", "mjlab", "tasks", "velocity", "config", "pygmalion", "env_cfgs.py"),
    os.path.join(MJLAB, "src", "mjlab", "tasks", "velocity", "config", "pygmalion", "rl_cfg.py"),
    os.path.join(MJLAB, "src", "mjlab", "tasks", "velocity", "mdp", "curriculums.py"),
  ]
  tag = env.get("PYG_MODEL_TAG")
  if tag:
    sources.append(os.path.join(
      MJLAB, "src", "mjlab", "asset_zoo", "robots", "pygmalion", "xmls", f"{tag}_loop.xml"
    ))
  mass_dr = env.get("PYG_MASS_DR_JSON")
  if mass_dr:
    sources.append(mass_dr)
  manifest = {
    "run_name": name,
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cwd": MJLAB,
    "command": cmd,
    "environment": {k: str(v) for k, v in sorted(env.items())},
    "inputs": [],
  }
  for src in sources:
    if not os.path.isfile(src):
      continue
    dst = os.path.join(repro, os.path.basename(src))
    shutil.copy2(src, dst)
    manifest["inputs"].append({
      "source": os.path.abspath(src), "snapshot": os.path.basename(dst), "sha256": _sha256(src)
    })
  with open(os.path.join(repro, "launch_manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)
  say(f"[record] reproducibility snapshot -> {os.path.relpath(repro, MJLAB)}")


def snapshot_review(run_dir: str, label: str) -> None:
  if not _NOTE:
    return
  cmd = [
    ".venv/bin/python3", "analysis/snapshot_review.py", run_dir, _NOTE,
    f"--label={label}",
  ]
  r = subprocess.run(cmd, cwd=MJLAB, capture_output=True, text=True)
  if r.returncode:
    say(f"[record] snapshot review failed: {(r.stderr or r.stdout).strip()}")
  else:
    say(f"[record] snapshot review appended: {label}")


def build_accum_video(run_dir: str, label: str) -> None:
  os.makedirs(DOC_ASSETS, exist_ok=True)
  cmd = [
    ".venv/bin/python3", "analysis/accum_video.py", "--run-dir", run_dir,
    "--label", label, "--assets", DOC_ASSETS,
  ]
  r = subprocess.run(cmd, cwd=MJLAB, capture_output=True, text=True)
  if r.returncode:
    say(f"[record] accumulate video failed: {(r.stderr or r.stdout).strip()}")
  else:
    say(f"[record] {r.stdout.strip().replace(chr(10), ' | ')}")


def pid_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except (OSError, ProcessLookupError):
    return False
  return True


# A REAL trainer is a python process running the mjlab train entry point AND carrying an
# --agent.run-name. Matching on the run-name alone is not enough: this launcher (and any
# shell command that greps a log) has the same string on its own command line, and one
# false positive here would make the launcher believe a phase is alive when it is not.
TRAINER_RE = re.compile(r"(analysis/train_wandb_video\.py|venv/bin/train\s)")
HEAVY_RE = re.compile(
  r"(analysis/(measure_full|measure_loads|impact_probe_multi|train_wandb_video)\.py|venv/bin/train\s)"
)


def _all_trainers() -> list[tuple[int, str]]:
  r = subprocess.run(
    ["pgrep", "-af", "agent.run-name"], capture_output=True, text=True
  )
  out = []
  for line in r.stdout.splitlines():
    pid, _, cmd = line.partition(" ")
    if pid.isdigit() and TRAINER_RE.search(cmd) and "pgrep" not in cmd:
      out.append((int(pid), cmd))
  return out


def trainer_pids(run: str) -> list[int]:
  """PIDs of the trainer carrying this run-name (or any 'r'-suffixed resume of it)."""
  pat = re.compile(rf"--agent\.run-name\s+{re.escape(run)}r*(\s|$)")
  return [pid for pid, cmd in _all_trainers() if pat.search(cmd)]


def foreign_trainer_pids(ours: list[str]) -> list[tuple[int, str]]:
  """Trainers belonging to somebody else (other coders share this 16 GB GPU / 23 GB host)."""
  pats = [re.compile(rf"--agent\.run-name\s+{re.escape(o)}r*(\s|$)") for o in ours]
  return [
    (pid, cmd) for pid, cmd in _all_trainers() if not any(p.search(cmd) for p in pats)
  ]


def _mem_available_gib() -> float:
  with open("/proc/meminfo") as fh:
    fields = {line.split(":", 1)[0]: line.split()[1] for line in fh if ":" in line}
  return int(fields["MemAvailable"]) / 1024**2


def _disk_available_gib(path: str) -> float:
  st = os.statvfs(path)
  return st.f_bavail * st.f_frsize / 1024**3


def _gpu_status() -> tuple[float, float]:
  """Return free GiB and utilization; fail closed when the driver is unavailable."""
  r = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, timeout=15,
  )
  if r.returncode or not r.stdout.strip():
    raise RuntimeError(f"nvidia-smi unavailable: {(r.stderr or r.stdout).strip()}")
  free_mib, util = (float(x.strip()) for x in r.stdout.splitlines()[0].split(","))
  return free_mib / 1024, util


def other_heavy_jobs() -> list[str]:
  """High-RAM jobs that must not overlap on this 24 GiB host."""
  r = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=15)
  out = []
  for line in r.stdout.splitlines():
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2 or not parts[0].isdigit():
      continue
    pid, cmd = int(parts[0]), parts[1]
    if pid in (os.getpid(), os.getppid()) or "run_v2_scratch" in cmd:
      continue
    if HEAVY_RE.search(cmd):
      out.append(f"pid={pid} {cmd}")
  return out


def action_window_preflight(a, env: dict) -> None:
  """Refuse to launch a config whose policy cannot reach part of its own robot.

  legonly_ab_v1 burned 5,700 iterations with the LEFT KNEE's command band and mechanical
  range disjoint - the motor held 21.8 N*m against a hard stop the whole time and the
  policy could not ask for anything else (docs/reward_research/
  2026-09-03_stiff_knee_root_cause.md). Nothing in the launcher noticed, because every
  number involved was individually plausible. This check costs about a second, runs in the
  phase's own environment so it validates the config THIS run will build, and its table
  goes into the log as the launch record.
  """
  script = os.path.join(MJLAB, "analysis", "preflight_action_window.py")
  if not os.path.exists(script):
    raise SystemExit(f"PRE-FLIGHT BLOCKED: missing {script}")
  proc = subprocess.run(
    [".venv/bin/python3", script],
    cwd=MJLAB, env=dict(os.environ, **{k: str(v) for k, v in env.items()}),
    capture_output=True, text=True, timeout=600,
  )
  for line in (proc.stdout + proc.stderr).rstrip().splitlines():
    say(f"  {line}")
  if proc.returncode != 0:
    raise SystemExit(
      "PRE-FLIGHT BLOCKED: action-window invariant failed (see the table above). "
      "A joint whose commandable band does not overlap its mechanical range cannot be "
      "trained; fix the config (never the model XML) before relaunching."
    )


def resource_preflight(a, env: dict) -> None:
  """Fail before allocating environments if the host cannot safely save a checkpoint."""
  ram = _mem_available_gib()
  disk = _disk_available_gib(MJLAB)
  gpu_free, gpu_util = _gpu_status()
  min_ram = 12.0 if a.num_envs >= 16384 else 8.0
  min_gpu = 12.0 if a.num_envs >= 16384 else 7.0
  # --min-gpu-free overrides ONLY this one threshold (2026-09-25). 12 GiB is what a
  # lone 16,384-env run wants for headroom; a second arm beside a running one needs
  # its own ~6.6 GiB plus margin. Skipping the whole preflight to get there would also
  # drop the RAM, disk and heavy-job checks, which still matter.
  if getattr(a, "min_gpu_free", None) is not None:
    min_gpu = float(a.min_gpu_free)
  errors = []
  if ram < min_ram:
    errors.append(f"available RAM {ram:.1f} GiB < {min_ram:.1f} GiB")
  if disk < 30.0:
    errors.append(f"workspace free disk {disk:.1f} GiB < 30.0 GiB")
  if gpu_free < min_gpu:
    errors.append(f"GPU free memory {gpu_free:.1f} GiB < {min_gpu:.1f} GiB")
  heavy = other_heavy_jobs()
  if heavy and not a.allow_concurrent:
    errors.append("another high-RAM job is alive: " + " | ".join(heavy))
  if a.run.lower().startswith("v2u") and env.get("PYG_UPPER_DOF") != "1":
    errors.append("upper-DOF run name requires --env PYG_UPPER_DOF=1")
  say(
    f"[preflight] RAM available={ram:.1f} GiB, workspace free={disk:.1f} GiB, "
    f"GPU free={gpu_free:.1f} GiB, GPU util={gpu_util:.0f}%, envs={a.num_envs}"
  )
  if errors:
    raise SystemExit("PRE-FLIGHT BLOCKED:\n  - " + "\n  - ".join(errors))


def run_dirs(run: str) -> list[str]:
  base = os.path.join(MJLAB, "logs", "rsl_rl", EXPERIMENT)
  found = glob.glob(f"{base}/*_{run}") + glob.glob(f"{base}/*_{run}r*")
  # A failed attempt is set aside by renaming it ABORTED_<dir>. "*_<run>" still matches
  # that name, and "A" sorts after the digits of a date, so without this filter the set-
  # aside directory became the "newest" one: on 2026-09-25 the v10b snapshot was written
  # into it, and a resume would have looked for checkpoints there.
  return sorted(d for d in found if not os.path.basename(d).startswith("ABORTED_"))


def latest_ckpt(run: str) -> tuple[int, str, str] | None:
  """(iteration, run_dir, 'model_N.pt') of the newest checkpoint across the resume chain."""
  best = None
  for d in run_dirs(run):
    for p in glob.glob(os.path.join(d, "model_*.pt")):
      try:
        it = int(os.path.basename(p).split("_")[1][:-3])
      except (IndexError, ValueError):
        continue
      if best is None or it > best[0]:
        best = (it, d, os.path.basename(p))
  return best


def log_files(run: str) -> list[str]:
  d = os.path.join(MJLAB, "logs")
  return sorted(glob.glob(f"{d}/{run}.log") + glob.glob(f"{d}/{run}r*.log"))


# --------------------------------------------------------------------------------------
# continue mode -- "+N more iterations from an existing checkpoint" (2026-09-05)
# --------------------------------------------------------------------------------------
DEFAULT_RUN_NAME = "v2s1_AB"  # must match the --run argparse default below


def ckpt_iter(ckpt_name: str) -> int:
  """Parse 'model_18099.pt' -> 18099 (same convention latest_ckpt() uses)."""
  return int(os.path.basename(ckpt_name).split("_")[1][:-3])


def resolve_run_dir(path: str) -> str:
  """--continue-from accepts an absolute path or one relative to the mjlab checkout."""
  return path if os.path.isabs(path) else os.path.join(MJLAB, path)


def load_parent_manifest_env(parent_dir: str) -> dict[str, str]:
  """The PYG_* subset of the parent run's frozen launch environment.

  Reading this (rather than recomputing p1_env()/p2_env() for a "similar" config, or
  reading the parent's live watchdog_runs.json entry, which can be edited later) is what
  makes the continued run's ONLY deliberate difference from its parent the keys the
  caller passes via --env: every other PYG_* flag -- including which curriculum/DR window
  iterations the parent had already passed -- is carried over unchanged.
  """
  manifest_path = os.path.join(parent_dir, "repro", "launch_manifest.json")
  if not os.path.isfile(manifest_path):
    raise SystemExit(
      f"--continue-from: missing {manifest_path} -- this run predates (or lost) its "
      "reproducibility snapshot, so its exact launch environment cannot be recovered."
    )
  manifest = json.load(open(manifest_path))
  full_env = manifest.get("environment", {})
  return {k: v for k, v in full_env.items() if k.startswith("PYG_")}


def build_continue_env(
  parent_env: dict[str, str], env_args: list[str] | None
) -> tuple[dict[str, str], dict[str, str]]:
  """Inherit the parent's PYG_* env wholesale; apply --env on top; report only the diff.

  Returns (full env to launch with, overrides actually applied) -- the second is what the
  state file and the dry-run preview show as "the deliberate variable" of this A/B.
  """
  env = dict(parent_env)
  overrides: dict[str, str] = {}
  for kv in env_args or []:
    k, _, v = kv.partition("=")
    k, v = k.strip(), v.strip()
    env[k] = v
    overrides[k] = v
  return env, overrides


def validate_continue_args(a) -> bool:
  """Return True if this invocation selects CONTINUE mode; raise on an inconsistent mix.

  When none of --continue-from/--continue-ckpt/--extra-iters is given, this returns False
  and touches nothing else -- every other function in this module runs exactly as it did
  before CONTINUE mode was added (a flag that is not passed must not change behavior).
  """
  given = {
    "--continue-from": a.continue_from,
    "--continue-ckpt": a.continue_ckpt,
    "--extra-iters": a.extra_iters,
  }
  present = [k for k, v in given.items() if v is not None]
  if not present:
    return False
  missing = [k for k, v in given.items() if v is None]
  if missing:
    raise SystemExit(
      "--continue-from/--continue-ckpt/--extra-iters must be given together; missing "
      + ", ".join(missing)
    )
  if a.smoke:
    raise SystemExit("--continue-from is incompatible with --smoke")
  if a.run == DEFAULT_RUN_NAME:
    raise SystemExit(
      "--continue-from requires an explicit --run name for the new (continued) run "
      f"(the default '{DEFAULT_RUN_NAME}' is reserved for from-scratch P1/P2 runs)"
    )
  return True


def continue_plan(a) -> dict:
  """Resolve --continue-from/--continue-ckpt/--extra-iters into everything both the
  dry-run preview and the real launch need. Pure aside from reading the two files the
  caller named (the parent's manifest and its checkpoint); raises SystemExit on anything
  that would make the launch incoherent.
  """
  parent_dir = resolve_run_dir(a.continue_from)
  if not os.path.isdir(parent_dir):
    raise SystemExit(f"--continue-from: no such run directory: {parent_dir}")
  ckpt_path = os.path.join(parent_dir, a.continue_ckpt)
  if not os.path.isfile(ckpt_path):
    raise SystemExit(f"--continue-ckpt: no such checkpoint: {ckpt_path}")
  parent_env = load_parent_manifest_env(parent_dir)
  parent_iter = ckpt_iter(a.continue_ckpt)
  env, overrides = build_continue_env(parent_env, a.env)
  total_abs = parent_iter + a.extra_iters
  return {
    "parent_dir": parent_dir,
    "parent_dir_name": os.path.basename(parent_dir),
    "parent_ckpt": a.continue_ckpt,
    "parent_iter": parent_iter,
    "parent_env_keys": len(parent_env),
    "extra_iters": a.extra_iters,
    "total_iters_abs": total_abs,
    "env": env,
    "env_overrides": overrides,
  }


# --------------------------------------------------------------------------------------
# log / tfevents parsing
# --------------------------------------------------------------------------------------
class LogReader:
  """Incremental reader over a phase's console logs (+ tfevents fallback).

  Keeps a byte offset per file so a 60 MB P1 log is not re-read every poll.
  """

  def __init__(self, run: str, source: str = "auto"):
    self.run = run
    self.source = source
    self._off: dict[str, int] = {}
    self._cur_iter: dict[str, int] = {}
    self.iters: dict[int, dict[str, float]] = {}
    self.promotions: list[dict] = []
    self._tf_seen: set[str] = set()

  def poll(self) -> None:
    if self.source in ("auto", "log"):
      for f in log_files(self.run):
        self._read_log(f)
    if self.source == "tfevents" or (self.source == "auto" and not self.iters):
      self._read_tfevents()

  def _read_log(self, path: str) -> None:
    try:
      size = os.path.getsize(path)
    except OSError:
      return
    off = self._off.get(path, 0)
    if size < off:  # truncated / replaced
      off = 0
      self._cur_iter.pop(path, None)
    if size == off:
      return
    # BINARY, not text: the offset bookkeeping is in BYTES, and TextIOWrapper.seek()
    # only accepts opaque cookies from tell() -- feeding it a byte count silently
    # mis-positions the stream (caught in the 2026-08-27 smoke: the reader sat at
    # iteration 5 while the trainer was at 31).
    with open(path, "rb") as fh:
      fh.seek(off)
      raw = fh.read()
    cut = raw.rfind(b"\n")  # keep the last partial line for the next poll
    if cut == -1:
      return
    self._off[path] = off + cut + 1
    data = raw[: cut + 1].decode("utf-8", "replace")
    cur = self._cur_iter.get(path)
    for raw in data.splitlines():
      line = ANSI.sub("", raw)
      m = ITER_RE.search(line)
      if m:
        cur = int(m.group(1))
        self.iters.setdefault(cur, {})
        continue
      m = PROMO_RE.search(line)
      if m:
        rec = {
          "forced": bool(m.group(1)),
          "from": int(m.group(2)),
          "to": int(m.group(3)),
          "iter": int(m.group(4)),
          "detail": m.group(5).strip(),
        }
        if rec not in self.promotions:
          self.promotions.append(rec)
        continue
      if cur is None:
        continue
      m = KV_RE.match(line)
      if m and ("/" in m.group(1)):
        try:
          self.iters[cur][m.group(1)] = float(m.group(2))
        except ValueError:
          pass
    self._cur_iter[path] = cur

  def _read_tfevents(self) -> None:
    try:
      from tensorboard.backend.event_processing import event_accumulator as ea
    except Exception as e:  # noqa: BLE001
      say(f"[reader] tfevents fallback unavailable ({e})")
      self.source = "log"
      return
    for d in run_dirs(self.run):
      for f in sorted(glob.glob(os.path.join(d, "events.out.tfevents.*"))):
        acc = ea.EventAccumulator(f, size_guidance={ea.SCALARS: 0})
        try:
          acc.Reload()
        except Exception:  # noqa: BLE001
          continue
        tags = set(acc.Tags().get("scalars", []))
        for k in GATE_KEYS:
          if k not in tags:
            continue
          for ev in acc.Scalars(k):
            self.iters.setdefault(int(ev.step), {})[k] = float(ev.value)
        self._tf_seen.add(f)

  # -- views ---------------------------------------------------------------------------
  def max_iter(self) -> int:
    return max(self.iters) if self.iters else -1

  def gate_points(self) -> list[tuple[int, dict[str, float]]]:
    """Iterations that actually carry the gate telemetry, in order."""
    return [
      (it, m)
      for it, m in sorted(self.iters.items())
      if "Curriculum/command_vel/stage" in m
    ]

  def dr_factor_at(self, it: int) -> float | None:
    m = self.iters.get(it, {})
    return m.get("Curriculum/dr_levels/dr_factor")

  def dr_trajectory(self) -> list[tuple[int, float]]:
    return [
      (it, m["Curriculum/dr_levels/dr_factor"])
      for it, m in sorted(self.iters.items())
      if "Curriculum/dr_levels/dr_factor" in m
    ]


def settle_check(
  points: list[tuple[int, dict[str, float]]],
  top_stage: int,
  min_dwell: int,
  max_dwell: int,
  min_episodes: int,
  fell_max: float,
  err_ratio: float,
  hold: int,
) -> tuple[bool, int, dict]:
  """Trailing run of iterations that pass the gate's own criteria AT the top stage."""
  streak = 0
  last: dict = {}
  for it, m in points:
    stage = m.get("Curriculum/command_vel/stage", -1.0)
    dwell = m.get("Curriculum/command_vel/stage_dwell_iters", 0.0)
    eps = m.get("Curriculum/command_vel/gate_window_episodes", 0.0)
    fell = m.get("Curriculum/command_vel/gate_fell_over", float("nan"))
    err = m.get("Curriculum/command_vel/gate_err_steady", float("nan"))
    base = m.get("Curriculum/command_vel/gate_baseline", float("nan"))
    # A stage that has out-dwelt max_dwell without promoting can ONLY be the top stage
    # (commands_vel_gated force-promotes every other stage at max_dwell). Belt and braces
    # against a wrong --n-stages.
    at_top = stage >= top_stage - 1e-6 or dwell > max_dwell
    ok_err = math.isnan(base) or (not math.isnan(err) and err <= base * err_ratio)
    ok = (
      at_top
      and dwell >= min_dwell
      and eps >= min_episodes
      and not math.isnan(fell)
      and fell <= fell_max
      and ok_err
    )
    if ok:
      streak += 1
      last = {
        "iter": it,
        "stage": stage,
        "dwell": dwell,
        "episodes": eps,
        "fell_over": fell,
        "err_steady": err,
        "baseline": base,
        "streak": streak,
      }
    else:
      streak = 0
      last = {}
  return (streak >= hold, streak, last)


# --------------------------------------------------------------------------------------
# GPU lock (two other coders share this box)
# --------------------------------------------------------------------------------------
class GpuLock:
  """The box-wide GPU lock the other coders use: a DIRECTORY `gpu.lock/` containing an
  `owner` file holding "<pid> <who> <iso8601>". mkdir is the atomic primitive (an O_EXCL
  regular file is NOT interchangeable with it -- discovered the hard way in the
  2026-08-27 smoke, where a file-based implementation could neither take nor read a
  directory-based lock and waited forever on an unparseable holder).
  """

  def __init__(self, path: str, who: str, enabled: bool = True, stale_min: int = 30):
    self.path, self.who, self.enabled, self.stale_min = path, who, enabled, stale_min
    self.owner = os.path.join(path, "owner")
    self.held = False

  def _read_owner(self) -> tuple[int | None, str, float]:
    """(pid, who, age_minutes). pid None when the lock is unreadable."""
    try:
      age = (time.time() - os.path.getmtime(self.owner)) / 60.0
      parts = open(self.owner).read().split()
      return int(parts[0]), (parts[1] if len(parts) > 1 else "?"), age
    except (OSError, ValueError, IndexError):
      try:
        age = (time.time() - os.path.getmtime(self.path)) / 60.0
      except OSError:
        age = 0.0
      return None, "?", age

  def acquire(self, poll: int = 30) -> None:
    if not self.enabled:
      say("[gpu-lock] disabled by flag")
      return
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    waited = 0
    while True:
      try:
        os.mkdir(self.path)
      except FileExistsError:
        pid, who, age = self._read_owner()
        # An unreadable lock is NOT automatically stale: it may be a holder that has not
        # written its owner file yet. Only a dead pid, or an unreadable lock older than
        # the stale window, may be broken.
        dead = pid is not None and not pid_alive(pid)
        if dead or (pid is None and age > self.stale_min):
          say(f"[gpu-lock] STEALING stale lock (holder {who} pid {pid}, {age:.0f} min old)")
          try:
            if os.path.isdir(self.path):
              for f in os.listdir(self.path):
                os.unlink(os.path.join(self.path, f))
              os.rmdir(self.path)
            else:
              os.unlink(self.path)
          except OSError:
            pass
          continue
        if waited % 300 == 0:
          say(
            f"[gpu-lock] waiting: held by {who} pid {pid} ({age:.0f} min) "
            f"-- polling politely every {poll}s"
          )
        time.sleep(poll)
        waited += poll
        continue
      except FileNotFoundError:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        continue
      stamp = datetime.now().astimezone().isoformat(timespec="seconds")
      with open(self.owner, "w") as fh:
        fh.write(f"{os.getpid()} {self.who} {stamp}\n")
      self.held = True
      say(f"[gpu-lock] acquired {self.path} (who={self.who})")
      return

  def release(self) -> None:
    if not self.held:
      return
    self.held = False
    pid, _, _ = self._read_owner()
    if pid is not None and pid != os.getpid():
      say(f"[gpu-lock] NOT releasing: owner is now pid {pid}, not us")
      return
    try:
      if os.path.isdir(self.path):
        for f in os.listdir(self.path):
          os.unlink(os.path.join(self.path, f))
        os.rmdir(self.path)
      else:
        os.unlink(self.path)
      say("[gpu-lock] released")
    except OSError as e:
      say(f"[gpu-lock] release failed: {e}")


def wait_for_idle_gpu(ours: list[str], poll: int = 60, enabled: bool = True) -> None:
  """Block until no OTHER trainer is running (15 GB host RAM: two trainers = OOM)."""
  if not enabled:
    return
  announced = False
  while True:
    foreign = foreign_trainer_pids(ours)
    if not foreign:
      if announced:
        say("[gpu-wait] other trainer finished; proceeding")
      return
    if not announced:
      for pid, cmd in foreign:
        name = re.search(r"run-name (\S+)", cmd)
        say(f"[gpu-wait] another trainer is running: pid {pid} run-name {name.group(1) if name else '?'}")
      say(f"[gpu-wait] polling every {poll}s until it finishes (RAM guard: 15 GB host)")
      announced = True
    time.sleep(poll)


# --------------------------------------------------------------------------------------
# watchdog registration
# --------------------------------------------------------------------------------------
def watchdog_register(name: str, enabled: bool, total_iters: int, num_envs: int, env: dict) -> None:
  try:
    spec = json.load(open(WATCHDOG_SPEC))
  except Exception:  # noqa: BLE001
    spec = {}
  spec[name] = {
    "enabled": bool(enabled),
    "total_iters": int(total_iters),
    "num_envs": int(num_envs),
    "task": TASK,
    "env": {k: str(v) for k, v in env.items()},
  }
  tmp = WATCHDOG_SPEC + ".tmp"
  with open(tmp, "w") as fh:
    json.dump(spec, fh, indent=1)
  os.replace(tmp, WATCHDOG_SPEC)
  say(f"[watchdog] {name}: enabled={enabled} total_iters={total_iters} num_envs={num_envs}")


def watchdog_disable(name: str) -> None:
  try:
    spec = json.load(open(WATCHDOG_SPEC))
  except Exception:  # noqa: BLE001
    return
  if name in spec and spec[name].get("enabled"):
    spec[name]["enabled"] = False
    tmp = WATCHDOG_SPEC + ".tmp"
    with open(tmp, "w") as fh:
      json.dump(spec, fh, indent=1)
    os.replace(tmp, WATCHDOG_SPEC)
    say(f"[watchdog] {name}: DISABLED (phase graduated -- do not auto-resume)")


# --------------------------------------------------------------------------------------
# phase env / command construction
# --------------------------------------------------------------------------------------
def base_env(a) -> dict:
  """The confirmed landing recipe + standard stack (docs/103 §4a)."""
  env = {
    # standard stack
    "PYG_V2": "1",
    "PYG_INIT_BENT": "1",
    "PYG_ARM_ABD_DEG": "15",
    "PYG_INERTIAL_DR": "1",
    "PYG_ANKLE_MODE": a.ankle,
    "PYG_TN": "1",
    "PYG_MOTOR_MEAS": "1",
    "PYG_SOFT_LANDING": "1",
    # Deployment contract: actor gets only raw-hardware-available proprioception
    # (IMU angular velocity + projected gravity, q[t-1:t], previous action, command).
    # Keep this in both phases so actor width/order cannot change across resume.
    "PYG_STUDENT_TEACHER": "1",
    # Simulator and hardware runtime share the centered 90%-of-ROM target clamp.
    "PYG_SAFE_TARGET_CLIP": "1",
    # confirmed landing recipe (docs/103 §4a: bundleD1)
    "PYG_INIT_MID": "1",
    "PYG_KNEE_EXT": "1",
    "PYG_KNEE_EXT_W": "2.0",
    "PYG_KNEE_EXT_DEG": "25",
    "PYG_SOFT_LANDING_MODE": "half",
    # V9: critic sees the DR draw. Set in BOTH phases so the critic obs width never
    # changes -> P2 can FULL-resume (actor + critic + optimizer).
    "PYG_CRITIC_DR_OBS": "1",
  }
  if a.model_v4:
    env["PYG_MODEL_V4"] = "1"
  if a.vy_stages:
    env["PYG_CMD_VY_STAGES"] = "1"
  for kv in a.env or []:
    k, _, v = kv.partition("=")
    env[k.strip()] = v.strip()
  return env


def p1_env(a) -> dict:
  env = dict(base_env(a))
  # A pseudo_inertia event is sampled at startup and is NOT controlled by dr_levels.
  # Leaving PYG_INERTIAL_DR enabled here made the supposedly clean P1 start with full
  # mass/COM/inertia randomization.  Remove only the event toggle; PYG_CRITIC_DR_OBS stays
  # enabled, so actor/critic dimensions remain identical and P2 can still full-resume.
  env.pop("PYG_INERTIAL_DR", None)
  env.update({
    "PYG_GATED_CURRICULUM": "1",
    "PYG_GATE_MIN_DWELL": str(a.gate_min_dwell),
    "PYG_GATE_MAX_DWELL": str(a.gate_max_dwell),
    "PYG_GATE_WINDOW": str(a.gate_window),
    "PYG_GATE_ERR_RATIO": str(a.gate_err_ratio),
    "PYG_GATE_FELL_MAX": str(a.gate_fell_max),
    "PYG_GATE_MIN_EPISODES": str(a.gate_min_episodes),
    # DR OFF: window pushed past any horizon -> dr_levels factor pinned at 0.0.
    "PYG_DR_START_ITER": str(DR_OFF_START),
    "PYG_DR_END_ITER": str(DR_OFF_END),
  })
  # entropy anneal is P2-only (V4): P1 needs its exploration noise.
  env.pop("PYG_ENTROPY_ANNEAL", None)
  return env


def p2_env(a, p1_end: int, dr_start: int, dr_end: int, total_abs: int) -> dict:
  env = dict(base_env(a))
  env.update({
    # The gate's per-stage state lives on the env object (_pyg_gate_state) and is NOT in
    # the checkpoint, so a resume would restart the curriculum at stage 0 and SHRINK the
    # command box under the policy. min/max dwell 0 makes commands_vel_gated force-promote
    # straight back to the top stage within the first reset batch, which is where P1 left
    # it. docs/103 §3b: the command range is fixed across phase 2; only DR ramps.
    "PYG_GATED_CURRICULUM": "1",
    "PYG_GATE_MIN_DWELL": "0",
    "PYG_GATE_MAX_DWELL": "0",
    "PYG_GATE_WINDOW": str(a.gate_window),
    "PYG_GATE_ERR_RATIO": str(a.gate_err_ratio),
    "PYG_GATE_FELL_MAX": str(a.gate_fell_max),
    "PYG_GATE_MIN_EPISODES": str(a.gate_min_episodes),
    # THE POINT OF THIS LAUNCHER: computed from the MEASURED P1 length, not a constant.
    "PYG_DR_START_ITER": str(dr_start),
    "PYG_DR_END_ITER": str(dr_end),
    # V4: anneal exploration noise across the whole P2 window.
    "PYG_ENTROPY_ANNEAL": "1",
    "PYG_ENTROPY_ANNEAL_START": str(p1_end),
    "PYG_ENTROPY_ANNEAL_END": str(total_abs - 1),
    "PYG_ENTROPY_START_COEF": str(a.entropy_start),
    "PYG_ENTROPY_END_COEF": str(a.entropy_end),
  })
  return env


def train_cmd(a, name: str, iters: int, resume: tuple[str, str] | None) -> list[str]:
  cmd = [
    ".venv/bin/python3",
    "analysis/train_wandb_video.py",
    TASK,
    "--video", "True",
    "--video-interval", str(a.video_interval),
    "--video-length", str(a.video_length),
    "--env.scene.num-envs", str(a.num_envs),
    "--agent.max-iterations", str(iters),
    "--agent.run-name", name,
    "--agent.logger", a.logger,
  ]
  if a.seed != 42:
    cmd += ["--agent.seed", str(a.seed)]
  if resume is not None:
    cmd += [
      "--agent.resume", "True",
      "--agent.load-run", resume[0],
      "--agent.load-checkpoint", resume[1],
    ]
  return cmd


def render(env: dict, cmd: list[str]) -> str:
  e = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
  return f"env {e} \\\n     {' '.join(shlex.quote(c) for c in cmd)}"


def spawn(env: dict, cmd: list[str], name: str) -> int:
  full = dict(os.environ, **{k: str(v) for k, v in env.items()})
  logp = os.path.join(MJLAB, "logs", f"{name}.log")
  fh = open(logp, "a")
  fh.write(f"\n===== launched by run_v2_scratch.py at {datetime.now().isoformat()} =====\n")
  fh.flush()
  p = subprocess.Popen(
    cmd, cwd=MJLAB, env=full, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
  )
  say(f"[launch] {name} pid {p.pid} -> logs/{name}.log")
  snapshot_repro(name, env, cmd)
  return p.pid


def stop_trainer(run: str, grace: int = 90) -> None:
  pids = trainer_pids(run)
  if not pids:
    return
  say(f"[stop] SIGTERM {run} pids {pids}")
  for p in pids:
    try:
      os.kill(p, signal.SIGTERM)
    except OSError:
      pass
  t0 = time.time()
  while time.time() - t0 < grace:
    if not trainer_pids(run):
      say(f"[stop] {run} exited cleanly after {int(time.time() - t0)}s")
      return
    time.sleep(3)
  for p in trainer_pids(run):
    say(f"[stop] SIGKILL {run} pid {p}")
    try:
      os.kill(p, signal.SIGKILL)
    except OSError:
      pass
  time.sleep(3)


# --------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------
def state_path(run: str) -> str:
  return os.path.join(OUT, f"v2_scratch_{run}.json")


def load_state(run: str) -> dict:
  try:
    return json.load(open(state_path(run)))
  except Exception:  # noqa: BLE001
    return {}


def save_state(st: dict) -> None:
  tmp = state_path(st["run"]) + ".tmp"
  with open(tmp, "w") as fh:
    json.dump(st, fh, indent=1)
  os.replace(tmp, state_path(st["run"]))


# --------------------------------------------------------------------------------------
# phase drivers
# --------------------------------------------------------------------------------------
def ensure_running(a, st: dict, phase: str, name: str, env: dict, total_abs: int,
                   first_resume: tuple[str, str] | None, start_iter: int = 0) -> None:
  """Start / attach / resume the phase's trainer. Never restarts from scratch if a
  checkpoint of this phase exists.

  start_iter is the iteration the FIRST launch of this phase resumes from (P2: P1's
  end_iter; CONTINUE mode: the parent checkpoint's iteration; P1 itself never uses it,
  since it always launches with first_resume=None). It is only consulted the very first
  time a phase is launched -- every subsequent crash-resume already has its own
  checkpoint (`ck` below) to compute the remaining iterations from.
  """
  if trainer_pids(name):
    say(f"[{phase}] already running (pids {trainer_pids(name)}) -- attaching")
    return
  ck = latest_ckpt(name)
  wait_for_idle_gpu([name], poll=a.idle_poll, enabled=not a.allow_concurrent)
  if ck is None:
    if first_resume is None:
      say(f"[{phase}] launching FROM SCRATCH: {total_abs} iters, {a.num_envs} envs")
      cmd = train_cmd(a, name, total_abs, None)
    else:
      say(f"[{phase}] launching FULL RESUME from {first_resume[0]}/{first_resume[1]}")
      cmd = train_cmd(a, name, total_abs - start_iter, first_resume)
  else:
    it, d, ckn = ck
    if it >= total_abs - 1:
      say(f"[{phase}] already finished at iter {it} (>= {total_abs - 1})")
      return
    n_r = os.path.basename(d).count("r", os.path.basename(d).rfind(name) + len(name))
    newname = name + "r" * (n_r + 1)
    say(
      f"[{phase}] RESUMING (not restarting): latest ckpt iter {it} in {os.path.basename(d)}"
      f" -> new run-name {newname}, {total_abs - it} iters left"
    )
    cmd = train_cmd(a, newname, total_abs - it, (os.path.basename(d), ckn))
    name = newname
  spawn(env, cmd, name)
  time.sleep(5)


def record_spec_tables(p1_dir: str, p2_dir: str | None = None) -> None:
  """Refresh the note's §1b~§1b-4 spec tables from the run's OWN saved config.

  User rule 2026-09-03 ("이건 모든 Docs에 다 넣으라고"): every training-run note carries
  the reward weights, per-joint Kp/Kd, actuator limits, ROM/action window and PYG_* stack
  flags, generated -- never retyped -- from that run's params/env.yaml. Idempotent: the
  block lives between SPEC-TABLES markers and is replaced, so calling it after P1 and
  again after P2 just updates it.  Never fatal: a note is not worth killing a run over.
  """
  if not _NOTE:
    return
  tool = os.path.join(MJLAB, "analysis", "run_spec_tables.py")
  cmd = [sys.executable, tool, p1_dir, "--insert", _NOTE, "--force"]
  if "| reward | weight |" not in open(_NOTE).read():
    cmd.append("--with-1b")
  if p2_dir:
    cmd += ["--p2", p2_dir]
  try:
    r = subprocess.run(cmd, cwd=MJLAB, capture_output=True, text=True, timeout=180)
    say(f"[record] spec tables (§1b~§1b-4) -> {os.path.relpath(_NOTE, REPO)}: "
        + (r.stdout.strip().splitlines() or ["?"])[0])
    if r.returncode:
      say(f"[record] WARN run_spec_tables failed: {r.stderr.strip()[-300:]}")
  except Exception as e:  # noqa: BLE001 - never let bookkeeping kill a training run
    say(f"[record] WARN run_spec_tables raised: {e}")


def drive_p1(a, st: dict) -> None:
  name = st["p1"]["name"]
  env = p1_env(a)
  st["p1"]["env"] = env
  watchdog_register(name, True, a.p1_max_iters, a.num_envs, env)
  ensure_running(a, st, "P1", name, env, a.p1_max_iters, None)
  save_state(st)

  reader = LogReader(name, a.source)
  last_stage = -1
  dead_since = None
  while True:
    time.sleep(a.poll)
    reader.poll()
    pts = reader.gate_points()
    it = reader.max_iter()
    for pr in reader.promotions[len(st["p1"].get("promotions", [])):]:
      st["p1"].setdefault("promotions", []).append(pr)
      say(
        f"[P1] promotion stage {pr['from']} -> {pr['to']} at iter {pr['iter']}"
        f"{' (FORCED)' if pr['forced'] else ''}: {pr['detail']}"
      )
    settled, streak, ev = settle_check(
      pts, a.n_stages - 1, a.gate_min_dwell, a.gate_max_dwell,
      a.gate_min_episodes, a.gate_fell_max, a.gate_err_ratio, a.settle_hold,
    )
    if pts:
      stage = pts[-1][1].get("Curriculum/command_vel/stage", -1)
      if int(stage) != last_stage:
        last_stage = int(stage)
      say(
        f"[P1] iter {it} stage {stage:.2f} dwell "
        f"{pts[-1][1].get('Curriculum/command_vel/stage_dwell_iters', 0):.0f} "
        f"err {pts[-1][1].get('Curriculum/command_vel/gate_err_steady', float('nan')):.4f} "
        f"base {pts[-1][1].get('Curriculum/command_vel/gate_baseline', float('nan')):.4f} "
        f"fell {pts[-1][1].get('Curriculum/command_vel/gate_fell_over', float('nan')):.4f} "
        f"dr {reader.dr_factor_at(it)} settle-streak {streak}/{a.settle_hold}"
      )
    if settled:
      st["p1"]["end_reason"] = "settled at top stage"
      st["p1"]["settle_evidence"] = ev
      say(f"[P1] SETTLED at top stage: {ev}")
      break
    if it >= a.p1_max_iters - 1:
      st["p1"]["end_reason"] = "hard iteration cap"
      say(f"[P1] hard cap {a.p1_max_iters} reached at iter {it}")
      break

    if not trainer_pids(name):
      if dead_since is None:
        dead_since = time.time()
        say(f"[P1] trainer not alive at iter {it}; giving watchdog.sh {a.resume_grace}s")
      elif time.time() - dead_since > a.resume_grace:
        say("[P1] watchdog did not resume in time -- resuming from the launcher")
        ensure_running(a, st, "P1", name, env, a.p1_max_iters, None)
        dead_since = None
    else:
      dead_since = None
    save_state(st)

  # graduate: disable the watchdog entry BEFORE stopping, or it resumes what we just ended
  watchdog_disable(name)
  target = reader.max_iter()
  ck = latest_ckpt(name)
  if st["p1"]["end_reason"] != "hard iteration cap":
    deadline = time.time() + a.ckpt_wait
    while time.time() < deadline:
      ck = latest_ckpt(name)
      if ck and ck[0] >= target:
        break
      say(f"[P1] waiting for a checkpoint at/after iter {target} (have {ck[0] if ck else None})")
      time.sleep(min(60, a.poll))
    stop_trainer(name)
  else:
    # the trainer exits by itself at the cap; give it a moment to write the last model
    t0 = time.time()
    while trainer_pids(name) and time.time() - t0 < a.ckpt_wait:
      time.sleep(min(30, a.poll))
    stop_trainer(name)
  ck = latest_ckpt(name)
  if ck is None:
    raise SystemExit("P1 produced no checkpoint -- refusing to continue")
  st["p1"].update({
    "status": "done",
    "end_iter": ck[0],
    "ckpt_dir": os.path.basename(ck[1]),
    "ckpt": ck[2],
    "top_stage_reached": bool(
      reader.gate_points() and reader.gate_points()[-1][1].get(
        "Curriculum/command_vel/stage", -1) >= a.n_stages - 1 - 1e-6
    ),
    "forced_promotions": int(
      reader.gate_points()[-1][1].get("Curriculum/command_vel/forced_promotions", 0)
    ) if reader.gate_points() else 0,
    "finished_at": datetime.now().isoformat(),
  })
  save_state(st)
  say(
    f"[P1] DONE at iter {st['p1']['end_iter']} ({st['p1']['end_reason']}), "
    f"checkpoint {st['p1']['ckpt_dir']}/{st['p1']['ckpt']}, "
    f"top_stage_reached={st['p1']['top_stage_reached']}, "
    f"forced_promotions={st['p1']['forced_promotions']}"
  )
  snapshot_review(ck[1], "P1 phase-end: review before P2")
  build_accum_video(ck[1], f"{st['run']}_p1")
  record_spec_tables(ck[1])


def drive_p2(a, st: dict) -> None:
  p1_end = st["p1"]["end_iter"]
  if not st["p1"].get("top_stage_reached") and not a.force_p2:
    say(
      "!! P1 ended WITHOUT reaching the top command stage. Starting P2 would force the "
      "command box to the top stage the policy never trained on (the gate's stage state "
      "is not in the checkpoint, so it cannot be resumed mid-curriculum). STOPPING. "
      "Re-run with --force-p2 if that is what you want, or raise --p1-max-iters."
    )
    raise SystemExit(3)

  dr_start = p1_end
  dr_end = p1_end + a.ramp_iters
  total_abs = dr_end + a.digest_iters
  name = st["p2"]["name"]
  env = p2_env(a, p1_end, dr_start, dr_end, total_abs)
  st["p2"].update({
    "p1_end": p1_end,
    "dr_start_iter": dr_start,
    "dr_end_iter": dr_end,
    "total_iters_abs": total_abs,
    "env": env,
  })
  say(
    f"[P2] DR window computed from the MEASURED P1 length: P1_END={p1_end} -> "
    f"PYG_DR_START_ITER={dr_start} PYG_DR_END_ITER={dr_end} "
    f"(ramp {a.ramp_iters} + digest {a.digest_iters} => total {total_abs} abs iters)"
  )
  watchdog_register(name, True, total_abs, a.num_envs, env)
  ensure_running(a, st, "P2", name, env, total_abs, (st["p1"]["ckpt_dir"], st["p1"]["ckpt"]),
                start_iter=p1_end)
  save_state(st)

  reader = LogReader(name, a.source)
  dead_since = None
  while True:
    time.sleep(a.poll)
    reader.poll()
    it = reader.max_iter()
    dr = reader.dr_factor_at(it)
    if it >= 0:
      say(
        f"[P2] iter {it}/{total_abs - 1} dr_factor {dr} "
        f"(window {dr_start}..{dr_end})"
      )
    ck = latest_ckpt(name)
    if (ck and ck[0] >= total_abs - 1) or it >= total_abs - 1:
      break
    if not trainer_pids(name):
      if dead_since is None:
        dead_since = time.time()
        say(f"[P2] trainer not alive at iter {it}; giving watchdog.sh {a.resume_grace}s")
      elif time.time() - dead_since > a.resume_grace:
        say("[P2] watchdog did not resume in time -- resuming from the launcher")
        ensure_running(a, st, "P2", name, env, total_abs,
                       (st["p1"]["ckpt_dir"], st["p1"]["ckpt"]), start_iter=p1_end)
        dead_since = None
    else:
      dead_since = None
    save_state(st)

  t0 = time.time()
  while trainer_pids(name) and time.time() - t0 < a.ckpt_wait:
    time.sleep(min(30, a.poll))
  stop_trainer(name)
  ck = latest_ckpt(name)
  st["p2"].update({
    "status": "done",
    "end_iter": ck[0] if ck else None,
    "ckpt_dir": os.path.basename(ck[1]) if ck else None,
    "ckpt": ck[2] if ck else None,
    "dr_trajectory": reader.dr_trajectory()[-3:],
    "finished_at": datetime.now().isoformat(),
  })
  save_state(st)
  say(f"[P2] DONE at iter {st['p2'].get('end_iter')} -- {st['p2'].get('ckpt_dir')}/{st['p2'].get('ckpt')}")
  if ck:
    snapshot_review(ck[1], "P2 phase-end: final smoke/training health review")
    build_accum_video(ck[1], f"{st['run']}_p2")
    p1d = os.path.join(MJLAB, "logs", "rsl_rl", EXPERIMENT, st["p1"]["ckpt_dir"])
    record_spec_tables(p1d if os.path.isdir(p1d) else ck[1], ck[1])


def drive_continue(a, st: dict, plan: dict) -> None:
  """CONTINUE mode's single, untracked-by-curriculum phase: full resume from a parent
  checkpoint, driven to (parent_iter + extra_iters) the same way P2's tail is (poll,
  crash-resume via the watchdog-style 'r'-suffix chain) but without any gate/settle
  logic -- there is no stage to detect, only an iteration target.
  """
  name = st["continue"]["name"]
  env = st["continue"]["env"]
  total_abs = st["continue"]["total_iters_abs"]
  parent_iter = st["continue"]["parent_iter"]
  first_resume = (plan["parent_dir_name"], plan["parent_ckpt"])
  watchdog_register(name, True, total_abs, a.num_envs, env)
  ensure_running(a, st, "C", name, env, total_abs, first_resume, start_iter=parent_iter)
  save_state(st)

  reader = LogReader(name, a.source)
  dead_since = None
  while True:
    time.sleep(a.poll)
    reader.poll()
    it = reader.max_iter()
    if it >= 0:
      say(f"[C] iter {it}/{total_abs - 1} (parent {plan['parent_dir_name']}/"
          f"{plan['parent_ckpt']} + {st['continue']['extra_iters']})")
    ck = latest_ckpt(name)
    if (ck and ck[0] >= total_abs - 1) or it >= total_abs - 1:
      break
    if not trainer_pids(name):
      if dead_since is None:
        dead_since = time.time()
        say(f"[C] trainer not alive at iter {it}; giving watchdog.sh {a.resume_grace}s")
      elif time.time() - dead_since > a.resume_grace:
        say("[C] watchdog did not resume in time -- resuming from the launcher")
        ensure_running(a, st, "C", name, env, total_abs, first_resume, start_iter=parent_iter)
        dead_since = None
    else:
      dead_since = None
    save_state(st)

  t0 = time.time()
  while trainer_pids(name) and time.time() - t0 < a.ckpt_wait:
    time.sleep(min(30, a.poll))
  stop_trainer(name)
  ck = latest_ckpt(name)
  st["continue"].update({
    "status": "done",
    "end_iter": ck[0] if ck else None,
    "ckpt_dir": os.path.basename(ck[1]) if ck else None,
    "ckpt": ck[2] if ck else None,
    "finished_at": datetime.now().isoformat(),
  })
  save_state(st)
  say(f"[C] DONE at iter {st['continue'].get('end_iter')} -- "
      f"{st['continue'].get('ckpt_dir')}/{st['continue'].get('ckpt')}")
  if ck:
    snapshot_review(ck[1], "CONTINUE phase-end: final smoke/training health review")
    build_accum_video(ck[1], f"{st['run']}_c")
    record_spec_tables(plan["parent_dir"], ck[1])


# --------------------------------------------------------------------------------------
def dry_run(a) -> None:
  e1, c1 = p1_env(a), train_cmd(a, f"{a.run}_p1", a.p1_max_iters, None)
  ex_end = a.p1_max_iters if a.smoke else 9000  # illustrative only
  ex_dr_end = ex_end + a.ramp_iters
  ex_total = ex_dr_end + a.digest_iters
  e2 = p2_env(a, ex_end, ex_end, ex_dr_end, ex_total)
  c2 = train_cmd(a, f"{a.run}_p2", ex_total - ex_end,
                 ("<P1_RUN_DIR>", "model_<P1_END>.pt"))
  print("=" * 100)
  print(f"DRY RUN  run={a.run}  ankle={a.ankle}  num_envs={a.num_envs}  seed={a.seed}"
        f"  smoke={a.smoke}")
  print(f"state file : analysis/out/v2_scratch_{a.run}.json")
  print(f"watchdog   : analysis/out/watchdog_runs.json  ({a.run}_p1, {a.run}_p2)")
  print(f"gpu lock   : {GPU_LOCK}  (enabled={not a.no_gpu_lock})")
  print("=" * 100)
  print("\n--- PHASE 1 : from scratch, flat, DR OFF, gated curriculum, no entropy anneal ---")
  print(f"  cap                : {a.p1_max_iters} iters (safety net only -- the gate ends P1)")
  print(f"  end when           : stage == {a.n_stages - 1} AND dwell >= {a.gate_min_dwell}"
        f" AND episodes >= {a.gate_min_episodes} AND fell_over <= {a.gate_fell_max}"
        f" AND err_steady <= baseline x {a.gate_err_ratio}, held {a.settle_hold} iters")
  print(f"  DR off via         : PYG_DR_START_ITER={DR_OFF_START} (dr_levels factor pinned 0.0)")
  print()
  print(render(e1, c1))
  print(f"\n  (log: logs/{a.run}_p1.log)")
  print("\n--- PHASE 2 : FULL resume, DR ramp computed FROM THE MEASURED P1 LENGTH ---")
  print("  PYG_DR_START_ITER  = P1_END                       (measured at P1 end)")
  print(f"  PYG_DR_END_ITER    = P1_END + {a.ramp_iters}")
  print(f"  --agent.max-iterations = {a.ramp_iters} + {a.digest_iters} = {a.ramp_iters + a.digest_iters}")
  print(f"  entropy anneal     = {a.entropy_start} -> {a.entropy_end} over P1_END .. P1_END+"
        f"{a.ramp_iters + a.digest_iters - 1}")
  print(f"  ILLUSTRATION BELOW USES P1_END={ex_end}; the real numbers are filled in at runtime.")
  print()
  print(render(e2, c2))
  print(f"\n  (log: logs/{a.run}_p2.log)")
  print("\n" + "=" * 100)
  print("nothing was launched (--dry-run)")


def dry_run_continue(a) -> dict[str, str]:
  """CONTINUE mode's dry-run preview. Unlike the scratch dry_run() above, this ALSO
  enforces the run-note gate: a continue attempt is a training attempt like any other,
  and the note is what the note-glob in require_experiment_note() needs to find before
  the real launch would be allowed to proceed. Returns the assembled env so the caller
  can feed it to the same action-window preflight the real launch runs.
  """
  note = require_experiment_note(a.run)
  plan = continue_plan(a)
  cmd = train_cmd(a, a.run, plan["extra_iters"],
                 (plan["parent_dir_name"], plan["parent_ckpt"]))
  st_preview = {
    "run": a.run,
    "mode": "continue",
    "parent_run_dir": plan["parent_dir"],
    "parent_ckpt": plan["parent_ckpt"],
    "parent_iter": plan["parent_iter"],
    "extra_iters": plan["extra_iters"],
    "env_overrides": plan["env_overrides"],
  }
  print("=" * 100)
  print(f"DRY RUN (continue)  run={a.run}  parent={plan['parent_dir_name']}"
        f"  ckpt={plan['parent_ckpt']}")
  print(f"note       : {os.path.relpath(note, REPO)}")
  print(f"state file : analysis/out/v2_scratch_{a.run}.json")
  print(f"watchdog   : analysis/out/watchdog_runs.json  ({a.run})")
  print(f"gpu lock   : {GPU_LOCK}  (enabled={not a.no_gpu_lock})")
  print("=" * 100)
  print(f"\nparent iter (parsed from checkpoint name) : {plan['parent_iter']}")
  print(f"extra iters                               : {plan['extra_iters']}")
  print(f"iteration cap (parent_iter + extra_iters) : {plan['total_iters_abs']}")
  print("  (--agent.max-iterations passed to THIS launch is extra_iters -- iterations "
        "remaining from the loaded checkpoint, same convention every resume in this "
        "file uses)")
  print(f"inherited PYG_* env from parent manifest  : {plan['parent_env_keys']} keys")
  print(f"--env overrides                           : {len(plan['env_overrides'])} key(s)"
        f"{' -> ' + str(plan['env_overrides']) if plan['env_overrides'] else ' (none)'}")
  print(f"total env keys launched with               : {len(plan['env'])} "
        f"({plan['parent_env_keys']} inherited + {len(plan['env_overrides'])} override(s), "
        f"minus any override that replaced -- not added -- a key)")
  print()
  print(render(plan["env"], cmd))
  print(f"\n  (log: logs/{a.run}.log)")
  print("\nstate file will additionally carry:")
  print(json.dumps(st_preview, indent=1, ensure_ascii=False))
  print("\n" + "=" * 100)
  print("nothing was launched (--dry-run)")
  return plan["env"]


def status(a) -> None:
  st = load_state(a.run)
  if not st:
    print(f"no state for run '{a.run}'")
    return
  print(json.dumps(st, indent=1, ensure_ascii=False))
  for ph in ("p1", "p2", "continue"):
    n = st.get(ph, {}).get("name")
    if n:
      print(f"{ph}: pids={trainer_pids(n)} latest_ckpt={latest_ckpt(n)}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--run", default=DEFAULT_RUN_NAME,
                 help="base run name; phases are <run>_p1 / <run>_p2. In CONTINUE mode "
                      "(--continue-from) this is used verbatim as the new run's name.")
  p.add_argument("--ankle", default="AB", choices=["AB", "RP"])
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--num-envs", type=int, default=16384)
  p.add_argument("--p1-max-iters", type=int, default=16000,
                 help="hard cap. 5 stages x PYG_GATE_MAX_DWELL(3000) = 15000 worst case "
                      "to reach the top stage, + 800 min dwell there")
  p.add_argument("--ramp-iters", type=int, default=10000, help="P2 DR ramp length")
  p.add_argument("--digest-iters", type=int, default=4000, help="P2 tail at full DR")
  p.add_argument("--settle-hold", type=int, default=10,
                 help="consecutive gate-passing iterations at the top stage")
  p.add_argument("--n-stages", type=int, default=5, help="len(velocity_stages) in env_cfgs")
  p.add_argument("--poll", type=int, default=60)
  p.add_argument("--idle-poll", type=int, default=60)
  p.add_argument("--resume-grace", type=int, default=420,
                 help="seconds to let watchdog.sh resume a dead trainer before we do")
  p.add_argument("--ckpt-wait", type=int, default=1800)
  p.add_argument("--source", default="auto", choices=["auto", "log", "tfevents"])
  # gate knobs (mirrored into PYG_GATE_* so the launcher and the trainer agree)
  p.add_argument("--gate-min-dwell", type=int, default=800)
  p.add_argument("--gate-max-dwell", type=int, default=3000)
  p.add_argument("--gate-window", type=int, default=100)
  p.add_argument("--gate-err-ratio", type=float, default=1.10)
  p.add_argument("--gate-fell-max", type=float, default=0.005)
  p.add_argument("--gate-min-episodes", type=int, default=64)
  p.add_argument("--entropy-start", type=float, default=0.01)
  p.add_argument("--entropy-end", type=float, default=0.002)
  p.add_argument("--video-interval", type=int, default=8000)
  p.add_argument("--video-length", type=int, default=500)
  p.add_argument("--logger", default="wandb")
  p.add_argument("--vy-stages", action="store_true", help="PYG_CMD_VY_STAGES=1 (docs/103 V5)")
  p.add_argument("--model-v4", action="store_true", help="PYG_MODEL_V4=1 (31.316 kg rebuild)")
  p.add_argument("--env", action="append", metavar="K=V", help="extra env var (repeatable)")
  p.add_argument("--smoke", action="store_true", help="tiny end-to-end validation of the orchestration")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--status", action="store_true")
  p.add_argument("--no-gpu-lock", action="store_true")
  p.add_argument("--allow-concurrent", action="store_true",
                 help="do not wait for other trainers to finish (15 GB host: OOM risk)")
  p.add_argument("--force-p2", action="store_true",
                 help="run P2 even if P1 never reached the top command stage")
  p.add_argument("--min-gpu-free", type=float, default=None, metavar="GIB",
                 help="override only the free-GPU-memory threshold of the preflight "
                      "(e.g. 7.5 for a second arm beside a running 16384-env run)")
  p.add_argument("--skip-resource-preflight", action="store_true",
                 help="emergency/debug only: bypass RAM/disk/GPU/heavy-job checks")
  # CONTINUE mode (2026-09-05): "+N more iterations from an existing checkpoint" -- see
  # the module docstring's CONTINUE MODE section. All three are required together and
  # select an entirely separate code path from P1/P2; giving none of them leaves every
  # existing flag combination's behavior unchanged.
  p.add_argument("--continue-from", default=None, metavar="RUN_DIR",
                 help="CONTINUE mode: prior run's log directory (absolute, or relative "
                      "to the mjlab checkout) to FULL-resume actor+critic+optimizer from")
  p.add_argument("--continue-ckpt", default=None, metavar="CKPT",
                 help="CONTINUE mode: checkpoint filename inside --continue-from, e.g. "
                      "model_18099.pt")
  p.add_argument("--extra-iters", type=int, default=None,
                 help="CONTINUE mode: iterations to add on top of the checkpoint's own "
                      "iteration count (parsed from its filename)")
  a = p.parse_args()
  continue_mode = validate_continue_args(a)

  if a.smoke:
    # A smoke is still a training attempt: it gets the same note, repro snapshots and
    # progress-video evidence as a long run. Only the performance interpretation differs.
    if a.run == "v2s1_AB":
      a.run = "v2scratch_smoke_test"
    a.num_envs = 1024
    a.p1_max_iters = 400
    a.ramp_iters = 120
    a.digest_iters = 80
    # The smoke validates the ORCHESTRATION, not the policy: a from-scratch 1024-env run
    # falls constantly and its steady error grows as the box widens, so the real quality
    # criteria would never fire and P1 would end on the cap -- the one path that does NOT
    # exercise settle-detection. Relax them so the gate promotes on DWELL alone and the
    # top-stage settle path is what actually runs. Quality knobs stay at their real values
    # for anything that is not --smoke.
    a.gate_min_dwell = 20
    a.gate_max_dwell = 60
    a.gate_window = 20
    a.gate_min_episodes = 32
    a.gate_fell_max = 1.0
    a.gate_err_ratio = 100.0
    a.settle_hold = 3
    a.poll = 20
    a.idle_poll = 30
    a.resume_grace = 60
    a.ckpt_wait = 900
    a.video_interval = 2400  # one clip per ~100 iters so orchestration progress is visible

  if a.status:
    status(a)
    return
  if a.dry_run:
    if continue_mode:
      env_for_preflight = dry_run_continue(a)
    else:
      dry_run(a)
      env_for_preflight = p1_env(a)
    # exercise the launch blocker too: a dry run that prints a command the real launcher
    # would refuse is worse than useless.
    print("\n--- ACTION-WINDOW PRE-FLIGHT (same check the real launch must pass) ---")
    action_window_preflight(a, env_for_preflight)
    return

  os.makedirs(OUT, exist_ok=True)
  global _LOGF, _NOTE
  _LOGF = open(os.path.join(OUT, f"v2_scratch_{a.run}.log"), "a")
  _NOTE = require_experiment_note(a.run)
  say("=" * 90)
  say(f"run_v2_scratch: run={a.run} ankle={a.ankle} envs={a.num_envs} smoke={a.smoke}"
      f"{' mode=continue' if continue_mode else ''}")
  say(f"[record] experiment note -> {os.path.relpath(_NOTE, REPO)}")
  start_review_loop(a.run, _NOTE)

  if continue_mode:
    plan = continue_plan(a)
    # The action-window gate is NOT covered by --skip-resource-preflight (see the P1/P2
    # comment below); it applies here on the INHERITED+overridden env, i.e. the config
    # this specific continue launch will actually build.
    action_window_preflight(a, plan["env"])
    if not a.skip_resource_preflight:
      resource_preflight(a, plan["env"])

    st = load_state(a.run)
    if not st:
      st = {
        "run": a.run,
        "mode": "continue",
        "created": datetime.now().isoformat(),
        "args": vars(a),
        "continue": {
          "name": a.run,
          "status": "running",
          "parent_run_dir": plan["parent_dir"],
          "parent_ckpt": plan["parent_ckpt"],
          "parent_iter": plan["parent_iter"],
          "extra_iters": plan["extra_iters"],
          "total_iters_abs": plan["total_iters_abs"],
          "env_overrides": plan["env_overrides"],
          "env": plan["env"],
        },
      }
      save_state(st)
      say(f"[state] created {state_path(a.run)}")
    else:
      say(f"[state] resuming from {state_path(a.run)}: "
          f"continue={st.get('continue', {}).get('status')}")

    lock = GpuLock(GPU_LOCK, f"run_v2_scratch:{a.run}", enabled=not a.no_gpu_lock)
    lock.acquire()
    for sig in (signal.SIGINT, signal.SIGTERM):
      signal.signal(sig, lambda *_: (lock.release(), sys.exit(130)))
    try:
      # ty infers a loose union for st["continue"] from the ad-hoc dict literal above
      # (same reason st["p1"]/st["p2"] already carry this warning elsewhere in this
      # file); .get() on a plain dict is safe at runtime regardless.
      if st["continue"].get("status") != "done":  # type: ignore[possibly-missing-attribute]
        drive_continue(a, st, plan)
      else:
        say("[C] already done")
    finally:
      lock.release()

    say("=" * 90)
    say("ALL PHASES DONE (continue)")
    say(f"  C: parent={st['continue'].get('parent_run_dir')} "  # type: ignore[possibly-missing-attribute]
        f"ckpt={st['continue'].get('parent_ckpt')} "  # type: ignore[possibly-missing-attribute]
        f"parent_iter={st['continue'].get('parent_iter')} "  # type: ignore[possibly-missing-attribute]
        f"+{st['continue'].get('extra_iters')} iters -> "  # type: ignore[possibly-missing-attribute]
        f"end {st['continue'].get('end_iter')} -> "  # type: ignore[possibly-missing-attribute]
        f"{st['continue'].get('ckpt_dir')}/{st['continue'].get('ckpt')}")  # type: ignore[possibly-missing-attribute]
    return

  # The action-window gate is NOT covered by --skip-resource-preflight: that flag exists
  # for a host that is temporarily short on RAM/disk, which is a reason to wait, whereas a
  # joint the policy cannot command is a reason never to start.
  action_window_preflight(a, p1_env(a))
  if not a.skip_resource_preflight:
    resource_preflight(a, p1_env(a))

  st = load_state(a.run)
  if not st:
    st = {
      "run": a.run,
      "created": datetime.now().isoformat(),
      "args": vars(a),
      "p1": {"name": f"{a.run}_p1", "status": "running", "cap": a.p1_max_iters},
      "p2": {"name": f"{a.run}_p2", "status": "pending"},
    }
    save_state(st)
    say(f"[state] created {state_path(a.run)}")
  else:
    say(f"[state] resuming from {state_path(a.run)}: "
        f"p1={st['p1'].get('status')} p2={st['p2'].get('status')}")

  lock = GpuLock(GPU_LOCK, f"run_v2_scratch:{a.run}", enabled=not a.no_gpu_lock)
  lock.acquire()
  for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: (lock.release(), sys.exit(130)))
  try:
    if st["p1"].get("status") != "done":
      drive_p1(a, st)
    else:
      say(f"[P1] already done at iter {st['p1']['end_iter']} -- skipping to P2")
    if st["p2"].get("status") != "done":
      drive_p2(a, st)
    else:
      say("[P2] already done")
  finally:
    lock.release()

  say("=" * 90)
  say("ALL PHASES DONE")
  say(f"  P1: {st['p1'].get('end_iter')} iters ({st['p1'].get('end_reason')}) "
      f"-> {st['p1'].get('ckpt_dir')}/{st['p1'].get('ckpt')}")
  say(f"  P2: DR {st['p2'].get('dr_start_iter')}..{st['p2'].get('dr_end_iter')}, "
      f"end {st['p2'].get('end_iter')} -> {st['p2'].get('ckpt_dir')}/{st['p2'].get('ckpt')}")


if __name__ == "__main__":
  main()
