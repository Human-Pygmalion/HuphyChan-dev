#!/usr/bin/env python3
"""Compare this package's resolved env/agent config with each training run's own dump.

reference/<stage>/env.yaml and agent.yaml are params/env.yaml and params/agent.yaml written by
mjlab when the run started (stages: flat_p1, flat_p2, rough_main). This script builds the
package config for the matching task with stock mjlab, dumps it the same way, and compares the
two trees value by value.

Normalisation before comparing:
  * function / class references are compared by their final name only (the module path
    differs: the runs used a fork of mjlab, this package keeps its terms in pygmalion_mjlab.mdp);
  * the `clamped_` prefix added by env_cfg.clamped (per-term reward clamp) is removed.

Before dumping, the fields mjlab fills in at env construction are set the same way
(terrain num_envs, sub-terrain size).

Expected differences (reported separately, not counted as failures):
  * all stages: meshes / sim.ls_parallel - API fields that differ between the fork and stock
    mjlab; agent run bookkeeping (run_name, resume, load_run, load_checkpoint, ...);
  * rough_main: commands.twist.ranges.* and curriculum.command_vel (the run's gated command
    curriculum is replaced by its final range, see env_cfg.py), agent entropy_coef;
  * flat_p2: commands.twist.ranges.* - the reference was dumped after the gate's first forced
    promotion (stage 1 range); the package starts at stage 0 and the gate promotes the same way.

  python scripts/verify_config.py [--stage flat_p1|flat_p2|rough_main|all]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

API_ONLY = (
  "scene.entities.robot.meshes", "scene.terrain.meshes",  # stock MeshCfg field (empty)
  "sim.ls_parallel",  # removed in MuJoCo Warp 3.10; the fork kept a no-op field
)
BOOKKEEPING = ("run_name", "resume", "load_run", "load_checkpoint", "wandb_project", "upload_model")
EXPECTED_ENV = {
  "rough_main": ("commands.twist.ranges.", "curriculum.command_vel") + API_ONLY,
  "flat_p1": API_ONLY,
  "flat_p2": ("commands.twist.ranges.",) + API_ONLY,
}
EXPECTED_AGENT = {
  "rough_main": ("algorithm.entropy_coef",) + BOOKKEEPING,
  "flat_p1": BOOKKEEPING,
  "flat_p2": BOOKKEEPING,
}


class _Loader(yaml.SafeLoader):
  pass


def _any(loader, suffix, node):
  if isinstance(node, yaml.MappingNode):
    d = loader.construct_mapping(node, deep=True)
    return d
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node, deep=True)
  val = loader.construct_scalar(node)
  if suffix.startswith("python/name:") or suffix.startswith("python/object"):
    name = suffix.split(":", 1)[1].rsplit(".", 1)[-1]
    return f"<ref {name.removeprefix('clamped_')}>"
  return val


_Loader.add_multi_constructor("tag:yaml.org,2002:", _any)
_Loader.add_multi_constructor("", _any)


def load(p: Path):
  return yaml.load(p.read_text(), Loader=_Loader)


def walk(a, b, path, out):
  if isinstance(a, dict) and isinstance(b, dict):
    for k in sorted(set(a) | set(b), key=str):
      p = f"{path}.{k}" if path else str(k)
      if k not in a:
        out.append((p, "<missing>", b[k]))
      elif k not in b:
        out.append((p, a[k], "<missing>"))
      else:
        walk(a[k], b[k], p, out)
  elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
    if len(a) != len(b):
      out.append((path, f"len {len(a)}", f"len {len(b)}"))
    for i, (x, y) in enumerate(zip(a, b)):
      walk(x, y, f"{path}[{i}]", out)
  else:
    if isinstance(a, float) and isinstance(b, float):
      if abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b)):
        return
    if a != b:
      out.append((path, a, b))


def compare(name: str, ref: Path, mine: Path, expected: tuple[str, ...]) -> int:
  diffs: list = []
  walk(load(ref), load(mine), "", diffs)
  real = [d for d in diffs if not d[0].startswith(expected)]
  exp = [d for d in diffs if d[0].startswith(expected)]
  print(f"\n== {name}: {len(real)} unexpected difference(s), {len(exp)} expected")
  for p, a, b in real[:60]:
    print(f"  DIFF {p}: reference={str(a)[:120]!s}  package={str(b)[:120]!s}")
  for p, a, b in exp:
    print(f"  expected {p}: reference={str(a)[:80]}  package={str(b)[:80]}")
  return len(real)


def check_stage(stage: str) -> bool:
  import pygmalion_mjlab
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.os import dump_yaml

  task = {v: k for k, v in pygmalion_mjlab.STAGES.items()}[stage]
  ref_dir = ROOT / "reference" / stage
  print(f"\n######## {stage}  (task {task}, reference {ref_dir.relative_to(ROOT)})")
  ref_env = load(ref_dir / "env.yaml")
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = int(ref_env["scene"]["num_envs"])
  env_cfg.seed = 43
  agent_cfg = load_rl_cfg(task)
  # Fields mjlab fills in when the env is built (the reference was dumped after that):
  # Scene copies num_envs to the terrain, TerrainGenerator copies its size to each sub-terrain.
  env_cfg.scene.terrain.num_envs = env_cfg.scene.num_envs
  tg = env_cfg.scene.terrain.terrain_generator
  if tg is not None:
    for sub in tg.sub_terrains.values():
      sub.size = tg.size
  weighted = [(n, t) for n, t in env_cfg.rewards.items() if t.weight != 0.0]
  clamped_terms = [n for n, t in weighted if getattr(t.func, "__name__", "").startswith("clamped_")]
  with tempfile.TemporaryDirectory() as d:
    env_p, agent_p = Path(d) / "env.yaml", Path(d) / "agent.yaml"
    dump_yaml(env_p, env_cfg)
    dump_yaml(agent_p, agent_cfg)
    n = compare("env", ref_dir / "env.yaml", env_p, EXPECTED_ENV[stage])
    n += compare("agent", ref_dir / "agent.yaml", agent_p, EXPECTED_AGENT[stage])
  # The rough run clamped every weighted reward term (PYG_REWARD_TERM_CLAMP=50); the flat runs
  # did not. The clamp is not visible in env.yaml, so check it here.
  if stage == "rough_main":
    bad = [n for n, _ in weighted if n not in clamped_terms]
    print(f"reward terms without the clamp wrapper (all must have it): {bad or 'none'}")
  else:
    bad = clamped_terms
    print(f"reward terms with a clamp wrapper (none must have it): {bad or 'none'}")
  ok = n == 0 and not bad
  print(f"{stage}: " + ("MATCH (only the expected differences)" if ok else "MISMATCH"))
  return ok


def main() -> int:
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", default="all", choices=("all", "flat_p1", "flat_p2", "rough_main"))
  stages = ("flat_p1", "flat_p2", "rough_main") if (a := ap.parse_args()).stage == "all" else (a.stage,)
  results = {st: check_stage(st) for st in stages}
  print("\nRESULT: " + ", ".join(f"{k} {'MATCH' if v else 'MISMATCH'}" for k, v in results.items()))
  return 0 if all(results.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
