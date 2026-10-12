#!/usr/bin/env python3
"""Compare this package's resolved env/agent config with the training run's own dump.

reference/env_reference.yaml and reference/agent_reference.yaml are params/env.yaml and
params/agent.yaml written by mjlab when run 2026-10-11_23-35-20_rough_box_main_v11a started.
This script builds the package config with stock mjlab, dumps it the same way, and compares
the two trees value by value.

Normalisation before comparing:
  * function / class references are compared by their final name only (the module path
    differs: the run used a fork of mjlab, this package keeps its terms in pygmalion_mjlab.mdp);
  * the `clamped_` prefix added by env_cfg.clamped (per-term reward clamp) is removed.

Before dumping, the two fields mjlab fills in at env construction are set the same way
(terrain num_envs, sub-terrain size).

Expected differences (reported separately, not counted as failures):
  * commands.twist.ranges.*  - the reference was dumped after the first forced promotion of
    the run's command gate (stage 1 range); training used the final range (see env_cfg.py);
  * curriculum.command_vel   - the gate term itself (removed here);
  * meshes / sim.ls_parallel - API fields that differ between the fork and stock mjlab;
  * agent entropy_coef / run bookkeeping fields (run_name, resume, load_run, ...).

  python scripts/verify_config.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

EXPECTED_ENV = (
  "commands.twist.ranges.", "curriculum.command_vel",
  # API-only differences between the run's fork and stock mjlab 1.5.0 (no physical effect):
  "scene.entities.robot.meshes", "scene.terrain.meshes",  # stock MeshCfg field (empty)
  "sim.ls_parallel",  # removed in MuJoCo Warp 3.10; the fork kept a no-op field
)
EXPECTED_AGENT = (
  "algorithm.entropy_coef", "run_name", "resume", "load_run", "load_checkpoint",
  "wandb_project", "upload_model",
)


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


def main() -> int:
  import pygmalion_mjlab  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.os import dump_yaml

  env_cfg = load_env_cfg(pygmalion_mjlab.TASK_ID)
  env_cfg.scene.num_envs = 4096
  env_cfg.seed = 43
  agent_cfg = load_rl_cfg(pygmalion_mjlab.TASK_ID)
  # Fields mjlab fills in when the env is built (the reference was dumped after that):
  # Scene copies num_envs to the terrain, TerrainGenerator copies its size to each sub-terrain.
  env_cfg.scene.terrain.num_envs = env_cfg.scene.num_envs
  for sub in env_cfg.scene.terrain.terrain_generator.sub_terrains.values():
    sub.size = env_cfg.scene.terrain.terrain_generator.size
  # every non-zero-weight reward term must carry the clamp wrapper
  unclamped = [n for n, t in env_cfg.rewards.items()
               if t.weight != 0.0 and not getattr(t.func, "__name__", "").startswith("clamped_")]
  with tempfile.TemporaryDirectory() as d:
    env_p, agent_p = Path(d) / "env.yaml", Path(d) / "agent.yaml"
    dump_yaml(env_p, env_cfg)
    dump_yaml(agent_p, agent_cfg)
    n = compare("env", ROOT / "reference/env_reference.yaml", env_p, EXPECTED_ENV)
    n += compare("agent", ROOT / "reference/agent_reference.yaml", agent_p, EXPECTED_AGENT)
  print(f"\nreward terms without the clamp wrapper: {unclamped or 'none'}")
  ok = n == 0 and not unclamped
  print("RESULT: " + ("MATCH (only the expected differences)" if ok else "MISMATCH"))
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
