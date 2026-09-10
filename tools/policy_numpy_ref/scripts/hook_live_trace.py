#!/usr/bin/env python3
"""Roll a real mjlab episode with the legonly_ab_v3_keyfix checkpoint and, via
PyTorch forward hooks (no source-file edits to mjlab/rsl_rl), dump the REAL
numeric value at every pipeline stage for several consecutive control steps:

    raw sensor/sim ingredients -> concatenated raw obs -> normalized obs
    -> each MLP layer's output -> raw action -> scaled+offset target
    -> clipped target -> final target (== the real-hardware q_target quantity)
    -> (for contrast only) MuJoCo's data.ctrl, which is a further TN-curve-PD
       torque for this checkpoint's actuators, NOT the same quantity.

These are the "real worked examples" embedded in
docs/134_legonly_ab_v3_keyfix_policy_eli5.md, and the reference data
test_action_pipeline.py's test_real_rollout_targets_against_live_trace checks
against.

This script constructs the env/runner using the SAME mjlab/pygviewer.bake
helper functions the existing bake tool uses (variant_env/load_contract/TASK --
plain config plumbing, not policy-inference logic) so it builds an identical
environment, but it does NOT import pygviewer.policy's ObsBuilder/
action_to_target/OnnxPolicy -- this script computes every pipeline stage itself,
independently, via hooks + the checkpoint's own contract values.

Run (mjlab venv):
    CUDA_VISIBLE_DEVICES= mujoco-sim/mjlab/.venv/bin/python3 \\
        tools/policy_numpy_ref/scripts/hook_live_trace.py --steps 40
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "pygviewer"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CKPT = (
  REPO
  / "mujoco-sim/mjlab/logs/rsl_rl/pygmalion_velocity"
  / "2026-09-06_11-06-27_legonly_ab_v3_keyfix_p2/model_27999.pt"
)
VARIANT = "LegOnly-AB"
CACHE_DIR = "/home/syaro/pyg_fea/pygviewer/cache"


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--steps", type=int, default=40)
  p.add_argument(
    "--out", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "legonly_ab_v3_keyfix_live_trace.npz")
  )
  a = p.parse_args()

  os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

  # Env-var plumbing reused from pygviewer.bake (sets the same PYG_* toggles this
  # checkpoint was trained under) -- this is configuration, not inference logic.
  from pygviewer.bake import TASK, variant_env

  os.environ.update(variant_env(VARIANT, init_bent=True))

  import numpy as np
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from tensordict import TensorDict

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  # Nominal run: strip DR events so this is the CLEAN pipeline (encoder_bias == 0),
  # matching the existing bake's convention -- same reasoning: describe the model,
  # not one random DR draw. (A second, DR-on pass is left as a documented follow-up
  # in docs/136 rather than done here, per the approved plan's scope.)
  dropped = []
  for ev in list(cfg.events):
    if ev in ("push_robot", "foot_friction", "encoder_bias", "base_com") or ev.startswith(
      "inertial_dr_"
    ):
      cfg.events.pop(ev)
      dropped.append(ev)
  big = cfg.episode_length_s + 100.0
  cfg.commands["twist"].resampling_time_range = (big, big)

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  rl_cfg = load_rl_cfg(TASK)
  envw = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(envw, asdict(rl_cfg), device="cpu")
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")

  robot = env.scene["robot"]
  names = list(robot.joint_names)
  om = env.observation_manager
  obs_group = policy.obs_groups[0] if getattr(policy, "obs_groups", None) else "actor"

  # -------------------------------------------------------------- hooks
  captured: dict[str, list] = {"normalized_obs": [], "layer0": [], "layer2": [], "layer4": [], "layer6": []}

  def hook_normalizer(_module, _inp, out):
    captured["normalized_obs"].append(out.detach().cpu().numpy().copy())

  def make_hook(key):
    def hook(_module, _inp, out):
      captured[key].append(out.detach().cpu().numpy().copy())

    return hook

  policy.obs_normalizer.register_forward_hook(hook_normalizer)
  policy.mlp[0].register_forward_hook(make_hook("layer0"))  # pre-ELU Linear(45,512) output
  policy.mlp[2].register_forward_hook(make_hook("layer2"))  # pre-ELU Linear(512,256) output
  policy.mlp[4].register_forward_hook(make_hook("layer4"))  # pre-ELU Linear(256,128) output
  policy.mlp[6].register_forward_hook(make_hook("layer6"))  # final Linear(128,12) output == raw action

  # -------------------------------------------------------------- contract values (for action post-processing)
  from pygviewer.contract import load_contract

  model_contract = load_contract(CACHE_DIR, VARIANT)
  action_joint_names = list(model_contract.raw["action_joint_names"])
  default_q_action = np.array(
    [float(robot.data.default_joint_pos[0][names.index(n)]) for n in action_joint_names]
  )
  action_scale = np.array([model_contract.raw["action_scale"][n] for n in action_joint_names])
  safe_clip = model_contract.raw["safe_clip"]
  clip_lo = np.array([safe_clip[n][0] for n in action_joint_names])
  clip_hi = np.array([safe_clip[n][1] for n in action_joint_names])

  import mujoco

  m = env.sim.mj_model
  s_adr = {}
  for s in range(m.nsensor):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or ""
    s_adr[nm.split("/")[-1]] = (int(m.sensor_adr[s]), int(m.sensor_dim[s]))
  actuator_ids = env.action_manager._terms["joint_pos"]._target_ids
  # data.ctrl is indexed by ACTUATOR id, not joint id -- for this checkpoint the 12
  # leg joints are the model's only actuators, so ctrl already has exactly 12
  # entries and needs no re-indexing (captured here only as a "for contrast, this
  # is NOT the deploy q_target" illustration -- see module docstring).
  n_ctrl = env.sim.mj_model.nu

  def t2n(x):
    return np.asarray(x.detach().cpu()).astype(np.float64)

  got = envw.get_observations()
  obs = got[0] if isinstance(got, tuple) else got

  rows = {
    "raw_obs": [],
    "processed_target": [],
    "clipped_target": [],
    "final_target": [],
    "encoder_bias": [],
    "ctrl": [],
    "raw_action": [],
  }
  for _ in range(a.steps):
    actor_obs = obs[obs_group]
    rows["raw_obs"].append(t2n(actor_obs)[0])
    with torch.no_grad():
      act = policy(TensorDict({obs_group: actor_obs}, batch_size=[actor_obs.shape[0]]))
    raw_action_np = t2n(act)[0]
    rows["raw_action"].append(raw_action_np)

    processed = raw_action_np * action_scale + default_q_action
    rows["processed_target"].append(processed)
    clipped = np.clip(processed, clip_lo, clip_hi)
    rows["clipped_target"].append(clipped)
    enc_bias = t2n(robot.data.encoder_bias)[0][actuator_ids]
    rows["encoder_bias"].append(enc_bias)
    final_target = clipped - enc_bias
    rows["final_target"].append(final_target)
    rows["ctrl"].append(t2n(env.sim.data.ctrl)[0][:n_ctrl])

    stepped = envw.step(act)
    obs = stepped[0]

  out = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
  out["normalized_obs"] = np.concatenate(captured["normalized_obs"], axis=0).astype(np.float64)
  out["mlp_layer0_out"] = np.concatenate(captured["layer0"], axis=0).astype(np.float64)
  out["mlp_layer2_out"] = np.concatenate(captured["layer2"], axis=0).astype(np.float64)
  out["mlp_layer4_out"] = np.concatenate(captured["layer4"], axis=0).astype(np.float64)
  out["mlp_layer6_out"] = np.concatenate(captured["layer6"], axis=0).astype(np.float64)
  out["default_q_action"] = default_q_action
  out["action_scale"] = action_scale
  out["clip_lo"] = clip_lo
  out["clip_hi"] = clip_hi
  out["action_joint_names"] = np.asarray(action_joint_names)
  out["dropped_events"] = np.asarray(dropped)
  out["checkpoint"] = str(CKPT)
  out["bake_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

  Path(a.out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(a.out, **out)
  print(f"wrote {a.out}")
  print(f"  steps: {a.steps}")
  print(f"  dropped_events (DR-off nominal): {dropped}")
  print()
  print("=== step 0 worked example ===")
  print("raw_obs[:6]           :", out["raw_obs"][0][:6])
  print("normalized_obs[:6]    :", out["normalized_obs"][0][:6])
  print("raw_action            :", out["raw_action"][0])
  print("processed_target      :", out["processed_target"][0])
  print("clipped_target        :", out["clipped_target"][0])
  print("encoder_bias (DR-off) :", out["encoder_bias"][0])
  print("final_target          :", out["final_target"][0])
  print("ctrl (torque, NOT the same quantity as final_target):", out["ctrl"][0])


if __name__ == "__main__":
  main()
