import os
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  # -------------------------------------------------------------------------
  # V4 (docs/103 §2, §3b): linear entropy-coefficient annealing.
  #
  # c3's exploration noise went 0.296 -> 0.474 over training and was never pulled
  # back. Deployment runs the mean action, so the policy that ships is one that
  # only ever practised while shaking. Anneal entropy_coef 0.01 -> 0.002 across
  # the phase-2 window and sigma should follow it down; docs/103 sets the
  # termination criterion at sigma <= 0.35.
  #
  # Implemented as a wrapper around alg.update() rather than an edit to rsl_rl:
  # update() is called exactly once per iteration, and setting the coefficient
  # immediately before it means the value that lands in the PPO loss is the one
  # this schedule computed. OFF unless PYG_ENTROPY_ANNEAL is set.
  #
  #   PYG_ENTROPY_ANNEAL=1       enable
  #   PYG_ENTROPY_ANNEAL_START   absolute iteration where annealing starts
  #                              (default: max(PYG_DR_START_ITER, this run's first
  #                               iteration) - i.e. the start of phase 2)
  #   PYG_ENTROPY_ANNEAL_END     absolute iteration where it reaches the floor
  #                              (default: this run's last iteration)
  #   PYG_ENTROPY_START_COEF     default 0.01   (matches rl_cfg's entropy_coef)
  #   PYG_ENTROPY_END_COEF       default 0.002
  #   PYG_ENTROPY_LOG_EVERY      default 25 iterations
  # -------------------------------------------------------------------------
  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if not os.environ.get("PYG_ENTROPY_ANNEAL"):
      return super().learn(num_learning_iterations, init_at_random_ep_len)

    first_it = self.current_learning_iteration
    last_it = first_it + num_learning_iterations - 1

    def _iter_env(name: str, default: int) -> int:
      v = os.environ.get(name)
      return default if v in (None, "") else int(v)

    start_it = _iter_env(
      "PYG_ENTROPY_ANNEAL_START",
      max(_iter_env("PYG_DR_START_ITER", first_it), first_it),
    )
    end_it = _iter_env("PYG_ENTROPY_ANNEAL_END", last_it)
    c0 = float(os.environ.get("PYG_ENTROPY_START_COEF", "0.01"))
    c1 = float(os.environ.get("PYG_ENTROPY_END_COEF", "0.002"))
    log_every = max(1, _iter_env("PYG_ENTROPY_LOG_EVERY", 25))

    print(
      f"[entropy-anneal] ON: entropy_coef {c0} -> {c1} linearly over iters "
      f"{start_it}..{end_it} (this run covers {first_it}..{last_it}, "
      f"cfg entropy_coef was {self.alg.entropy_coef})",
      flush=True,
    )

    orig_update = self.alg.update
    cursor = {"it": first_it}

    def _annealed_update(*args, **kwargs):
      it = cursor["it"]
      if it <= start_it:
        coef = c0
      elif it >= end_it:
        coef = c1
      else:
        frac = (it - start_it) / float(max(1, end_it - start_it))
        coef = c0 + (c1 - c0) * frac
      self.alg.entropy_coef = coef
      out = orig_update(*args, **kwargs)
      # Read the value BACK off the algorithm - this is the number the loss used,
      # not the number the schedule intended.
      if it % log_every == 0 or it in (first_it, last_it):
        std = self.alg.get_policy().output_std
        std_mean = float(std.mean().item()) if hasattr(std, "mean") else float("nan")
        print(
          f"[entropy-anneal] iter {it} entropy_coef={self.alg.entropy_coef:.6f} "
          f"action_std={std_mean:.4f}",
          flush=True,
        )
      cursor["it"] = it + 1
      return out

    self.alg.update = _annealed_update  # type: ignore[method-assign]
    try:
      return super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.alg.update = orig_update  # type: ignore[method-assign]
