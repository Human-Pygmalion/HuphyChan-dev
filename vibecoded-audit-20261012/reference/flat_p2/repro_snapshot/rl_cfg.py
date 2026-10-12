"""RL configuration for Pygmalion velocity task."""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pygmalion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Pygmalion velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      # PYG_CONNECTOME=1 (2026-09-18, docs/151): the actor's core becomes a piece of the
      # fly's wiring diagram (rl/connectome_policy.py). Off, the string is "MLPModel" and
      # the run is byte-identical to before. The critic is never touched.
      class_name=(
        "mjlab.rl.connectome_policy:ConnectomeActor"
        if os.environ.get("PYG_CONNECTOME") == "1"
        else "MLPModel"
      ),
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      # PYG_CAPS_S / PYG_CAPS_T (2026-09-17): 관측 잡음이 행동으로 증폭되는 것을 줄이는
      # CAPS 정규화(arXiv 2012.06644)를 쓸 때만 알고리즘을 바꾼다. 켜지 않으면 문자열이
      # "PPO" 그대로라 예전 런과 **한 글자도 다르지 않다**. 설명은 rl/smooth_ppo.py.
      class_name=(
        "mjlab.rl.smooth_ppo:SmoothPPO"
        if (
          os.environ.get("PYG_CAPS_S")
          or os.environ.get("PYG_CAPS_T")
          or os.environ.get("PYG_LCP")
          or os.environ.get("PYG_SN")
        )
        else "PPO"
      ),
    ),
    experiment_name="pygmalion_velocity",
    wandb_project="pygmalion",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=30_001,
  )
