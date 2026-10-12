"""PPO settings of training run 2026-10-11_23-35-20_rough_box_main_v11a (params/agent.yaml).

entropy_coef is 0.002. The run's agent.yaml shows 0.01, but the run overrode it to a constant
0.002 before every update (with 0.01 the action std grew 0.13 -> 0.60 in rough-terrain
fine-tuning and turning in place collapsed).
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def pygmalion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pygmalion_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=12_000,
    seed=43,
    logger="tensorboard",
  )
