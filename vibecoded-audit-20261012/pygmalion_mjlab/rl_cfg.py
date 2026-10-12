"""PPO settings of the three training stages (reference/<stage>/agent.yaml). All three are
identical except entropy_coef and max_iterations.

entropy_coef:
  flat_p1     0.01 constant.
  flat_p2     0.01 -> 0.002 linearly over iterations 15,999 -> 29,998. agent.yaml shows
              0.01; the run's runner overrode it before every update. Here:
              scripts/train.py --entropy-anneal 15999 29998 0.01 0.002.
  rough_main  0.002 constant. agent.yaml shows 0.01; the run overrode it to 0.002 before
              every update (with 0.01 the action std grew 0.13 -> 0.60 in rough-terrain
              fine-tuning and turning in place collapsed).
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


ENTROPY_COEF = {"flat_p1": 0.01, "flat_p2": 0.01, "rough_main": 0.002}
MAX_ITERATIONS = {"flat_p1": 16_000, "flat_p2": 14_000, "rough_main": 12_000}


def pygmalion_ppo_runner_cfg(stage: str = "rough_main") -> RslRlOnPolicyRunnerCfg:
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
      entropy_coef=ENTROPY_COEF[stage],
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
    max_iterations=MAX_ITERATIONS[stage],
    seed=43,
    logger="tensorboard",
  )
