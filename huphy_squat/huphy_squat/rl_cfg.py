"""PPO settings (rsl_rl) plus the CAPS smoothness loss (caps.py)."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env_cfg import ACTOR_NOISE


@dataclass
class HuphyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "huphy_squat.caps:CapsPPO"
  caps_coef: float = 0.3
  """Weight of mean ||mu(o) - mu(o + eps)||^2 (0 disables it). In 1000-iteration tests 1.0 cut the
  hold-still jitter by 81 % but fell more under 50 N pushes; 0.3 cuts it by 69 % with push
  robustness close to a policy trained without the smoothness terms (action_rate -0.1, no
  torque_rate, no CAPS)."""
  caps_noise: tuple[float, ...] = ACTOR_NOISE
  """Half-width of eps per actor dim (= the training observation noise)."""


def huphy_squat_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=HuphyPpoAlgorithmCfg(
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
    ),
    experiment_name="huphy_squat",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=5_001,
    seed=43,
    logger="wandb",
    wandb_project="huphy_squat",
  )
