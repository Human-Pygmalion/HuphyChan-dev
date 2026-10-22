"""PPO + CAPS spatial smoothness: make the policy mean insensitive to observation noise.

Loss added to the PPO loss on every mini-batch (Mysore et al., "Regularizing Action Policies
for Smooth Control with Reinforcement Learning", ICRA 2021, spatial term):

  L_caps = caps_coef * mean_batch || mu(o) - mu(o + eps) ||^2,   eps ~ U(-n, n) per actor dim

with n = the training observation noise of each actor dim (`env_cfg.ACTOR_NOISE`), mu = the
deterministic actor output. It only compares the policy with itself on the same state with
different sensor noise, so it does not penalize intended motion and, unlike action_rate,
does not depend on the exploration std.

rsl_rl's PPO.update has one hook for an extra actor loss (the symmetry extension: augment the
batch, compute a loss, add coeff * loss). `_CapsLoss` fills that slot with no augmentation;
the logged loss is renamed "caps".
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class _CapsLoss:
  use_mirror_loss = True

  def __init__(self, coef: float, noise: torch.Tensor, group: str = "actor"):
    self.mirror_loss_coeff = coef
    self.noise = noise
    self.group = group

  def augment_batch(self, batch, original_batch_size: int) -> None:
    del batch, original_batch_size

  def compute_loss(self, actor, batch, original_batch_size: int) -> torch.Tensor:
    obs = batch.observations[:original_batch_size]
    x = obs[self.group]
    eps = (torch.rand_like(x) * 2 - 1) * self.noise
    noisy = obs.clone()
    noisy[self.group] = x + eps
    return (actor(obs) - actor(noisy)).square().sum(dim=-1).mean()


class CapsPPO(PPO):
  def __init__(self, *args, caps_coef: float = 0.0, caps_noise: tuple[float, ...] = (), **kwargs):
    super().__init__(*args, **kwargs)
    assert self.symmetry is None, "CapsPPO uses the symmetry hook; symmetry_cfg must be unset"
    if caps_coef > 0:
      noise = torch.tensor(caps_noise, dtype=torch.float, device=self.device)
      dim = self.storage.observations["actor"].shape[-1]
      assert noise.numel() == dim, f"caps_noise has {noise.numel()} entries, actor obs has {dim}"
      self.symmetry = _CapsLoss(caps_coef, noise)

  def update(self) -> dict[str, float]:
    out = super().update()
    if "symmetry" in out:
      out["caps"] = out.pop("symmetry")
    return out
