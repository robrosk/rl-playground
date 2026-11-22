"""What is left of the old PPO ``Policy`` class once GAE, the losses, the optimizer and the
constants moved out: a module that maps observations to a distribution and a value."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Independent


def _mode(dist: torch.distributions.Distribution) -> torch.Tensor:
    """The most likely action: argmax for Categorical, the mean for a Gaussian."""
    if isinstance(dist, Categorical):
        return dist.probs.argmax(-1)
    if isinstance(dist, Independent):
        return dist.base_dist.mean
    return dist.mean


class ActorCritic(nn.Module):
    def __init__(self, actor_encoder: nn.Module, critic_encoder: nn.Module, policy_head: nn.Module, value_head: nn.Module):
        super().__init__()
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder
        self.policy_head = policy_head
        self.value_head = value_head

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        return self.policy_head(self.actor_encoder(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(obs), shape [B]."""
        return self.value_head(self.critic_encoder(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[PPO-1] (action, log pi(action | obs), V(obs)). Deterministic takes the mode."""
        dist = self.distribution(obs)
        action = _mode(dist) if deterministic else dist.sample()
        return action, dist.log_prob(action), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[PPO-5] (log pi(actions | obs), entropy, V(obs)) under the current parameters, with gradient."""
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.value(obs)
