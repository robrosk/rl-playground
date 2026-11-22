"""What is left of the old PPO ``Policy`` class once GAE, the losses, the optimizer and the
constants moved out: a module that maps observations to a distribution and a value."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, actor_encoder: nn.Module, critic_encoder: nn.Module, policy_head: nn.Module, value_head: nn.Module):
        super().__init__()
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder
        self.policy_head = policy_head
        self.value_head = value_head

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(obs), shape [B]."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[PPO-1] (action, log pi(action | obs), V(obs)). Deterministic takes the mode."""
        raise NotImplementedError

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[PPO-5] (log pi(actions | obs), entropy, V(obs)) under the current parameters, with gradient."""
        raise NotImplementedError
