"""Observation encoders. No dropout anywhere: rollout and update must see the same function,
or the PPO ratio is off from the first step."""

from __future__ import annotations

from typing import Sequence, Type

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Linear + activation per hidden size. Sets ``self.out_dim`` to the last hidden size."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (64, 64), activation: Type[nn.Module] = nn.Tanh):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CNNEncoder(nn.Module):
    """The Nature DQN conv stack for [C, H, W] images scaled to [0, 1]. Sets ``self.out_dim``."""

    def __init__(self, in_shape: Sequence[int]):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
