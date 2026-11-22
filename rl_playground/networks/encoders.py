"""Observation encoders. No dropout anywhere: rollout and update must see the same function,
or the PPO ratio is off from the first step."""

from __future__ import annotations

from typing import Sequence, Type

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Linear + activation per hidden size. ``self.out_dim`` is the last hidden size."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (64, 64), activation: Type[nn.Module] = nn.Tanh):
        super().__init__()
        if not hidden:
            raise ValueError("MLPEncoder needs at least one hidden layer")
        layers, width = [], in_dim
        for size in hidden:
            layers += [nn.Linear(width, size), activation()]
            width = size
        self.net = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class CNNEncoder(nn.Module):
    """The Nature DQN conv stack (Mnih et al. 2015) for [C, H, W] images.

    Integer inputs are raw pixels and are scaled by 1/255; float inputs are taken as already in
    [0, 1]. ``self.out_dim`` is 512.
    """

    def __init__(self, in_shape: Sequence[int]):
        super().__init__()
        channels, height, width = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, channels, height, width)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU())
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0 if not torch.is_floating_point(x) else x
        return self.fc(self.conv(x))
