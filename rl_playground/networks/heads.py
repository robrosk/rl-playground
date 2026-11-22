"""Output heads. Policy heads return ``torch.distributions.Distribution``, so callers never
branch on discrete versus continuous actions."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


def _linear(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class CategoricalHead(nn.Module):
    """Logits to ``Categorical``. A small init keeps the first policy close to uniform."""

    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.logits = _linear(in_dim, n_actions, gain=0.01)

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        return Categorical(logits=self.logits(features))


class GaussianHead(nn.Module):
    """Mean from a linear layer and a state-independent learned log std, to a diagonal Gaussian.
    ``Independent`` sums log-probs and entropies over action dimensions."""

    def __init__(self, in_dim: int, action_dim: int, init_log_std: float = 0.0):
        super().__init__()
        self.mean = _linear(in_dim, action_dim, gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        mean = self.mean(features)
        return Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)


class ValueHead(nn.Module):
    """Linear to one output, squeezed to [B]."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out = _linear(in_dim, 1, gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(features).squeeze(-1)


class QHead(nn.Module):
    """Linear to one output per action, [B, A]."""

    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.out = nn.Linear(in_dim, n_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(features)
