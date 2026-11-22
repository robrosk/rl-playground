"""Output heads. Policy heads return ``torch.distributions.Distribution``, so callers never
branch on discrete versus continuous actions."""

from __future__ import annotations

import torch
from torch import nn


class CategoricalHead(nn.Module):
    """Logits to ``Categorical``."""

    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError


class GaussianHead(nn.Module):
    """Mean from a linear layer, a state-independent learned log std, to ``Normal``.
    Sum log-probs over action dimensions (``Independent``)."""

    def __init__(self, in_dim: int, action_dim: int, init_log_std: float = 0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        raise NotImplementedError


class ValueHead(nn.Module):
    """Linear to one output, squeezed to [B]."""

    def __init__(self, in_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QHead(nn.Module):
    """Linear to one output per action, [B, A]."""

    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
