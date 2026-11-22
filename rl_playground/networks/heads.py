"""Output heads. Policy heads return ``torch.distributions.Distribution``, so callers never
branch on discrete versus continuous actions."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[MASK-3] Logits with every illegal action pushed out of the softmax.

    ``logits`` is [B, A] float and ``mask`` is [B, A] bool, True where the action is legal.
    Return logits whose ``Categorical`` puts probability 0 on illegal actions and renormalizes
    over the legal ones, so log-probs, sampling and entropy all ignore illegal moves.

    Pick the fill value with care: -inf is exact but turns 0 * log 0 into nan in some entropy
    formulas, so many libraries use ``torch.finfo(logits.dtype).min``. The tests check that
    the entropy stays finite and equals the entropy over the legal actions alone.
    """
    raise NotImplementedError


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

    def forward(self, features: torch.Tensor, mask=None) -> torch.distributions.Distribution:
        logits = self.logits(features)
        if mask is not None:
            logits = masked_logits(logits, mask)
        return Categorical(logits=logits)


class GaussianHead(nn.Module):
    """Mean from a linear layer and a state-independent learned log std, to a diagonal Gaussian.
    ``Independent`` sums log-probs and entropies over action dimensions."""

    def __init__(self, in_dim: int, action_dim: int, init_log_std: float = 0.0):
        super().__init__()
        self.mean = _linear(in_dim, action_dim, gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def forward(self, features: torch.Tensor, mask=None) -> torch.distributions.Distribution:
        if mask is not None:
            raise ValueError("action masks are for discrete actions; this head is continuous")
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
