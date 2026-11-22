"""Value networks behind the same interface as the tables. Needs torch."""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from rl_playground.core.types import Metrics


class NeuralQ:
    """Online network, target network and optimizer behind ``QFunction``.

    DQN calls ``__call__``, ``target`` and ``update``; it never touches the network directly.
    """

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer, loss, max_grad_norm: Optional[float] = None):
        self.network = network
        self.target_network = copy.deepcopy(network).requires_grad_(False)
        self.optimizer = optimizer
        self.loss = loss
        self.max_grad_norm = max_grad_norm

    def __call__(self, obs) -> torch.Tensor:
        """Q_online(obs, .) as a [B, A] tensor, no gradient. Used to act and to pick a*."""
        raise NotImplementedError

    def target(self, obs) -> torch.Tensor:
        """Q_target(obs, .) as a [B, A] tensor, no gradient."""
        raise NotImplementedError

    def update(self, obs, actions, targets) -> Metrics:
        """[DQN-5] [DQN-6] One gradient step on loss(Q_online(s, a), y) with y detached.

        Gather Q(s, a) with ``core.arrays.take_along_last``, call ``self.loss``, backprop,
        clip the global grad norm to ``max_grad_norm``, step. Return the loss metrics.
        """
        raise NotImplementedError

    def sync_target(self, tau: Optional[float] = None) -> None:
        """[DQN-7] tau None: copy the online weights. Otherwise theta_t <- tau theta + (1 - tau) theta_t."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"network": self.network.state_dict(), "target": self.target_network.state_dict()}


class NeuralV:
    """V(s) from a network, for critics outside an actor-critic module."""

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer):
        self.network = network
        self.optimizer = optimizer

    def __call__(self, obs) -> torch.Tensor:
        raise NotImplementedError

    def update(self, obs, targets) -> Metrics:
        """One gradient step on mean (V(s) - target)^2."""
        raise NotImplementedError
