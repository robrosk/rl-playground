"""Regression losses for value targets. Inputs are torch tensors of shape [B]."""

from __future__ import annotations


class HuberTDLoss:
    """[DQN-5] mean Huber_delta(q_sa - targets), targets detached. Reads "q_sa", "targets"."""

    name = "td"

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, inputs):
        raise NotImplementedError


class MSETDLoss:
    """mean (q_sa - targets)^2, targets detached."""

    name = "td"

    def __call__(self, inputs):
        raise NotImplementedError
