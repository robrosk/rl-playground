"""The value interfaces every algorithm codes against."""

from __future__ import annotations

from typing import Protocol

from rl_playground.core.types import Metrics


class VFunction(Protocol):
    def __call__(self, obs):
        """V(obs) for a batch of observations, shape [B]."""

    def update(self, obs, targets) -> Metrics:
        """Move V(obs) toward ``targets``."""


class QFunction(Protocol):
    def __call__(self, obs):
        """Q(obs, .) for a batch of observations, shape [B, A]."""

    def update(self, obs, actions, targets) -> Metrics:
        """Move Q(obs, actions) toward ``targets``."""
