"""Off-policy storage."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rl_playground.core.types import Batch, Transition


class ReplayBuffer:
    """A ring buffer of the last ``capacity`` transitions, sampled uniformly.

    Store terminated and truncated separately. Preallocating numpy columns on the first add is
    faster than a list of Transitions, but either passes the tests.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

    def add(self, transition: Transition) -> None:
        """[DQN-2] Overwrite the oldest entry once full."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """``batch_size`` rows drawn uniformly with replacement."""
        raise NotImplementedError


class LatestTransition:
    """A buffer of one: online tabular learners update on each transition as it happens."""

    def __init__(self):
        self._last: Optional[Transition] = None

    def add(self, transition: Transition) -> None:
        self._last = transition

    def __len__(self) -> int:
        return 0 if self._last is None else 1

    def sample(self, batch_size: int, rng) -> Batch:
        return Batch.from_transitions([self._last])
