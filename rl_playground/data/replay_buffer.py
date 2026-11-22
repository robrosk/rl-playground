"""Off-policy storage."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rl_playground.core.types import Batch, Transition


class ReplayBuffer:
    """A ring buffer of the last ``capacity`` transitions, sampled uniformly.

    Terminated and truncated stay separate inside each Transition. A list of Transitions keeps
    this readable; preallocated numpy columns would be faster for image observations.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._data: List[Transition] = []
        self._next = 0  # index the next add writes to once the buffer is full

    def add(self, transition: Transition) -> None:
        """[DQN-2] Overwrite the oldest entry once full."""
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """``batch_size`` rows drawn uniformly with replacement."""
        if not self._data:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(len(self._data), size=batch_size)
        return Batch.from_transitions([self._data[i] for i in idx])


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


class NewTransitions:
    """What arrived since the last sample, all of it, then empty. Online learners in a shared
    multi-agent seat use it, since one env step can bring a transition from every agent and
    ``LatestTransition`` would keep only the last."""

    def __init__(self):
        self._data: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self._data.append(transition)

    def __len__(self) -> int:
        return len(self._data)

    def sample(self, batch_size: int, rng) -> Batch:
        batch, self._data = Batch.from_transitions(self._data), []
        return batch
