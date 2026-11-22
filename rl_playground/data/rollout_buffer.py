"""On-policy storage: one rollout, used for one update, then cleared (Iterator over minibatches)."""

from __future__ import annotations

from typing import Iterator, List

import numpy as np

from rl_playground.core.types import Batch, Transition


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._transitions: List[Transition] = []

    def add(self, transition: Transition) -> None:
        if self.full:
            raise RuntimeError("rollout is full; update and clear it first")
        self._transitions.append(transition)

    @property
    def full(self) -> bool:
        return len(self._transitions) >= self.capacity

    def __len__(self) -> int:
        return len(self._transitions)

    def get(self) -> Batch:
        return Batch.from_transitions(self._transitions)

    def clear(self) -> None:
        self._transitions = []


def iterate_minibatches(batch: Batch, size: int, rng: np.random.Generator) -> Iterator[Batch]:
    """Shuffle once, then yield ``batch.take(chunk)`` for consecutive chunks of ``size`` indices.

    Every row appears exactly once per pass; the last chunk may be smaller. The old PPO ran
    full-batch epochs, which takes one large step per epoch instead of many small ones.
    """
    if size < 1:
        raise ValueError("minibatch size must be at least 1")
    order = rng.permutation(len(batch))
    for start in range(0, len(order), size):
        yield batch.take(order[start:start + size])
