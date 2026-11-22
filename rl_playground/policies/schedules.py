"""Hyperparameters that change with the step count (Strategy pattern)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Schedule(Protocol):
    def value(self, step: int) -> float:
        ...


@dataclass(frozen=True)
class Constant:
    v: float

    def value(self, step: int) -> float:
        return self.v


@dataclass(frozen=True)
class LinearDecay:
    start: float
    end: float
    steps: int

    def value(self, step: int) -> float:
        """start at step 0, end at ``steps``, end afterwards, linear in between."""
        if step >= self.steps:
            return self.end
        return self.start + (self.end - self.start) * step / self.steps


@dataclass(frozen=True)
class ExponentialDecay:
    start: float
    end: float
    decay: float

    def value(self, step: int) -> float:
        """max(end, start * decay ** step)."""
        return max(self.end, self.start * self.decay ** step)
