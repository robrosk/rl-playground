"""The one interface every algorithm sees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from rl_playground.core.types import StepResult


@dataclass(frozen=True)
class Discrete:
    n: int


@dataclass(frozen=True)
class Box:
    shape: Tuple[int, ...]
    low: float = float("-inf")
    high: float = float("inf")


class Env(Protocol):
    observation_space: Any  # Discrete or Box
    action_space: Any

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        ...

    def step(self, action) -> StepResult:
        ...
