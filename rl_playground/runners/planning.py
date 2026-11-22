"""Planning with a known model: one step is one sweep over every state."""

from __future__ import annotations

import math
from typing import Iterable

from rl_playground.core.callbacks import Callback
from rl_playground.core.types import Metrics
from rl_playground.runners.base import Runner


class PlanningRunner(Runner):
    def __init__(self, planner, tol: float = 1e-8, callbacks: Iterable[Callback] = ()):
        super().__init__(callbacks)
        self.planner = planner
        self.tol = tol
        self._delta = math.inf

    def step(self) -> None:
        self._delta = float(self.planner.sweep())

    def ready_to_update(self) -> bool:
        return True  # every sweep is reported

    def update(self) -> Metrics:
        # [DP-8] ContractionMonitor reads this delta to check the contraction bound.
        return {"delta": self._delta}

    def done(self, budget: int) -> bool:
        return self.t >= budget or self._delta < self.tol
