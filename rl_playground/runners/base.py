"""The one loop every method runs in (Template Method pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from rl_playground.core.callbacks import Callback, CallbackList
from rl_playground.core.types import Metrics


class Runner(ABC):
    """``run`` is the template and is never overridden. Subclasses say what one step is,
    when an update is due and what an update does."""

    def __init__(self, callbacks: Iterable[Callback] = ()):
        self.callbacks = CallbackList(callbacks)
        self.t = 0  # steps taken: sweeps for planners, env steps for learners
        self.updates = 0

    def run(self, budget: int) -> "Runner":
        self.callbacks.on_run_start(self)
        while not self.done(budget):
            self.step()
            self.t += 1
            if self.ready_to_update():
                metrics = self.update()
                self.updates += 1
                self.after_update(metrics)
                self.callbacks.on_update(self.t, metrics)
        self.callbacks.on_run_end(self)
        return self

    def done(self, budget: int) -> bool:
        return self.t >= budget

    @abstractmethod
    def step(self) -> None:
        """Advance one unit of time."""

    @abstractmethod
    def ready_to_update(self) -> bool:
        """Whether this step ends with an update."""

    @abstractmethod
    def update(self) -> Metrics:
        """Do the update and return its metrics."""

    def after_update(self, metrics: Metrics) -> None:
        """Hook after each update, for example clearing an on-policy rollout."""
