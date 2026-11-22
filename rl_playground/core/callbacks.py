"""Runner events and their subscribers (Observer pattern).

Runners never print. They notify a ``CallbackList``, and loggers, checkpoints and monitors
subscribe to it.
"""

from __future__ import annotations

import csv
import pathlib
import pickle
from typing import Iterable, List, Optional, Tuple

from rl_playground.core.types import Metrics


class Callback:
    def on_run_start(self, runner) -> None:
        pass

    def on_update(self, step: int, metrics: Metrics) -> None:
        pass

    def on_run_end(self, runner) -> None:
        pass


class CallbackList(Callback):
    """Fans every event out to its subscribers, in order."""

    def __init__(self, callbacks: Iterable[Callback] = ()):
        self.callbacks: List[Callback] = list(callbacks)

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_run_start(self, runner) -> None:
        for cb in self.callbacks:
            cb.on_run_start(runner)

    def on_update(self, step: int, metrics: Metrics) -> None:
        for cb in self.callbacks:
            cb.on_update(step, metrics)

    def on_run_end(self, runner) -> None:
        for cb in self.callbacks:
            cb.on_run_end(runner)


class ConsoleLogger(Callback):
    def __init__(self, every: int = 1):
        self.every = every
        self._seen = 0

    def on_update(self, step: int, metrics: Metrics) -> None:
        self._seen += 1
        if self._seen % self.every == 0:
            parts = "  ".join(f"{k} {v:.4g}" for k, v in metrics.items())
            print(f"step {step:>8}  {parts}")


class CSVLogger(Callback):
    """One row per update. Columns come from the first update's metric names."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._columns: Optional[List[str]] = None

    def on_update(self, step: int, metrics: Metrics) -> None:
        if self._columns is None:
            self._columns = ["step", *metrics]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self._columns)
        with self.path.open("a", newline="") as f:
            row = [step] + [metrics.get(c, "") for c in self._columns[1:]]
            csv.writer(f).writerow(row)


class Checkpoint(Callback):
    """Pickles ``state_dict()`` of the runner's learner or planner every ``every`` updates."""

    def __init__(self, directory, every: int = 100):
        self.directory = pathlib.Path(directory)
        self.every = every
        self._runner = None
        self._seen = 0

    def on_run_start(self, runner) -> None:
        self._runner = runner

    def on_update(self, step: int, metrics: Metrics) -> None:
        self._seen += 1
        if self._seen % self.every:
            return
        owner = getattr(self._runner, "learner", None) or getattr(self._runner, "planner", None)
        if owner is None or not hasattr(owner, "state_dict"):
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f"step_{step}.pkl").open("wb") as f:
            pickle.dump(owner.state_dict(), f)


class ContractionMonitor(Callback):
    """[DP-8] Checks the contraction bound on every sweep.

    Policy evaluation and value iteration each apply a gamma-contraction in the sup norm, so
    successive changes must satisfy ||V_{k+1} - V_k||_inf <= gamma ||V_k - V_{k-1}||_inf.
    A violation means the backup is wrong. Reads ``metrics["delta"]``.
    """

    def __init__(self, gamma: float, atol: float = 1e-12):
        self.gamma = gamma
        self.atol = atol
        self.deltas: List[float] = []
        self.violations: List[Tuple[int, float, float]] = []

    def on_update(self, step: int, metrics: Metrics) -> None:
        delta = metrics.get("delta")
        if delta is None:
            return
        if self.deltas and delta > self.gamma * self.deltas[-1] + self.atol:
            self.violations.append((step, self.deltas[-1], delta))
        self.deltas.append(delta)

    def on_run_end(self, runner) -> None:
        for step, before, after in self.violations:
            print(f"contraction violated at sweep {step}: {after:.3g} > gamma * {before:.3g}")
