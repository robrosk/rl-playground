"""Several learners in one game, each on its own schedule (Template Method, like the others).

One step is one env step (parallel games) or one move (turn-based games). After every step,
each seat that is due updates; seats that are not due wait. Metrics carry the seat's name, so
``player_1/loss`` and ``player_2/loss`` sit side by side in the log.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from rl_playground.core.callbacks import Callback
from rl_playground.core.types import Metrics
from rl_playground.runners.base import Runner


class MultiAgentTrainer(Runner):
    def __init__(self, driver_cls, env, seats: Sequence, callbacks: Iterable[Callback] = (),
                 seed: int = 0, returns_window: int = 100):
        super().__init__(callbacks)
        self.seats = list(seats)
        self.seat_of: Dict[str, object] = {a: seat for seat in self.seats for a in seat.agents}
        missing = [a for a in env.possible_agents if a not in self.seat_of]
        if missing:
            raise ValueError(f"agents without a seat: {missing}")
        self.driver = driver_cls(env, self._act, seed=seed)
        self.returns_window = returns_window

    def _act(self, agent, obs, mask):
        return self.seat_of[agent].act(agent, obs, mask)

    def step(self) -> None:
        for agent, transition in self.driver.step():
            self.seat_of[agent].record(agent, transition)

    def ready_to_update(self) -> bool:
        return any(seat.ready() for seat in self.seats)

    def update(self) -> Metrics:
        """[MA-7] Update every seat that is due, then add each agent's recent return."""
        metrics: Metrics = {}
        for seat in self.seats:
            if seat.ready():
                for key, value in seat.update().items():
                    metrics[f"{seat.name}/{key}"] = value
        for agent, returns in self.driver.episode_returns.items():
            if returns:
                metrics[f"return/{agent}"] = float(np.mean(returns[-self.returns_window:]))
        metrics["episodes"] = float(self.driver.episodes)
        return metrics
