"""On-policy learning: collect a rollout with the current policy, update once, discard it."""

from __future__ import annotations

from typing import Iterable

from rl_playground.core.callbacks import Callback
from rl_playground.core.types import Metrics
from rl_playground.runners.base import Runner


class OnPolicyTrainer(Runner):
    """Updates when the rollout is full, or at an episode end for learners that set
    ``update_on_episode_end`` (Monte Carlo control, REINFORCE)."""

    def __init__(self, learner, collector, rollout, callbacks: Iterable[Callback] = ()):
        super().__init__(callbacks)
        self.learner = learner
        self.collector = collector
        self.rollout = rollout
        self._episode_ended = False

    def step(self) -> None:
        transition = self.collector.step()  # [PPO-1]
        self.rollout.add(transition)
        self._episode_ended = transition.terminated or transition.truncated

    def ready_to_update(self) -> bool:
        at_episode_end = getattr(self.learner, "update_on_episode_end", False) and self._episode_ended
        return self.rollout.full or at_episode_end

    def update(self) -> Metrics:
        return self.learner.update(self.rollout.get())

    def after_update(self, metrics: Metrics) -> None:
        self.rollout.clear()  # [PPO-10] on-policy data is stale after one update
