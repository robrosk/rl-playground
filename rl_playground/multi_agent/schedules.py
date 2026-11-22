"""When a seat updates and what it updates on (Strategy pattern).

These are the single-agent trainers' rules, moved per seat: ``OnPolicyTrainer`` became
``OnPolicySchedule`` and ``OffPolicyTrainer`` became ``OffPolicySchedule``. Each seat counts
its own transitions, so a PPO seat and a DQN seat in the same game keep their own clocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set

import numpy as np

from rl_playground.core.types import Metrics, Transition
from rl_playground.data.multi_stream import MultiStreamRollout
from rl_playground.data.replay_buffer import NewTransitions, ReplayBuffer


class UpdateSchedule(ABC):
    @abstractmethod
    def add(self, agent: str, transition: Transition) -> None:
        ...

    @abstractmethod
    def ready(self) -> bool:
        ...

    @abstractmethod
    def update(self, learner) -> Metrics:
        ...


class OnPolicySchedule(UpdateSchedule):
    """Update when the rollout is full, or once every agent's episode has ended for learners
    that set ``update_on_episode_end``. Clear after each update."""

    def __init__(self, rollout: MultiStreamRollout, update_on_episode_end: bool = False):
        self.rollout = rollout
        self.update_on_episode_end = update_on_episode_end
        self._mid_episode: Set[str] = set()

    def add(self, agent: str, transition: Transition) -> None:
        self.rollout.add(agent, transition)
        if transition.terminated or transition.truncated:
            self._mid_episode.discard(agent)
        else:
            self._mid_episode.add(agent)

    def ready(self) -> bool:
        at_episode_end = self.update_on_episode_end and len(self.rollout) > 0 and not self._mid_episode
        return self.rollout.full or at_episode_end

    def update(self, learner) -> Metrics:
        metrics = learner.update(self.rollout.get())
        self.rollout.clear()  # on-policy data is stale after one update
        return metrics


class OffPolicySchedule(UpdateSchedule):
    """After ``warmup`` transitions, one update per ``train_every`` transitions, each on a
    sample of ``batch_size``. A parallel step can bring several transitions to a shared seat,
    so several updates can fall due at once; their metrics are averaged."""

    def __init__(self, buffer, batch_size: int = 1, warmup: int = 0, train_every: int = 1, seed: int = 0):
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._due = 0

    def add(self, agent: str, transition: Transition) -> None:
        self.buffer.add(transition)
        self.steps += 1
        if self.steps >= self.warmup and self.steps % self.train_every == 0:
            self._due += 1

    def ready(self) -> bool:
        return self._due > 0 and len(self.buffer) >= self.batch_size

    def update(self, learner) -> Metrics:
        results: List[Metrics] = []
        for _ in range(self._due):
            if len(self.buffer) < self.batch_size:  # NewTransitions empties on each sample
                break
            results.append(learner.update(self.buffer.sample(self.batch_size, self.rng)))
        self._due = 0
        return _mean(results)


class NoUpdates(UpdateSchedule):
    """For fixed policies: they play, and nothing is stored."""

    def add(self, agent: str, transition: Transition) -> None:
        pass

    def ready(self) -> bool:
        return False

    def update(self, learner) -> Metrics:
        return {}


def schedule_for(learner, config, seed: int = 0) -> UpdateSchedule:
    """The schedule a learner's ``family`` calls for (Factory). With no config (a learner you
    built yourself), on-policy seats update every transition and off-policy seats learn
    online from each new transition."""
    family = getattr(learner, "family", None)
    if family == "fixed":
        return NoUpdates()
    if family == "on_policy":
        return OnPolicySchedule(MultiStreamRollout(getattr(config, "rollout_len", 1)), learner.update_on_episode_end)
    if family == "off_policy":
        buffer = ReplayBuffer(config.buffer_size) if hasattr(config, "buffer_size") else NewTransitions()
        return OffPolicySchedule(
            buffer,
            batch_size=getattr(config, "batch_size", 1),
            warmup=getattr(config, "warmup", 0),
            train_every=getattr(config, "train_every", 1),
            seed=seed,
        )
    raise ValueError(f"family must be 'on_policy', 'off_policy' or 'fixed', got {family!r}")


def _mean(results: List[Metrics]) -> Metrics:
    if not results:
        return {}
    keys = {k for r in results for k in r}
    return {k: float(np.mean([r[k] for r in results if k in r])) for k in keys}
