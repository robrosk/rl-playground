"""Off-policy learning: every step goes into a buffer, and updates sample from it."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from rl_playground.core.callbacks import Callback
from rl_playground.core.types import Metrics
from rl_playground.runners.base import Runner


class OffPolicyTrainer(Runner):
    """With ``LatestTransition`` as the buffer this is plain online Q-learning (no replay)."""

    def __init__(
        self,
        learner,
        collector,
        buffer,
        batch_size: int = 1,
        warmup: int = 0,
        train_every: int = 1,
        callbacks: Iterable[Callback] = (),
        seed: int = 0,
    ):
        super().__init__(callbacks)
        self.learner = learner
        self.collector = collector
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.rng = np.random.default_rng(seed)

    def step(self) -> None:
        self.buffer.add(self.collector.step())  # [DQN-2]

    def ready_to_update(self) -> bool:
        # [DQN-3] after warmup, every train_every env steps
        return (
            self.t >= self.warmup
            and self.t % self.train_every == 0
            and len(self.buffer) >= self.batch_size
        )

    def update(self) -> Metrics:
        return self.learner.update(self.buffer.sample(self.batch_size, self.rng))
