from __future__ import annotations

import numpy as np

from rl_playground.policies.schedules import Schedule


class EpsilonGreedy:
    """[TAB-1] [DQN-1] Uniform random action with probability epsilon(step), else greedy.

    The schedule is keyed on env steps, not gradient steps. The old DQN decayed epsilon per
    gradient step, which made exploration depend on ``train_every``.
    """

    def __init__(self, schedule: Schedule, n_actions: int, seed: int = 0):
        self.schedule = schedule
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def probs(self, q_row, step: int) -> np.ndarray:
        """pi(. | s): epsilon / A on every action, plus 1 - epsilon on the greedy one. Shape [A].
        Expected SARSA needs this."""
        raise NotImplementedError

    def act(self, q_row, step: int) -> int:
        raise NotImplementedError
