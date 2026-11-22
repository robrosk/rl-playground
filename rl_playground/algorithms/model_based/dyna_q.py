"""Dyna-Q (Sutton 1990): learn a model from experience and plan with it between steps.
See docs/model_based.md."""

from __future__ import annotations

import numpy as np

from rl_playground.algorithms.tabular.q_learning import QLearning


class DynaQ(QLearning):
    def __init__(self, obs_space, action_space, config, seed: int = 0):
        super().__init__(obs_space, action_space, config, seed)
        self.model = {}  # (s, a) -> (r, s', terminated): deterministic model
        self.rng = np.random.default_rng(seed + 2)

    def update(self, batch):
        """[DYNA-1] Direct RL: the Q-learning update on the real transition.
        [DYNA-2] Model learning: self.model[(s, a)] = (r, s', terminated).
        [DYNA-3] Planning: config.planning_steps times, draw a seen (s, a) uniformly from the
        model and apply the same Q-learning update to the simulated transition."""
        raise NotImplementedError
