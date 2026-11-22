"""Double Q-learning (van Hasselt 2010): two tables, each corrects the other's max bias."""

from __future__ import annotations

import numpy as np

from rl_playground.algorithms.tabular.base import TabularLearner
from rl_playground.core.types import ActionOut
from rl_playground.policies.greedy import greedy_action
from rl_playground.values.tabular import TabularQ


class DoubleQLearning(TabularLearner):
    family = "off_policy"

    def __init__(self, obs_space, action_space, config, seed: int = 0):
        super().__init__(obs_space, action_space, config, seed)
        self.q_b = TabularQ(self.n_states, self.n_actions, alpha=config.alpha)
        self.rng = np.random.default_rng(seed + 1)

    def act(self, obs, explore: bool = True, mask=None) -> ActionOut:
        """Act on Q_a + Q_b."""
        q_sum = self.q(obs) + self.q_b(obs)
        if not explore:
            return ActionOut(greedy_action(q_sum, mask))
        action = self.behaviour.act(q_sum, self.steps, mask)
        self.steps += 1
        return ActionOut(action)

    def update(self, batch):
        """[TAB-6] With probability 1/2 update Q_a toward double_q_target(r, Q_a(s'), Q_b(s'), ...);
        otherwise update Q_b with the roles swapped. Pass ``next_mask=batch.next_masks``."""
        raise NotImplementedError
