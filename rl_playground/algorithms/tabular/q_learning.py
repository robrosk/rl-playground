"""Q-learning: off-policy TD control. Behave epsilon-greedily, learn the greedy policy."""

from __future__ import annotations

from rl_playground.algorithms.tabular.base import TabularLearner


class QLearning(TabularLearner):
    family = "off_policy"

    def update(self, batch):
        """[TAB-5] y = q_learning_target(r, Q(s'), terminated, gamma); self.q.update(s, a, y)."""
        raise NotImplementedError
