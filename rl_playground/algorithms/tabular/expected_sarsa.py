"""Expected SARSA: average over the behaviour policy at s' instead of sampling a'."""

from __future__ import annotations

from rl_playground.algorithms.tabular.base import TabularLearner


class ExpectedSARSA(TabularLearner):
    family = "on_policy"

    def update(self, batch):
        """[TAB-4] next_pi = self.behaviour.probs(Q(s'), self.steps, next_mask) per row, then
        expected_sarsa_target, then self.q.update."""
        raise NotImplementedError
