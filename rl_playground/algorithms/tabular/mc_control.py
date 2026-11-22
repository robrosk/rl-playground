"""On-policy Monte Carlo control with epsilon-greedy exploration."""

from __future__ import annotations

from rl_playground.algorithms.tabular.base import TabularLearner


class MCControl(TabularLearner):
    family = "on_policy"
    update_on_episode_end = True

    def update(self, batch):
        """[TAB-7] The batch is one whole episode. G = discounted_returns(r, terminated, gamma).
        Every-visit: self.q.update(s, a, G). For first-visit, keep only each (s, a)'s first row.

        If the episode was truncated, G is biased low; say so in a metric rather than hide it.
        """
        raise NotImplementedError
