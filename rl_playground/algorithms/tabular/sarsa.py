"""SARSA: on-policy TD control on (s, a, r, s', a')."""

from __future__ import annotations

from rl_playground.algorithms.tabular.base import TabularLearner


class SARSA(TabularLearner):
    family = "on_policy"

    def update(self, batch):
        """[TAB-3] Q(s, a) moves toward r + gamma (1 - terminated) Q(s', a').

        a' is ``batch.next_actions`` when given (the tests pass it). Otherwise pick a' with the
        behaviour policy now and cache it, and make the next ``act`` on s' return the cached
        a', so the action you learned from is the action you take.
        """
        raise NotImplementedError
