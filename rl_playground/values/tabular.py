"""Value tables. The update is the tabular form of a gradient step on (target - Q)^2 / 2."""

from __future__ import annotations

import numpy as np

from rl_playground.core.types import Metrics


class TabularV:
    def __init__(self, n_states: int, alpha: float, init: float = 0.0):
        self.table = np.full(n_states, init, dtype=np.float64)
        self.alpha = alpha

    def __call__(self, obs):
        return self.table[np.asarray(obs, dtype=np.int64)]

    def update(self, obs, targets) -> Metrics:
        """[TAB-8] V(s) <- V(s) + alpha * (target - V(s)) for each row.

        Apply rows in order, so a state that appears twice is updated twice. Return
        {"td_error": mean |target - V(s)| before the update}.
        """
        raise NotImplementedError


class TabularQ:
    def __init__(self, n_states: int, n_actions: int, alpha: float, init: float = 0.0):
        self.table = np.full((n_states, n_actions), init, dtype=np.float64)
        self.alpha = alpha

    def __call__(self, obs):
        return self.table[np.asarray(obs, dtype=np.int64)]

    def update(self, obs, actions, targets) -> Metrics:
        """[TAB-8] Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a)) for each row, in order.

        Return {"td_error": mean |target - Q(s, a)| before the update}.
        """
        raise NotImplementedError
