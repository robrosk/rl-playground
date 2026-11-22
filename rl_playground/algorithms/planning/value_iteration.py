"""Value iteration: V <- T* V, then act greedily on the result."""

from __future__ import annotations

import numpy as np

from rl_playground.algorithms.base import Planner
from rl_playground.bellman.exact import greedy_policy


class ValueIteration(Planner):
    contracts = True

    def sweep(self) -> float:
        """[DP-6] V <- optimality_backup(mdp, V). Return ||V_new - V_old||_inf.

        If the change is below eps, ||V - V*||_inf <= eps * gamma / (1 - gamma).
        """
        raise NotImplementedError

    def policy(self) -> np.ndarray:
        return greedy_policy(self.mdp, self.V)
