"""Iterative policy evaluation: V <- T^pi V until it stops changing. See docs/planning.md."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rl_playground.algorithms.base import Planner


class PolicyEvaluation(Planner):
    contracts = True

    def __init__(self, mdp, config, pi: Optional[np.ndarray] = None):
        super().__init__(mdp, config)
        if pi is not None:
            self.pi = np.asarray(pi, dtype=np.float64)

    def sweep(self) -> float:
        """[DP-2] V <- expectation_backup(mdp, V, pi) (synchronous: compute all of the new V
        from the old one). Return ||V_new - V_old||_inf. Converges to V^pi."""
        raise NotImplementedError
