"""Policy iteration: evaluate pi exactly, then make it greedy, until it stops changing."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rl_playground.algorithms.base import Planner


class PolicyIteration(Planner):
    def __init__(self, mdp, config, pi: Optional[np.ndarray] = None):
        super().__init__(mdp, config)
        if pi is not None:
            self.pi = np.asarray(pi, dtype=np.float64)
        self.improvements = 0
        self.stable = False

    def sweep(self) -> float:
        """[DP-5] One evaluate-and-improve round.

        1. Evaluate: repeat [DP-1] until ||change||_inf < config.eval_tol.
        2. Improve: pi_new = greedy_policy(mdp, V) [DP-4]. Count it in ``self.improvements``
           if it differs from pi.
        3. If pi did not change, set ``self.stable`` and return 0.0, which ends the run.
        Otherwise return ||V_after_eval - V_before||_inf.

        The policy improvement theorem says each round's V^pi is at least the last, so this
        ends in finitely many rounds on a finite MDP.
        """
        raise NotImplementedError
