"""Modified policy iteration: k evaluation sweeps per improvement.

k = 1 is value iteration; k -> infinity is policy iteration.
"""

from __future__ import annotations

from rl_playground.algorithms.base import Planner


class ModifiedPolicyIteration(Planner):
    def sweep(self) -> float:
        """[DP-7] Apply [DP-1] ``config.eval_sweeps`` times, then pi <- greedy_policy [DP-4].
        Return ||V_after - V_before||_inf."""
        raise NotImplementedError
