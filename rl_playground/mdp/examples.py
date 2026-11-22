"""Small MDPs whose answers can be worked out by hand. The tests use these numbers."""

from __future__ import annotations

import numpy as np

from rl_playground.mdp.tabular_mdp import TabularMDP


def two_state_mdp(gamma: float = 0.9) -> TabularMDP:
    """State 0 has two actions; state 1 is terminal.

    action 0 ("stay"): reward 1, stay in state 0
    action 1 ("go"):   reward 5, move to terminal state 1

    With gamma = 0.9: V_stay(0) = 1 / (1 - 0.9) = 10, V_go(0) = 5, so the optimal policy stays
    and V*(0) = 10. The uniform policy has V(0) = 0.5 (1 + 0.9 V(0)) + 0.5 * 5, so V(0) = 3 / 0.55.
    """
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, :, 1] = 1.0
    R = np.array([[1.0, 5.0], [0.0, 0.0]])
    return TabularMDP(P=P, R=R, gamma=gamma, terminal=np.array([False, True]))
