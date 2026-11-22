"""Random MDPs for property tests: contraction, convergence, PI and VI agreeing."""

from __future__ import annotations

import numpy as np

from rl_playground.mdp.tabular_mdp import TabularMDP


def random_mdp(n_states: int, n_actions: int, gamma: float = 0.9, seed: int = 0, n_terminal: int = 1) -> TabularMDP:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    R = rng.normal(size=(n_states, n_actions))
    terminal = np.zeros(n_states, dtype=bool)
    for s in range(n_states - n_terminal, n_states):
        terminal[s] = True
        P[s] = 0.0
        P[s, :, s] = 1.0
        R[s] = 0.0
    return TabularMDP(P=P, R=R, gamma=gamma, terminal=terminal)
