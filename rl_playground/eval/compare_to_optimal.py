"""Scores a learner against the exact answer from value iteration on the same MDP."""

from __future__ import annotations

import numpy as np


def value_error(V, V_star) -> float:
    """||V - V*||_inf."""
    return float(np.max(np.abs(np.asarray(V) - np.asarray(V_star))))


def policy_agreement(Q_or_pi, pi_star, terminal) -> float:
    """Fraction of non-terminal states where the greedy action matches pi*'s."""
    chosen = np.argmax(np.asarray(Q_or_pi), axis=1)
    best = np.argmax(np.asarray(pi_star), axis=1)
    live = ~np.asarray(terminal)
    return float(np.mean(chosen[live] == best[live]))
