"""Exact Bellman backups with a known model. Numpy only. See docs/planning.md."""

from __future__ import annotations

import numpy as np

from rl_playground.mdp.tabular_mdp import TabularMDP


def q_from_v(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """[DP-3] Q(s, a) = R(s, a) + gamma * sum_s' P(s' | s, a) V(s').

    Returns [S, A]. Rows of terminal states are 0 (they are absorbing with zero reward, so this
    holds as long as V[terminal] is 0; set them to 0 explicitly to be safe).
    Hint: one ``np.einsum``.
    """
    raise NotImplementedError


def expectation_backup(mdp: TabularMDP, V: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """[DP-1] (T^pi V)(s) = sum_a pi(a | s) Q(s, a).

    ``pi`` is [S, A] with rows summing to 1. Returns [S] with V[terminal] = 0.
    T^pi is a gamma-contraction in the sup norm, and V^pi is its unique fixed point.
    """
    raise NotImplementedError


def optimality_backup(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """[DP-6] (T* V)(s) = max_a Q(s, a). Returns [S] with V[terminal] = 0.

    T* is also a gamma-contraction, and V* is its unique fixed point.
    """
    raise NotImplementedError


def greedy_policy(mdp: TabularMDP, V: np.ndarray) -> np.ndarray:
    """[DP-4] One-hot [S, A] policy that picks argmax_a Q(s, a) in every state.

    Ties go to the lowest action index, so runs are deterministic and policy iteration can
    tell when the policy stopped changing.
    """
    raise NotImplementedError
