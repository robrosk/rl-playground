"""Multi-step returns over a rollout. Arrays are 1-D over time for a single env.

``terminated[t]``: s_(t+1) is terminal, so nothing after t counts.
``truncated[t]``: the episode was cut after t, so bootstrap from ``next_values[t]`` and do
not chain into t + 1, which belongs to a new episode.
See docs/derivations/gae.md.
"""

from __future__ import annotations


def discounted_returns(rewards, terminated, gamma: float, bootstrap: float = 0.0):
    """[TAB-7] [RF-1] G_t = r_t + gamma * (1 - terminated_t) * G_(t+1), with G_T = bootstrap.

    Walk backwards from the last step. Returns an array shaped like ``rewards``.
    """
    raise NotImplementedError


def n_step_returns(rewards, next_values, terminated, truncated, gamma: float, n: int):
    """[A2C-1] G_t^(n) = sum_(k<n) gamma^k r_(t+k) + gamma^n V(s_(t+n)).

    Stops early at an episode end: no bootstrap after ``terminated``, bootstrap from
    ``next_values`` at ``truncated`` or at the end of the rollout.
    """
    raise NotImplementedError


def gae(rewards, values, next_values, terminated, truncated, gamma: float, lam: float):
    """[PPO-3] Generalized advantage estimation. Returns (advantages, returns).

    delta_t = r_t + gamma * (1 - terminated_t) * next_values[t] - values[t]
    A_t     = delta_t + gamma * lam * (1 - terminated_t) * (1 - truncated_t) * A_(t+1)
    R_t     = A_t + values[t]

    ``next_values[t]`` is V of the true next observation, including the final observation
    when the episode was truncated at t. lam = 1 gives Monte Carlo advantages; lam = 0 gives
    one-step TD errors.
    """
    raise NotImplementedError
