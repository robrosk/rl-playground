"""Env with a known model to ``TabularMDP`` (Adapter pattern)."""

from __future__ import annotations

from rl_playground.mdp.tabular_mdp import TabularMDP


def from_env(env, gamma: float) -> TabularMDP:
    """Unwraps wrappers until it finds an env with ``to_mdp(gamma)``."""
    current = env
    while current is not None:
        if hasattr(type(current), "to_mdp"):
            return current.to_mdp(gamma)
        current = getattr(current, "env", None)
    raise TypeError(f"{type(env).__name__} has no known model (no to_mdp method)")
