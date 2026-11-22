"""Greedy evaluation episodes, separate from training."""

from __future__ import annotations

from typing import Callable

import numpy as np


def evaluate(env, act: Callable, episodes: int = 10, seed: int = 0, max_steps: int = 1_000) -> dict:
    """Runs ``act(obs, explore=False)`` for ``episodes`` episodes and reports mean and std return."""
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(max_steps):
            result = env.step(act(obs, explore=False).action)
            total += result.reward
            obs = result.obs
            if result.done:
                break
        returns.append(total)
    return {"eval/return_mean": float(np.mean(returns)), "eval/return_std": float(np.std(returns))}
