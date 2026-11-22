"""An open grid with a step cost and one goal: the classic textbook gridworld."""

from __future__ import annotations

from rl_playground.envs.frozen_lake import FrozenLake


class GridWorld(FrozenLake):
    def __init__(self, size: int = 4, step_reward: float = -0.04, goal_reward: float = 1.0, slip: float = 0.0):
        if size < 2:
            raise ValueError("size must be at least 2")
        desc = ["S" + "F" * (size - 1)] + ["F" * size] * (size - 2) + ["F" * (size - 1) + "G"]
        super().__init__(desc=desc, slip=slip, step_reward=step_reward, goal_reward=goal_reward)
