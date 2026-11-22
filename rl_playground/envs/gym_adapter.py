"""Gymnasium env to ``Env`` (Adapter pattern)."""

from __future__ import annotations

from typing import Optional

from rl_playground.core.types import StepResult
from rl_playground.envs.base import Box, Discrete


def convert_space(space):
    import gymnasium as gym

    if isinstance(space, gym.spaces.Discrete):
        return Discrete(int(space.n))
    if isinstance(space, gym.spaces.Box):
        return Box(tuple(space.shape), float(space.low.min()), float(space.high.max()))
    raise TypeError(f"unsupported gymnasium space {space!r}")


class GymAdapter:
    def __init__(self, env_or_id, **make_kwargs):
        import gymnasium as gym

        self.env = gym.make(env_or_id, **make_kwargs) if isinstance(env_or_id, str) else env_or_id
        self.observation_space = convert_space(self.env.observation_space)
        self.action_space = convert_space(self.env.action_space)

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, action) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

    def close(self) -> None:
        self.env.close()
