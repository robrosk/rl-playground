"""Env wrappers that stack without touching the env (Decorator pattern)."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from rl_playground.core.types import StepResult
from rl_playground.envs.base import Box, Discrete


class EnvWrapper:
    """Forwards everything to the wrapped env. Subclasses override what they change."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        env = self.env
        while isinstance(env, EnvWrapper):
            env = env.env
        return env

    def reset(self, seed: Optional[int] = None):
        return self.env.reset(seed=seed)

    def step(self, action) -> StepResult:
        return self.env.step(action)


class OneHot(EnvWrapper):
    """Discrete observations to one-hot float vectors, for neural learners on tabular envs."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, Discrete):
            raise TypeError("OneHot needs a Discrete observation space")
        self.n = env.observation_space.n
        self.observation_space = Box((self.n,), 0.0, 1.0)

    def _encode(self, obs) -> np.ndarray:
        vec = np.zeros(self.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return self._encode(obs), info

    def step(self, action) -> StepResult:
        result = self.env.step(action)
        return replace(result, obs=self._encode(result.obs))


class TimeLimit(EnvWrapper):
    """Sets ``truncated`` after ``max_steps``. Truncated is not terminated: learners bootstrap."""

    def __init__(self, env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self._t = 0

    def reset(self, seed: Optional[int] = None):
        self._t = 0
        return self.env.reset(seed=seed)

    def step(self, action) -> StepResult:
        result = self.env.step(action)
        self._t += 1
        if self._t >= self.max_steps and not result.terminated:
            result = replace(result, truncated=True)
        return result
