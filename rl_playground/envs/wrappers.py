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


class FlattenObservation(EnvWrapper):
    """Box observations of any shape to 1-D float vectors, for MLP encoders on board games."""

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, Box):
            raise TypeError("FlattenObservation needs a Box observation space")
        space = env.observation_space
        self.observation_space = Box((int(np.prod(space.shape)),), space.low, space.high)

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32).reshape(-1), info

    def step(self, action) -> StepResult:
        result = self.env.step(action)
        return replace(result, obs=np.asarray(result.obs, dtype=np.float32).reshape(-1))


class ChannelFirst(EnvWrapper):
    """[H, W, C] images to [C, H, W], the layout ``CNNEncoder`` and PyTorch expect."""

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, Box) or len(space.shape) != 3:
            raise TypeError("ChannelFirst needs a [H, W, C] Box observation space")
        h, w, c = space.shape
        self.observation_space = Box((c, h, w), space.low, space.high)

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return np.moveaxis(np.asarray(obs), -1, 0), info

    def step(self, action) -> StepResult:
        result = self.env.step(action)
        return replace(result, obs=np.moveaxis(np.asarray(result.obs), -1, 0))
