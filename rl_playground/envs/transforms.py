"""Observation transforms as values: each maps a space to (new space, obs -> new obs).

The single-agent wrappers in ``envs/wrappers.py`` and the per-agent multi-agent wrappers in
``envs/multi_agent/wrappers.py`` both use these, so a transform is written once.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from rl_playground.envs.base import Box, Discrete

Transform = Callable[[object], Tuple[object, Callable]]


def flatten(space) -> Tuple[Box, Callable]:
    if not isinstance(space, Box):
        raise TypeError("flatten needs a Box observation space")
    return Box((int(np.prod(space.shape)),), space.low, space.high), (
        lambda obs: np.asarray(obs, dtype=np.float32).reshape(-1)
    )


def channel_first(space) -> Tuple[Box, Callable]:
    if not isinstance(space, Box) or len(space.shape) != 3:
        raise TypeError("channel_first needs a [H, W, C] Box observation space")
    h, w, c = space.shape
    return Box((c, h, w), space.low, space.high), (lambda obs: np.moveaxis(np.asarray(obs), -1, 0))


def one_hot(space) -> Tuple[Box, Callable]:
    if not isinstance(space, Discrete):
        raise TypeError("one_hot needs a Discrete observation space")
    n = space.n

    def encode(obs) -> np.ndarray:
        vec = np.zeros(n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec

    return Box((n,), 0.0, 1.0), encode


PRESETS = {"flatten": flatten, "image": channel_first, "one_hot": one_hot}
