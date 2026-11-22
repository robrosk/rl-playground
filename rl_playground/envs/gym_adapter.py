"""Gymnasium env to ``Env`` (Adapter pattern).

Spaces are converted to the library's two kinds, ``Discrete`` and ``Box``. A tuple of discrete
spaces (Blackjack) or a ``MultiDiscrete`` becomes one ``Discrete`` by mixed-radix indexing, so
tabular learners can use it. Discrete spaces that start at an offset are shifted to start at 0.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from rl_playground.core.types import StepResult
from rl_playground.envs.base import Box, Discrete
from rl_playground.mdp.tabular_mdp import TabularMDP


def _discrete_parts(space) -> Optional[Tuple[Sequence[int], Sequence[int]]]:
    """(sizes, starts) for a tuple of Discrete spaces or a MultiDiscrete, else None."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Tuple) and all(isinstance(s, gym.spaces.Discrete) for s in space.spaces):
        return [int(s.n) for s in space.spaces], [int(s.start) for s in space.spaces]
    if isinstance(space, gym.spaces.MultiDiscrete) and space.nvec.ndim == 1:
        return [int(n) for n in space.nvec], [int(s) for s in getattr(space, "start", np.zeros_like(space.nvec))]
    return None


def convert_space(space):
    """gymnasium space to ``Discrete`` or ``Box``."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Discrete):
        return Discrete(int(space.n))
    if isinstance(space, gym.spaces.Box):
        return Box(tuple(space.shape), float(space.low.min()), float(space.high.max()))
    parts = _discrete_parts(space)
    if parts is not None:
        return Discrete(int(np.prod(parts[0])))
    raise TypeError(f"unsupported gymnasium space {space!r}")


def observation_encoder(space) -> Callable:
    """A function from a raw gymnasium observation to one in ``convert_space(space)``."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Discrete):
        start = int(space.start)
        return lambda obs: int(obs) - start
    parts = _discrete_parts(space)
    if parts is not None:
        sizes, starts = parts

        def encode(obs) -> int:
            index = 0
            for value, size, start in zip(obs, sizes, starts):
                index = index * size + (int(value) - start)
            return index

        return encode
    return lambda obs: obs


def action_decoder(space) -> Callable:
    """The inverse shift for a Discrete action space that starts at an offset."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Discrete) and int(space.start) != 0:
        start = int(space.start)
        return lambda action: int(action) + start
    return lambda action: action


class GymAdapter:
    def __init__(self, env_or_id, **make_kwargs):
        import gymnasium as gym

        self.env = gym.make(env_or_id, **make_kwargs) if isinstance(env_or_id, str) else env_or_id
        self.observation_space = convert_space(self.env.observation_space)
        self.action_space = convert_space(self.env.action_space)
        self._encode = observation_encoder(self.env.observation_space)
        self._decode = action_decoder(self.env.action_space)

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return self._encode(obs), info

    def step(self, action) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(self._decode(action))
        return StepResult(self._encode(obs), float(reward), bool(terminated), bool(truncated), info)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class GymToyTextAdapter(GymAdapter):
    """A toy-text env that publishes its model as ``unwrapped.P`` (FrozenLake, Taxi, CliffWalking).

    ``P[s][a]`` is a list of ``(prob, next_state, reward, done)``. ``to_mdp`` adds one absorbing
    terminal state at index ``S`` and routes every ``done`` transition there with its reward, so
    V(after the episode) = 0 without guessing which of the env's own states are terminal. The
    extra state is never observed; compare learners on states ``0 .. S-1``.
    """

    def __init__(self, env_or_id, **make_kwargs):
        super().__init__(env_or_id, **make_kwargs)
        if not hasattr(self.env.unwrapped, "P"):
            raise TypeError(f"{self.env.spec.id if self.env.spec else self.env} publishes no model P")

    def to_mdp(self, gamma: float) -> TabularMDP:
        model = self.env.unwrapped.P
        n_states, n_actions = self.observation_space.n, self.action_space.n
        end = n_states  # the added absorbing state
        P = np.zeros((n_states + 1, n_actions, n_states + 1))
        R = np.zeros((n_states + 1, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                for prob, s_next, reward, done in model[s][a]:
                    P[s, a, end if done else s_next] += prob
                    R[s, a] += prob * reward
        P[end, :, end] = 1.0
        terminal = np.zeros(n_states + 1, dtype=bool)
        terminal[end] = True
        return TabularMDP(P=P, R=R, gamma=gamma, terminal=terminal)
