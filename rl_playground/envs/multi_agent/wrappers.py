"""Per-agent observation transforms for multi-agent envs (Decorator pattern).

``transforms`` maps an agent to a transform from ``envs/transforms.py``; agents not in it are
left alone. That is how one game can feed a flattened board to a PPO seat and a raw index to
a tabular seat at the same time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Mapping

from rl_playground.envs.transforms import Transform


class _PerAgentObservations:
    def __init__(self, env, transforms: Mapping[str, Transform]):
        self.env = env
        self._spaces: Dict[str, object] = {}
        self._fns: Dict[str, Callable] = {}
        for agent, transform in transforms.items():
            self._spaces[agent], self._fns[agent] = transform(env.observation_space(agent))

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation_space(self, agent):
        return self._spaces[agent] if agent in self._spaces else self.env.observation_space(agent)

    def _obs(self, agent, obs):
        fn = self._fns.get(agent)
        return obs if fn is None else fn(obs)


class ParallelObservations(_PerAgentObservations):
    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        return {a: self._obs(a, o) for a, o in observations.items()}, infos

    def step(self, actions):
        return {a: replace(r, obs=self._obs(a, r.obs)) for a, r in self.env.step(actions).items()}


class TurnBasedObservations(_PerAgentObservations):
    def last(self):
        agent = self.env.current_agent
        obs, result = self.env.last()
        obs = self._obs(agent, obs)
        return obs, replace(result, obs=obs)


def transform_observations(env, transforms: Mapping[str, Transform]):
    """Wraps either kind of multi-agent env."""
    if not transforms:
        return env
    cls = TurnBasedObservations if hasattr(env, "last") else ParallelObservations
    return cls(env, transforms)
