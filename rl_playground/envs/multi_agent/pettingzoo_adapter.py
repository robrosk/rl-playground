"""PettingZoo environments to ``ParallelEnv`` and ``TurnBasedEnv`` (Adapter pattern).

PettingZoo's classic games return ``{"observation": ..., "action_mask": ...}``; others put the
mask in ``info["action_mask"]`` or have none. Both adapters split the mask out.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from rl_playground.core.types import StepResult
from rl_playground.envs.gym_adapter import action_decoder, convert_space, observation_encoder


class _Spaces:
    """Per-agent space conversion shared by both adapters."""

    def _init_spaces(self, env) -> None:
        import gymnasium as gym

        self.possible_agents = list(env.possible_agents)
        self._dict_obs: Dict[str, bool] = {}
        self._obs_spaces, self._act_spaces, self._encode, self._decode = {}, {}, {}, {}
        for agent in self.possible_agents:
            raw = env.observation_space(agent)
            self._dict_obs[agent] = isinstance(raw, gym.spaces.Dict) and "observation" in raw.spaces
            obs_space = raw["observation"] if self._dict_obs[agent] else raw
            self._obs_spaces[agent] = convert_space(obs_space)
            self._encode[agent] = observation_encoder(obs_space)
            act_space = env.action_space(agent)
            self._act_spaces[agent] = convert_space(act_space)
            self._decode[agent] = action_decoder(act_space)
        self._masks: Dict[str, Optional[np.ndarray]] = {}

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def action_mask(self, agent) -> Optional[np.ndarray]:
        return self._masks.get(agent)

    def _split(self, agent, raw_obs, info) -> Any:
        """Encoded observation; remembers the agent's mask."""
        mask = None
        if self._dict_obs[agent]:
            mask = raw_obs.get("action_mask")
            raw_obs = raw_obs["observation"]
        elif info and "action_mask" in info:
            mask = info["action_mask"]
        self._masks[agent] = None if mask is None else np.asarray(mask, dtype=bool)
        return self._encode[agent](raw_obs)


class PettingZooParallel(_Spaces):
    def __init__(self, env):
        self.env = env
        self._init_spaces(env)

    @property
    def agents(self):
        return list(self.env.agents)

    def reset(self, seed: Optional[int] = None):
        observations, infos = self.env.reset(seed=seed)
        return {a: self._split(a, o, infos.get(a)) for a, o in observations.items()}, infos

    def step(self, actions: Dict[str, Any]) -> Dict[str, StepResult]:
        decoded = {a: self._decode[a](act) for a, act in actions.items()}
        observations, rewards, terminations, truncations, infos = self.env.step(decoded)
        return {
            a: StepResult(
                self._split(a, observations[a], infos.get(a)),
                float(rewards.get(a, 0.0)),
                bool(terminations.get(a, False)),
                bool(truncations.get(a, False)),
                infos.get(a, {}),
            )
            for a in observations
        }

    def close(self) -> None:
        self.env.close()


class PettingZooTurnBased(_Spaces):
    """PettingZoo's AEC API. ``last()`` reports the reward the current agent has collected
    since it last acted, which is how turn-based games credit a move that the opponent's
    reply decides."""

    def __init__(self, env):
        self.env = env
        self._init_spaces(env)

    @property
    def agents(self):
        return list(self.env.agents)

    @property
    def current_agent(self) -> str:
        return self.env.agent_selection

    def reset(self, seed: Optional[int] = None) -> None:
        self.env.reset(seed=seed)

    def last(self):
        agent = self.env.agent_selection
        raw_obs, reward, terminated, truncated, info = self.env.last()
        obs = self._split(agent, raw_obs, info)
        return obs, StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

    def step(self, action) -> None:
        agent = self.env.agent_selection
        self.env.step(None if action is None else self._decode[agent](action))

    def close(self) -> None:
        self.env.close()
