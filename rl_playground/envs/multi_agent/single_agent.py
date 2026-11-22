"""A multi-agent env seen by one learner as a plain ``Env`` (Adapter + Strategy patterns).

The learner controls ``agent``; every other agent acts from a policy in ``others`` (random
by default). That is enough to train any library algorithm against random play, a scripted
bot, or a frozen copy of itself (self-play). Learning several agents at once belongs to a
multi-agent runner, which these views do not replace.

The current agent's legal-move mask, when the env has one, is in ``info["action_mask"]``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Union

from rl_playground.core.types import StepResult
from rl_playground.envs.multi_agent.base import AgentPolicy, random_policy

Others = Union[None, AgentPolicy, Mapping[str, AgentPolicy]]


def _policies(env, learner, others: Others, seed: Optional[int]):
    policies = {}
    for i, agent in enumerate(env.possible_agents):
        if agent == learner:
            continue
        if others is None:
            policies[agent] = random_policy(env.action_space(agent), None if seed is None else seed + i)
        elif callable(others):
            policies[agent] = others
        else:
            policies[agent] = others[agent]
    return policies


class TurnBasedSingleAgent:
    def __init__(self, env, agent: Optional[str] = None, others: Others = None, seed: Optional[int] = 0):
        self.env = env
        self.agent = agent if agent is not None else env.possible_agents[0]
        if self.agent not in env.possible_agents:
            raise KeyError(f"no agent {self.agent!r}; agents are {env.possible_agents}")
        self.others = _policies(env, self.agent, others, seed)
        self.observation_space = env.observation_space(self.agent)
        self.action_space = env.action_space(self.agent)
        self._last_obs = None

    def _info(self, info) -> dict:
        out = dict(info or {})
        mask = self.env.action_mask(self.agent)
        if mask is not None:
            out["action_mask"] = mask
        return out

    def _until_learner_turn(self) -> StepResult:
        """Play the other agents until the learner must act or the game is over."""
        while self.env.agents:
            obs, result = self.env.last()
            agent = self.env.current_agent
            if agent == self.agent:
                self._last_obs = obs
                return StepResult(obs, result.reward, result.terminated, result.truncated, self._info(result.info))
            action = None if result.done else self.others[agent](obs, self.env.action_mask(agent))
            self.env.step(action)
        return StepResult(self._last_obs, 0.0, True, False, {})

    def reset(self, seed: Optional[int] = None):
        """Start a game and play the others until the learner's first turn. A game the others
        end before the learner moves (an opponent folds first in poker) has nothing to learn
        from, so it is skipped and another one dealt."""
        for attempt in range(1000):
            self.env.reset(seed=seed if attempt == 0 else None)
            result = self._until_learner_turn()
            if not result.done:
                return result.obs, result.info
        raise RuntimeError("the other agents ended 1000 games in a row before the learner could act")

    def step(self, action) -> StepResult:
        self.env.step(action)
        return self._until_learner_turn()

    def close(self) -> None:
        self.env.close()


class ParallelSingleAgent:
    def __init__(self, env, agent: Optional[str] = None, others: Others = None, seed: Optional[int] = 0):
        self.env = env
        self.agent = agent if agent is not None else env.possible_agents[0]
        if self.agent not in env.possible_agents:
            raise KeyError(f"no agent {self.agent!r}; agents are {env.possible_agents}")
        self.others = _policies(env, self.agent, others, seed)
        self.observation_space = env.observation_space(self.agent)
        self.action_space = env.action_space(self.agent)
        self._obs = {}

    def _info(self, info) -> dict:
        out = dict(info or {})
        mask = self.env.action_mask(self.agent)
        if mask is not None:
            out["action_mask"] = mask
        return out

    def reset(self, seed: Optional[int] = None):
        self._obs, infos = self.env.reset(seed=seed)
        return self._obs[self.agent], self._info(infos.get(self.agent))

    def step(self, action) -> StepResult:
        actions = {self.agent: action}
        for agent in self.env.agents:
            if agent != self.agent:
                actions[agent] = self.others[agent](self._obs[agent], self.env.action_mask(agent))
        results = self.env.step(actions)
        self._obs = {a: r.obs for a, r in results.items()}
        mine = results[self.agent]
        done = mine.done or not self.env.agents
        truncated = mine.truncated or (done and not mine.terminated)
        return StepResult(mine.obs, mine.reward, mine.terminated, truncated, self._info(mine.info))

    def close(self) -> None:
        self.env.close()
