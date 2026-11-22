"""A seat: one learner, the agents it plays, and its update schedule.

Swapping an agent's algorithm means changing its seat's learner and nothing else: drivers
and the trainer only call ``act``, ``record``, ``ready`` and ``update``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from rl_playground.core.registry import Registry
from rl_playground.core.types import ActionOut, Metrics, Transition
from rl_playground.envs.base import Discrete
from rl_playground.envs.multi_agent.base import AgentPolicy, random_policy
from rl_playground.multi_agent.schedules import UpdateSchedule


class PolicyLearner:
    """An ``AgentPolicy`` that sits in a seat like a learner and never updates (Adapter).
    Use it for random or scripted opponents, or for a frozen copy of a trained agent."""

    family = "fixed"
    tabular = False
    update_on_episode_end = False
    uses_action_mask = True

    def __init__(self, policy: AgentPolicy):
        self.policy = policy

    def act(self, obs, explore: bool = True, mask: Optional[np.ndarray] = None) -> ActionOut:
        return ActionOut(self.policy(obs, mask))

    def update(self, batch) -> Metrics:
        return {}

    def state_dict(self) -> dict:
        return {}


def first_legal(obs_space, action_space, seed: int = 0) -> PolicyLearner:
    """Always the lowest legal action. Deterministic, which makes tests and baselines easy."""
    if not isinstance(action_space, Discrete):
        raise TypeError("first_legal needs a Discrete action space")

    def act(obs, mask=None):
        return int(np.flatnonzero(mask)[0]) if mask is not None else 0

    return PolicyLearner(act)


def random_legal(obs_space, action_space, seed: int = 0) -> PolicyLearner:
    """Uniform over the legal actions (over the bounds for a Box)."""
    return PolicyLearner(random_policy(action_space, seed))


#: Fixed policies by name, each ``(obs_space, action_space, seed) -> PolicyLearner`` (Registry).
POLICIES = Registry("policy")
POLICIES.add("random", random_legal)
POLICIES.add("first_legal", first_legal)


@dataclass
class Seat:
    name: str
    agents: Tuple[str, ...]
    learner: object
    schedule: UpdateSchedule
    updates: int = field(default=0)

    def act(self, agent: str, obs, mask: Optional[np.ndarray], explore: bool = True) -> ActionOut:
        """[MA-2] The legal-move mask goes only to learners that say they use it."""
        if getattr(self.learner, "uses_action_mask", False):
            return self.learner.act(obs, explore, mask=mask)
        return self.learner.act(obs, explore)

    def record(self, agent: str, transition: Transition) -> None:
        self.schedule.add(agent, transition)

    def ready(self) -> bool:
        return self.schedule.ready()

    def update(self) -> Metrics:
        self.updates += 1
        return self.schedule.update(self.learner)
