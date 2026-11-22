"""Multi-agent interfaces. Two shapes, matching PettingZoo's two APIs.

* ``ParallelEnv``: every live agent acts at once (Pistonball, MPE, rock-paper-scissors).
* ``TurnBasedEnv``: one agent acts at a time (tic-tac-toe, chess, Go, poker).

Each agent's spaces are the library's ``Discrete`` and ``Box``. Legal-move masks travel beside
the observation, never inside it, so encoders see a plain array.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from rl_playground.core.types import StepResult
from rl_playground.envs.base import Box, Discrete

AgentId = str

#: (observation, legal-action mask or None) -> action. Opponents and teammates are these (Strategy).
AgentPolicy = Callable[[Any, Optional[np.ndarray]], Any]


class ParallelEnv(Protocol):
    possible_agents: List[AgentId]
    agents: List[AgentId]  # the agents still live

    def observation_space(self, agent: AgentId): ...

    def action_space(self, agent: AgentId): ...

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[AgentId, Any], Dict[AgentId, dict]]: ...

    def step(self, actions: Dict[AgentId, Any]) -> Dict[AgentId, StepResult]: ...

    def action_mask(self, agent: AgentId) -> Optional[np.ndarray]: ...


class TurnBasedEnv(Protocol):
    possible_agents: List[AgentId]
    agents: List[AgentId]

    def observation_space(self, agent: AgentId): ...

    def action_space(self, agent: AgentId): ...

    def reset(self, seed: Optional[int] = None) -> None: ...

    @property
    def current_agent(self) -> AgentId: ...

    def last(self) -> Tuple[Any, StepResult]:
        """(the current agent's observation, its StepResult since it last acted)."""
        ...

    def step(self, action) -> None: ...

    def action_mask(self, agent: AgentId) -> Optional[np.ndarray]: ...


def random_policy(action_space, seed: Optional[int] = None) -> AgentPolicy:
    """Uniform over legal actions for ``Discrete``; uniform in the bounds for ``Box``."""
    rng = np.random.default_rng(seed)

    def act(obs, mask=None):
        if isinstance(action_space, Discrete):
            legal = np.flatnonzero(mask) if mask is not None else np.arange(action_space.n)
            return int(rng.choice(legal))
        if isinstance(action_space, Box):
            low = action_space.low if np.isfinite(action_space.low) else -1.0
            high = action_space.high if np.isfinite(action_space.high) else 1.0
            return rng.uniform(low, high, size=action_space.shape).astype(np.float32)
        raise TypeError(f"no random policy for {action_space!r}")

    return act
