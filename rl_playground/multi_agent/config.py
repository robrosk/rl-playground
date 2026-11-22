"""Which algorithm plays which agents (Value Object pattern)."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SeatSpec:
    """One line of the seating plan.

    ``agents``: an agent name or a glob, such as ``"player_1"``, ``"adversary_*"`` or ``"*"``.
    ``algo``: a name in ``ALGORITHMS`` (``"ppo"``, ``"q_learning"``, ...) or in ``POLICIES``
    (``"random"``, ``"first_legal"``) for an agent that plays but never learns.
    ``overrides``: changes to that algorithm's default config.
    ``shared``: one learner for every matched agent (parameter sharing, or self-play when the
    agents are opponents). Otherwise each matched agent gets its own learner.
    ``make``: build the learner yourself, ``(obs_space, action_space, seed) -> Learner``.
    It wins over ``algo``, so an algorithm that is not registered can still take a seat.
    """

    agents: str
    algo: str = "random"
    overrides: Mapping = field(default_factory=dict)
    shared: bool = False
    name: Optional[str] = None
    make: Optional[Callable] = None

    def matches(self, agent: str) -> bool:
        return fnmatch.fnmatchcase(agent, self.agents)


@dataclass(frozen=True)
class MultiAgentRunConfig:
    env: str
    seats: Tuple[SeatSpec, ...]
    budget: int = 100_000  # env steps for parallel games, moves for turn-based ones
    seed: int = 0
    log_every: int = 1
    returns_window: int = 100  # episodes averaged in the return/<agent> metrics


def assign(agents: Sequence[str], specs: Sequence[SeatSpec]) -> List[Tuple[SeatSpec, List[str]]]:
    """[MA-1] Each agent goes to the first spec that matches it. Returns one (spec, agents)
    group per seat: a shared spec is one seat, any other spec is one seat per agent."""
    groups: Dict[int, List[str]] = {}
    for agent in agents:
        index = next((i for i, spec in enumerate(specs) if spec.matches(agent)), None)
        if index is None:
            raise ValueError(
                f"no seat for agent {agent!r}; agents are {list(agents)}. "
                "Add a SeatSpec for it, or SeatSpec('*', 'random') last to fill the rest."
            )
        groups.setdefault(index, []).append(agent)
    seats = []
    for index, members in sorted(groups.items()):
        spec = specs[index]
        if spec.shared:
            seats.append((spec, members))
        else:
            seats.extend((spec, [agent]) for agent in members)
    return seats
