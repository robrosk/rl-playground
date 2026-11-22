"""Every supported PettingZoo environment, declared as data (Registry + Factory patterns).

Each ``MultiAgentSpec`` builds two things:

* ``make_multi_agent()``: the multi-agent env itself (``ParallelEnv`` or ``TurnBasedEnv``),
  for multi-agent learners.
* ``make()``: the same game as a single-agent ``Env`` for the first agent against random
  opponents, so every library algorithm runs on it today. Pass ``agent`` and ``others`` to
  pick the seat and the opponents (a scripted bot, or a frozen copy of the learner).

Modules are named without their ``_vN`` suffix; the newest installed version is used.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from rl_playground.envs.catalog import Extras, is_installed, require


def latest_module(base: str, version: Optional[int] = None) -> str:
    """``"pettingzoo.classic.tictactoe"`` -> ``"pettingzoo.classic.tictactoe_v3"``."""
    if version is not None:
        return f"{base}_v{version}"
    package, _, stem = base.rpartition(".")
    found = []
    for module in pkgutil.iter_modules(importlib.import_module(package).__path__):
        head, sep, tail = module.name.rpartition("_v")
        if sep and head == stem and tail.isdigit():
            found.append(int(tail))
    if not found:
        raise KeyError(f"no module {base}_vN in {package}")
    return f"{base}_v{max(found)}"


@dataclass(frozen=True)
class MultiAgentSpec:
    """``api`` is ``"turn_based"`` (PettingZoo AEC) or ``"parallel"``. ``preset`` shapes the
    single-agent view's observations: ``"flatten"`` for boards and small grids fed to an MLP,
    ``"image"`` for [H, W, C] frames made channel-first."""

    name: str
    module: str
    api: str
    suite: str
    extra: Extras = ("pettingzoo",)
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    preset: Optional[str] = None
    version: Optional[int] = None
    notes: str = ""

    def available(self) -> bool:
        return is_installed(self.extra)

    def make_multi_agent(self, **overrides):
        require(self.extra, self.name)
        from rl_playground.envs.multi_agent.pettingzoo_adapter import PettingZooParallel, PettingZooTurnBased

        module = importlib.import_module(latest_module(self.module, self.version))
        kwargs = {**self.kwargs, **overrides}
        if self.api == "parallel":
            return PettingZooParallel(module.parallel_env(**kwargs))
        if self.api == "turn_based":
            return PettingZooTurnBased(module.env(**kwargs))
        raise ValueError(f"api must be 'parallel' or 'turn_based', got {self.api!r}")

    def make(self, agent: Optional[str] = None, others=None, seed: Optional[int] = 0, **overrides):
        from rl_playground.envs.multi_agent.single_agent import ParallelSingleAgent, TurnBasedSingleAgent
        from rl_playground.envs.wrappers import ChannelFirst, FlattenObservation

        env = self.make_multi_agent(**overrides)
        view_cls = ParallelSingleAgent if self.api == "parallel" else TurnBasedSingleAgent
        view = view_cls(env, agent=agent, others=others, seed=seed)
        if self.preset == "flatten":
            return FlattenObservation(view)
        if self.preset == "image":
            return ChannelFirst(view)
        return view


def _specs(suite: str, api: str, extra: Extras, rows) -> Tuple[MultiAgentSpec, ...]:
    out = []
    for row in rows:
        name, module, *rest = row
        out.append(MultiAgentSpec(name, module, api, suite, extra, **(rest[0] if rest else {})))
    return tuple(out)


PZ = ("pettingzoo",)

BOARD_GAMES = _specs("classic", "turn_based", PZ, [
    ("tictactoe", "pettingzoo.classic.tictactoe", {"preset": "flatten"}),
    ("connect_four", "pettingzoo.classic.connect_four", {"preset": "flatten"}),
    ("go_9x9", "pettingzoo.classic.go", {"preset": "flatten", "kwargs": {"board_size": 9}}),
    ("go", "pettingzoo.classic.go", {"preset": "flatten"}),
]) + _specs("classic", "turn_based", PZ + ("pettingzoo-chess",), [
    ("chess", "pettingzoo.classic.chess", {"preset": "flatten"}),
])

CARD_GAMES = _specs("classic", "turn_based", PZ + ("pettingzoo-cards",), [
    ("leduc_holdem", "pettingzoo.classic.leduc_holdem"),
    ("texas_holdem", "pettingzoo.classic.texas_holdem"),
    ("texas_holdem_no_limit", "pettingzoo.classic.texas_holdem_no_limit"),
])

SIMULTANEOUS_GAMES = _specs("classic", "parallel", PZ, [
    ("rock_paper_scissors", "pettingzoo.classic.rps"),
])

BUTTERFLY = _specs("butterfly", "parallel", PZ + ("pettingzoo-butterfly",), [
    ("pistonball", "pettingzoo.butterfly.pistonball", {"preset": "image", "notes": "20 pistons, cooperative"}),
    ("cooperative_pong", "pettingzoo.butterfly.cooperative_pong", {"preset": "image"}),
    ("knights_archers_zombies", "pettingzoo.butterfly.knights_archers_zombies", {"preset": "flatten"}),
])

SISL = _specs("sisl", "parallel", PZ + ("box2d",), [
    ("multiwalker", "pettingzoo.sisl.multiwalker", {"notes": "continuous actions"}),
]) + _specs("sisl", "parallel", PZ + ("pettingzoo-butterfly",), [
    ("pursuit", "pettingzoo.sisl.pursuit", {"preset": "flatten"}),
])

MPE = _specs("mpe", "parallel", PZ + ("pettingzoo-mpe",), [
    (name, f"mpe2.{name}")
    for name in [
        "simple",
        "simple_spread",
        "simple_adversary",
        "simple_tag",
        "simple_push",
        "simple_reference",
        "simple_speaker_listener",
        "simple_crypto",
        "simple_world_comm",
    ]
])

MULTI_AGENT_SPECS: Tuple[MultiAgentSpec, ...] = BOARD_GAMES + CARD_GAMES + SIMULTANEOUS_GAMES + BUTTERFLY + SISL + MPE


def register(single_agent_registry, multi_agent_registry, specs=MULTI_AGENT_SPECS) -> None:
    for spec in specs:
        single_agent_registry.add(spec.name, spec.make)
        multi_agent_registry.add(spec.name, spec.make_multi_agent)
