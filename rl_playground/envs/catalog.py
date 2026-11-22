"""Every supported single-agent environment, declared as data (Registry + Factory patterns).

One ``EnvSpec`` line adds an environment. Specs name a gymnasium id without its version, and
the newest installed version is used, so a gymnasium release that bumps ``Taxi-v3`` to
``Taxi-v4`` changes nothing here. Pass ``version`` to pin one.

Suites that need more than numpy and gymnasium name their pip extra; asking for one that is
not installed raises ``MissingDependency`` with the install line instead of an import error.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Union

#: pip extra -> the modules whose presence means it is installed
EXTRAS: Dict[str, Tuple[str, ...]] = {
    "box2d": ("Box2D", "pygame"),
    "mujoco": ("mujoco", "imageio"),
    "atari": ("ale_py", "cv2"),
    "pettingzoo": ("pettingzoo",),
    "pettingzoo-butterfly": ("pygame", "pymunk"),
    "pettingzoo-cards": ("rlcard",),
    "pettingzoo-chess": ("chess",),
    "pettingzoo-hanabi": ("pyspiel",),
    "pettingzoo-mpe": ("mpe2",),
}


class MissingDependency(ImportError):
    pass


Extras = Union[None, str, Tuple[str, ...]]


def _as_tuple(extras: Extras) -> Tuple[str, ...]:
    if extras is None:
        return ()
    return (extras,) if isinstance(extras, str) else tuple(extras)


def missing(extras: Extras) -> Tuple[str, ...]:
    """The extras in ``extras`` whose packages are not installed."""
    return tuple(e for e in _as_tuple(extras) if any(importlib.util.find_spec(m) is None for m in EXTRAS[e]))


def is_installed(extras: Extras) -> bool:
    return not missing(extras)


def require(extras: Extras, name: str) -> None:
    absent = missing(extras)
    if absent:
        raise MissingDependency(f"{name} needs: pip install -e \".[{','.join(absent)}]\"")


def latest_gym_id(base_id: str, version: Optional[int] = None) -> str:
    """``"ALE/Pong"`` -> ``"ALE/Pong-v5"``: the newest registered version, or ``version``."""
    if version is not None:
        return f"{base_id}-v{version}"
    import gymnasium as gym

    prefix = base_id + "-v"
    versions = [int(i[len(prefix):]) for i in gym.registry if i.startswith(prefix) and i[len(prefix):].isdigit()]
    if not versions:
        if base_id in gym.registry:
            return base_id
        raise KeyError(f"gymnasium has no environment {base_id!r}")
    return f"{base_id}-v{max(versions)}"


@dataclass(frozen=True)
class EnvSpec:
    """How to build one gymnasium environment behind ``Env``.

    ``preset`` picks the observation pipeline: ``None`` as is, ``"image"`` for [H, W, C] frames
    (made channel-first), ``"atari"`` for the standard DQN preprocessing (84x84 grey, frame
    skip 4, 4 stacked frames). ``model`` marks toy-text envs whose transition model can be
    exported with ``to_mdp`` for planning.
    """

    name: str
    gym_id: str
    suite: str
    extra: Extras = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    preset: Optional[str] = None
    model: bool = False
    version: Optional[int] = None
    notes: str = ""

    def available(self) -> bool:
        return is_installed(self.extra)

    def make(self, **overrides):
        require(self.extra, self.name)
        import gymnasium as gym

        from rl_playground.envs.gym_adapter import GymAdapter, GymToyTextAdapter
        from rl_playground.envs.wrappers import ChannelFirst

        if self.suite == "atari":
            import ale_py

            gym.register_envs(ale_py)
        kwargs = {**self.kwargs, **overrides}
        env_id = latest_gym_id(self.gym_id, self.version)
        if self.preset == "atari":
            from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

            env = gym.make(env_id, frameskip=1, **kwargs)  # AtariPreprocessing does the skipping
            env = FrameStackObservation(AtariPreprocessing(env, frame_skip=4, screen_size=84), 4)
            return GymAdapter(env)
        env = gym.make(env_id, **kwargs)
        adapted = GymToyTextAdapter(env) if self.model else GymAdapter(env)
        return ChannelFirst(adapted) if self.preset == "image" else adapted


def _specs(suite: str, extra: Extras, rows) -> Tuple[EnvSpec, ...]:
    out = []
    for row in rows:
        name, gym_id, *rest = row
        options = rest[0] if rest else {}
        out.append(EnvSpec(name, gym_id, suite, extra, **options))
    return tuple(out)


CLASSIC_CONTROL = _specs("classic_control", None, [
    ("cartpole", "CartPole"),
    ("acrobot", "Acrobot"),
    ("mountain_car", "MountainCar"),
    ("mountain_car_continuous", "MountainCarContinuous"),
    ("pendulum", "Pendulum"),
])

TOY_TEXT = _specs("toy_text", None, [
    ("taxi", "Taxi", {"model": True}),
    ("cliff_walking", "CliffWalking", {"model": True}),
    ("cliff_walking_slippery", "CliffWalkingSlippery", {"model": True}),
    ("gym_frozen_lake", "FrozenLake", {"model": True, "notes": "gymnasium's own; frozen_lake is ours"}),
    ("gym_frozen_lake_8x8", "FrozenLake8x8", {"model": True}),
    ("blackjack", "Blackjack", {"notes": "tuple observation flattened to one Discrete index"}),
])

BOX2D = _specs("box2d", "box2d", [
    ("lunar_lander", "LunarLander"),
    ("lunar_lander_continuous", "LunarLanderContinuous"),
    ("bipedal_walker", "BipedalWalker"),
    ("bipedal_walker_hardcore", "BipedalWalkerHardcore"),
    ("car_racing", "CarRacing", {"preset": "image"}),
    ("car_racing_discrete", "CarRacing", {"preset": "image", "kwargs": {"continuous": False}}),
])

MUJOCO = _specs("mujoco", "mujoco", [
    ("inverted_pendulum", "InvertedPendulum"),
    ("inverted_double_pendulum", "InvertedDoublePendulum"),
    ("reacher", "Reacher"),
    ("swimmer", "Swimmer"),
    ("hopper", "Hopper"),
    ("half_cheetah", "HalfCheetah"),
    ("walker2d", "Walker2d"),
    ("ant", "Ant"),
    ("pusher", "Pusher"),
    ("humanoid", "Humanoid"),
    ("humanoid_standup", "HumanoidStandup"),
])

ATARI = _specs("atari", "atari", [
    (game, f"ALE/{gym_name}", {"preset": "atari"})
    for game, gym_name in [
        ("pong", "Pong"),
        ("breakout", "Breakout"),
        ("space_invaders", "SpaceInvaders"),
        ("seaquest", "Seaquest"),
        ("beam_rider", "BeamRider"),
        ("qbert", "Qbert"),
        ("enduro", "Enduro"),
        ("ms_pacman", "MsPacman"),
        ("asteroids", "Asteroids"),
        ("montezuma_revenge", "MontezumaRevenge"),
    ]
])

GYM_SPECS: Tuple[EnvSpec, ...] = CLASSIC_CONTROL + TOY_TEXT + BOX2D + MUJOCO + ATARI


def register(registry, specs=GYM_SPECS) -> None:
    for spec in specs:
        registry.add(spec.name, spec.make)
