"""Builds a ready-to-run runner from configs (Factory pattern).

Everything is wired here and passed in through constructors (Dependency injection), so no
part builds its own collaborators and tests can pass fakes instead.
"""

from __future__ import annotations

from typing import Iterable, Optional

from rl_playground.algorithms import ALGORITHMS
from rl_playground.algorithms.base import Planner
from rl_playground.collectors.collector import StepCollector
from rl_playground.core.callbacks import Callback, ConsoleLogger, ContractionMonitor
from rl_playground.core.config import CONFIGS, RunConfig, config_for
from rl_playground.data.replay_buffer import LatestTransition, ReplayBuffer
from rl_playground.data.rollout_buffer import RolloutBuffer
from rl_playground.envs import ENVS, MULTI_AGENT_ENVS
from rl_playground.envs.base import Discrete
from rl_playground.envs.gym_adapter import GymAdapter
from rl_playground.envs.wrappers import OneHot, TimeLimit
from rl_playground.mdp.from_env import from_env
from rl_playground.runners import MultiAgentTrainer, OffPolicyTrainer, OnPolicyTrainer, PlanningRunner, Runner


def make_env(name: str):
    """A registered name (``scripts/list_envs.py`` lists them), or ``gym:<id>`` for any
    gymnasium id. A PettingZoo name gives the game as seen by its first agent against random
    opponents; ``envs.MULTI_AGENT_ENVS`` has the multi-agent env itself."""
    if name.startswith("gym:"):
        return GymAdapter(name[len("gym:"):])
    return ENVS.make(name)


def build_runner(run: RunConfig, algo_config=None, callbacks: Optional[Iterable[Callback]] = None) -> Runner:
    cls = ALGORITHMS.get(run.algo)
    cfg = algo_config if algo_config is not None else config_for(run.algo)
    callbacks = list(callbacks) if callbacks is not None else [ConsoleLogger(every=run.log_every)]
    env = make_env(run.env)

    if issubclass(cls, Planner):
        if cls.contracts:
            callbacks.append(ContractionMonitor(cfg.gamma))
        planner = cls(from_env(env, cfg.gamma), cfg)
        return PlanningRunner(planner, tol=cfg.tol, callbacks=callbacks)

    env = TimeLimit(env, run.max_episode_steps)
    if not cls.tabular and isinstance(env.observation_space, Discrete):
        env = OneHot(env)
    learner = cls(env.observation_space, env.action_space, cfg, seed=run.seed)
    collector = StepCollector(env, learner.act, seed=run.seed)

    if cls.family == "on_policy":
        return OnPolicyTrainer(learner, collector, RolloutBuffer(cfg.rollout_len), callbacks=callbacks)
    if cls.family == "off_policy":
        buffer = ReplayBuffer(cfg.buffer_size) if hasattr(cfg, "buffer_size") else LatestTransition()
        return OffPolicyTrainer(
            learner,
            collector,
            buffer,
            batch_size=getattr(cfg, "batch_size", 1),
            warmup=getattr(cfg, "warmup", 0),
            train_every=getattr(cfg, "train_every", 1),
            callbacks=callbacks,
            seed=run.seed,
        )
    raise ValueError(f"{cls.__name__}.family must be 'on_policy' or 'off_policy', got {cls.family!r}")


# --- multi-agent -----------------------------------------------------------------------


def build_multi_agent_runner(run, callbacks: Optional[Iterable[Callback]] = None, env=None) -> MultiAgentTrainer:
    """Seats ``run.seats`` in the game ``run.env`` and returns a ready ``MultiAgentTrainer``.

    Pass ``env`` to use an env you built yourself (a fake in a test, or a PettingZoo game with
    non-default arguments) instead of looking ``run.env`` up in ``MULTI_AGENT_ENVS``.
    """
    from rl_playground.envs.multi_agent.wrappers import transform_observations
    from rl_playground.envs.transforms import one_hot
    from rl_playground.multi_agent.config import assign
    from rl_playground.multi_agent.drivers import ParallelDriver, TurnBasedDriver
    from rl_playground.multi_agent.schedules import schedule_for
    from rl_playground.multi_agent.seats import POLICIES, Seat

    callbacks = list(callbacks) if callbacks is not None else [ConsoleLogger(every=run.log_every)]
    env = env if env is not None else MULTI_AGENT_ENVS.make(run.env)
    groups = assign(env.possible_agents, run.seats)

    def maker(spec):
        """(factory, config, takes one-hot of Discrete observations)."""
        cfg = config_for(spec.algo, **spec.overrides) if spec.algo in CONFIGS else None
        if spec.make is not None:
            return spec.make, cfg, False
        if spec.algo in POLICIES:
            return POLICIES.get(spec.algo), None, False
        cls = ALGORITHMS.get(spec.algo)
        if issubclass(cls, Planner):
            raise ValueError(f"{spec.algo!r} plans with a known model; a seat needs a learner")
        return (lambda o, a, seed: cls(o, a, cfg, seed=seed)), cfg, not cls.tabular

    makers = [maker(spec) for spec, _ in groups]

    def learner_cls_of(spec):
        if spec.make is not None or spec.algo in POLICIES:
            return None
        return ALGORITHMS.get(spec.algo)

    # one-hot Discrete observations for the seats whose learners want vectors
    transforms = {}
    for (spec, members), (_, _, wants_vectors) in zip(groups, makers):
        for agent in members:
            if wants_vectors and isinstance(env.observation_space(agent), Discrete):
                transforms[agent] = one_hot
    env = transform_observations(env, transforms)

    seats = []
    for i, ((spec, members), (make, cfg, _)) in enumerate(zip(groups, makers)):
        obs_space, act_space = env.observation_space(members[0]), env.action_space(members[0])
        for agent in members[1:]:
            if env.observation_space(agent) != obs_space or env.action_space(agent) != act_space:
                raise ValueError(f"shared seat {spec.agents!r}: {agent} and {members[0]} have different spaces")
        if getattr(learner_cls_of(spec), "tabular", False) and not isinstance(obs_space, Discrete):
            raise ValueError(
                f"{spec.algo} is tabular and needs Discrete observations, but {members[0]} sees "
                f"{obs_space}. Give that seat a deep algorithm such as ppo or dqn."
            )
        seed = run.seed + 1000 * i
        learner = make(obs_space, act_space, seed)
        name = spec.name if (spec.name and (spec.shared or len(members) == 1)) else (
            spec.agents if spec.shared else members[0])
        schedule = schedule_for(learner, getattr(learner, "config", cfg), seed)
        seats.append(Seat(name, tuple(members), learner, schedule))

    driver_cls = TurnBasedDriver if hasattr(env, "last") else ParallelDriver
    return MultiAgentTrainer(driver_cls, env, seats, callbacks, seed=run.seed, returns_window=run.returns_window)
