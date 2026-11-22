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
from rl_playground.core.config import RunConfig, config_for
from rl_playground.data.replay_buffer import LatestTransition, ReplayBuffer
from rl_playground.data.rollout_buffer import RolloutBuffer
from rl_playground.envs import ENVS
from rl_playground.envs.base import Discrete
from rl_playground.envs.gym_adapter import GymAdapter
from rl_playground.envs.wrappers import OneHot, TimeLimit
from rl_playground.mdp.from_env import from_env
from rl_playground.runners import OffPolicyTrainer, OnPolicyTrainer, PlanningRunner, Runner


def make_env(name: str):
    """A registered name, or ``gym:<id>`` for any gymnasium env."""
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
