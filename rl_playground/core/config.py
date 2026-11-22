"""Every hyperparameter lives here as a frozen dataclass. Modules hold no magic constants."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class RunConfig:
    algo: str
    env: str = "frozen_lake"
    budget: int = 10_000  # sweeps for planners, env steps for learners
    seed: int = 0
    max_episode_steps: int = 100
    log_every: int = 1
    out_dir: str = "runs"


@dataclass(frozen=True)
class PlanningConfig:
    gamma: float = 0.99
    tol: float = 1e-8  # stop when ||V_new - V||_inf < tol
    eval_tol: float = 1e-10  # policy iteration: how exactly to evaluate each policy
    eval_sweeps: int = 5  # modified policy iteration: evaluation sweeps per improvement


@dataclass(frozen=True)
class TabularConfig:
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    rollout_len: int = 1  # on-policy tabular methods update every step


@dataclass(frozen=True)
class MCConfig(TabularConfig):
    rollout_len: int = 10_000  # Monte Carlo updates at episode ends, not on a full buffer


@dataclass(frozen=True)
class DynaConfig(TabularConfig):
    planning_steps: int = 10


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    hidden: Tuple[int, ...] = (128, 128)
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup: int = 1_000
    train_every: int = 4
    target_update_every: int = 1_000  # hard sync, in gradient updates
    tau: Optional[float] = None  # set for Polyak averaging instead of hard sync
    double: bool = True
    huber_delta: float = 1.0
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000  # env steps, not gradient steps


@dataclass(frozen=True)
class PGConfig:
    """REINFORCE and A2C."""

    gamma: float = 0.99
    lam: float = 1.0  # A2C with lam < 1 uses GAE
    lr: float = 7e-4
    hidden: Tuple[int, ...] = (64, 64)
    rollout_len: int = 5
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    hidden: Tuple[int, ...] = (64, 64)
    rollout_len: int = 2048
    epochs: int = 10
    minibatch_size: int = 64
    clip_eps: float = 0.2
    value_clip: Optional[float] = 0.2  # None gives plain MSE
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


CONFIGS = {
    "policy_evaluation": PlanningConfig,
    "policy_iteration": PlanningConfig,
    "value_iteration": PlanningConfig,
    "modified_policy_iteration": PlanningConfig,
    "mc_control": MCConfig,
    "td0": TabularConfig,
    "sarsa": TabularConfig,
    "expected_sarsa": TabularConfig,
    "q_learning": TabularConfig,
    "double_q_learning": TabularConfig,
    "dyna_q": DynaConfig,
    "reinforce": PGConfig,
    "a2c": PGConfig,
    "ppo": PPOConfig,
    "dqn": DQNConfig,
}


def config_for(algo: str, **overrides):
    """The default config for ``algo`` with ``overrides`` applied."""
    try:
        cls = CONFIGS[algo]
    except KeyError:
        raise KeyError(f"no config for {algo!r}; known: {', '.join(sorted(CONFIGS))}") from None
    return replace(cls(), **overrides)
