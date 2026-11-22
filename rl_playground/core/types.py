"""Value objects passed between parts (Value Object pattern).

All are frozen dataclasses, so nothing changes them in flight and every field has a name
instead of a position in a tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Mapping, Sequence

import numpy as np

Metrics = Dict[str, float]


@dataclass(frozen=True)
class StepResult:
    """What ``Env.step`` returns."""

    obs: Any
    reward: float
    terminated: bool  # reached a true terminal state: never bootstrap past it
    truncated: bool  # cut off by a time limit: bootstrap from ``obs``
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class ActionOut:
    """What ``Learner.act`` returns. On-policy learners also report log pi(a|s) and V(s)."""

    action: Any
    log_prob: float | None = None
    value: float | None = None


@dataclass(frozen=True)
class Transition:
    """One step of experience.

    ``next_obs`` is the true next observation. When the episode ended at this step it is the
    final observation, never the observation the env returned after a reset.
    """

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    log_prob: float | None = None
    value: float | None = None
    # [MASK-2] legal actions where the env has them: in ``obs`` (for acting and for PPO's
    # re-evaluation) and in ``next_obs`` (for the max in a masked Bellman target)
    mask: Any = None
    next_mask: Any = None


@dataclass(frozen=True)
class Batch:
    """Stacked transitions, batch-first. Arrays are numpy here; deep learners convert them."""

    obs: Any
    actions: Any
    rewards: Any
    next_obs: Any
    terminated: Any
    truncated: Any
    log_probs: Any = None
    values: Any = None
    next_actions: Any = None  # filled by SARSA-style learners when they pick a'
    advantages: Any = None  # filled by on-policy learners before minibatching
    returns: Any = None
    masks: Any = None  # [B, A] bool, legal actions in obs; None when the env has no masks
    next_masks: Any = None  # [B, A] bool, legal actions in next_obs

    def __len__(self) -> int:
        return len(self.rewards)

    def take(self, indices) -> "Batch":
        """The rows at ``indices`` of every column that is present."""
        picked = {
            f.name: getattr(self, f.name)[indices]
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
        return replace(self, **picked)

    @staticmethod
    def from_transitions(transitions: Sequence[Transition]) -> "Batch":
        if not transitions:
            raise ValueError("cannot build a Batch from no transitions")

        def column(name, dtype=None):
            values = [getattr(t, name) for t in transitions]
            if any(v is None for v in values):
                return None
            return np.asarray(values, dtype=dtype)

        return Batch(
            obs=column("obs"),
            actions=column("action"),
            rewards=column("reward", np.float64),
            next_obs=column("next_obs"),
            terminated=column("terminated", np.float64),
            truncated=column("truncated", np.float64),
            log_probs=column("log_prob", np.float64),
            values=column("value", np.float64),
            masks=column("mask", bool),
            next_masks=column("next_mask", bool),
        )
