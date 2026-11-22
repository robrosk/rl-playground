"""The two kinds of algorithm. Runners depend on these, never on a concrete algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rl_playground.core.types import ActionOut, Batch, Metrics
from rl_playground.mdp.tabular_mdp import TabularMDP


class Planner(ABC):
    """Improves V and pi with a known model, one sweep at a time."""

    contracts = False  # True when each sweep is a gamma-contraction (ContractionMonitor applies)

    def __init__(self, mdp: TabularMDP, config):
        self.mdp = mdp
        self.config = config
        self.V = np.zeros(mdp.n_states)
        self.pi = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)

    @abstractmethod
    def sweep(self) -> float:
        """One pass over all states. Returns ||V_new - V_old||_inf."""

    def policy(self) -> np.ndarray:
        return self.pi

    def state_dict(self) -> dict:
        return {"V": self.V.copy(), "pi": self.pi.copy()}


class Learner(ABC):
    """Learns from samples. ``family`` tells the factory which runner drives it."""

    family = "off_policy"  # or "on_policy"
    tabular = False  # tabular learners take Discrete observations as indices
    update_on_episode_end = False  # Monte Carlo methods update when an episode ends

    @abstractmethod
    def act(self, obs, explore: bool = True, mask=None) -> ActionOut:
        """``mask``: bool [A] of legal actions, or None when every action is legal. A learner
        given a mask never returns an illegal action. See docs/masking.md."""

    @abstractmethod
    def update(self, batch: Batch) -> Metrics:
        ...

    def state_dict(self) -> dict:
        return {}
