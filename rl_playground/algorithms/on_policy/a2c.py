"""Advantage actor-critic, synchronous, single env. See docs/policy_gradient.md."""

from __future__ import annotations

import numpy as np
import torch

from rl_playground.algorithms.base import Learner
from rl_playground.core.types import ActionOut
from rl_playground.networks.factory import build_actor_critic


class A2C(Learner):
    family = "on_policy"

    def __init__(self, obs_space, action_space, config, seed: int = 0, actor_critic=None):
        torch.manual_seed(seed)
        self.config = config
        self.ac = actor_critic or build_actor_critic(obs_space, action_space, config.hidden)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr)

    def act(self, obs, explore: bool = True) -> ActionOut:
        raise NotImplementedError

    def update(self, batch):
        """[A2C-1] Advantages: n_step_returns minus V, or gae with config.lam < 1.
        [A2C-2] actor loss -mean(log pi(a | s) * A), A detached.
        [A2C-3] + value_coef * mean (V - G)^2 - entropy_coef * mean entropy; one clipped step."""
        raise NotImplementedError
