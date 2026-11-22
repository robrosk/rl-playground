"""REINFORCE (Williams 1992) with an optional learned baseline. See docs/policy_gradient.md."""

from __future__ import annotations

import numpy as np
import torch

from rl_playground.algorithms.base import Learner
from rl_playground.core.types import ActionOut
from rl_playground.networks.factory import build_actor_critic


class REINFORCE(Learner):
    family = "on_policy"
    update_on_episode_end = True

    def __init__(self, obs_space, action_space, config, seed: int = 0, actor_critic=None):
        torch.manual_seed(seed)
        self.config = config
        self.ac = actor_critic or build_actor_critic(obs_space, action_space, config.hidden)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, explore: bool = True) -> ActionOut:
        """Sample from pi(. | s) with ``self.ac.act``; take the mode when not exploring."""
        raise NotImplementedError

    def update(self, batch):
        """[RF-1] G_t from discounted_returns.
        [RF-2] loss = -mean(log pi(a_t | s_t) * (G_t - b_t)), with b_t = V(s_t) detached.
        [RF-3] add value_coef * mean (V(s_t) - G_t)^2 to fit the baseline; one gradient step."""
        raise NotImplementedError
