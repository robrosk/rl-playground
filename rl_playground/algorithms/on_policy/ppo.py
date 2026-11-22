"""Proximal policy optimization with GAE. Read alongside docs/ppo.md."""

from __future__ import annotations

import numpy as np
import torch

from rl_playground.algorithms.base import Learner
from rl_playground.core.types import ActionOut, Batch, Metrics
from rl_playground.losses.composite import CompositeLoss
from rl_playground.losses.ppo import ClippedSurrogateLoss, ClippedValueLoss, EntropyBonus
from rl_playground.networks.factory import build_actor_critic


class PPO(Learner):
    family = "on_policy"

    def __init__(self, obs_space, action_space, config, seed: int = 0, actor_critic=None):
        torch.manual_seed(seed)
        self.config = config
        self.ac = actor_critic or build_actor_critic(obs_space, action_space, config.hidden)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.lr)
        self.loss = CompositeLoss([
            (1.0, ClippedSurrogateLoss(config.clip_eps)),
            (config.value_coef, ClippedValueLoss(config.value_clip)),
            (config.entropy_coef, EntropyBonus()),
        ])
        self.rng = np.random.default_rng(seed)

    def act(self, obs, explore: bool = True, mask=None) -> ActionOut:
        """[PPO-1] ``self.ac.act`` on one observation; return action, log-prob and value as
        plain Python or numpy values so the collector stores them in the Transition.
        [MASK-3] Pass ``mask`` (as a [1, A] bool tensor) so the logits of illegal moves drop out."""
        raise NotImplementedError

    def update(self, batch: Batch) -> Metrics:
        """One PPO update on one rollout.

        [PPO-2] next_values = V(batch.next_obs), no gradient (true next observations).
        [PPO-3] advantages, returns = gae(...) from bellman.returns.
        [PPO-4] normalize advantages over the whole rollout.
        For each of config.epochs, for each minibatch from data.iterate_minibatches:
            [PPO-5] log_probs, entropy, values = self.ac.evaluate(obs, actions, masks)
                    with the minibatch's ``masks`` (None where the env has none): the
                    new and old log-probs must come from the same masked distribution.
            [PPO-6] [PPO-7] [PPO-8] total, metrics = self.loss({...named inputs...})
            [PPO-9] zero_grad, backward, clip_grad_norm_(config.max_grad_norm), step
        Return metrics averaged over minibatches.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"actor_critic": self.ac.state_dict()}
