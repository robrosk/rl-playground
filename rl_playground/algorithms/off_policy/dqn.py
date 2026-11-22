"""DQN with the Double DQN target. Read alongside docs/dqn.md."""

from __future__ import annotations

import torch

from rl_playground.algorithms.base import Learner
from rl_playground.algorithms.tabular.base import epsilon_schedule
from rl_playground.core.types import ActionOut, Batch, Metrics
from rl_playground.losses.td import HuberTDLoss
from rl_playground.networks.factory import build_q_network
from rl_playground.policies.epsilon_greedy import EpsilonGreedy
from rl_playground.values.neural import NeuralQ


class DQN(Learner):
    family = "off_policy"

    def __init__(self, obs_space, action_space, config, seed: int = 0, q: NeuralQ = None):
        torch.manual_seed(seed)
        self.config = config
        if q is None:
            network = build_q_network(obs_space, action_space, config.hidden)
            optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)
            q = NeuralQ(network, optimizer, HuberTDLoss(config.huber_delta), config.max_grad_norm)
        self.q = q
        self.behaviour = EpsilonGreedy(epsilon_schedule(config), action_space.n, seed=seed)
        self.steps = 0  # env steps
        self.updates = 0  # gradient steps

    def act(self, obs, explore: bool = True, mask=None) -> ActionOut:
        """[DQN-1] Q values for one observation from ``self.q``; epsilon-greedy on env steps
        when exploring, greedy otherwise. [MASK-4] [MASK-5] Hand ``mask`` to both, so the
        agent never picks an illegal move."""
        raise NotImplementedError

    def update(self, batch: Batch) -> Metrics:
        """[DQN-4] y = double_q_target(r, Q_online(s'), Q_target(s'), terminated, gamma)
        (q_learning_target on Q_target(s') when config.double is False).
        [MASK-6] Pass ``next_mask=batch.next_masks`` (None where the env has no masks), so the
        target only looks at moves that were legal in s'.
        [DQN-5] [DQN-6] self.q.update(s, a, y).
        [DQN-7] sync the target every config.target_update_every updates, or Polyak with tau."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        return self.q.state_dict()
