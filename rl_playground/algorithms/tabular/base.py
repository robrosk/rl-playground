"""Shared wiring for tabular learners: a Q table, a behaviour policy and a step count."""

from __future__ import annotations

from rl_playground.algorithms.base import Learner
from rl_playground.core.types import ActionOut
from rl_playground.policies.epsilon_greedy import EpsilonGreedy
from rl_playground.policies.greedy import greedy_action
from rl_playground.policies.schedules import Constant, LinearDecay
from rl_playground.values.tabular import TabularQ


def epsilon_schedule(config):
    if config.epsilon_start == config.epsilon_end:
        return Constant(config.epsilon_start)
    return LinearDecay(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)


class TabularLearner(Learner):
    tabular = True

    def __init__(self, obs_space, action_space, config, seed: int = 0):
        self.config = config
        self.n_states = obs_space.n
        self.n_actions = action_space.n
        self.q = TabularQ(self.n_states, self.n_actions, alpha=config.alpha)
        self.behaviour = EpsilonGreedy(epsilon_schedule(config), self.n_actions, seed=seed)
        self.steps = 0  # env steps, for the epsilon schedule

    def act(self, obs, explore: bool = True, mask=None) -> ActionOut:
        # [TAB-1] behave epsilon-greedily while learning, greedily when evaluated
        if not explore:
            return ActionOut(greedy_action(self.q(obs), mask))
        action = self.behaviour.act(self.q(obs), self.steps, mask)
        self.steps += 1
        return ActionOut(action)

    def state_dict(self) -> dict:
        return {"Q": self.q.table.copy()}
