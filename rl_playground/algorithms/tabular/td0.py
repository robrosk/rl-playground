"""TD(0) prediction: learn V of the behaviour policy (uniform random by default)."""

from __future__ import annotations

from rl_playground.algorithms.tabular.base import TabularLearner
from rl_playground.core.types import ActionOut
from rl_playground.policies.epsilon_greedy import EpsilonGreedy
from rl_playground.policies.schedules import Constant
from rl_playground.values.tabular import TabularV


class TD0(TabularLearner):
    family = "on_policy"

    def __init__(self, obs_space, action_space, config, seed: int = 0):
        super().__init__(obs_space, action_space, config, seed)
        self.v = TabularV(self.n_states, alpha=config.alpha)
        self.behaviour = EpsilonGreedy(Constant(1.0), self.n_actions, seed=seed)  # uniform random

    def act(self, obs, explore: bool = True) -> ActionOut:
        return ActionOut(self.behaviour.act(self.q(obs), self.steps))

    def update(self, batch):
        """[TAB-2] y = td_target(r, V(s'), terminated, gamma); self.v.update(s, y)."""
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {"V": self.v.table.copy()}
