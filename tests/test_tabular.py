from dataclasses import replace

import numpy as np
import pytest

from rl_playground.algorithms.tabular.expected_sarsa import ExpectedSARSA
from rl_playground.algorithms.tabular.q_learning import QLearning
from rl_playground.algorithms.tabular.sarsa import SARSA
from rl_playground.algorithms.tabular.td0 import TD0
from rl_playground.core.config import TabularConfig
from rl_playground.envs.base import Discrete
from tests.conftest import one_transition

CFG = TabularConfig(gamma=0.9, alpha=0.5, epsilon_start=0.2, epsilon_end=0.2)
SPACE = Discrete(2)
BATCH = one_transition(s=0, a=1, r=1.0, s_next=1)


def make(cls):
    learner = cls(SPACE, SPACE, CFG, seed=0)
    learner.q.table[1] = [2.0, 4.0]
    return learner


def test_tab8_tab5_q_learning_update():
    learner = make(QLearning)
    learner.update(BATCH)  # target 1 + 0.9 * 4 = 4.6; Q <- 0 + 0.5 * 4.6
    assert learner.q.table[0, 1] == pytest.approx(2.3)


def test_tab3_sarsa_uses_the_next_action_taken():
    learner = make(SARSA)
    learner.update(replace(BATCH, next_actions=np.array([0])))  # 1 + 0.9 * 2 = 2.8
    assert learner.q.table[0, 1] == pytest.approx(1.4)


def test_tab4_expected_sarsa_averages_over_the_behaviour_policy():
    learner = make(ExpectedSARSA)
    learner.update(BATCH)  # pi(.|1) = [0.1, 0.9]; 1 + 0.9 * 3.8 = 4.42
    assert learner.q.table[0, 1] == pytest.approx(2.21)


def test_tab2_td0_prediction():
    learner = TD0(SPACE, SPACE, CFG, seed=0)
    learner.v.table[1] = 2.0
    learner.update(BATCH)  # 1 + 0.9 * 2 = 2.8
    assert learner.v.table[0] == pytest.approx(1.4)
