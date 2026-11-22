import numpy as np

from rl_playground.bellman.sampled import (
    double_q_target, expected_sarsa_target, q_learning_target, sarsa_target, td_target,
)

R = np.array([1.0, 2.0])
DONE = np.array([0.0, 1.0])  # the second transition ends in a terminal state


def test_tab2_td_target():
    np.testing.assert_allclose(td_target(R, np.array([3.0, 9.0]), DONE, 0.9), [3.7, 2.0])


def test_tab5_q_learning_target():
    next_q = np.array([[1.0, 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(q_learning_target(R, next_q, DONE, 0.9), [3.7, 2.0])


def test_tab6_dqn4_double_q_target_selects_online_evaluates_target():
    online = np.array([[1.0, 3.0], [0.0, 0.0]])  # picks action 1 in the first row
    target = np.array([[10.0, 20.0], [5.0, 5.0]])
    np.testing.assert_allclose(double_q_target(R, online, target, DONE, 0.9), [19.0, 2.0])


def test_dqn4_vanilla_target_on_the_target_net():
    target = np.array([[10.0, 20.0], [5.0, 5.0]])
    np.testing.assert_allclose(q_learning_target(R, target, DONE, 0.9), [19.0, 2.0])


def test_tab3_sarsa_target():
    np.testing.assert_allclose(sarsa_target(R, np.array([3.0, 7.0]), DONE, 0.9), [3.7, 2.0])


def test_tab4_expected_sarsa_target():
    next_q = np.array([[2.0, 4.0], [1.0, 1.0]])
    next_pi = np.array([[0.1, 0.9], [0.5, 0.5]])
    np.testing.assert_allclose(expected_sarsa_target(R, next_q, next_pi, DONE, 0.9), [4.42, 2.0])
