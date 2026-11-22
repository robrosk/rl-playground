import numpy as np
import pytest

from rl_playground.bellman.exact import expectation_backup, greedy_policy, optimality_backup, q_from_v
from tests.conftest import UNIFORM

V = np.array([2.0, 0.0])


def test_dp3_q_from_v(mdp):
    # stay: 1 + 0.9 * 2 = 2.8; go: 5 + 0.9 * 0 = 5
    np.testing.assert_allclose(q_from_v(mdp, V), [[2.8, 5.0], [0.0, 0.0]])


def test_dp1_expectation_backup(mdp):
    np.testing.assert_allclose(expectation_backup(mdp, V, UNIFORM), [3.9, 0.0])


def test_dp6_optimality_backup(mdp):
    np.testing.assert_allclose(optimality_backup(mdp, V), [5.0, 0.0])


def test_dp4_greedy_policy_ties_go_to_the_lowest_action(mdp):
    np.testing.assert_array_equal(greedy_policy(mdp, V), [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(greedy_policy(mdp, np.array([40 / 9, 0.0])), [[1.0, 0.0], [1.0, 0.0]])


def test_dp1_fixed_point_is_v_pi(mdp):
    v_uniform = np.array([3 / 0.55, 0.0])
    np.testing.assert_allclose(expectation_backup(mdp, v_uniform, UNIFORM), v_uniform)
