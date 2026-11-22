import numpy as np
import pytest

from rl_playground.policies import Constant, EpsilonGreedy, ExponentialDecay, LinearDecay, greedy_action


def test_constant():
    assert Constant(0.3).value(10**6) == 0.3


def test_linear_decay():
    s = LinearDecay(start=1.0, end=0.1, steps=10)
    assert [s.value(t) for t in (0, 5, 10, 20)] == pytest.approx([1.0, 0.55, 0.1, 0.1])


def test_exponential_decay():
    s = ExponentialDecay(start=1.0, end=0.1, decay=0.5)
    assert [s.value(t) for t in (0, 1, 2, 10)] == pytest.approx([1.0, 0.5, 0.25, 0.1])


def test_greedy_ties_go_low():
    assert greedy_action(np.array([1.0, 3.0, 3.0])) == 1


def test_tab1_epsilon_greedy_probs():
    policy = EpsilonGreedy(Constant(0.2), n_actions=2)
    np.testing.assert_allclose(policy.probs(np.array([2.0, 4.0]), step=0), [0.1, 0.9])


def test_tab1_epsilon_greedy_frequencies():
    policy = EpsilonGreedy(Constant(0.2), n_actions=4, seed=0)
    actions = [policy.act(np.array([0.0, 0.0, 1.0, 0.0]), step=t) for t in range(20_000)]
    assert np.mean(np.array(actions) == 2) == pytest.approx(0.85, abs=0.01)
