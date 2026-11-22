import numpy as np

from rl_playground.bellman.returns import discounted_returns, gae, n_step_returns

GAMMA, LAM = 0.9, 0.8
R = np.array([1.0, 1.0, 1.0])
V = np.full(3, 0.5)
NEXT_V = np.full(3, 0.5)
NO = np.zeros(3)
AT_1 = np.array([0.0, 1.0, 0.0])


def test_tab7_discounted_returns():
    np.testing.assert_allclose(discounted_returns(R, np.array([0.0, 0.0, 1.0]), GAMMA), [2.71, 1.9, 1.0])


def test_tab7_returns_restart_after_a_terminal_step():
    np.testing.assert_allclose(discounted_returns(R, AT_1, GAMMA), [1.9, 1.0, 1.0])


def test_a2c1_two_step_returns():
    # G_0 = 1 + 0.9 * 1 + 0.81 * 0.5; G_1 = 1 + 0.9 * 1 + 0.81 * 0.5 cut at the rollout end
    got = n_step_returns(R, NEXT_V, NO, NO, GAMMA, n=2)
    np.testing.assert_allclose(got, [2.305, 2.305, 1.45])


def test_ppo3_gae_without_episode_ends():
    adv, ret = gae(R, V, NEXT_V, NO, NO, GAMMA, LAM)
    np.testing.assert_allclose(adv, [2.12648, 1.634, 0.95])
    np.testing.assert_allclose(ret, adv + V)


def test_ppo3_terminated_does_not_bootstrap():
    adv, _ = gae(R, V, NEXT_V, AT_1, NO, GAMMA, LAM)
    np.testing.assert_allclose(adv, [1.31, 0.5, 0.95])


def test_ppo3_truncation_bootstraps_but_cuts_the_recursion():
    adv, _ = gae(R, V, NEXT_V, NO, AT_1, GAMMA, LAM)
    np.testing.assert_allclose(adv, [1.634, 0.95, 0.95])
