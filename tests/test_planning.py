import numpy as np
import pytest

from rl_playground.algorithms.planning.modified_pi import ModifiedPolicyIteration
from rl_playground.algorithms.planning.policy_evaluation import PolicyEvaluation
from rl_playground.algorithms.planning.policy_iteration import PolicyIteration
from rl_playground.algorithms.planning.value_iteration import ValueIteration
from rl_playground.core.callbacks import ContractionMonitor
from rl_playground.core.config import PlanningConfig
from rl_playground.mdp import random_mdp
from rl_playground.runners.planning import PlanningRunner
from tests.conftest import GO

CFG = PlanningConfig(gamma=0.9, tol=1e-12)


def solve(planner, budget=10_000, monitor=None):
    return PlanningRunner(planner, tol=CFG.tol, callbacks=[monitor] if monitor else []).run(budget)


def test_dp2_evaluates_the_uniform_policy(mdp):
    planner = PolicyEvaluation(mdp, CFG)
    solve(planner)
    np.testing.assert_allclose(planner.V, [3 / 0.55, 0.0], atol=1e-9)


def test_dp2_evaluates_a_given_policy(mdp):
    planner = PolicyEvaluation(mdp, CFG, pi=GO)
    solve(planner)
    np.testing.assert_allclose(planner.V, [5.0, 0.0], atol=1e-9)


def test_dp6_first_sweeps_by_hand(mdp):
    planner = ValueIteration(mdp, CFG)
    assert [planner.sweep() for _ in range(3)] == pytest.approx([5.0, 0.5, 0.45])
    np.testing.assert_allclose(planner.V, [5.95, 0.0])


def test_dp6_converges_to_v_star_and_stays(mdp):
    planner = ValueIteration(mdp, CFG)
    monitor = ContractionMonitor(CFG.gamma)
    solve(planner, monitor=monitor)
    np.testing.assert_allclose(planner.V, [10.0, 0.0], atol=1e-9)
    assert planner.policy()[0].tolist() == [1.0, 0.0]
    assert monitor.violations == []  # [DP-8]


def test_dp5_policy_iteration_from_go(mdp):
    planner = PolicyIteration(mdp, CFG, pi=GO)
    solve(planner)
    assert planner.stable and planner.improvements == 1
    np.testing.assert_allclose(planner.V, [10.0, 0.0], atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pi_vi_and_mpi_agree_on_random_mdps(seed):
    mdp = random_mdp(8, 3, gamma=0.9, seed=seed)
    vi, pi, mpi = ValueIteration(mdp, CFG), PolicyIteration(mdp, CFG), ModifiedPolicyIteration(mdp, CFG)
    for planner in (vi, pi, mpi):
        solve(planner)
    np.testing.assert_allclose(vi.V, pi.V, atol=1e-6)
    np.testing.assert_allclose(mpi.V, pi.V, atol=1e-6)
