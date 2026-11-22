from rl_playground.build import build_runner
from rl_playground.core.config import RunConfig
from rl_playground.runners import OffPolicyTrainer, OnPolicyTrainer, PlanningRunner


def test_factory_picks_the_runner_from_the_algorithm():
    assert isinstance(build_runner(RunConfig("value_iteration"), callbacks=[]), PlanningRunner)
    assert isinstance(build_runner(RunConfig("q_learning"), callbacks=[]), OffPolicyTrainer)
    assert isinstance(build_runner(RunConfig("sarsa"), callbacks=[]), OnPolicyTrainer)
