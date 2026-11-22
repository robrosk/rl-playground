import numpy as np
import pytest

from rl_playground.core.arrays import amax, argmax, take_along_last
from rl_playground.envs.frozen_lake import FrozenLake
from rl_playground.envs.gridworld import GridWorld
from rl_playground.envs.wrappers import OneHot, TimeLimit
from rl_playground.mdp import TabularMDP, from_env, random_mdp


def test_random_mdp_is_valid():
    mdp = random_mdp(6, 3, seed=1)
    assert mdp.n_states == 6 and mdp.n_actions == 3
    assert mdp.terminal[-1]


def test_mdp_arrays_are_read_only(mdp):
    with pytest.raises(ValueError):
        mdp.P[0, 0, 0] = 0.5


def test_invalid_rows_are_refused():
    P = np.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        TabularMDP(P=P, R=np.zeros((1, 1)), gamma=0.9, terminal=np.array([False]))


def test_frozen_lake_model_matches_its_layout():
    env = FrozenLake(slip=0.0)
    mdp = from_env(TimeLimit(env, 100), gamma=0.9)  # unwraps wrappers to find the model
    assert mdp.terminal.sum() == 5  # four holes and the goal
    assert mdp.P[14, 2, 15] == 1.0  # right from the cell left of the goal
    assert mdp.R[14, 2] == 1.0
    slippery = from_env(FrozenLake(), gamma=0.9)
    assert np.isclose(slippery.P[0, 1].sum(), 1.0) and slippery.P[0, 1].max() < 1.0


def test_frozen_lake_reports_the_chosen_action():
    env = FrozenLake()
    env.reset(seed=0)
    assert env.step(2).info["action"] == 2


def test_time_limit_truncates_without_terminating():
    env = TimeLimit(GridWorld(size=4), max_steps=2)
    env.reset(seed=0)
    first, second = env.step(3), env.step(3)  # walk into the top wall
    assert not first.truncated
    assert second.truncated and not second.terminated


def test_one_hot_observations():
    env = OneHot(GridWorld(size=3))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,) and obs[0] == 1.0


def test_array_helpers_on_numpy():
    q = np.array([[1.0, 3.0], [5.0, 0.0]])
    assert amax(q).tolist() == [3.0, 5.0]
    assert argmax(q).tolist() == [1, 0]
    assert take_along_last(q, np.array([0, 1])).tolist() == [1.0, 0.0]
