"""PettingZoo games seen by one learner. Skipped where pettingzoo is not installed."""

import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from rl_playground.envs import ENVS, MULTI_AGENT_ENVS  # noqa: E402


def lowest_legal(obs, mask):
    return int(np.flatnonzero(mask)[0])


def highest_legal(obs, mask):
    return int(np.flatnonzero(mask)[-1])


def canonical_return(learner, policies):
    """The learner's total reward from PettingZoo's own documented AEC loop."""
    from pettingzoo.classic import tictactoe_v3

    env = tictactoe_v3.env()
    env.reset(seed=0)
    total = 0.0
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        if agent == learner:
            total += reward
        action = None if terminated or truncated else policies[agent](obs["observation"], obs["action_mask"])
        env.step(action)
    return total


@pytest.mark.parametrize("seat", ["player_1", "player_2"])
def test_turn_based_credit_matches_pettingzoo(seat):
    other = "player_2" if seat == "player_1" else "player_1"
    expected = canonical_return(seat, {seat: lowest_legal, other: highest_legal})
    env = ENVS.make("tictactoe", agent=seat, others=highest_legal)
    obs, info = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        result = env.step(lowest_legal(obs, info["action_mask"]))
        total += result.reward
        obs, info, done = result.obs, result.info, result.done
    assert total == expected and expected != 0.0
    assert result.terminated


def test_second_seat_starts_after_the_opponents_move():
    env = ENVS.make("tictactoe", agent="player_2")
    obs, info = env.reset(seed=0)
    assert obs.shape == (18,) and info["action_mask"].sum() == 8


def test_illegal_move_loses():
    env = ENVS.make("tictactoe", others=lowest_legal)
    env.reset(seed=0)
    env.step(4)  # centre
    result = env.step(4)  # centre again
    assert result.terminated and result.reward == -1.0


def test_parallel_view_against_a_fixed_opponent():
    env = ENVS.make("rock_paper_scissors", others=lambda obs, mask: 1)  # always paper
    env.reset(seed=0)
    rewards = []
    for _ in range(3):
        rewards.append(env.step(0).reward)  # rock
    assert rewards == [-1.0, -1.0, -1.0]


def test_multi_agent_env_exposes_every_agent():
    env = MULTI_AGENT_ENVS.make("rock_paper_scissors")
    observations, _ = env.reset(seed=0)
    assert set(observations) == set(env.possible_agents) == {"player_0", "player_1"}
    results = env.step({"player_0": 0, "player_1": 2})  # rock beats scissors
    assert results["player_0"].reward == 1.0 and results["player_1"].reward == -1.0


# --- the multi-agent trainer on real games ---------------------------------------------

from rl_playground.build import build_multi_agent_runner  # noqa: E402
from rl_playground.multi_agent import MultiAgentRunConfig, PolicyLearner, SeatSpec  # noqa: E402


def _fixed(policy):
    return lambda obs_space, action_space, seed: PolicyLearner(policy)


def test_trainer_credits_tictactoe_moves_like_pettingzoo():
    expected = canonical_return("player_1", {"player_1": lowest_legal, "player_2": highest_legal})
    cfg = MultiAgentRunConfig("tictactoe", (
        SeatSpec("player_1", make=_fixed(lowest_legal)),
        SeatSpec("player_2", make=_fixed(highest_legal)),
    ))
    runner = build_multi_agent_runner(cfg, callbacks=[])
    while runner.driver.episodes == 0:
        runner.step()
    assert runner.driver.episode_returns["player_1"] == [expected]
    assert runner.driver.episode_returns["player_2"] == [-expected]


def test_trainer_on_a_simultaneous_game():
    cfg = MultiAgentRunConfig("rock_paper_scissors", (
        SeatSpec("player_0", "first_legal"),  # always rock
        SeatSpec("player_1", make=_fixed(lambda obs, mask: 1)),  # always paper
    ))
    runner = build_multi_agent_runner(cfg, callbacks=[]).run(30)
    assert runner.driver.episodes > 0
    assert all(r < 0 for r in runner.driver.episode_returns["player_0"])  # paper beats rock every round
    assert runner.driver.episode_returns["player_1"] == [-r for r in runner.driver.episode_returns["player_0"]]


def test_one_shared_seat_plays_every_agent_of_a_cooperative_game():
    pytest.importorskip("mpe2")
    cfg = MultiAgentRunConfig("simple_spread", (SeatSpec("agent_*", "random", shared=True),))
    runner = build_multi_agent_runner(cfg, callbacks=[]).run(60)
    assert [s.agents for s in runner.seats] == [("agent_0", "agent_1", "agent_2")]
    assert runner.driver.episodes >= 1


def test_ppo_and_dqn_can_sit_at_the_same_board():
    pytest.importorskip("torch")
    cfg = MultiAgentRunConfig("connect_four", (SeatSpec("player_0", "ppo"), SeatSpec("player_1", "dqn")))
    runner = build_multi_agent_runner(cfg, callbacks=[])
    learners = {s.name: type(s.learner).__name__ for s in runner.seats}
    assert learners == {"player_0": "PPO", "player_1": "DQN"}


def test_a_tabular_seat_on_a_vector_observation_is_refused_clearly():
    cfg = MultiAgentRunConfig("leduc_holdem", (SeatSpec("player_0", "q_learning"), SeatSpec("*", "random")))
    with pytest.raises(ValueError, match="needs Discrete observations"):
        build_multi_agent_runner(cfg, callbacks=[])
