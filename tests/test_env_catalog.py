"""The environment catalog: every entry builds and steps where its packages are installed.

Entries whose suite is not installed are skipped, so this file passes on a bare install and
checks more as extras are added (``pip install -e ".[envs]"`` for all of them).
"""

import numpy as np
import pytest

from rl_playground.envs import ENVS, MULTI_AGENT_ENVS
from rl_playground.envs.base import Box, Discrete
from rl_playground.envs.catalog import EXTRAS, GYM_SPECS, EnvSpec, MissingDependency, latest_gym_id
from rl_playground.envs.multi_agent.base import random_policy
from rl_playground.envs.multi_agent.catalog import MULTI_AGENT_SPECS, latest_module
from rl_playground.envs.wrappers import ChannelFirst, FlattenObservation
from rl_playground.mdp import from_env

ALL_SPECS = GYM_SPECS + MULTI_AGENT_SPECS


def in_space(obs, space) -> bool:
    if isinstance(space, Discrete):
        return 0 <= int(obs) < space.n
    return np.asarray(obs).shape == tuple(space.shape)


def test_every_spec_is_registered_once():
    names = [s.name for s in ALL_SPECS]
    assert len(names) == len(set(names))
    assert all(name in ENVS for name in names)
    assert all(s.name in MULTI_AGENT_ENVS for s in MULTI_AGENT_SPECS)
    assert all(extra in EXTRAS for s in ALL_SPECS for extra in ((s.extra,) if isinstance(s.extra, str) else s.extra or ()))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_builds_resets_and_steps(spec):
    if not spec.available():
        pytest.skip(f"{spec.name} needs {spec.extra}")
    env = ENVS.make(spec.name)
    try:
        obs, info = env.reset(seed=0)
        assert in_space(obs, env.observation_space)
        act = random_policy(env.action_space, seed=0)
        for _ in range(5):
            result = env.step(act(obs, info.get("action_mask")))
            assert in_space(result.obs, env.observation_space)
            assert isinstance(result.reward, float)
            obs, info = (env.reset() if result.done else (result.obs, result.info))
    finally:
        env.close()


def test_missing_extra_names_the_install_line(monkeypatch):
    monkeypatch.setitem(EXTRAS, "imaginary", ("a_module_that_is_not_installed",))
    with pytest.raises(MissingDependency, match=r"pip install -e \".\[imaginary\]\""):
        EnvSpec("ghost", "CartPole", "test", "imaginary").make()


def test_latest_version_is_picked():
    import gymnasium as gym

    newest = max(int(i.rsplit("-v", 1)[1]) for i in gym.registry if i.startswith("CartPole-v"))
    assert latest_gym_id("CartPole") == f"CartPole-v{newest}"
    assert latest_gym_id("CartPole", version=0) == "CartPole-v0"


def test_toy_text_model_matches_the_env():
    env = ENVS.make("taxi")
    mdp = from_env(env, gamma=0.99)
    assert mdp.n_states == env.observation_space.n + 1  # plus the absorbing end state
    obs, _ = env.reset(seed=3)
    for action in range(env.action_space.n):  # Taxi is deterministic
        env.reset(seed=3)
        result = env.step(action)
        landing = mdp.n_states - 1 if result.terminated else result.obs
        assert mdp.P[obs, action, landing] == pytest.approx(1.0)
        assert mdp.R[obs, action] == pytest.approx(result.reward)


def test_blackjack_tuple_observation_is_one_discrete_index():
    env = ENVS.make("blackjack")
    assert env.observation_space == Discrete(32 * 11 * 2)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, int) and 0 <= obs < env.observation_space.n


class _ImageEnv:
    observation_space = Box((4, 5, 3), 0.0, 255.0)
    action_space = Discrete(2)

    def reset(self, seed=None):
        return np.arange(60).reshape(4, 5, 3), {}

    def step(self, action):
        from rl_playground.core.types import StepResult

        return StepResult(np.arange(60).reshape(4, 5, 3), 0.0, False, False, {})


def test_channel_first_and_flatten():
    chw = ChannelFirst(_ImageEnv())
    assert chw.observation_space.shape == (3, 4, 5)
    obs, _ = chw.reset()
    assert obs.shape == (3, 4, 5) and obs[1, 0, 0] == 1  # channel 1 of pixel (0, 0)
    flat = FlattenObservation(_ImageEnv())
    assert flat.observation_space.shape == (60,) and flat.step(0).obs.shape == (60,)


def test_latest_module_is_picked():
    pytest.importorskip("pettingzoo")
    assert latest_module("pettingzoo.classic.tictactoe").startswith("pettingzoo.classic.tictactoe_v")
