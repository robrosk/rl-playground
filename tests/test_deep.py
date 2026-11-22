"""Deep parts. Skipped where torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from rl_playground.losses.ppo import ClippedSurrogateLoss, ClippedValueLoss, EntropyBonus  # noqa: E402


def test_ppo6_clipped_surrogate():
    old = torch.zeros(3)
    new = torch.log(torch.tensor([1.5, 0.5, 1.0]))
    adv = torch.tensor([1.0, -1.0, 2.0])
    loss, _ = ClippedSurrogateLoss(0.2)({"log_probs": new, "old_log_probs": old, "advantages": adv})
    assert loss.item() == pytest.approx(-0.8)  # -(1.2 - 0.8 + 2) / 3


def test_ppo7_clipped_value_loss():
    inputs = {"values": torch.tensor([1.0, 0.1]), "old_values": torch.zeros(2), "returns": torch.ones(2)}
    loss, _ = ClippedValueLoss(0.2)(inputs)
    assert loss.item() == pytest.approx(0.725)  # mean(max(0, 0.64), max(0.81, 0.81))


def test_ppo8_entropy_bonus():
    loss, _ = EntropyBonus()({"entropy": torch.tensor([1.0, 3.0])})
    assert loss.item() == pytest.approx(-2.0)


def test_dqn_and_ppo_build_on_cartpole_shapes():
    from rl_playground.algorithms.off_policy.dqn import DQN
    from rl_playground.algorithms.on_policy.ppo import PPO
    from rl_playground.core.config import DQNConfig, PPOConfig
    from rl_playground.envs.base import Box, Discrete

    obs, act = Box((4,)), Discrete(2)
    assert DQN(obs, act, DQNConfig()).act([0.0] * 4).action in (0, 1)
    assert PPO(obs, act, PPOConfig()).act([0.0] * 4).action in (0, 1)
