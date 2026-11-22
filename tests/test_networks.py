"""Shapes and contracts of the filled-in network plumbing. Skipped where torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from rl_playground.envs.base import Box, Discrete  # noqa: E402
from rl_playground.networks.encoders import CNNEncoder, MLPEncoder  # noqa: E402
from rl_playground.networks.factory import build_actor_critic, build_q_network  # noqa: E402
from rl_playground.networks.heads import CategoricalHead, GaussianHead, ValueHead  # noqa: E402
from rl_playground.values.neural import NeuralQ, NeuralV  # noqa: E402


def test_mlp_encoder_sets_out_dim():
    enc = MLPEncoder(4, (32, 16))
    assert enc.out_dim == 16
    assert enc(torch.zeros(5, 4)).shape == (5, 16)


def test_cnn_encoder_takes_raw_pixels():
    enc = CNNEncoder((4, 84, 84))
    pixels = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    out = enc(pixels)
    assert enc.out_dim == 512 and out.shape == (2, 512)
    assert torch.allclose(out, enc(pixels.float() / 255.0))


def test_heads_shapes():
    feats = torch.zeros(3, 8)
    cat = CategoricalHead(8, 4)(feats)
    assert cat.sample().shape == (3,) and cat.log_prob(torch.zeros(3, dtype=torch.long)).shape == (3,)
    gauss = GaussianHead(8, 2)(feats)
    a = gauss.sample()
    assert a.shape == (3, 2) and gauss.log_prob(a).shape == (3,) and gauss.entropy().shape == (3,)
    assert ValueHead(8)(feats).shape == (3,)


@pytest.mark.parametrize("action_space", [Discrete(3), Box((2,))])
def test_actor_critic_act_and_evaluate(action_space):
    net = build_actor_critic(Box((4,)), action_space, (16,))
    obs = torch.randn(6, 4)
    action, log_prob, value = net.act(obs)
    assert log_prob.shape == (6,) and value.shape == (6,)
    assert not log_prob.requires_grad
    log_probs, entropy, values = net.evaluate(obs, action)
    assert log_probs.requires_grad and entropy.shape == (6,) and values.shape == (6,)
    assert torch.allclose(log_probs, log_prob, atol=1e-6)
    mode, _, _ = net.act(obs, deterministic=True)
    assert torch.equal(mode, net.act(obs, deterministic=True)[0])


def test_neural_q_online_and_target_start_equal():
    net = build_q_network(Box((4,)), Discrete(2), (16,))
    q = NeuralQ(net, torch.optim.Adam(net.parameters()), loss=None)
    obs = [[0.1, 0.2, 0.3, 0.4]] * 3
    assert q(obs).shape == (3, 2)
    assert torch.allclose(q(obs), q.target(obs))
    assert not any(p.requires_grad for p in q.target_network.parameters())


def test_neural_v_update_reduces_the_error():
    torch.manual_seed(0)
    enc = MLPEncoder(2, (16,))
    net = torch.nn.Sequential(enc, ValueHead(enc.out_dim))
    v = NeuralV(net, torch.optim.SGD(net.parameters(), lr=0.1))
    obs, targets = torch.randn(32, 2), torch.ones(32)
    first = v.update(obs, targets)["value_loss"]
    for _ in range(50):
        last = v.update(obs, targets)["value_loss"]
    assert last < first
