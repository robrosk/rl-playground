"""Networks from spaces (Factory pattern). Replaces ``if is_image_obs`` branches in algorithms."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from rl_playground.envs.base import Box, Discrete
from rl_playground.networks.actor_critic import ActorCritic
from rl_playground.networks.encoders import CNNEncoder, MLPEncoder
from rl_playground.networks.heads import CategoricalHead, GaussianHead, QHead, ValueHead


def build_encoder(obs_space, hidden: Sequence[int]) -> nn.Module:
    if isinstance(obs_space, Box) and len(obs_space.shape) == 3:
        return CNNEncoder(obs_space.shape)
    if isinstance(obs_space, Box) and len(obs_space.shape) == 1:
        return MLPEncoder(obs_space.shape[0], hidden)
    if isinstance(obs_space, Discrete):
        raise TypeError("wrap Discrete observations with envs.wrappers.OneHot first")
    raise TypeError(f"no encoder for observation space {obs_space!r}")


def build_policy_head(action_space, in_dim: int) -> nn.Module:
    if isinstance(action_space, Discrete):
        return CategoricalHead(in_dim, action_space.n)
    if isinstance(action_space, Box) and len(action_space.shape) == 1:
        return GaussianHead(in_dim, action_space.shape[0])
    raise TypeError(f"no policy head for action space {action_space!r}")


def build_q_network(obs_space, action_space, hidden: Sequence[int]) -> nn.Module:
    if not isinstance(action_space, Discrete):
        raise TypeError("Q networks need a Discrete action space")
    encoder = build_encoder(obs_space, hidden)
    return nn.Sequential(encoder, QHead(encoder.out_dim, action_space.n))


def build_actor_critic(obs_space, action_space, hidden: Sequence[int], shared: bool = False) -> ActorCritic:
    """Separate actor and critic encoders by default; ``shared=True`` shares one."""
    actor_encoder = build_encoder(obs_space, hidden)
    critic_encoder = actor_encoder if shared else build_encoder(obs_space, hidden)
    return ActorCritic(
        actor_encoder=actor_encoder,
        critic_encoder=critic_encoder,
        policy_head=build_policy_head(action_space, actor_encoder.out_dim),
        value_head=ValueHead(critic_encoder.out_dim),
    )
