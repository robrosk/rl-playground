"""Multi-agent environments: PettingZoo behind ``ParallelEnv`` and ``TurnBasedEnv`` (Adapter),
with single-agent views so every library algorithm runs on them (Strategy for the other agents)."""

from rl_playground.envs.multi_agent.base import AgentPolicy, ParallelEnv, TurnBasedEnv, random_policy

__all__ = ["AgentPolicy", "ParallelEnv", "TurnBasedEnv", "random_policy"]
