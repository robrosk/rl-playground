"""Environments behind one ``Env`` interface (Adapter) with stackable wrappers (Decorator)."""

from rl_playground.core.registry import Registry
from rl_playground.envs.base import Box, Discrete, Env

ENVS = Registry("env")
ENVS.add("frozen_lake", "rl_playground.envs.frozen_lake:FrozenLake")
ENVS.add("frozen_lake_det", "rl_playground.envs.frozen_lake:deterministic_frozen_lake")
ENVS.add("frozen_lake_8x8", "rl_playground.envs.frozen_lake:frozen_lake_8x8")
ENVS.add("gridworld", "rl_playground.envs.gridworld:GridWorld")

__all__ = ["Box", "Discrete", "ENVS", "Env"]
