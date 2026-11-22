"""Environments behind one ``Env`` interface (Adapter) with stackable wrappers (Decorator).

``ENVS`` holds every single-agent env by name: the hand-built tabular ones, the gymnasium
catalog, and each PettingZoo game seen by one learner. ``MULTI_AGENT_ENVS`` holds the
PettingZoo games themselves. ``python scripts/list_envs.py`` prints them all.
"""

from rl_playground.core.registry import Registry
from rl_playground.envs import catalog
from rl_playground.envs.base import Box, Discrete, Env
from rl_playground.envs.multi_agent import catalog as multi_agent_catalog

ENVS = Registry("env")
ENVS.add("frozen_lake", "rl_playground.envs.frozen_lake:FrozenLake")
ENVS.add("frozen_lake_det", "rl_playground.envs.frozen_lake:deterministic_frozen_lake")
ENVS.add("frozen_lake_8x8", "rl_playground.envs.frozen_lake:frozen_lake_8x8")
ENVS.add("gridworld", "rl_playground.envs.gridworld:GridWorld")
catalog.register(ENVS)

MULTI_AGENT_ENVS = Registry("multi-agent env")
multi_agent_catalog.register(ENVS, MULTI_AGENT_ENVS)

__all__ = ["Box", "Discrete", "ENVS", "Env", "MULTI_AGENT_ENVS"]
