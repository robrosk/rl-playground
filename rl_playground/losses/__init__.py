"""Loss terms and their weighted sum (Composite pattern). Terms read named inputs."""

from rl_playground.losses.base import LossTerm
from rl_playground.losses.composite import CompositeLoss

__all__ = ["CompositeLoss", "LossTerm"]
