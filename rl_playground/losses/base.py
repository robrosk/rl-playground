from __future__ import annotations

from typing import Any, Mapping, Protocol, Tuple

from rl_playground.core.types import Metrics


class LossTerm(Protocol):
    name: str

    def __call__(self, inputs: Mapping[str, Any]) -> Tuple[Any, Metrics]:
        """(loss to minimize, metrics). Reads only the keys it documents."""
