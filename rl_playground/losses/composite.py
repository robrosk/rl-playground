from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from rl_playground.core.types import Metrics
from rl_playground.losses.base import LossTerm


def _scalar(x) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


class CompositeLoss:
    """sum_i weight_i * term_i(inputs). Each term's own metrics come back under its name."""

    def __init__(self, terms: Sequence[Tuple[float, LossTerm]]):
        self.terms = list(terms)

    def __call__(self, inputs: Mapping[str, Any]) -> Tuple[Any, Metrics]:
        total = 0.0
        metrics: Metrics = {}
        for weight, term in self.terms:
            loss, term_metrics = term(inputs)
            total = total + weight * loss
            metrics[f"loss/{term.name}"] = _scalar(loss)
            metrics.update({f"{term.name}/{k}": v for k, v in term_metrics.items()})
        metrics["loss/total"] = _scalar(total)
        return total, metrics
