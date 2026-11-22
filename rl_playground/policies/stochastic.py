"""Deep policies act through ``ActorCritic.act``; this module is the seam for adding other
sampling rules (for example a squashed Gaussian for SAC) without changing the learners."""

from __future__ import annotations

from typing import Protocol, Tuple


class StochasticPolicy(Protocol):
    def act(self, obs, deterministic: bool = False, mask=None) -> Tuple[object, object, object]:
        """(action, log_prob, value)."""
