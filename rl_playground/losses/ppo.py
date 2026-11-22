"""PPO's three loss terms. Inputs are torch tensors of shape [B]. See docs/ppo.md."""

from __future__ import annotations

from typing import Optional


class ClippedSurrogateLoss:
    """[PPO-6] rho = exp(log_probs - old_log_probs)
    loss = -mean(min(rho * A, clip(rho, 1 - eps, 1 + eps) * A))

    Reads "log_probs", "old_log_probs", "advantages". Metrics: "clip_fraction" (share of rows
    where |rho - 1| > eps) and "approx_kl" (mean of old_log_probs - log_probs).
    """

    name = "policy"

    def __init__(self, clip_eps: float = 0.2):
        self.clip_eps = clip_eps

    def __call__(self, inputs):
        raise NotImplementedError


class ClippedValueLoss:
    """[PPO-7] V_clip = old_values + clip(values - old_values, -c, c)
    loss = mean(max((values - returns)^2, (V_clip - returns)^2)); plain MSE when c is None.

    No 0.5 factor, matching the old code. Reads "values", "old_values", "returns".
    """

    name = "value"

    def __init__(self, clip: Optional[float] = 0.2):
        self.clip = clip

    def __call__(self, inputs):
        raise NotImplementedError


class EntropyBonus:
    """[PPO-8] loss = -mean(entropy). Minimizing it raises entropy. Reads "entropy"."""

    name = "entropy"

    def __call__(self, inputs):
        raise NotImplementedError
