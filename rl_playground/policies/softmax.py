from __future__ import annotations

import numpy as np


class Softmax:
    """Boltzmann exploration: pi(a | s) proportional to exp(Q(s, a) / temperature).
    Subtract the max before exponentiating."""

    def __init__(self, temperature: float = 1.0, seed: int = 0):
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def probs(self, q_row, step: int = 0, mask=None) -> np.ndarray:
        logits = np.asarray(q_row, dtype=np.float64) / self.temperature
        legal = np.ones(len(logits), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        weights = np.where(legal, np.exp(logits - logits[legal].max()), 0.0)
        return weights / weights.sum()

    def act(self, q_row, step: int = 0, mask=None) -> int:
        p = self.probs(q_row, step, mask)
        return int(self.rng.choice(len(p), p=p))
