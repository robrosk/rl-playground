"""A finite MDP as arrays (Value Object pattern)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TabularMDP:
    """P[s, a, s'] = Pr(s' | s, a); R[s, a] = E[r | s, a]; terminal[s] marks absorbing states.

    Terminal states must be absorbing with zero reward, so V(terminal) = 0 under every policy.
    The arrays are copied and made read-only.
    """

    P: np.ndarray
    R: np.ndarray
    gamma: float
    terminal: np.ndarray

    def __post_init__(self):
        for name, dtype in (("P", float), ("R", float), ("terminal", bool)):
            array = np.array(getattr(self, name), dtype=dtype)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        self.validate()

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    def validate(self, atol: float = 1e-8) -> None:
        P, R, terminal = self.P, self.R, self.terminal
        if P.ndim != 3 or P.shape[0] != P.shape[2]:
            raise ValueError(f"P must be [S, A, S], got {P.shape}")
        if R.shape != P.shape[:2]:
            raise ValueError(f"R must be [S, A] = {P.shape[:2]}, got {R.shape}")
        if terminal.shape != (P.shape[0],):
            raise ValueError(f"terminal must be [S], got {terminal.shape}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if (P < -atol).any() or not np.allclose(P.sum(axis=2), 1.0, atol=atol):
            raise ValueError("every P[s, a, :] must be a probability distribution")
        for s in np.flatnonzero(terminal):
            if not np.allclose(P[s, :, s], 1.0, atol=atol) or not np.allclose(R[s], 0.0, atol=atol):
                raise ValueError(f"terminal state {s} must be absorbing with zero reward")
