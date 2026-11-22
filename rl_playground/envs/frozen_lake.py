"""Frozen Lake with a known model, so planners and learners can be scored on one map.

``step`` samples from the same P that ``to_mdp`` exports, so value iteration on the MDP gives
the exact V* that every learner on the env should approach.

Actions follow gymnasium: 0 left, 1 down, 2 right, 3 up. With ``slip`` > 0 the intended move
happens with probability 1 - slip and each perpendicular move with probability slip / 2.
``slip = 2/3`` matches gymnasium's slippery lake.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from rl_playground.core.types import StepResult
from rl_playground.envs.base import Discrete
from rl_playground.mdp.tabular_mdp import TabularMDP

MOVES = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

MAPS = {
    "4x4": ("SFFF", "FHFH", "FFFH", "HFFG"),
    "8x8": (
        "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
        "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
    ),
}


class FrozenLake:
    def __init__(
        self,
        map_name: str = "4x4",
        slip: float = 2 / 3,
        goal_reward: float = 1.0,
        hole_reward: float = 0.0,
        step_reward: float = 0.0,
        desc: Optional[Sequence[str]] = None,
    ):
        self.desc = tuple(desc) if desc is not None else MAPS[map_name]
        self.n_rows, self.n_cols = len(self.desc), len(self.desc[0])
        self.slip = slip
        self.observation_space = Discrete(self.n_rows * self.n_cols)
        self.action_space = Discrete(4)
        self.start = "".join(self.desc).index("S")
        cell_reward = {"G": goal_reward, "H": hole_reward}
        self._arrival_reward = np.array(
            [cell_reward.get(self.cell(s), step_reward) for s in range(self.observation_space.n)]
        )
        self._P = self._build_transitions()
        self._rng = np.random.default_rng()
        self.state = self.start

    def cell(self, s: int) -> str:
        row, col = divmod(s, self.n_cols)
        return self.desc[row][col]

    def is_terminal(self, s: int) -> bool:
        return self.cell(s) in "GH"

    def _move(self, s: int, a: int) -> int:
        row, col = divmod(s, self.n_cols)
        d_row, d_col = MOVES[a]
        row = min(max(row + d_row, 0), self.n_rows - 1)
        col = min(max(col + d_col, 0), self.n_cols - 1)
        return row * self.n_cols + col

    def _build_transitions(self) -> np.ndarray:
        n_s, n_a = self.observation_space.n, self.action_space.n
        P = np.zeros((n_s, n_a, n_s))
        for s in range(n_s):
            for a in range(n_a):
                if self.is_terminal(s):
                    P[s, a, s] = 1.0  # absorbing
                    continue
                outcomes = ((a, 1 - self.slip), ((a - 1) % 4, self.slip / 2), ((a + 1) % 4, self.slip / 2))
                for moved, p in outcomes:
                    if p > 0:
                        P[s, a, self._move(s, moved)] += p
        return P

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state = self.start
        return self.state, {}

    def step(self, action: int) -> StepResult:
        if self.is_terminal(self.state):
            raise RuntimeError("episode is over; call reset()")
        action = int(action)
        next_state = int(self._rng.choice(self.observation_space.n, p=self._P[self.state, action]))
        self.state = next_state
        # info reports the action the agent chose. The old DQN env returned the slipped action.
        return StepResult(
            obs=next_state,
            reward=float(self._arrival_reward[next_state]),
            terminated=self.is_terminal(next_state),
            truncated=False,
            info={"action": action},
        )

    def to_mdp(self, gamma: float) -> TabularMDP:
        terminal = np.array([self.is_terminal(s) for s in range(self.observation_space.n)])
        R = self._P @ self._arrival_reward  # expected reward of arriving, [S, A]
        R[terminal] = 0.0
        return TabularMDP(P=self._P.copy(), R=R, gamma=gamma, terminal=terminal)


def deterministic_frozen_lake(**kwargs) -> FrozenLake:
    return FrozenLake(slip=0.0, **kwargs)


def frozen_lake_8x8(**kwargs) -> FrozenLake:
    return FrozenLake(map_name="8x8", **kwargs)
