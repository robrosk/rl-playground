"""Steps the env and turns each step into a Transition (Iterator pattern)."""

from __future__ import annotations

from typing import Callable, List, Optional

from rl_playground.core.types import ActionOut, Transition


class StepCollector:
    """Owns the env's episode state so runners and learners never see a reset.

    ``act`` is ``learner.act``: obs -> ActionOut.
    """

    def __init__(self, env, act: Callable[..., ActionOut], seed: Optional[int] = None):
        self.env = env
        self.act = act
        self.seed = seed
        self.obs = None
        self.episode_returns: List[float] = []
        self._episode_return = 0.0

    def step(self) -> Transition:
        """[PPO-1] [DQN-2] One env step.

        1. On the first call, ``reset(seed=self.seed)``.
        2. ``out = self.act(self.obs)``, then ``result = env.step(out.action)``.
        3. Build the Transition with ``next_obs = result.obs``, the log-prob and value from
           ``out``, and terminated and truncated kept separate.
        4. Add the reward to the running return. If ``result.done``: append the return to
           ``episode_returns``, reset (no seed), and set ``self.obs`` to the reset obs.
           Otherwise set ``self.obs = result.obs``.

        The stored ``next_obs`` is always the true next observation, never the reset one.
        """
        raise NotImplementedError
