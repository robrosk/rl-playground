"""Steps the env and turns each step into a Transition (Iterator pattern)."""

from __future__ import annotations

from typing import Callable, List, Mapping, Optional

from rl_playground.core.types import ActionOut, Transition


class StepCollector:
    """Owns the env's episode state so runners and learners never see a reset.

    ``act`` is ``learner.act``: obs -> ActionOut. Where the env reports legal actions in
    ``info["action_mask"]`` (board and card games), ``act`` is also given ``mask=``.
    """

    def __init__(self, env, act: Callable[..., ActionOut], seed: Optional[int] = None):
        self.env = env
        self.act = act
        self.seed = seed
        self.obs = None
        self.mask = None  # legal actions in self.obs, or None
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
        [MASK-1] The mask read with an observation travels with it: ``mask`` with ``obs`` and
        ``next_mask`` with ``next_obs``.
        """
        if self.obs is None:
            self.obs, info = self.env.reset(seed=self.seed)
            self.mask = _mask(info)
        out = self.act(self.obs) if self.mask is None else self.act(self.obs, mask=self.mask)
        result = self.env.step(out.action)
        next_mask = _mask(result.info)
        transition = Transition(
            obs=self.obs,
            action=out.action,
            reward=result.reward,
            next_obs=result.obs,
            terminated=result.terminated,
            truncated=result.truncated,
            log_prob=out.log_prob,
            value=out.value,
            mask=self.mask,
            next_mask=next_mask,
        )
        self._episode_return += result.reward
        if result.done:
            self.episode_returns.append(self._episode_return)
            self._episode_return = 0.0
            self.obs, info = self.env.reset()
            self.mask = _mask(info)
        else:
            self.obs, self.mask = result.obs, next_mask
        return transition


def _mask(info: Optional[Mapping]):
    return None if not info else info.get("action_mask")
