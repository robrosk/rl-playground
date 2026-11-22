"""Play one step of a multi-agent game and say what each agent experienced (Strategy pattern).

A driver owns the game's episode state, as ``StepCollector`` does for one agent. It is given
``act(agent, obs, mask) -> ActionOut`` and returns ``(agent, Transition)`` pairs, so it knows
nothing about seats, learners or updates.

Both keep ``episode_returns[agent]``, the return of each finished episode.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from rl_playground.core.types import ActionOut, Transition

Act = Callable[..., ActionOut]
Emitted = List[Tuple[str, Transition]]


class _Returns:
    def _init_returns(self, agents) -> None:
        self.episode_returns: Dict[str, List[float]] = {a: [] for a in agents}
        self.episodes = 0
        self._running: Dict[str, float] = {}

    def _credit(self, agent: str, reward: float) -> None:
        self._running[agent] = self._running.get(agent, 0.0) + reward

    def _finish_episode(self) -> None:
        for agent, total in self._running.items():
            self.episode_returns.setdefault(agent, []).append(total)
        self._running = {}
        self.episodes += 1


class ParallelDriver(_Returns):
    """Every live agent acts, then the env steps once."""

    def __init__(self, env, act: Act, seed: Optional[int] = None):
        self.env = env
        self.act = act
        self.seed = seed
        self.obs: Optional[Dict[str, object]] = None
        self._init_returns(env.possible_agents)

    def step(self) -> Emitted:
        """[MA-3] One env step: one transition for each agent that acted."""
        if self.obs is None:
            self.obs, _ = self.env.reset(seed=self.seed if self.episodes == 0 else None)
        acting = [a for a in self.env.agents if a in self.obs]
        masks = {a: self.env.action_mask(a) for a in acting}
        outs = {a: self.act(a, self.obs[a], masks[a]) for a in acting}
        results = self.env.step({a: out.action for a, out in outs.items()})
        game_over = not self.env.agents

        emitted: Emitted = []
        for agent, out in outs.items():
            result = results[agent]
            # an agent the game stopped without a verdict was cut off, not finished
            truncated = result.truncated or (game_over and not result.terminated)
            emitted.append((agent, Transition(
                obs=self.obs[agent],
                action=out.action,
                reward=result.reward,
                next_obs=result.obs,
                terminated=result.terminated,
                truncated=truncated,
                log_prob=out.log_prob,
                value=out.value,
                mask=masks[agent],
                next_mask=self.env.action_mask(agent),
            )))
            self._credit(agent, result.reward)

        if game_over:
            self._finish_episode()
            self.obs = None
        else:
            live = set(self.env.agents)
            self.obs = {a: r.obs for a, r in results.items() if a in live}
        return emitted


class TurnBasedDriver(_Returns):
    """One move per step. An agent's move is only judged once the others have replied, so its
    transition is held until that agent's next turn, when ``last()`` reports the reward it
    collected meanwhile and the observation it now faces."""

    def __init__(self, env, act: Act, seed: Optional[int] = None):
        self.env = env
        self.act = act
        self.seed = seed
        self._pending: Dict[str, Tuple[object, object, ActionOut]] = {}  # obs, mask, out
        self._started = False
        self._init_returns(env.possible_agents)

    def step(self) -> Emitted:
        """[MA-4] Credit the current agent's previous move, then let it move."""
        if not self._started:
            self.env.reset(seed=self.seed if self.episodes == 0 else None)
            self._pending = {}
            self._started = True

        agent = self.env.current_agent
        obs, result = self.env.last()
        emitted: Emitted = []
        mask = self.env.action_mask(agent)  # read after last(), which refreshes it
        if agent in self._pending:
            prev_obs, prev_mask, out = self._pending.pop(agent)
            emitted.append((agent, Transition(
                obs=prev_obs,
                action=out.action,
                reward=result.reward,
                next_obs=obs,
                terminated=result.terminated,
                truncated=result.truncated,
                log_prob=out.log_prob,
                value=out.value,
                mask=prev_mask,
                next_mask=mask,
            )))
        self._credit(agent, result.reward)

        if result.done:
            self.env.step(None)  # [MA-5] a finished agent's last turn takes no action
        else:
            out = self.act(agent, obs, mask)
            self._pending[agent] = (obs, mask, out)
            self.env.step(out.action)

        if not self.env.agents:
            emitted.extend(self._cut_off_pending())
            self._finish_episode()
            self._started = False
        return emitted

    def _cut_off_pending(self) -> Emitted:
        """Moves the game ended before their agent's next turn. PettingZoo gives every agent
        a final turn, so this is for envs that do not; the move is marked truncated."""
        emitted = [
            (agent, Transition(obs, out.action, 0.0, obs, False, True, out.log_prob, out.value, mask, mask))
            for agent, (obs, mask, out) in self._pending.items()
        ]
        self._pending = {}
        return emitted
