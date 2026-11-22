"""Seats, schedules, drivers and the multi-agent trainer, on small pure-Python games.

``Nim`` follows PettingZoo's AEC rules (reward since an agent last acted, a final turn for
each finished agent), so the turn-based driver is tested without PettingZoo installed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from rl_playground.build import build_multi_agent_runner
from rl_playground.core.types import ActionOut, StepResult, Transition
from rl_playground.data.multi_stream import MultiStreamRollout
from rl_playground.envs.base import Box, Discrete
from rl_playground.multi_agent import (
    MultiAgentRunConfig,
    NoUpdates,
    OffPolicySchedule,
    OnPolicySchedule,
    ParallelDriver,
    SeatSpec,
    TurnBasedDriver,
)
from rl_playground.multi_agent.config import assign


class Nim:
    """Take 1 or 2 from a pile; whoever takes the last one wins (+1, the other -1)."""

    possible_agents = ["a", "b"]

    def __init__(self, pile: int = 5):
        self.start = pile
        self.agents = []

    def observation_space(self, agent):
        return Discrete(self.start + 1)

    def action_space(self, agent):
        return Discrete(2)  # action k takes k + 1

    def reset(self, seed=None):
        self.pile = self.start
        self.agents = list(self.possible_agents)
        self._current = "a"
        self._reward = {a: 0.0 for a in self.agents}
        self._over = False

    @property
    def current_agent(self):
        return self._current

    def last(self):
        a = self._current
        return self.pile, StepResult(self.pile, self._reward[a], self._over, False, {})

    def step(self, action):
        a = self._current
        self._reward[a] = 0.0
        if self._over:
            assert action is None, "a finished agent's last turn takes None"
            self.agents.remove(a)
            self._current = self.agents[0] if self.agents else None
            return
        other = "b" if a == "a" else "a"
        self.pile = max(0, self.pile - (int(action) + 1))
        if self.pile == 0:
            self._over = True
            self._reward[a] += 1.0
            self._reward[other] -= 1.0
        self._current = other

    def action_mask(self, agent):
        return np.array([True, self.pile >= 2])


class Echo:
    """Parallel: both agents act at once, each is paid its own action, three rounds."""

    possible_agents = ["x", "y"]

    def __init__(self, rounds: int = 3):
        self.rounds = rounds
        self.agents = []

    def observation_space(self, agent):
        return Box((1,), 0.0, float(self.rounds))

    def action_space(self, agent):
        return Discrete(2)

    def reset(self, seed=None):
        self.t = 0
        self.agents = list(self.possible_agents)
        return {a: np.array([0.0]) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        self.t += 1
        over = self.t >= self.rounds
        out = {a: StepResult(np.array([float(self.t)]), float(act), False, over, {}) for a, act in actions.items()}
        if over:
            self.agents = []
        return out

    def action_mask(self, agent):
        return None


class Recorder:
    """A fake learner: always action 0, remembers every batch it was given."""

    tabular = True
    update_on_episode_end = False

    def __init__(self, family="on_policy", episode_end=False, rollout_len=None):
        self.family = family
        if rollout_len is not None:
            self.config = SimpleNamespace(rollout_len=rollout_len)  # the builder reads learner.config
        self.update_on_episode_end = episode_end
        self.batches = []

    def act(self, obs, explore=True, mask=None):
        self.masks_seen = getattr(self, "masks_seen", []) + [mask]
        return ActionOut(0, log_prob=0.0, value=0.0)

    def update(self, batch):
        self.batches.append(batch)
        return {"rows": float(len(batch))}


def take_one(agent, obs, mask):
    return ActionOut(0)


# --- seating -------------------------------------------------------------------------


def test_first_matching_spec_wins_and_shared_specs_make_one_seat():
    specs = [SeatSpec("adversary_0", "ppo"), SeatSpec("adversary_*", "dqn", shared=True), SeatSpec("*", "random")]
    groups = assign(["adversary_0", "adversary_1", "adversary_2", "agent_0", "agent_1"], specs)
    assert [(spec.algo, members) for spec, members in groups] == [
        ("ppo", ["adversary_0"]),
        ("dqn", ["adversary_1", "adversary_2"]),
        ("random", ["agent_0"]),
        ("random", ["agent_1"]),
    ]


def test_an_agent_without_a_seat_is_an_error():
    with pytest.raises(ValueError, match="no seat for agent 'b'"):
        assign(["a", "b"], [SeatSpec("a", "ppo")])


# --- storage -------------------------------------------------------------------------


def _t(reward, terminated=False, truncated=False):
    return Transition(0, 0, reward, 0, terminated, truncated)


def test_streams_stay_contiguous_and_each_cut_is_marked_truncated():
    rollout = MultiStreamRollout(capacity=5)
    rollout.add("a", _t(1.0))
    rollout.add("b", _t(10.0))
    rollout.add("a", _t(2.0, terminated=True))
    rollout.add("b", _t(20.0))
    rollout.add("a", _t(3.0))
    assert rollout.full
    batch = rollout.get()
    assert batch.rewards.tolist() == [1.0, 2.0, 3.0, 10.0, 20.0]
    assert batch.terminated.tolist() == [0, 1, 0, 0, 0]
    assert batch.truncated.tolist() == [0, 0, 1, 0, 1]  # both streams cut mid-episode


# --- drivers -------------------------------------------------------------------------


def test_turn_based_driver_credits_each_move_at_its_agents_next_turn():
    driver = TurnBasedDriver(Nim(5), take_one, seed=0)
    emitted = []
    while driver.episodes == 0:
        emitted.extend(driver.step())
    by_agent = {a: [t for agent, t in emitted if agent == a] for a in "ab"}
    # a takes at piles 5, 3, 1 and wins; b takes at 4, 2 and loses
    assert [(t.obs, t.next_obs, t.reward, t.terminated) for t in by_agent["a"]] == [
        (5, 3, 0.0, False), (3, 1, 0.0, False), (1, 0, 1.0, True)]
    assert [(t.obs, t.next_obs, t.reward, t.terminated) for t in by_agent["b"]] == [
        (4, 2, 0.0, False), (2, 0, -1.0, True)]
    assert driver.episode_returns == {"a": [1.0], "b": [-1.0]}


def test_parallel_driver_marks_the_last_round_truncated_and_resets():
    driver = ParallelDriver(Echo(rounds=3), lambda agent, obs, mask: ActionOut(1), seed=0)
    steps = [driver.step() for _ in range(4)]
    assert all(len(s) == 2 for s in steps)
    assert [t.truncated for _, t in steps[2]] == [True, True]
    assert steps[3][0][1].obs.tolist() == [0.0]  # a new episode
    assert driver.episode_returns == {"x": [3.0], "y": [3.0]}


# --- trainer -------------------------------------------------------------------------


def _run(env, specs, budget):
    runner = build_multi_agent_runner(MultiAgentRunConfig("fake", tuple(specs), budget=budget), callbacks=[], env=env)
    return runner.run(budget)


def test_each_seat_keeps_its_own_algorithm_and_schedule():
    on, off = Recorder("on_policy"), Recorder("off_policy")
    runner = _run(Nim(5), [SeatSpec("a", make=lambda o, a, s: on), SeatSpec("b", make=lambda o, a, s: off)], budget=40)
    seats = {s.name: s for s in runner.seats}
    assert isinstance(seats["a"].schedule, OnPolicySchedule)
    assert isinstance(seats["b"].schedule, OffPolicySchedule)
    assert seats["a"].learner is on and seats["b"].learner is off
    # without a config, an on-policy seat waits for a full rollout of default length 1...
    assert on.batches and all(len(b) == 1 for b in on.batches)
    # ...and an online off-policy seat learns from each transition once
    assert sum(len(b) for b in off.batches) == seats["b"].schedule.steps > 0


def test_a_fixed_policy_seat_never_updates():
    learner = Recorder("on_policy")
    runner = _run(Nim(5), [SeatSpec("a", make=lambda o, a, s: learner), SeatSpec("b", "first_legal")], budget=20)
    seat_b = next(s for s in runner.seats if s.name == "b")
    assert isinstance(seat_b.schedule, NoUpdates) and seat_b.updates == 0
    assert learner.batches  # the learning seat still trained


def test_a_shared_episode_end_seat_waits_until_every_agent_has_finished():
    learner = Recorder("on_policy", episode_end=True, rollout_len=10_000)
    runner = _run(Nim(5), [SeatSpec("*", shared=True, make=lambda o, a, s: learner)], budget=7)
    assert len(runner.seats) == 1 and runner.seats[0].agents == ("a", "b")
    (batch,) = learner.batches  # the first game is 7 driver steps: 5 moves and 2 final turns
    assert len(batch) == 5 and batch.rewards.tolist() == [0.0, 0.0, 1.0, 0.0, -1.0]


def test_metrics_carry_the_seat_name():
    seen = []

    class Collect:
        def on_run_start(self, runner): pass
        def on_update(self, step, metrics): seen.append(metrics)
        def on_run_end(self, runner): pass

    cfg = MultiAgentRunConfig("fake", (SeatSpec("a", make=lambda o, a, s: Recorder()), SeatSpec("b", "random")), budget=20)
    build_multi_agent_runner(cfg, callbacks=[Collect()], env=Nim(5)).run(cfg.budget)
    assert seen and "a/rows" in seen[-1] and "return/a" in seen[-1] and not any(k.startswith("b/") for k in seen[-1])


def test_tabular_and_deep_seats_get_the_observations_they_need():
    pytest.importorskip("torch")
    cfg = MultiAgentRunConfig("fake", (SeatSpec("a", "q_learning"), SeatSpec("b", "ppo", {"rollout_len": 8})))
    runner = build_multi_agent_runner(cfg, callbacks=[], env=Nim(5))
    assert runner.driver.env.observation_space("a") == Discrete(6)  # index for the Q table
    assert runner.driver.env.observation_space("b") == Box((6,), 0.0, 1.0)  # one-hot for the network
    assert runner.seats[1].schedule.rollout.capacity == 8


def test_shared_seats_need_matching_spaces():
    class Uneven(Echo):
        def action_space(self, agent):
            return Discrete(2 if agent == "x" else 3)

    cfg = MultiAgentRunConfig("fake", (SeatSpec("*", "random", shared=True),))
    with pytest.raises(ValueError, match="different spaces"):
        build_multi_agent_runner(cfg, callbacks=[], env=Uneven())


def test_command_line_seats():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from run_multi import parse_seats

    specs = parse_seats(["player_0=ppo", "adversary_*=dqn:shared"], ["player_0.lr=3e-4", "adversary_*.warmup=500"])
    assert specs[0] == SeatSpec("player_0", "ppo", {"lr": 3e-4})
    assert specs[1] == SeatSpec("adversary_*", "dqn", {"warmup": 500}, shared=True)
