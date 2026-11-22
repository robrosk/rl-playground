import numpy as np

from rl_playground.collectors.collector import StepCollector
from rl_playground.core.types import ActionOut, Batch, StepResult, Transition
from rl_playground.data.replay_buffer import ReplayBuffer
from rl_playground.data.rollout_buffer import iterate_minibatches


def transition(i):
    return Transition(obs=i, action=0, reward=float(i), next_obs=i + 1, terminated=False, truncated=False)


def test_dqn2_replay_buffer_keeps_the_newest():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        buffer.add(transition(i))
    assert len(buffer) == 3
    batch = buffer.sample(500, np.random.default_rng(0))
    assert len(batch) == 500 and set(batch.rewards.tolist()) == {2.0, 3.0, 4.0}


def test_rollout_minibatches_cover_every_row_once():
    batch = Batch.from_transitions([transition(i) for i in range(10)])
    chunks = list(iterate_minibatches(batch, size=4, rng=np.random.default_rng(0)))
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert sorted(np.concatenate([c.rewards for c in chunks]).tolist()) == list(map(float, range(10)))


class TwoStepEpisodes:
    """Each episode is two steps; reset obs are 100, 200, ...; each step adds 1."""

    def __init__(self):
        self.episode = 0

    def reset(self, seed=None):
        self.episode += 1
        self.obs, self.t = 100 * self.episode, 0
        return self.obs, {}

    def step(self, action):
        self.obs, self.t = self.obs + 1, self.t + 1
        return StepResult(self.obs, 1.0, terminated=self.t == 2, truncated=False)


def test_ppo1_collector_stores_the_final_obs_not_the_reset_obs():
    collector = StepCollector(TwoStepEpisodes(), act=lambda obs: ActionOut(0, log_prob=-0.5, value=0.1))
    t0, t1, t2 = (collector.step() for _ in range(3))
    assert (t0.obs, t0.next_obs) == (100, 101)
    assert (t1.obs, t1.next_obs, t1.terminated) == (101, 102, True)
    assert t2.obs == 200
    assert t0.log_prob == -0.5 and t0.value == 0.1
    assert collector.episode_returns == [2.0]
