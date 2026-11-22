"""Legal-move masks. See docs/masking.md.

The first half checks the plumbing and passes now: masks reach the learner and travel with
every transition and batch. The second half is the masked math in your stubs and fails with
NotImplementedError until you write it.
"""

import numpy as np
import pytest

from rl_playground.collectors.collector import StepCollector
from rl_playground.core.types import ActionOut, Batch, StepResult, Transition
from rl_playground.data.replay_buffer import ReplayBuffer
from rl_playground.envs.base import Box, Discrete
from rl_playground.multi_agent import TurnBasedDriver

from tests.test_multi_agent_trainer import Nim


class Shrinking:
    """One state; four actions, and each step makes the highest legal one illegal."""

    observation_space = Discrete(1)
    action_space = Discrete(4)

    def reset(self, seed=None):
        self.legal = 4
        return 0, {"action_mask": self._mask()}

    def _mask(self):
        return np.arange(4) < self.legal

    def step(self, action):
        self.legal -= 1
        done = self.legal == 1
        return StepResult(0, 1.0, done, False, {"action_mask": self._mask()})


class Spy:
    def __init__(self):
        self.masks = []

    def __call__(self, obs, explore=True, mask=None):
        self.masks.append(mask)
        return ActionOut(0)


# --- plumbing ------------------------------------------------------------------------


def test_collector_hands_the_mask_to_act_and_stores_both_masks():
    spy = Spy()
    collector = StepCollector(Shrinking(), spy, seed=0)
    first, second = collector.step(), collector.step()
    assert spy.masks[0].tolist() == [True, True, True, True]
    assert first.mask.tolist() == [True, True, True, True]
    assert first.next_mask.tolist() == [True, True, True, False]
    assert second.mask.tolist() == first.next_mask.tolist()  # next_mask is the next step's mask


def test_collector_passes_no_mask_where_the_env_has_none():
    from rl_playground.envs.gridworld import GridWorld

    calls = []
    collector = StepCollector(GridWorld(), lambda obs: calls.append(obs) or ActionOut(0), seed=0)
    assert collector.step().mask is None and calls


def test_masks_survive_batching_and_replay():
    t = Transition(0, 1, 0.0, 0, False, False, mask=np.array([True, False]), next_mask=np.array([False, True]))
    buffer = ReplayBuffer(4)
    buffer.add(t)
    batch = buffer.sample(3, np.random.default_rng(0))
    assert batch.masks.shape == (3, 2) and batch.masks.dtype == bool
    assert batch.next_masks.tolist() == [[False, True]] * 3
    assert Batch.from_transitions([Transition(0, 1, 0.0, 0, False, False)]).masks is None


def test_turn_based_driver_pairs_each_move_with_its_masks():
    driver = TurnBasedDriver(Nim(3), lambda agent, obs, mask: ActionOut(0), seed=0)
    emitted = []
    while driver.episodes == 0:
        emitted.extend(driver.step())
    first_a = next(t for agent, t in emitted if agent == "a")
    assert (first_a.obs, first_a.next_obs) == (3, 1)
    assert first_a.mask.tolist() == [True, True] and first_a.next_mask.tolist() == [True, False]


# --- your stubs ----------------------------------------------------------------------

LEGAL = np.array([False, True, True, False])


def test_mask4_greedy_picks_the_best_legal_move():
    from rl_playground.policies.greedy import greedy_action

    q = np.array([9.0, 1.0, 3.0, 9.0])  # the illegal moves look best
    assert greedy_action(q, LEGAL) == 2
    assert greedy_action(np.array([9.0, 3.0, 3.0, 9.0]), LEGAL) == 1  # ties to the lowest legal


def test_mask5_epsilon_greedy_spreads_epsilon_over_legal_moves_only():
    from rl_playground.policies.epsilon_greedy import EpsilonGreedy
    from rl_playground.policies.schedules import Constant

    policy = EpsilonGreedy(Constant(0.2), 4, seed=0)
    q = np.array([9.0, 1.0, 3.0, 9.0])
    np.testing.assert_allclose(policy.probs(q, 0, LEGAL), [0.0, 0.1, 0.9, 0.0])
    assert all(LEGAL[policy.act(q, t, LEGAL)] for t in range(500))


def test_mask6_q_learning_target_maxes_over_legal_next_moves():
    from rl_playground.bellman.sampled import q_learning_target

    next_q = np.array([[5.0, 1.0, 2.0, 5.0], [5.0, 1.0, 2.0, 5.0]])
    next_mask = np.array([LEGAL, [False] * 4])  # the second s' has no legal move
    y = q_learning_target(np.array([1.0, 1.0]), next_q, np.array([0.0, 0.0]), 0.5, next_mask=next_mask)
    np.testing.assert_allclose(y, [1.0 + 0.5 * 2.0, 1.0])
    np.testing.assert_allclose(q_learning_target(np.array([1.0]), next_q[:1], np.array([0.0]), 0.5), [3.5])


def test_mask6_double_q_selects_the_best_legal_move_online():
    from rl_playground.bellman.sampled import double_q_target

    online = np.array([[9.0, 1.0, 3.0, 9.0]])  # a* among legal = 2
    target = np.array([[0.0, 7.0, 4.0, 0.0]])
    y = double_q_target(np.array([0.0]), online, target, np.array([0.0]), 1.0, next_mask=LEGAL[None])
    np.testing.assert_allclose(y, [4.0])


def test_mask6_targets_take_torch_masks_too():
    torch = pytest.importorskip("torch")
    from rl_playground.bellman.sampled import q_learning_target

    y = q_learning_target(torch.tensor([1.0]), torch.tensor([[5.0, 1.0, 2.0, 5.0]]), torch.tensor([0.0]), 0.5,
                          next_mask=torch.tensor([LEGAL.tolist()]))
    assert torch.allclose(y, torch.tensor([2.0]))


def test_mask3_masked_logits_drop_illegal_moves_from_the_distribution():
    torch = pytest.importorskip("torch")
    from torch.distributions import Categorical

    from rl_playground.networks.heads import masked_logits

    logits = torch.tensor([[2.0, 0.0, 0.0, 5.0]])
    mask = torch.tensor([LEGAL.tolist()])
    dist = Categorical(logits=masked_logits(logits, mask))
    assert torch.allclose(dist.probs, torch.tensor([[0.0, 0.5, 0.5, 0.0]]))
    assert torch.allclose(dist.entropy(), torch.tensor([np.log(2.0)], dtype=torch.float32))
    assert torch.isfinite(dist.log_prob(torch.tensor([1]))).all()
    assert all(mask[0, dist.sample()].item() for _ in range(200))


def test_mask3_masked_logits_keep_gradients_finite():
    torch = pytest.importorskip("torch")
    from torch.distributions import Categorical

    from rl_playground.networks.heads import masked_logits

    logits = torch.tensor([[2.0, 0.0, 1.0, 5.0]], requires_grad=True)
    dist = Categorical(logits=masked_logits(logits, torch.tensor([LEGAL.tolist()])))
    (dist.log_prob(torch.tensor([2])) + dist.entropy()).sum().backward()
    assert torch.isfinite(logits.grad).all()
