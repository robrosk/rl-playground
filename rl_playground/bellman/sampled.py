"""Sampled Bellman targets: one observed (r, s') stands in for the average over P.

Every function works on numpy arrays and torch tensors alike, so tabular methods and deep
methods call the same code. Use ``rl_playground.core.arrays`` for max, argmax and gather.
``terminated`` is a float or bool array; truncated transitions still bootstrap, so they are
never passed here as ``terminated``. See docs/tabular.md and docs/dqn.md.

``next_mask`` ([B, A] bool, True where a' is legal in s') limits the max and the argmax to
legal moves. It is None when the env has no masks, and then nothing changes. A row with no
legal move contributes no bootstrap (as if terminated); it only happens at a game's end.
See docs/masking.md.
"""

from __future__ import annotations


def td_target(rewards, next_values, terminated, gamma: float):
    """[TAB-2] y = r + gamma * (1 - terminated) * next_value."""
    raise NotImplementedError


def q_learning_target(rewards, next_q, terminated, gamma: float, next_mask=None):
    """[TAB-5] y = r + gamma * (1 - terminated) * max_a' Q(s', a').

    ``next_q`` is [B, A]. This is also the vanilla DQN target of [DQN-4] when ``next_q``
    comes from the target network. [MASK-6] With ``next_mask``, the max is over legal a'.
    """
    raise NotImplementedError


def double_q_target(rewards, next_q_online, next_q_target, terminated, gamma: float, next_mask=None):
    """[TAB-6] [DQN-4] Select with one estimate, evaluate with the other.

    a* = argmax_a' Q_online(s', a');  y = r + gamma * (1 - terminated) * Q_target(s', a*).
    Decoupling selection from evaluation removes the max operator's upward bias.
    [MASK-6] With ``next_mask``, a* is the best legal a' under Q_online.
    """
    raise NotImplementedError


def sarsa_target(rewards, next_q_sa, terminated, gamma: float):
    """[TAB-3] y = r + gamma * (1 - terminated) * Q(s', a'), with a' the action actually taken."""
    raise NotImplementedError


def expected_sarsa_target(rewards, next_q, next_pi, terminated, gamma: float):
    """[TAB-4] y = r + gamma * (1 - terminated) * sum_a' pi(a' | s') Q(s', a').

    ``next_q`` and ``next_pi`` are [B, A]. With a greedy pi this is the Q-learning target.
    """
    raise NotImplementedError
