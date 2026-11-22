# DQN step map

Deep Q-Network (Mnih et al. 2015) with the Double DQN target (van Hasselt et al. 2016). This is
tabular Q-learning [TAB-5] with a network for Q, a replay buffer and a target network. Read
`algorithms/off_policy/dqn.py` alongside this page.

| Step | What | Where | Test |
| --- | --- | --- | --- |
| DQN-1 | Epsilon-greedy action, epsilon on env steps | `policies/epsilon_greedy.py`, `policies/schedules.py` | `test_policies.py` |
| DQN-2 | Store the transition with the chosen action | `collectors/collector.py`, `data/replay_buffer.py` | `test_buffers_and_collector.py` |
| DQN-3 | After warmup, every k env steps, sample a minibatch | `runners/off_policy.py` | |
| DQN-4 | Double DQN target | `bellman/sampled.py::double_q_target` | `test_bellman_sampled.py` |
| DQN-5 | Huber loss on Q(s, a) - y | `losses/td.py::HuberTDLoss`, `values/neural.py` | |
| DQN-6 | Gradient step with grad-norm clipping | `values/neural.py::NeuralQ.update` | |
| DQN-7 | Target network sync (hard every N updates, or Polyak) | `values/neural.py::NeuralQ.sync_target` | |

## Equations

**[DQN-1]** With probability epsilon(step) take a uniform random action, otherwise
argmax_a Q_online(s, a). The schedule is keyed on env steps, not gradient steps.

**[DQN-2]** Store (s, a, r, s', terminated, truncated) where a is the action the agent chose.
The old Frozen Lake stored the slipped action, which makes the lake look deterministic to Q
and hides slip risk.

**[DQN-4]** Vanilla target, for comparison:

```latex
y = r + \gamma (1 - d) \max_{a'} Q_{\text{target}}(s', a')
```

Double DQN target (what the old code used):

```latex
a^* = \arg\max_{a'} Q_{\text{online}}(s', a'), \quad y = r + \gamma (1 - d) Q_{\text{target}}(s', a^*)
```

d is `terminated` only. A truncated transition still bootstraps.

**[DQN-5]** L = mean over the batch of Huber_delta(Q_online(s, a) - y), with y detached.

**[DQN-7]** Hard: theta_target <- theta every `target_update_every` updates.
Polyak: theta_target <- tau theta + (1 - tau) theta_target every update.

On board and card games, pass the legal-move masks through: see `masking.md` ([MASK-4] to [MASK-6]).
