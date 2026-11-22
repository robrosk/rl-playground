# Tabular learning step map

Learning from samples with a table (Sutton and Barto, chapters 5 and 6). The same targets in
`bellman/sampled.py` serve DQN later. Score every method against V* from value iteration on
the same Frozen Lake (`eval/compare_to_optimal.py`).

| Step | What | Where | Test |
| --- | --- | --- | --- |
| TAB-1 | Epsilon-greedy behaviour | `policies/epsilon_greedy.py`, `algorithms/tabular/base.py` | `test_policies.py` |
| TAB-2 | TD(0) prediction | `bellman/sampled.py::td_target`, `tabular/td0.py` | `test_tabular.py`, `test_bellman_sampled.py` |
| TAB-3 | SARSA | `sarsa_target`, `tabular/sarsa.py` | same |
| TAB-4 | Expected SARSA | `expected_sarsa_target`, `tabular/expected_sarsa.py` | same |
| TAB-5 | Q-learning | `q_learning_target`, `tabular/q_learning.py` | same |
| TAB-6 | Double Q-learning | `double_q_target`, `tabular/double_q.py` | `test_bellman_sampled.py` |
| TAB-7 | Monte Carlo control | `bellman/returns.py::discounted_returns`, `tabular/mc_control.py` | `test_returns.py` |
| TAB-8 | The tabular update rule | `values/tabular.py` | `test_tabular.py` |

## Equations

d = terminated. Truncated transitions still bootstrap.

**[TAB-1]** pi(a | s) = epsilon / |A| + (1 - epsilon) 1[a = argmax Q(s, .)]. Epsilon follows a
schedule on env steps.

**[TAB-2]** y = r + gamma (1 - d) V(s'). Prediction only: learns V of the behaviour policy.

**[TAB-3]** y = r + gamma (1 - d) Q(s', a'), with a' the action the agent then takes.
On-policy: learns Q of the epsilon-greedy policy, so it stays away from cliffs.

**[TAB-4]**

```latex
y = r + \gamma (1 - d) \sum_{a'} \pi(a' \mid s')\, Q(s', a')
```

Same expectation as SARSA, lower variance.

**[TAB-5]** y = r + gamma (1 - d) max_a' Q(s', a'). Off-policy: learns Q* while behaving
epsilon-greedily.

**[TAB-6]** a* = argmax Q_A(s', .), y = r + gamma (1 - d) Q_B(s', a*), then swap roles. Removes
the max operator's upward bias.

**[TAB-7]** G_t = r_t + gamma (1 - d_t) G_(t+1) over one whole episode; Q(s_t, a_t) moves
toward G_t.

**[TAB-8]** Q(s, a) <- Q(s, a) + alpha (y - Q(s, a)): a gradient step on (y - Q)^2 / 2 with the
target held fixed.
