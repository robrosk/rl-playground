# Planning step map

Dynamic programming with a known model (Sutton and Barto, chapter 4). Every planner reads a
`TabularMDP` and calls `bellman/exact.py`. Derivations: `docs/derivations/bellman.md`.

| Step | What | Where | Test |
| --- | --- | --- | --- |
| DP-1 | Bellman expectation backup T^pi | `bellman/exact.py::expectation_backup` | `test_bellman_exact.py` |
| DP-2 | Iterative policy evaluation | `algorithms/planning/policy_evaluation.py` | `test_planning.py::test_dp2_*` |
| DP-3 | Q from V with the model | `bellman/exact.py::q_from_v` | `test_bellman_exact.py` |
| DP-4 | Greedy policy improvement | `bellman/exact.py::greedy_policy` | `test_bellman_exact.py` |
| DP-5 | Policy iteration | `algorithms/planning/policy_iteration.py` | `test_planning.py::test_dp5_*` |
| DP-6 | Bellman optimality backup, value iteration | `bellman/exact.py::optimality_backup`, `value_iteration.py` | `test_planning.py::test_dp6_*` |
| DP-7 | Modified policy iteration | `algorithms/planning/modified_pi.py` | `test_pi_vi_and_mpi_agree_*` |
| DP-8 | Contraction check every sweep | `core/callbacks.py::ContractionMonitor` | `test_runner_template.py` |

## Equations

**[DP-1]**

```latex
(T^\pi V)(s) = \sum_a \pi(a \mid s)\Big[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V(s')\Big]
```

**[DP-2]** V_(k+1) = T^pi V_k from V_0 = 0, until ||V_(k+1) - V_k||_inf < tol. Converges to
V^pi, the unique fixed point of T^pi.

**[DP-3]**

```latex
Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V(s')
```

**[DP-4]** pi'(s) = argmax_a Q(s, a), ties to the lowest index. The policy improvement
theorem gives V^(pi') >= V^pi in every state.

**[DP-5]** Evaluate pi exactly, improve it, repeat until it stops changing. Finitely many
deterministic policies and monotone improvement mean it ends.

**[DP-6]**

```latex
(T^* V)(s) = \max_a \Big[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V(s')\Big]
```

Stopping when ||V_(k+1) - V_k||_inf < eps gives ||V_(k+1) - V*||_inf <= eps gamma / (1 - gamma).

**[DP-7]** Apply T^pi k times, then improve. k = 1 is value iteration; k to infinity is
policy iteration.

**[DP-8]**

```latex
\|V_{k+1} - V_k\|_\infty \le \gamma \|V_k - V_{k-1}\|_\infty
```

Holds for policy evaluation and value iteration on every sweep. A violation means the backup
is wrong.

## Worked example

`mdp/examples.py::two_state_mdp`, gamma = 0.9. State 0: stay for reward 1, or go to the
terminal state for reward 5.

- Value iteration from V = 0: V(0) goes 5, 5.5, 5.95, ... to V*(0) = 10. Changes 5, 0.5, 0.45.
- Uniform policy: V(0) = 0.5 (1 + 0.9 V(0)) + 2.5, so V(0) = 3 / 0.55.
- Policy iteration from "go": V = 5, improve to "stay", V = 10, stable after one improvement.
