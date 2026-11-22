# GAE as a sum of TD errors

## The TD error is a one-step advantage

With delta_t = r_t + gamma V(s_(t+1)) - V(s_t), if V = V^pi then
E[delta_t | s_t, a_t] = Q^pi(s_t, a_t) - V^pi(s_t) = A^pi(s_t, a_t).

## k-step advantages telescope

```latex
\hat A_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l} = -V(s_t) + r_t + \dots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
```

Small k: low variance, biased by the error in V. Large k: unbiased, high variance.

## The exponential average

GAE weights the k-step estimates by (1 - lam) lam^(k-1):

```latex
\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{k\ge1}\lambda^{k-1}\hat A_t^{(k)} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}
```

lam = 0 gives delta_t. lam = 1 gives the Monte Carlo advantage G_t - V(s_t). The sum has the
recursion A_t = delta_t + gamma lam A_(t+1), which `bellman/returns.py::gae` walks backwards.

## Episode ends

- **Terminated at t:** s_(t+1) is terminal, so V(s_(t+1)) = 0 and nothing after t belongs to
  this episode. Drop the bootstrap in delta_t and cut the recursion.
- **Truncated at t:** s_(t+1) is not terminal; the episode was only cut. Keep
  gamma V(s_(t+1)) in delta_t, using the final observation, and cut the recursion, because
  t + 1 in the buffer is a new episode.
- **End of the rollout:** like truncation, with next_values[T-1] = V(s_T).

The old PPO bootstrapped across the truncation boundary into the next episode's first state.

## Returns for the critic

R_t = A_t + V(s_t) is the lambda-return, the critic's regression target.
