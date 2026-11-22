# Policy gradient step map

Methods that learn pi directly. PPO has its own page, `docs/ppo.md`.

## REINFORCE (`algorithms/on_policy/reinforce.py`)

**[RF-1]** G_t from `bellman/returns.py::discounted_returns` over one episode.

**[RF-2]**

```latex
\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,(G_t - b(s_t))\big]
```

Any baseline b that does not depend on the action leaves the gradient unbiased and can cut its
variance.

**[RF-3]** b(s) = V(s), fit by regressing on G_t.

## A2C (`algorithms/on_policy/a2c.py`)

**[A2C-1]** A_t = G_t^(n) - V(s_t) with n-step returns (`n_step_returns`), or GAE with lam < 1.

**[A2C-2]** Actor loss: -mean(log pi(a_t | s_t) A_t), with A_t held fixed.

**[A2C-3]** Total: actor + c_v mean (V(s_t) - G_t)^2 - c_h mean entropy, one clipped gradient step
per rollout.
