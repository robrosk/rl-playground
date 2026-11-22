# PPO step map

Proximal policy optimization with GAE (Schulman et al. 2017; GAE: Schulman et al. 2016).
Read `algorithms/on_policy/ppo.py` alongside this page.

| Step | What | Where | Test |
| --- | --- | --- | --- |
| PPO-1 | Collect T steps with the current policy | `collectors/collector.py`, `runners/on_policy.py` | `test_buffers_and_collector.py` |
| PPO-2 | Values of the true next observations | `algorithms/on_policy/ppo.py::update` | via `test_returns.py` |
| PPO-3 | GAE advantages and returns | `bellman/returns.py::gae` | `test_returns.py` |
| PPO-4 | Normalize advantages | `algorithms/on_policy/ppo.py::update` | |
| PPO-5 | Re-evaluate log-probs, entropy, values per minibatch | `networks/actor_critic.py::evaluate` | |
| PPO-6 | Clipped surrogate objective | `losses/ppo.py::ClippedSurrogateLoss` | `test_deep.py` |
| PPO-7 | Clipped value loss | `losses/ppo.py::ClippedValueLoss` | `test_deep.py` |
| PPO-8 | Entropy bonus | `losses/ppo.py::EntropyBonus` | `test_deep.py` |
| PPO-9 | Weighted total, backward, clip grad norm, step | `losses/composite.py`, `ppo.py::update` | |
| PPO-10 | Discard the rollout | `runners/on_policy.py::after_update` | |

## Equations

**[PPO-1]** Store for each step: s_t, a_t, r_t, s_(t+1), terminated_t, truncated_t,
log pi_old(a_t | s_t), V_old(s_t).

**[PPO-2]** next_value_t = V(s_(t+1)), where s_(t+1) is the true next observation (the final
observation when the episode ended at t).

**[PPO-3]**

```latex
\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t), \quad d_t = \text{terminated}_t
```

```latex
\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t)(1 - u_t) \hat{A}_{t+1}, \quad u_t = \text{truncated}_t
```

```latex
\hat{R}_t = \hat{A}_t + V(s_t)
```

Terminated: no bootstrap, cut the recursion. Truncated: bootstrap from V(final obs), cut the
recursion. The old code did neither on truncation. Derivation: `docs/derivations/gae.md`.

**[PPO-4]** A <- (A - mean(A)) / (std(A) + 1e-8), over the whole rollout, before minibatching.

**[PPO-6]**

```latex
\rho_t = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t))
```

```latex
L^{\text{clip}} = -\mathbb{E}_t\left[\min(\rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]
```

**[PPO-7]**

```latex
V^{\text{clip}} = V_{\text{old}} + \text{clip}(V_\theta - V_{\text{old}}, -\epsilon_v, \epsilon_v)
```

```latex
L^{V} = \mathbb{E}_t\left[\max((V_\theta - \hat{R}_t)^2,\ (V^{\text{clip}} - \hat{R}_t)^2)\right]
```

No 0.5 factor, matching the old code. Set `value_clip=None` for plain MSE.

**[PPO-8]** L^H = -E_t[H(pi_theta(. | s_t))]. Minimizing it raises entropy.

**[PPO-9]** L = L^clip + c_v L^V + c_h L^H, then clip the global grad norm to `max_grad_norm`.

On board and card games, the policy masks illegal moves out of its logits: see `masking.md` ([MASK-3]).
