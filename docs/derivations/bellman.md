# The Bellman equations, from the definition of return

## Setup

A finite MDP has states S, actions A, transitions P(s' | s, a), expected rewards R(s, a) and a
discount gamma in [0, 1). A policy pi(a | s) gives the return

```latex
G_t = \sum_{k \ge 0} \gamma^k r_{t+k}, \qquad V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]
```

## Expectation equation

Split off the first reward: G_t = r_t + gamma G_(t+1). Take expectations, condition on the
first action and next state, and use the Markov property, which makes
E[G_(t+1) | s_(t+1) = s'] = V^pi(s') whatever happened before:

```latex
V^\pi(s) = \sum_a \pi(a \mid s)\Big[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^\pi(s')\Big] = (T^\pi V^\pi)(s)
```

In matrix form, with P_pi and R_pi averaged over pi, V^pi = R_pi + gamma P_pi V^pi, so
V^pi = (I - gamma P_pi)^(-1) R_pi. The inverse exists because P_pi is stochastic and
gamma < 1, so gamma P_pi has spectral radius below 1. Iterative evaluation [DP-2] avoids the
O(S^3) solve.

## T^pi is a gamma-contraction

For any U and V,

```latex
|(T^\pi U)(s) - (T^\pi V)(s)| = \gamma \Big|\sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\,(U - V)(s')\Big| \le \gamma \|U - V\|_\infty
```

because the weights are non-negative and sum to 1. Banach's fixed-point theorem then gives a
unique fixed point, V^pi, and convergence from any start at rate gamma. Applying the bound to
V_(k+1) = T V_k and V_k = T V_(k-1) gives the per-sweep check [DP-8].

## Optimality equation

Define V*(s) = max over pi of V^pi(s). The optimal policy can be taken greedy with respect to
V*, which gives

```latex
V^*(s) = \max_a \Big[R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^*(s')\Big] = (T^* V^*)(s)
```

T* is also a gamma-contraction, using |max_a f(a) - max_a g(a)| <= max_a |f(a) - g(a)|.
So value iteration [DP-6] converges to V*, and from
||V* - V_(k+1)|| <= gamma ||V* - V_k|| <= gamma (||V* - V_(k+1)|| + ||V_(k+1) - V_k||):

```latex
\|V_{k+1} - V^*\|_\infty \le \frac{\gamma}{1-\gamma}\,\|V_{k+1} - V_k\|_\infty
```

## Policy improvement theorem

If Q^pi(s, pi'(s)) >= V^pi(s) for all s, then V^(pi') >= V^pi. Unroll:

```latex
V^\pi(s) \le Q^\pi(s, \pi'(s)) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid a_t = \pi'(s)] \le \mathbb{E}_{\pi'}[r_t + \gamma r_{t+1} + \gamma^2 V^\pi(s_{t+2})] \le \dots \le V^{\pi'}(s)
```

The greedy policy satisfies the premise by construction, so policy iteration [DP-5] improves
monotonically. When greedy improvement changes nothing, V^pi satisfies the optimality
equation, so pi is optimal.

## From exact to sampled

Every learner in `docs/tabular.md` replaces the sum over s' with one sampled s'. The TD target
r + gamma V(s') is an unbiased sample of (T^pi V)(s); the Q-learning target is a sample of the
inside of T*. Stochastic approximation (Robbins-Monro step sizes, every pair visited
infinitely often) turns the contraction into convergence for tabular Q-learning.
