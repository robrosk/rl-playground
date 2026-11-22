# RL-Playground

Reinforcement learning from the Bellman equation up. Planning, tabular and deep methods share
one Bellman core and differ only in how they get data and how they store value, so every
algorithm is a few swappable parts on the same loop.

| Question | Seam | Lives in |
| --- | --- | --- |
| Is the model known? | planning with P and R, or learning from samples | `Planner` + `PlanningRunner`, or `Learner` + a trainer |
| How is value stored? | a table or a function approximator | `values/`: `TabularQ` or `NeuralQ` behind one interface |
| Whose data? | on-policy or off-policy | `OnPolicyTrainer` or `OffPolicyTrainer` |

Runners own time. Algorithms own math. The math itself is in `bellman/`.

## Tracing an algorithm

Each algorithm has a step map in `docs/`, and every step has an id such as `[DP-6]`,
`[TAB-5]`, `[PPO-3]` or `[DQN-4]`. The same id tags the docstring that implements it and the
test that checks it:

```bash
grep -rn "DP-6" docs rl_playground tests
```

`tests/test_step_ids.py` fails if a documented step is not tagged in the code.

| Family | Step map | Methods |
| --- | --- | --- |
| Planning | `docs/planning.md` | policy evaluation, policy iteration, value iteration, modified PI |
| Tabular | `docs/tabular.md` | TD(0), SARSA, Expected SARSA, Q-learning, Double Q, MC control |
| Policy gradient | `docs/policy_gradient.md`, `docs/ppo.md` | REINFORCE, A2C, PPO |
| Value-based deep | `docs/dqn.md` | DQN, Double DQN |
| Model-based | `docs/model_based.md` | Dyna-Q |

Derivations are in `docs/derivations/`. The design and its patterns are in
`docs/architecture.md`.

## Status

A scaffold. The plumbing is written and tested: value objects, configs, the registry,
callbacks, the `Runner` template, the runners, the environments, the MDP model and the
factory. Every piece of math and learning is a stub that raises `NotImplementedError` with
its contract in the docstring. The tests carry hand-computed expected values and fail until
the stub they cover is filled in.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The planning and tabular half needs only numpy. Torch is needed from DQN on.

## Build order

1. `bellman/exact.py`, then the four planners in `algorithms/planning/`. Tests:
   `test_bellman_exact.py`, `test_planning.py`.
2. `policies/`, `values/tabular.py`, `bellman/sampled.py`, then `algorithms/tabular/`. Tests:
   `test_policies.py`, `test_bellman_sampled.py`, `test_tabular.py`. Score each learner
   against V* from value iteration on the same Frozen Lake.
3. `collectors/`, `data/`, `networks/`, `values/neural.py`, `losses/td.py`, then
   `algorithms/off_policy/dqn.py`.
4. `bellman/returns.py`, `losses/ppo.py`, then `algorithms/on_policy/ppo.py` on Acrobot.
5. REINFORCE, A2C and Dyna-Q, each one new file on existing parts.

## Running

```bash
python scripts/run.py value_iteration --env frozen_lake
python scripts/run.py q_learning --env frozen_lake --budget 50000 --set alpha=0.2
python scripts/run.py ppo --env gym:Acrobot-v1 --budget 500000
```

## Layout

```text
rl_playground/
  core/          types, config, registry, callbacks, arrays
  envs/          Env interface, gym adapter, Frozen Lake, gridworld, wrappers
  mdp/           TabularMDP, from_env, random and hand-checkable MDPs
  bellman/       exact.py, sampled.py, returns.py
  values/        tabular and neural V and Q behind one interface
  networks/      encoders, heads, actor-critic, factory
  policies/      greedy, epsilon-greedy, softmax, schedules
  data/          rollout and replay buffers
  collectors/    env stepping with terminated and truncated kept apart
  losses/        PPO terms, TD losses, composite
  algorithms/    planning/, tabular/, on_policy/, off_policy/, model_based/
  runners/       the Runner template and its three subclasses
  eval/          greedy evaluation, distance to the optimum
  build.py       config to a ready runner
scripts/run.py
docs/
tests/
```
