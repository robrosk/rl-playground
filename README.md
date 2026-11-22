# rl-playground

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
factory. So is the engineering boilerplate: the MLP and CNN encoders, the policy, value and Q
heads, `ActorCritic`, the replay and rollout buffers, the step collector, the decay schedules,
softmax exploration, and the forward passes of `NeuralQ` and `NeuralV`.

What is left is the RL itself, 43 stubs that raise `NotImplementedError` with their contract
in the docstring: the Bellman backups, targets and returns, the four planners, the tabular
updates, epsilon-greedy and greedy action choice, `NeuralQ.update` and `sync_target`, the TD
and PPO losses, and each algorithm's `act` and `update`. The tests carry hand-computed
expected values and fail until the stub they cover is filled in.

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
3. `NeuralQ.update` and `sync_target` in `values/neural.py`, `losses/td.py`, then
   `algorithms/off_policy/dqn.py`. The collector, buffers and networks are already written.
4. `bellman/returns.py`, `losses/ppo.py`, then `algorithms/on_policy/ppo.py` on Acrobot.
5. REINFORCE, A2C and Dyna-Q, each one new file on existing parts.

## Running

```bash
python scripts/run.py value_iteration --env frozen_lake
python scripts/run.py q_learning --env frozen_lake --budget 50000 --set alpha=0.2
python scripts/run.py ppo --env gym:Acrobot-v1 --budget 500000
```

Multi-agent games give each agent its own algorithm, or one shared learner for a group:

```bash
python scripts/run_multi.py --env connect_four --seat player_0=ppo --seat player_1=dqn
python scripts/run_multi.py --env tictactoe --seat "*=ppo:shared"   # self-play
```

See `docs/multi_agent.md`.

## Environments

65 environments by name: the hand-built tabular ones, Gymnasium's toy text, classic
control, Box2D, MuJoCo and Atari, and PettingZoo's board, card, cooperative and particle-world
games. Any other Gymnasium id runs as `gym:<id>`.

```bash
python scripts/list_envs.py
python scripts/run.py value_iteration --env taxi
python scripts/run.py dqn --env pong --budget 1000000
python scripts/run.py ppo --env connect_four --budget 200000   # first seat vs random
```

Suites beyond numpy and Gymnasium are pip extras (`box2d`, `mujoco`, `atari`, `pettingzoo`,
...), or `pip install -e ".[envs]"` for all of them. See `docs/environments.md`.

## Layout

```text
rl_playground/
  core/          types, config, registry, callbacks, arrays
  envs/          Env interface, catalogs, Gymnasium adapter, wrappers, Frozen Lake, gridworld
    multi_agent/ PettingZoo adapters, per-agent wrappers, single-agent views
  mdp/           TabularMDP, from_env, random and hand-checkable MDPs
  bellman/       exact.py, sampled.py, returns.py
  values/        tabular and neural V and Q behind one interface
  networks/      encoders, heads, actor-critic, factory
  policies/      greedy, epsilon-greedy, softmax, schedules
  data/          rollout and replay buffers
  collectors/    env stepping with terminated and truncated kept apart
  losses/        PPO terms, TD losses, composite
  algorithms/    planning/, tabular/, on_policy/, off_policy/, model_based/
  multi_agent/   seats, update schedules, parallel and turn-based drivers
  runners/       the Runner template and its subclasses, including MultiAgentTrainer
  eval/          greedy evaluation, distance to the optimum
  build.py       config to a ready runner
scripts/run.py, scripts/list_envs.py
docs/
tests/
```
