# Architecture

Every method in RL-Playground is a Bellman backup. Dynamic programming applies the exact
backup with a known model. TD, SARSA and Q-learning apply a sampled backup. DQN applies the
sampled backup with a network. So the library is cut along the three questions that
separate methods, and everything else is shared.

## Three seams

1. **Is the model known?** A `Planner` reads `TabularMDP.P` and `R` and sweeps over every
   state (`PlanningRunner`). A `Learner` sees one transition at a time (`OnPolicyTrainer`,
   `OffPolicyTrainer`).
2. **How is value stored?** Algorithms call a `VFunction` or `QFunction`. `TabularQ` and
   `NeuralQ` implement the same interface, so Q-learning and DQN differ in the value store,
   the buffer and the target sync, not in the target.
3. **Whose data?** On-policy runners collect a rollout with the current policy, update once
   and discard it. Off-policy runners keep a buffer and sample from it.

4. **How many learners?** `MultiAgentTrainer` runs several, one per seat, each with the
   schedule its family needs. See `multi_agent.md`.

Runners own time: when to collect, when to update, when to report. Algorithms own math, and
the math is in `bellman/`.

## Flow

```text
scripts/run.py -> build.build_runner(RunConfig, algo config)
    ALGORITHMS.get(name)                      Registry
    env from ENVS, wrapped                    Adapter, Decorator
    Planner(mdp, cfg)  or  Learner(spaces, cfg)
    PlanningRunner / OnPolicyTrainer / OffPolicyTrainer
Runner.run(budget)                            Template Method
    step()            sweep, or collector.step() into a buffer      Iterator
    ready_to_update() every sweep, rollout full, or warmup and train_every
    update()          planner.sweep metrics, or learner.update(batch)
    after_update()    clear the rollout
    callbacks.on_update(t, metrics)           Observer

scripts/run_multi.py -> build.build_multi_agent_runner(MultiAgentRunConfig)
    assign(agents, SeatSpecs)                 first match wins
    Seat(learner, schedule_for(learner))      Strategy per seat
    ParallelDriver / TurnBasedDriver          Strategy per game shape
MultiAgentTrainer.run(budget)                 the same Template Method
    step()            driver.step() -> (agent, Transition) -> seat.record
    ready_to_update() any seat due
    update()          every due seat updates; metrics prefixed by seat
```

## Design patterns

| Pattern | Where | Why it fits |
| --- | --- | --- |
| Bridge | `values/` | Algorithms ask a value interface for values and updates. `TabularQ` and `NeuralQ` implement it, so one algorithm runs on a table or a network. |
| Strategy | `bellman/`, `policies/`, planners, learners, `multi_agent/` | Each backup, exploration rule and algorithm is swappable behind one call signature; each seat picks its learner and update schedule, and the trainer picks a parallel or turn-based driver. |
| Template Method | `runners/base.py` | `Runner.run` is written once; subclasses, `MultiAgentTrainer` included, say what a step is and when an update is due. |
| Factory | `build.py`, `networks/factory.py`, `multi_agent/schedules.py::schedule_for` | Builds runners from configs and networks from spaces, instead of `if is_image_obs` branches. |
| Registry | `core/registry.py`, `algorithms/__init__.py`, `envs/__init__.py`, `envs/catalog.py` | Names to classes, lazily, so torch loads only for deep methods; each environment is one catalog line. |
| Adapter | `envs/gym_adapter.py`, `envs/multi_agent/`, `mdp/from_env.py`, `multi_agent/seats.py::PolicyLearner` | Gymnasium, PettingZoo and custom envs share one `Env`; a known env becomes a `TabularMDP`; a fixed policy sits in a seat like a learner. |
| Decorator | `envs/wrappers.py`, `envs/multi_agent/wrappers.py` | `OneHot`, `TimeLimit`, `ChannelFirst` and `FlattenObservation` stack around any env without changing it; per-agent transforms do the same for one agent of a game. |
| Composite | `losses/composite.py` | PPO's loss is a weighted sum of terms, each reporting its own metrics. |
| Iterator | `collectors/`, `data/` | Collectors yield transitions; buffers yield minibatches. |
| Observer | `core/callbacks.py` | Console, CSV, checkpoints and the contraction check subscribe to runner events. |
| Dependency injection | every constructor, `build.py` | Parts arrive through constructors, so tests pass fakes and configs swap parts. |
| Value object | `core/types.py`, `core/config.py`, `mdp/tabular_mdp.py` | Frozen dataclasses replace positional tuples and magic constants. |

## Conventions

- **Episode ends.** Every step records `terminated` (true terminal state: no bootstrap) and
  `truncated` (time limit: bootstrap from the final observation) separately, end to end.
- **`next_obs` is the true next observation.** On an episode end the collector stores the
  final observation, then resets. It never stores the reset observation as `next_obs`.
- **Terminal states are absorbing with zero reward** in every `TabularMDP`, so V(terminal) = 0
  under every policy and no backup needs a special case.
- **One Bellman core.** Sampled targets work on numpy and torch, so tabular Q-learning and DQN
  call the same `q_learning_target`.
- **Heads return distributions**, so callers never branch on discrete versus continuous.
- **No dropout in RL networks.** Rollout and update must see the same function.
- **Step ids.** `[DP-n]`, `[TAB-n]`, `[RF-n]`, `[A2C-n]`, `[PPO-n]`, `[DQN-n]` and `[DYNA-n]`
  tag the same step in the docs, the docstrings and the test names.

## Where the old code went

| Old code | New home |
| --- | --- |
| PPO GAE in `utilities.py` | `bellman/returns.py` `gae()` |
| PPO `Policy` loss terms | `losses/ppo.py` |
| PPO `Policy` update and optimizer | `algorithms/on_policy/ppo.py` |
| PPO `NeuralNetwork.py` | `networks/` encoders, heads, `actor_critic.py`, factory |
| PPO `ReinforcementLearningLoop.py` | `runners/on_policy.py`, `collectors/` |
| PPO `Environment.py` | `envs/gym_adapter.py` |
| Constants in both `main.py` files | `core/config.py` |
| Logging and checkpoints | `core/callbacks.py` |
| DQN Keras network in `agent.py` | `values/neural.py` `NeuralQ`, ported to PyTorch |
| DQN target loop in `agent.py` | `bellman/sampled.py` `double_q_target()` |
| DQN target sync and epsilon | `algorithms/off_policy/dqn.py`, `policies/` |
| DQN `replay_buffer.py` | `data/replay_buffer.py` |
| DQN custom `environment.py` | `envs/frozen_lake.py`, behind `Env` |
| DQN `train.py` | `runners/off_policy.py` |
| `main.py` argparse | `scripts/run.py`, `build.py` |

## Bugs the old code had, and where the fix lives

| Bug | Fix lives in | Guarded by |
| --- | --- | --- |
| PPO bootstrapped across episodes on truncation | `collectors/`, `bellman/returns.py` | `test_returns.py::test_ppo3_truncation_*`, `test_buffers_and_collector.py` |
| PPO dropout active during rollout and update | `networks/encoders.py` (no dropout) | convention above |
| PPO full-batch epochs | `data/rollout_buffer.py::iterate_minibatches` | `test_rollout_minibatches_*` |
| DQN stored the slipped action | `envs/frozen_lake.py` reports the chosen action | `test_frozen_lake_reports_the_chosen_action` |
| DQN epsilon decayed per gradient step | `policies/` keyed on env steps | `test_policies.py` |
