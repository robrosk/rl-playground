# Multi-agent training

Every agent in a game sits in a **seat**. A seat holds one learner, the agents it plays, and
the update schedule its algorithm family needs. Changing an agent's algorithm means changing
one line of the seating plan; the game loop, the drivers and the trainer never change.

```python
from rl_playground.build import build_multi_agent_runner
from rl_playground.multi_agent import MultiAgentRunConfig, SeatSpec

run = MultiAgentRunConfig("connect_four", seats=(
    SeatSpec("player_0", "ppo", {"lr": 3e-4}),
    SeatSpec("player_1", "dqn", {"warmup": 500}),
), budget=200_000)
build_multi_agent_runner(run).run(run.budget)
```

```bash
python scripts/run_multi.py --env connect_four --seat player_0=ppo --seat player_1=dqn
python scripts/run_multi.py --env tictactoe --seat "*=ppo:shared"            # self-play
python scripts/run_multi.py --env simple_tag --seat "adversary_*=ppo:shared" --seat "*=dqn"
python scripts/run_multi.py --env leduc_holdem --seat player_0=reinforce --seat "*=random"
```

## The seating plan

| Field | Meaning |
| --- | --- |
| `agents` | An agent name or a glob. Each agent goes to the first spec that matches it. |
| `algo` | Any name in `ALGORITHMS`, or a fixed policy in `POLICIES` (`random`, `first_legal`). |
| `overrides` | Changes to that algorithm's default config, per seat. |
| `shared` | One learner for every matched agent: parameter sharing for teammates, self-play for opponents. Without it, each matched agent gets its own learner. |
| `make` | `(obs_space, action_space, seed) -> learner`. Seats a learner that is not registered, a frozen copy of a trained agent, or a test fake. |

An agent no spec matches is an error, so nothing plays by accident. End the plan with
`SeatSpec("*", "random")` to fill the remaining seats.

A new algorithm needs no multi-agent code. Register it in `algorithms/__init__.py` with a
config in `core/config.py`, give it `family = "on_policy"` or `"off_policy"`, and it can sit
in any seat of any game.

## Steps

- **[MA-1]** Match agents to seats (`multi_agent/config.py::assign`). First match wins;
  a shared spec is one seat, any other spec is one seat per agent.
- **[MA-2]** A seat acts (`multi_agent/seats.py::Seat.act`). When the game has a legal-move
  mask, the learner gets it as `act(obs, explore, mask=...)`, and each transition carries
  `mask` and `next_mask`. What a learner does with them is in `docs/masking.md`.
- **[MA-3]** Parallel games (`multi_agent/drivers.py::ParallelDriver`): every live agent acts,
  the env steps once, and each agent that acted gets one transition. An agent the game
  stops without a verdict is marked truncated, so its value bootstraps.
- **[MA-4]** Turn-based games (`TurnBasedDriver`): a move is only judged once the others have
  replied, so its transition is held until that agent's next turn. `last()` then gives the
  reward it collected since it moved and the observation it now faces. This is the credit
  rule from PettingZoo's own AEC loop, and `test_trainer_credits_tictactoe_moves_like_pettingzoo`
  checks it.
- **[MA-5]** A finished agent still gets one last turn, which takes `step(None)`. That turn is
  where the loser of a game learns it lost.
- **[MA-6]** A shared on-policy seat keeps one stream per agent (`data/multi_stream.py`).
  Streams are laid end to end and each cut is marked truncated, so GAE never chains one
  agent's advantage into another's, and `bellman/returns.py` needs no change.
- **[MA-7]** After every step, each seat that is due updates (`runners/multi_agent.py`).
  Metrics carry the seat's name (`player_0/loss`), and `return/<agent>` is the mean return
  over the last 100 episodes.

## Schedules

The single-agent trainers' rules, moved into each seat (`multi_agent/schedules.py`):

| Learner family | Schedule | Updates when |
| --- | --- | --- |
| `on_policy` | `OnPolicySchedule` | the rollout reaches `rollout_len`, or every agent's episode has ended for Monte Carlo methods |
| `off_policy` | `OffPolicySchedule` | after `warmup`, every `train_every` of the seat's own transitions |
| `fixed` | `NoUpdates` | never |

Each seat counts its own transitions, so a PPO seat and a DQN seat in one game keep their
own clocks. One trainer step is one env step in a parallel game and one move in a
turn-based game, so `budget` means moves for chess and steps for Pistonball.

## Observations

The builder gives each seat the observations its learner needs. A deep learner on a
`Discrete` observation gets it one-hot; a tabular learner keeps the index. A tabular learner
on a vector observation (poker, chess) is refused with a message that names the seat.
Per-agent transforms live in `envs/multi_agent/wrappers.py` and reuse `envs/transforms.py`,
which the single-agent wrappers use too.

## Where to go next

- League play: snapshot a learner into a `PolicyLearner` every N updates and seat the
  snapshots as opponents.
- Centralised critics (MAPPO, MADDPG): a learner that sees every agent's observation. That
  is a new learner and a driver that passes the joint observation, behind the same seats.
- Evaluation: `eval/` against `first_legal` and `random` for board games.
