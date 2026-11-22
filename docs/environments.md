# Environments

Every environment reaches an algorithm through the same `Env` interface: `reset(seed)` and
`step(action) -> StepResult`, with `Discrete` or `Box` spaces. Names are in `envs.ENVS`, and
`scripts/run.py --env <name>` takes any of them.

```bash
python scripts/list_envs.py              # every env, and whether its packages are installed
python scripts/list_envs.py --installed  # only the ones you can run now
pip install -e ".[envs]"                 # every suite at once
```

## Single-agent (Gymnasium)

| Suite | Extra | Envs | Good for |
| --- | --- | --- | --- |
| hand-built | none | `frozen_lake`, `frozen_lake_det`, `frozen_lake_8x8`, `gridworld` | planning and tabular, exact V* to score against |
| toy text | none | `taxi`, `cliff_walking`, `cliff_walking_slippery`, `gym_frozen_lake`, `gym_frozen_lake_8x8`, `blackjack` | tabular control; all but Blackjack export a model for planning |
| classic control | none | `cartpole`, `acrobot`, `mountain_car`, `mountain_car_continuous`, `pendulum` | first DQN and PPO runs, sparse reward (Mountain Car) |
| Box2D | `box2d` | `lunar_lander`, `lunar_lander_continuous`, `bipedal_walker`, `bipedal_walker_hardcore`, `car_racing`, `car_racing_discrete` | harder control, pixels (Car Racing) |
| MuJoCo | `mujoco` | `inverted_pendulum`, `inverted_double_pendulum`, `reacher`, `swimmer`, `hopper`, `half_cheetah`, `walker2d`, `ant`, `pusher`, `humanoid`, `humanoid_standup` | continuous-action PPO |
| Atari | `atari` | `pong`, `breakout`, `space_invaders`, `seaquest`, `beam_rider`, `qbert`, `enduro`, `ms_pacman`, `asteroids`, `montezuma_revenge` | DQN from pixels, exploration (Montezuma) |

Anything else Gymnasium registers runs as `gym:<id>`, for example `--env gym:Acrobot-v1`.

Catalog names leave out the version: the newest installed one is used, so a Gymnasium release
that renames `Taxi-v3` to `Taxi-v4` changes nothing here.

What the adapter does on the way in:

- **Tuple and MultiDiscrete observations** (Blackjack) become one `Discrete` index, so tabular
  learners run on them.
- **Toy-text models.** `taxi`, `cliff_walking` and the Gymnasium lakes publish their
  transition table. `from_env` turns it into a `TabularMDP` with one extra absorbing end
  state, so `value_iteration --env taxi` gives the exact V* every learner should approach.
- **Atari** gets the standard DQN preprocessing: 84x84 greyscale, a frame skip of 4, and the
  last 4 frames stacked into `[4, 84, 84]`.
- **Pixel envs** (Car Racing) are made channel-first, `[C, H, W]`, for `CNNEncoder`.

## Multi-agent (PettingZoo)

| Suite | Extra | Envs | Kind |
| --- | --- | --- | --- |
| classic | `pettingzoo` | `tictactoe`, `connect_four`, `go_9x9`, `go` | turn-based board games, legal-move masks |
| classic | `pettingzoo-chess` | `chess` | turn-based |
| classic | `pettingzoo-cards` | `leduc_holdem`, `texas_holdem`, `texas_holdem_no_limit` | turn-based, hidden information |
| classic | `pettingzoo` | `rock_paper_scissors` | simultaneous moves |
| butterfly | `pettingzoo-butterfly` | `pistonball`, `cooperative_pong`, `knights_archers_zombies` | cooperative, many agents, pixels |
| SISL | `box2d`, `pettingzoo-butterfly` | `multiwalker`, `pursuit` | cooperative, continuous (Multiwalker) |
| MPE | `pettingzoo-mpe` | `simple`, `simple_spread`, `simple_adversary`, `simple_tag`, `simple_push`, `simple_reference`, `simple_speaker_listener`, `simple_crypto`, `simple_world_comm` | particle worlds: cooperation, competition, communication |

PettingZoo has two APIs, and the library mirrors both in `envs/multi_agent/base.py`:

- `ParallelEnv`: every live agent acts at once. `step` takes `{agent: action}` and returns
  `{agent: StepResult}`.
- `TurnBasedEnv`: one agent acts at a time. `last()` gives the current agent its observation
  and the reward it has collected since it last moved, which is how a move gets credit when
  the opponent's reply decides the game.

`MULTI_AGENT_ENVS.make("connect_four")` returns the multi-agent env itself. The same name in
`ENVS` returns a single-agent view of it, so every algorithm in the library runs on it today:

```python
from rl_playground.envs import ENVS

env = ENVS.make("connect_four")                                    # first seat vs random
env = ENVS.make("connect_four", agent="player_1", others=my_bot)   # second seat vs a bot
```

`others` is any function `(observation, mask) -> action` (Strategy pattern), so the opponents
can be random, scripted, or a frozen copy of the learner for self-play. In turn-based games
the view plays the others until it is the learner's turn again, and a game the others end
before the learner moves is skipped. The legal-move mask is in `info["action_mask"]`. The
learners do not use it yet; an illegal move ends the game with a loss, so they learn the
rules from reward until masking is added.

The views train one agent. Training several at once (independent learners, parameter
sharing, self-play leagues) needs a multi-agent runner on `ParallelEnv` and `TurnBasedEnv`,
which is the next piece to build here.

## Adding an environment

One line in a catalog:

```python
# envs/catalog.py
("gym_frozen_lake_dry", "FrozenLake", {"model": True, "kwargs": {"is_slippery": False}}),

# envs/multi_agent/catalog.py
("simple_tag", "mpe2.simple_tag"),
```

`tests/test_env_catalog.py` then builds it, resets it and steps it with random legal actions
whenever its packages are installed.

## Patterns

| Pattern | Where | Why |
| --- | --- | --- |
| Registry | `envs/__init__.py` | Names to builders, filled from the catalogs. |
| Factory | `EnvSpec.make`, `MultiAgentSpec.make` | One spec builds the env with its preset wrappers; no `if env_name == ...` anywhere. |
| Adapter | `envs/gym_adapter.py`, `envs/multi_agent/pettingzoo_adapter.py`, `envs/multi_agent/single_agent.py` | Gymnasium and PettingZoo behind the library's spaces; a multi-agent game behind `Env`. |
| Decorator | `envs/wrappers.py` | `ChannelFirst`, `FlattenObservation`, `OneHot`, `TimeLimit` stack on any env. |
| Strategy | `others` in the single-agent views | Opponents and teammates are swappable policies. |
| Value object | `EnvSpec`, `MultiAgentSpec` | Each environment is a frozen record, so the catalog reads as a table. |
