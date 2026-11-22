# Legal-move masks

Board and card games allow only some actions in each state. PettingZoo reports them as an
action mask, and an illegal move usually ends the game as a loss. With masks, an agent never
chooses an illegal move, and its value targets never count one.

The wiring is done. The masked math is yours, in four stubs, each with a test in
`tests/test_masking.py`.

## Where the mask travels (done)

- **[MASK-1]** Env to learner. Single-agent views put the current agent's mask in
  `info["action_mask"]`; `StepCollector` reads it and calls `act(obs, mask=mask)`. The
  multi-agent drivers read `env.action_mask(agent)` and the seat passes it on. Envs without
  masks pass nothing, so Frozen Lake and CartPole are unchanged.
- **[MASK-2]** With every transition. `Transition.mask` is the mask in `obs` and
  `Transition.next_mask` the mask in `next_obs`. Batches stack them as `masks` and
  `next_masks` ([B, A] bool), so replay and rollouts carry them for free. Both are None when
  the env has no masks.

Every learner's `act` takes `mask=None`. `ActorCritic.act`, `distribution` and `evaluate`
take a mask and hand it to `CategoricalHead`, which calls `masked_logits`.

## The math (yours)

- **[MASK-3]** `networks/heads.py::masked_logits`. Push illegal logits out of the softmax.
  This one function covers PPO, A2C and REINFORCE: sampling, log-probs and entropy all come
  from the masked `Categorical`. In `PPO.update`, pass `batch.masks` to `evaluate`, so the new
  log-probs come from the same masked distribution as the old ones, or the ratio is wrong.
- **[MASK-4]** `policies/greedy.py::greedy_action(q_row, mask)`. Argmax over legal moves,
  ties to the lowest legal index.
- **[MASK-5]** `policies/epsilon_greedy.py`. Epsilon spread over the legal moves only:
  epsilon / A_legal on each, 1 - epsilon more on the greedy legal one, 0 on illegal ones.
  Expected SARSA gets the masked pi for free through `probs`.
- **[MASK-6]** `bellman/sampled.py`. `q_learning_target(..., next_mask)` takes the max over
  moves legal in s', and `double_q_target(..., next_mask)` picks a* among them. Pass
  `batch.next_masks` from Q-learning, Double Q and DQN. Without the mask, the target
  bootstraps from a move the agent can never make, which inflates Q.

One edge case: a row whose `next_mask` has no legal move. It only happens at the end of a
game, so give it no bootstrap, as if it were terminated. The tests check this.

## Why masks and not a penalty

Letting illegal moves lose teaches the agent the rules eventually, but it spends most of its
early experience on them, and the value of a state mixes "good position" with "easy to blunder
here". Masking takes the rules as given and spends all of the learning on strategy.
