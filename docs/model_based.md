# Model-based step map

## Dyna-Q (`algorithms/model_based/dyna_q.py`)

Sutton and Barto, chapter 8. Learning and planning run on the same Q table: real experience
updates it directly and also trains a model that generates more updates.

**[DYNA-1]** Direct RL: the Q-learning update [TAB-5] on the real transition.

**[DYNA-2]** Model learning: Model(s, a) <- (r, s', terminated). Deterministic for now; a
stochastic model keeps counts instead.

**[DYNA-3]** Planning: n times, draw a previously seen (s, a), ask the model for (r, s'), and
apply the same Q-learning update.

With n = 0 this is Q-learning. As n grows, it moves toward value iteration on the learned
model, which is the bridge between `docs/tabular.md` and `docs/planning.md`.
