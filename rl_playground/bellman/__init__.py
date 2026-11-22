"""The Bellman core. Every method in the library is one of these backups.

exact.py    the model is known: average over P (dynamic programming)
sampled.py  one sample stands in for the average (TD, SARSA, Q-learning, DQN)
returns.py  many samples chained: Monte Carlo, n-step, lambda-returns, GAE
"""
