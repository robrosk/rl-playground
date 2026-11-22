"""How to act given values (Strategy pattern): each rule is swappable behind act and probs."""

from rl_playground.policies.epsilon_greedy import EpsilonGreedy
from rl_playground.policies.greedy import greedy_action
from rl_playground.policies.schedules import Constant, ExponentialDecay, LinearDecay, Schedule
from rl_playground.policies.softmax import Softmax

__all__ = ["Constant", "EpsilonGreedy", "ExponentialDecay", "LinearDecay", "Schedule", "Softmax", "greedy_action"]
