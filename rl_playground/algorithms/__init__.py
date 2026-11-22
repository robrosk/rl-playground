"""Every algorithm by name (Registry pattern). Entries are lazy, so torch loads only for
deep methods."""

from rl_playground.core.registry import Registry

ALGORITHMS = Registry("algorithm")

_PATHS = {
    # planning: the model is known
    "policy_evaluation": "planning.policy_evaluation:PolicyEvaluation",
    "policy_iteration": "planning.policy_iteration:PolicyIteration",
    "value_iteration": "planning.value_iteration:ValueIteration",
    "modified_policy_iteration": "planning.modified_pi:ModifiedPolicyIteration",
    # tabular learning from samples
    "mc_control": "tabular.mc_control:MCControl",
    "td0": "tabular.td0:TD0",
    "sarsa": "tabular.sarsa:SARSA",
    "expected_sarsa": "tabular.expected_sarsa:ExpectedSARSA",
    "q_learning": "tabular.q_learning:QLearning",
    "double_q_learning": "tabular.double_q:DoubleQLearning",
    # deep
    "reinforce": "on_policy.reinforce:REINFORCE",
    "a2c": "on_policy.a2c:A2C",
    "ppo": "on_policy.ppo:PPO",
    "dqn": "off_policy.dqn:DQN",
    # model-based
    "dyna_q": "model_based.dyna_q:DynaQ",
}

for _name, _path in _PATHS.items():
    ALGORITHMS.add(_name, f"rl_playground.algorithms.{_path}")
