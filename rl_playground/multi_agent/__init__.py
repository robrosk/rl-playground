"""Training several agents at once, each seat with its own swappable algorithm.

* ``SeatSpec`` says which agents an algorithm controls and whether they share one learner.
* ``Seat`` pairs a learner with the ``UpdateSchedule`` its family needs (Strategy).
* A driver plays one step of a parallel or turn-based game and emits transitions (Strategy).
* ``runners.MultiAgentTrainer`` runs the loop; ``build.build_multi_agent_runner`` wires it.

See docs/multi_agent.md.
"""

from rl_playground.multi_agent.config import MultiAgentRunConfig, SeatSpec
from rl_playground.multi_agent.drivers import ParallelDriver, TurnBasedDriver
from rl_playground.multi_agent.schedules import NoUpdates, OffPolicySchedule, OnPolicySchedule, UpdateSchedule
from rl_playground.multi_agent.seats import POLICIES, PolicyLearner, Seat

__all__ = [
    "MultiAgentRunConfig",
    "NoUpdates",
    "OffPolicySchedule",
    "OnPolicySchedule",
    "POLICIES",
    "ParallelDriver",
    "PolicyLearner",
    "Seat",
    "SeatSpec",
    "TurnBasedDriver",
    "UpdateSchedule",
]
