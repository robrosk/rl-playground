import numpy as np
import pytest

from rl_playground.core.types import Batch, Transition
from rl_playground.mdp.examples import two_state_mdp


@pytest.fixture
def mdp():
    """State 0: stay (r=1) or go to terminal state 1 (r=5). gamma = 0.9. V*(0) = 10."""
    return two_state_mdp(0.9)


def one_transition(s, a, r, s_next, terminated=False, truncated=False) -> Batch:
    return Batch.from_transitions([Transition(s, a, r, s_next, terminated, truncated)])


GO = np.array([[0.0, 1.0], [1.0, 0.0]])  # always "go" in state 0
UNIFORM = np.full((2, 2), 0.5)
