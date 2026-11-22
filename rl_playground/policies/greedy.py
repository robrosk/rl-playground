from __future__ import annotations

import numpy as np


def greedy_action(q_row, mask=None) -> int:
    """argmax_a Q(s, a) for one state's action values, ties to the lowest index. Same rule as [DP-4].

    [MASK-4] With ``mask`` (bool [A]), the argmax is over legal actions only: the best legal
    move, even when an illegal one has a higher value, ties to the lowest legal index.
    """
    raise NotImplementedError
