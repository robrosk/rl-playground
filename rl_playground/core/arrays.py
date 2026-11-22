"""Three reductions that read the same on numpy arrays and torch tensors.

The Bellman core is written once and called by tabular methods (numpy) and deep methods
(torch). These helpers hide the one place the two libraries spell things differently.
"""

from __future__ import annotations

import numpy as np


def is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def amax(x, axis: int = -1):
    return x.amax(dim=axis) if is_torch(x) else np.max(x, axis=axis)


def argmax(x, axis: int = -1):
    return x.argmax(dim=axis) if is_torch(x) else np.argmax(x, axis=axis)


def take_along_last(x, index):
    """``x[..., index]`` row by row: for x of shape [B, A] and index of shape [B], shape [B]."""
    if is_torch(x):
        return x.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)
    index = np.asarray(index).astype(np.int64)
    return np.take_along_axis(np.asarray(x), index[..., None], axis=-1)[..., 0]
