"""On-policy storage for one learner that controls several agents (parameter sharing).

Each agent's transitions form their own stream, in time order. ``get`` lays the streams end
to end, so the batch reads like one env's sequence of episodes and ``bellman.returns`` needs
no change. Where a stream stops mid-episode, its last row is marked truncated, so GAE
bootstraps from that agent's own next observation and never chains into the next stream.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from rl_playground.core.types import Batch, Transition


class MultiStreamRollout:
    """``capacity`` triggers the update rather than capping the size: one parallel step can
    add a transition for every agent, so a rollout can run a few rows past it."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._streams: Dict[str, List[Transition]] = {}

    def add(self, stream: str, transition: Transition) -> None:
        self._streams.setdefault(stream, []).append(transition)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    def __len__(self) -> int:
        return sum(len(s) for s in self._streams.values())

    def get(self) -> Batch:
        """[MA-6] Streams end to end, each cut marked truncated."""
        rows: List[Transition] = []
        for stream in self._streams.values():
            if not stream:
                continue
            last = stream[-1]
            if not (last.terminated or last.truncated):
                last = replace(last, truncated=True)
            rows.extend(stream[:-1])
            rows.append(last)
        return Batch.from_transitions(rows)

    def clear(self) -> None:
        self._streams = {}
