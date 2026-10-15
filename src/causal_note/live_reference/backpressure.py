"""Validated live overload policy for bounded causal inference."""

from __future__ import annotations

from collections import deque


class LiveBackpressure:
    """Allow at most two skipped inferences in any 32 calibrated hops."""

    def __init__(
        self,
        max_backlog_hops: int,
        window_hops: int = 32,
        max_skips: int = 2,
        hard_backlog_hops: int = 16,
    ) -> None:
        if max_backlog_hops < 0:
            raise ValueError("max_backlog_hops doit etre positif ou nul.")
        if window_hops < 1 or not 0 <= max_skips <= window_hops:
            raise ValueError("Budget de sauts invalide.")
        if hard_backlog_hops <= max_backlog_hops:
            raise ValueError("Le seuil dur doit depasser le backlog tolere.")
        self.max_backlog_hops = int(max_backlog_hops)
        self.window_hops = int(window_hops)
        self.max_skips = int(max_skips)
        self.hard_backlog_hops = int(hard_backlog_hops)
        self.history: deque[bool] = deque(maxlen=self.window_hops)

    def decide(self, queue_depth: int) -> bool:
        depth = int(queue_depth)
        if depth > self.hard_backlog_hops:
            raise RuntimeError(
                f"Backlog audio dangereux: {depth} hops "
                f"> {self.hard_backlog_hops}."
            )
        skip = bool(
            depth > self.max_backlog_hops
            and sum(self.history) < self.max_skips
        )
        self.history.append(skip)
        return skip

    @property
    def recent_skip_percent(self) -> float:
        if not self.history:
            return 0.0
        return 100.0 * sum(self.history) / len(self.history)
