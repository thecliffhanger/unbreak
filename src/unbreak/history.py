"""Per-call retry timeline tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RetryTimeline:
    """A single point in the retry timeline."""
    attempt: int
    wait_time: float
    error: Optional[BaseException] = None
    timestamp: float = field(default_factory=time.time)


class RetryHistory:
    """Tracks the timeline of a single retry operation."""

    def __init__(self) -> None:
        self._timeline: list[RetryTimeline] = []
        self._start_time: float = time.monotonic()

    def record(self, attempt: int, wait_time: float, error: Optional[BaseException] = None) -> None:
        """Record an attempt in the timeline."""
        self._timeline.append(RetryTimeline(
            attempt=attempt,
            wait_time=wait_time,
            error=error,
        ))

    @property
    def timeline(self) -> list[RetryTimeline]:
        return list(self._timeline)

    @property
    def retries(self) -> int:
        """Number of retries (attempts after the first)."""
        return max(0, len(self._timeline) - 1)

    @property
    def total_wait(self) -> float:
        """Total time spent waiting between attempts."""
        return sum(t.wait_time for t in self._timeline)

    @property
    def errors(self) -> list[BaseException]:
        """List of all errors encountered."""
        return [t.error for t in self._timeline if t.error is not None]

    @property
    def elapsed(self) -> float:
        """Total elapsed time since the history was created."""
        return time.monotonic() - self._start_time
