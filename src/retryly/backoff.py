"""Backoff strategies: fixed, exponential, adaptive."""

from __future__ import annotations

import math
import time
import threading
from abc import ABC, abstractmethod
from typing import Callable


class BackoffStrategy(ABC):
    """Base class for backoff strategies."""

    @abstractmethod
    def compute(self, attempt: int) -> float:
        """Return seconds to wait before attempt *attempt* (1-indexed)."""
        ...


class FixedBackoff(BackoffStrategy):
    """Constant delay backoff.

    Args:
        delay: Seconds between attempts (default 1.0).
    """

    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay

    def compute(self, attempt: int) -> float:
        return self.delay


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff: base * factor^(attempt-1), capped at *max_delay*.

    Args:
        base: Initial delay in seconds (default 0.5).
        factor: Multiplier per attempt (default 2.0).
        max_delay: Cap on delay (default 60.0).
    """

    def __init__(self, base: float = 0.5, factor: float = 2.0, max_delay: float = 60.0) -> None:
        self.base = base
        self.factor = factor
        self.max_delay = max_delay

    def compute(self, attempt: int) -> float:
        delay = self.base * (self.factor ** (attempt - 1))
        return min(delay, self.max_delay)


class AdaptiveBackoff(BackoffStrategy):
    """Learns optimal wait from actual successful response times via EMA.

    Stores per-function state keyed by function id.

    Args:
        base: Minimum delay (default 0.1).
        max_delay: Maximum delay (default 60.0).
        smoothing: EMA smoothing factor (default 0.3).
    """

    def __init__(self, base: float = 0.1, max_delay: float = 60.0, smoothing: float = 0.3) -> None:
        self.base = base
        self.max_delay = max_delay
        self.smoothing = smoothing
        self._state: dict[int, dict] = {}
        self._lock = threading.Lock()

    def _get_state(self, key: int) -> dict:
        with self._lock:
            if key not in self._state:
                self._state[key] = {"ema": self.base, "count": 0}
            return self._state[key]

    def compute(self, attempt: int) -> float:
        # Use hash of attempt as a pseudo-key if no function set
        state = self._get_state(id(self))
        delay = state["ema"] * (1.5 ** (attempt - 1))
        return min(delay, self.max_delay)

    def compute_for(self, attempt: int, func_id: int) -> float:
        """Compute delay keyed to a specific function."""
        state = self._get_state(func_id)
        if state["count"] == 0:
            delay = self.base * (1.5 ** (attempt - 1))
        else:
            delay = state["ema"] * (1.5 ** (attempt - 1))
        return min(delay, self.max_delay)

    def record_success(self, func_id: int, elapsed: float) -> None:
        """Update EMA with a successful call's elapsed time."""
        with self._lock:
            if func_id not in self._state:
                self._state[func_id] = {"ema": self.base, "count": 0}
            s = self._state[func_id]
            s["ema"] = self.smoothing * elapsed + (1 - self.smoothing) * s["ema"]
            s["count"] += 1


def get_backoff(
    strategy: str | BackoffStrategy,
    base: float = 0.5,
    factor: float = 2.0,
    max_delay: float = 60.0,
    delay: float = 1.0,
    smoothing: float = 0.3,
) -> BackoffStrategy:
    """Resolve a backoff strategy from name or instance."""
    if isinstance(strategy, BackoffStrategy):
        return strategy
    if strategy == "fixed":
        return FixedBackoff(delay=delay)
    if strategy == "exponential":
        return ExponentialBackoff(base=base, factor=factor, max_delay=max_delay)
    if strategy == "adaptive":
        return AdaptiveBackoff(base=base, max_delay=max_delay, smoothing=smoothing)
    raise ValueError(f"Unknown backoff strategy: {strategy!r}")
