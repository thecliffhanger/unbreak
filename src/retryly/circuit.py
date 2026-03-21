"""Circuit breaker with sliding window, predictive opening, and pattern learning."""

from __future__ import annotations

import enum
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from retryly.events import EventDispatcher, EventType, RetryEvent


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker with sliding window failure counting and predictive degradation detection.

    Args:
        failure_threshold: Failures in window before opening (default 5).
        recovery_timeout: Seconds before trying half-open (default 30).
        window_size: Sliding window size for failure counting (default 10).
        predictive: Enable predictive opening based on response time degradation (default False).
        predictive_threshold: Number of standard deviations for predictive trigger (default 2.0).
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        window_size: int = 10,
        predictive: bool = False,
        predictive_threshold: float = 2.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_size = window_size
        self.predictive = predictive
        self.predictive_threshold = predictive_threshold
        self._lock = threading.Lock()
        self._dispatcher = EventDispatcher()

        self._state = CircuitState.CLOSED
        self._window: deque[bool] = deque(maxlen=window_size)  # True=fail, False=success
        self._opened_at: Optional[float] = None
        self._half_open_permits: int = 0

        # Predictive tracking
        self._response_times: deque[float] = deque(maxlen=window_size * 3)
        self._response_time_ema: Optional[float] = None
        self._response_time_var: Optional[float] = None

        # Pattern learning: track which attempt number fails most
        self._attempt_fail_counts: dict[int, int] = {}
        self._attempt_total_counts: dict[int, int] = {}

        # Timing stats
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None

    def set_dispatcher(self, dispatcher: EventDispatcher) -> None:
        """Set the event dispatcher for circuit state transition events."""
        self._dispatcher = dispatcher

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        """Check if circuit should transition from OPEN → HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_permits = 1
                self._dispatcher.emit(RetryEvent(
                    type=EventType.CIRCUIT_HALF_OPEN,
                    function_name="",
                    circuit_state=CircuitState.HALF_OPEN,
                ))

    def _failures_in_window(self) -> int:
        return sum(1 for v in self._window if v)

    def allow(self) -> bool:
        """Check if a call is allowed through the circuit."""
        with self._lock:
            self._maybe_transition()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_permits > 0:
                    self._half_open_permits -= 1
                    return True
                return False
            return False  # OPEN

    def record_success(self, fn_name: str = "") -> None:
        with self._lock:
            self._maybe_transition()
            self._window.append(False)
            self._last_success_time = time.monotonic()
            old_state = self._state
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._half_open_permits = 0
            if old_state != CircuitState.CLOSED and self._state == CircuitState.CLOSED:
                self._dispatcher.emit(RetryEvent(
                    type=EventType.CIRCUIT_CLOSED,
                    function_name=fn_name,
                    circuit_state=CircuitState.CLOSED,
                ))

    def record_failure(self, attempt: int = 1, fn_name: str = "") -> None:
        with self._lock:
            self._window.append(True)
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            elif self._state == CircuitState.CLOSED:
                if self._failures_in_window() >= self._learned_threshold():
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    self._dispatcher.emit(RetryEvent(
                        type=EventType.CIRCUIT_OPEN,
                        function_name=fn_name,
                        circuit_state=CircuitState.OPEN,
                    ))

    def _learned_threshold(self) -> int:
        """Adjust threshold based on retry pattern learning."""
        threshold = self.failure_threshold
        reduced = False
        for attempt_num, fails in self._attempt_fail_counts.items():
            totals = self._attempt_total_counts.get(attempt_num, 1)
            if totals >= 3 and fails / totals > 0.8:
                if not reduced:
                    threshold = max(2, threshold - 1)
                    reduced = True
        return threshold

    def record_attempt(self, attempt: int, success: bool) -> None:
        """Record an attempt for pattern learning."""
        with self._lock:
            self._attempt_total_counts[attempt] = self._attempt_total_counts.get(attempt, 0) + 1
            if not success:
                self._attempt_fail_counts[attempt] = self._attempt_fail_counts.get(attempt, 0) + 1

    def record_response_time(self, elapsed: float) -> bool:
        """Record response time for predictive analysis. Returns True if degraded."""
        if not self.predictive:
            return False
        with self._lock:
            self._response_times.append(elapsed)
            n = len(self._response_times)
            if n < 5:
                return False

            # Capture pre-update values for spike detection
            old_ema = self._response_time_ema
            old_var = self._response_time_var

            if old_ema is None:
                mean = sum(self._response_times) / n
                old_ema = mean
                var = sum((x - mean) ** 2 for x in self._response_times) / n
                old_var = var
                self._response_time_ema = mean
                self._response_time_var = var
            else:
                alpha = 0.2
                delta = elapsed - old_ema
                self._response_time_ema = old_ema + alpha * delta
                self._response_time_var = (1 - alpha) * (old_var + alpha * delta ** 2)

            if old_var is not None and old_var >= 0:
                std = math.sqrt(old_var) if old_var > 0 else 0.0
                if std == 0.0:
                    # All prior observations were identical; any deviation is a spike
                    if elapsed > old_ema:
                        return True
                elif elapsed > old_ema + self.predictive_threshold * std:
                    return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._window.clear()
            self._opened_at = None
            self._half_open_permits = 0

    @property
    def stats(self) -> CircuitStats:
        with self._lock:
            return CircuitStats(
                state=self._state,
                consecutive_failures=self._failures_in_window(),
                total_failures=sum(1 for v in self._window if v),
                total_successes=sum(1 for v in self._window if not v),
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
            )
